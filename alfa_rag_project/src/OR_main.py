"""
OpenRouter-optimized RAG pipeline.
Uses OpenRouter API for open-source models (Qwen, Llama, etc).
No local model downloads - API-based inference.
"""

import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    INDEX_PATH,
    QUESTIONS_CSV,
    SUBMISSION_CSV,
    WEBSITES_CSV,
    MAX_RESPONSE_WORDS,
    TEMPERATURE,
)
from chunker import chunk_all_websites
from generator import extract_answer_from_context
from indexer import build_and_save_index, load_index
from retriever import create_retriever

# ─────────────────────────────────────────────
# Логирование
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# OpenRouter модели
# ─────────────────────────────────────────────

OPENROUTER_MODELS = {
    "qwen2.5-7b": "qwen/qwen-2.5-7b-instruct",
    "qwen2-7b": "qwen/qwen-2-7b-instruct",
    "llama3-8b": "meta-llama/llama-3-8b-instruct",
    "mistral-7b": "mistralai/mistral-7b-instruct",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# System prompt
SYSTEM_PROMPT = """Ты - полезный ассистент. Отвечай максимально кратко: 1-2 предложения, без воды и лишних объяснений.
Не используй вводные фразы вроде "Вот что я нашел" или "На основе информации".
Сразу переходи к сути. Если информации недостаточно - скажи "Недостаточно информации".
""".strip()


class OpenRouterGenerator:
    """
    OpenRouter API генератор.
    Использует OpenAI-compatible API для open-source моделей.
    """

    def __init__(
        self,
        model: str = "qwen/qwen-2.5-7b-instruct",
        api_key: Optional[str] = None,
    ):
        """
        Initialize OpenRouter generator.

        Args:
            model: OpenRouter model identifier
            api_key: OpenRouter API key (or from OPENROUTER_API_KEY env)
        """
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY required. Set env var or pass api_key parameter."
            )

        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=self.api_key,
        )
        logger.info(f"OpenRouter client initialized for model: {model}")

    def generate(self, query: str, context: str) -> str:
        """
        Generate answer using OpenRouter API.

        Args:
            query: User question
            context: Retrieved context

        Returns:
            Generated answer
        """
        if not context:
            return "Недостаточно информации"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Вопрос: {query}\n\nКонтекст:\n{context}\n\nОтветь кратко на основе контекста.",
                    },
                ],
                temperature=TEMPERATURE,
                max_tokens=256,
            )

            answer = response.choices[0].message.content or ""
            answer = self._truncate_to_words(answer, MAX_RESPONSE_WORDS)

            return answer.strip()

        except Exception as e:
            logger.error("OpenRouter API failed, using fallback: %s", e)
            return extract_answer_from_context(query, context)

    def _truncate_to_words(self, text: str, max_words: int) -> str:
        """Truncate to max words."""
        words = text.split()
        if len(words) <= max_words:
            return text

        truncated = " ".join(words[:max_words])

        for punct in [".", "!", "?"]:
            last_punct = truncated.rfind(punct)
            if last_punct > len(truncated) * 0.5:
                return truncated[:last_punct + 1]

        return truncated


# ─────────────────────────────────────────────
# Кеш ответов
# ─────────────────────────────────────────────

class AnswerCache:
    """Персистентный кеш ответов."""

    def __init__(self, cache_path: Path):
        self._path = cache_path
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    self._data = json.load(f)
                logger.info("Cache loaded: %d entries", len(self._data))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Cache corrupted: %s", e)
                self._data = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _make_key(query: str, model: str) -> str:
        raw = f"{query.strip()}|{model}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def get(self, query: str, model: str) -> Optional[str]:
        key = self._make_key(query, model)
        entry = self._data.get(key)
        return entry["answer"] if entry else None

    def set(self, query: str, model: str, q_id: str, answer: str) -> None:
        key = self._make_key(query, model)
        self._data[key] = {"q_id": q_id, "query": query, "answer": answer, "model": model}
        self._save()

    def __len__(self) -> int:
        return len(self._data)


# ─────────────────────────────────────────────
# Валидация
# ─────────────────────────────────────────────

_STOP_WORDS = frozenset([
    "как", "что", "где", "когда", "почем", "зачем", "кто",
    "можно", "нужно", "надо", "это", "есть", "для", "при",
    "или", "если", "чтобы", "который", "которая", "которое",
    "в", "на", "с", "по", "из", "от", "до", "за", "к", "у",
])


def validate_answer(query: str, answer: str, min_overlap: int = 1) -> bool:
    """Проверяет релевантность ответа."""
    if not answer or not answer.strip():
        return False

    query_words = {
        w.strip(".,!?;:\"'()[]")
        for w in query.lower().split()
        if len(w) > 2 and w not in _STOP_WORDS
    }
    answer_words = {
        w.strip(".,!?;:\"'()[]")
        for w in answer.lower().split()
        if len(w) > 2 and w not in _STOP_WORDS
    }

    if not query_words:
        return True

    return len(query_words & answer_words) >= min_overlap


# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────

def run_pipeline(
    build_index: bool = False,
    llm_model: str = "qwen2.5-7b",
    cache_path: Path = Path("data/answer_cache.json"),
    validate_answers: bool = True,
    min_overlap: int = 1,
    api_key: Optional[str] = None,
) -> None:
    """
    Запускает RAG pipeline через OpenRouter.

    Args:
        build_index: Пересобрать индекс
        llm_model: Ключ модели из OPENROUTER_MODELS
        cache_path: Путь к кешу
        validate_answers: Валидация ответов
        min_overlap: Минимальный overlap
        api_key: OpenRouter API key
    """
    # ── Индекс ────────────────────────────────────────────────
    if build_index or not INDEX_PATH.exists():
        logger.info("Building index from scratch")
        websites_df = pd.read_csv(WEBSITES_CSV)
        websites_data = [(row["web_id"], row["text"]) for _, row in websites_df.iterrows()]
        chunks = chunk_all_websites(websites_data)
        logger.info("Created %d chunks", len(chunks))
        indexer = build_and_save_index(chunks)
    else:
        logger.info("Loading existing index")
        indexer = load_index()

    # ── Retriever ─────────────────────────────────────────────
    retriever = create_retriever(indexer)

    # ── Generator ─────────────────────────────────────────────
    or_model = OPENROUTER_MODELS.get(llm_model, llm_model)
    generator = OpenRouterGenerator(model=or_model, api_key=api_key)

    # ── Кеш ───────────────────────────────────────────────────
    cache = AnswerCache(cache_path)

    # ── Вопросы ───────────────────────────────────────────────
    questions_df = pd.read_csv(QUESTIONS_CSV)
    total = len(questions_df)

    results = []
    stats = {"cached": 0, "generated": 0, "failed": 0, "invalid": 0}
    CHECKPOINT_INTERVAL = 2000

    # Проверяем последний чекпоинт
    start_idx = 0
    for cp_num in [6000, 4000, 2000]:
        cp_path = SUBMISSION_CSV.parent / f"submission_checkpoint_{cp_num}.csv"
        if cp_path.exists():
            cp_df = pd.read_csv(cp_path)
            if len(cp_df) == cp_num:
                results = cp_df.to_dict("records")
                start_idx = cp_num
                logger.info("Resuming from checkpoint: %d answers", start_idx)
                break

    for idx, row in enumerate(tqdm(questions_df.iterrows(), total=total, desc="Generating")):
        if idx < start_idx:
            continue
        _, row = row
        q_id = str(row["q_id"])
        query = str(row["query"]).strip()

        # Кеш
        cached = cache.get(query, or_model)
        if cached:
            results.append({"q_id": q_id, "answer": cached})
            stats["cached"] += 1
            continue

        # Генерация
        try:
            context = retriever.get_context(query)
            answer = generator.generate(query, context)
        except Exception as e:
            logger.error("Failed q_id=%s: %s", q_id, e)
            answer = ""
            stats["failed"] += 1

        # Валидация
        if validate_answers and answer and not validate_answer(query, answer, min_overlap):
            stats["invalid"] += 1

        # Сохраняем
        if answer:
            cache.set(query, or_model, q_id, answer)

        results.append({"q_id": q_id, "answer": answer})
        stats["generated"] += 1

        # Чекпоинт
        if (idx + 1) % CHECKPOINT_INTERVAL == 0:
            cp_path = SUBMISSION_CSV.parent / f"submission_checkpoint_{idx + 1}.csv"
            pd.DataFrame(results).to_csv(cp_path, index=False)
            logger.info("Checkpoint saved: %d answers", idx + 1)

    # Финал
    logger.info(
        "Done | total=%d | cached=%d | generated=%d | failed=%d | invalid=%d",
        total, stats["cached"], stats["generated"], stats["failed"], stats["invalid"],
    )

    pd.DataFrame(results).to_csv(SUBMISSION_CSV, index=False)
    logger.info("Results: %s", SUBMISSION_CSV)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenRouter RAG Pipeline")

    parser.add_argument("--build-index", action="store_true", help="Build index from scratch")
    parser.add_argument(
        "--model",
        choices=list(OPENROUTER_MODELS.keys()),
        default="qwen2.5-7b",
        help="OpenRouter model",
    )
    parser.add_argument("--api-key", help="OpenRouter API key (or set OPENROUTER_API_KEY)")
    parser.add_argument("--no-validate", action="store_true", help="Disable validation")
    parser.add_argument("--min-overlap", type=int, default=1, help="Min word overlap")

    args = parser.parse_args()

    run_pipeline(
        build_index=args.build_index,
        llm_model=args.model,
        api_key=args.api_key,
        validate_answers=not args.no_validate,
        min_overlap=args.min_overlap,
    )