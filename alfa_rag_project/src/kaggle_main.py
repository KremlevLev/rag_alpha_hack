"""
Kaggle-optimized RAG pipeline.
Uses Hugging Face Inference API or local transformers for open-source LLMs.
Supports 2x T4 GPU for faster inference.
"""

import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import razdel
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.utils import is_flash_attn_2_available

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    INDEX_PATH,
    QUESTIONS_CSV,
    SUBMISSION_CSV,
    WEBSITES_CSV,
    MAX_SENTENCES,
    MAX_RESPONSE_WORDS,
    MAX_RESPONSE_CHARS,
    TEMPERATURE,
    LLM_TIMEOUT,
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
# Kaggle-оптимизированный генератор
# ─────────────────────────────────────────────

# Open-source модели, совместимые с Kaggle
KAGGLE_MODELS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2-7b": "Qwen/Qwen2-7B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
}

# Flag for int8 quantization (speeds up inference on T4 GPU)
USE_INT8: bool = True

# System prompt для русскоязычных моделей с few-shot примерами
SYSTEM_PROMPT = """Ты суровый банковский AI-аналитик. Отвечай на вопрос строго на основе предоставленного текста.

ПРАВИЛА:
1. Выдавай ТОЛЬКО факты из контекста. Никаких приветствий и лишних слов.
2. Если в контексте нет прямого ответа, выбери самое релевантное предложение из контекста.
3. Отвечай максимально емко. Объединяй длинные списки в одно-два предложения через запятую.
4. Твой ответ не должен превышать 3 предложений.

ПРИМЕРЫ:
Вопрос: Как узнать номер счёта?
Контекст: Номер счёта отображается в мобильном приложении банка на вкладке "Мои счета". Также вы можете позвонить в поддержку.
Ответ: Номер счёта отображается в мобильном приложении банка на вкладке "Мои счета".

Вопрос: Что такое БИК?
Контекст: БИК — это банковский идентификационный код, используемый для перечисления средств.
Ответ: БИК — это банковский идентификационный код, используемый для перечисления средств.

Вопрос: Как открыть вклад?
Контекст: Для открытия вклада необходимо подъехать в отделение банка с паспортом и заполнить форму.
Ответ: Для открытия вклада необходимо подъехать в отделение банка с паспортом и заполнить форму.
""".strip()


def truncate_to_chars(text: str, max_chars: int) -> str:
    """
    Truncate text to maximum character count.
    
    This is a SAFETY limit to prevent 3x length penalty.
    
    Args:
        text: Input text
        max_chars: Maximum number of characters
        
    Returns:
        Truncated text
    """
    if len(text) <= max_chars:
        return text
    
    truncated = text[:max_chars]
    
    for punct in [".", "!", "?", "»"]:
        last_punct = truncated.rfind(punct)
        if last_punct > max_chars * 0.3:
            return truncated[:last_punct + 1]
    
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.5:
        return truncated[:last_space]
    
    return truncated


def truncate_to_sentences(text: str, max_sentences: int) -> str:
    """
    Truncate text to maximum number of sentences.
    
    This is the PRIMARY truncation for BERT-Recall-L compliance.
    
    Args:
        text: Input text
        max_sentences: Maximum number of sentences
        
    Returns:
        Truncated text
    """
    sentences = [s.text.strip() for s in razdel.sentenize(text) if s.text.strip()]
    
    if len(sentences) <= max_sentences:
        return text
    
    return " ".join(sentences[:max_sentences])


class KaggleGenerator:
    """
    Hugging Face генератор для Kaggle.
    Автоматически использует доступные GPU.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize generator with Hugging Face model.

        Args:
            model_name: Hugging Face model identifier
            device_map: "auto" for multi-GPU, "cuda:0" for single GPU
            torch_dtype: float16 for T4 GPU (memory efficient)
        """
        self.model_name = model_name

        # Определяем количество доступных GPU
        num_gpus = torch.cuda.device_count()
        logger.info(f"Detected {num_gpus} GPU(s) on Kaggle")

        if num_gpus >= 2:
            # Два T4: используем model parallelism
            device_map = "auto"  # Автоматическое распределение по GPU
            logger.info("Using multi-GPU (2x T4) with device_map='auto'")
        elif num_gpus == 1:
            device_map = "cuda:0"
            logger.info("Using single GPU (T4)")
        else:
            device_map = "cpu"
            torch_dtype = torch.float32
            logger.warning("No GPU available, falling back to CPU (slow)")

        # Загружаем модель
        logger.info(f"Loading model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
        )

        # Устанавливаем pad_token если нужно
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model loading with int8 quantization for speed
        model_kwargs = {
            "device_map": device_map,
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
            "use_cache": True,
        }
        
        # Add int8 quantization for GPU (speeds up inference)
        if USE_INT8 and num_gpus >= 1:
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                model_kwargs["quantization_config"] = bnb_config
                logger.info("Using int8 quantization for faster inference")
            except ImportError:
                logger.warning("bitsandbytes not available, using float16")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )

        # Создаем pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )

        logger.info("Model loaded successfully")

    def generate(self, query: str, context: str) -> str:
        """
        Generate answer for query using context.

        Args:
            query: User question
            context: Retrieved context from retriever

        Returns:
            Generated answer (truncated to max sentences and chars)
        """
        if not context:
            return extract_answer_from_context(query, context)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Вопрос: {query}\n\nКонтекст:\n{context}\n\nОтветь кратко на основе контекста."},
        ]

        try:
            # Формируем промпт в зависимости от модели
            if "Qwen" in self.model_name:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            elif "Llama" in self.model_name:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                # Универсальный формат
                prompt = f"{SYSTEM_PROMPT}\n\nВопрос: {query}\n\nКонтекст:\n{context}\n\nОтвет:"

            # Генерируем
            outputs = self.pipe(
                prompt,
                max_new_tokens=64,  # Reduced for speed (answers limited to 2 sentences anyway)
                temperature=TEMPERATURE,
                do_sample=TEMPERATURE > 0,
                top_p=0.9 if TEMPERATURE > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            answer = outputs[0]["generated_text"]

            # Убираем промпт из ответа (для Qwen/Llama)
            if "Qwen" in self.model_name or "Llama" in self.model_name:
                # Ответ после последнего </
                if "</" in answer:
                    answer = answer.split("</")[-1]
                    # Убираем теги
                    for tag in ["system", "user", "assistant"]:
                        if f"<{tag}>" in answer:
                            answer = answer.split(f"<{tag}>")[-1]
                # Берем только сгенерированный текст
                if prompt in answer:
                    answer = answer[len(prompt):]

            # Очищаем и обрезаем
            answer = answer.strip()
            answer = truncate_to_sentences(answer, MAX_SENTENCES)
            answer = truncate_to_chars(answer, MAX_RESPONSE_CHARS)

            return answer

        except Exception as e:
            logger.error("Generation failed, using fallback: %s", e)
            return extract_answer_from_context(query, context)


# ─────────────────────────────────────────────
# Кеш ответов (тот же, что и в main.py)
# ─────────────────────────────────────────────

class AnswerCache:
    """Персистентный кеш ответов на диске (JSON)."""

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
                logger.warning("Cache corrupted, starting fresh: %s", e)
                self._data = {}
        else:
            logger.info("No cache found, starting fresh")

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
        self._data[key] = {
            "q_id": q_id,
            "query": query,
            "answer": answer,
            "model": model,
        }
        self._save()

    def __len__(self) -> int:
        return len(self._data)


# ─────────────────────────────────────────────
# Валидация ответов
# ─────────────────────────────────────────────

_STOP_WORDS = frozenset([
    "как", "что", "где", "когда", "почему", "зачем", "кто",
    "можно", "нужно", "надо", "это", "есть", "для", "при",
    "или", "если", "чтобы", "который", "которая", "которое",
    "в", "на", "с", "по", "из", "от", "до", "за", "к", "у",
    "я", "мы", "вы", "он", "она", "они",
])


def _extract_meaningful_words(text: str) -> set[str]:
    words = text.lower().split()
    return {
        w.strip(".,!?;:\"'()[]")
        for w in words
        if len(w) > 2 and w not in _STOP_WORDS
    }


def validate_answer(query: str, answer: str, min_overlap: int = 1) -> bool:
    """Проверяет релевантность ответа вопросу."""
    if not answer or not answer.strip():
        return False

    query_words = _extract_meaningful_words(query)
    answer_words = _extract_meaningful_words(answer)

    if not query_words:
        return True

    overlap = query_words & answer_words
    return len(overlap) >= min_overlap


# ─────────────────────────────────────────────
# Основной pipeline
# ─────────────────────────────────────────────

def run_pipeline(
    build_index: bool = False,
    llm_model: str = "qwen2.5-7b",
    cache_path: Path = Path("data/answer_cache.json"),
    validate_answers: bool = True,
    min_overlap: int = 1,
) -> None:
    """
    Запускает полный RAG pipeline для Kaggle.

    Args:
        build_index: Строить индекс с нуля
        llm_model: Ключ модели из KAGGLE_MODELS
        cache_path: Путь к кешу
        validate_answers: Включить валидацию
        min_overlap: Минимальный overlap слов
    """
    # ── Индекс ────────────────────────────────────────────────
    if build_index or not INDEX_PATH.exists():
        logger.info("Building index from scratch")
        websites_df = pd.read_csv(WEBSITES_CSV)
        websites_data = [
            (row["web_id"], row["text"])
            for _, row in websites_df.iterrows()
        ]
        chunks = chunk_all_websites(websites_data)
        logger.info("Created %d chunks", len(chunks))
        indexer = build_and_save_index(chunks)
    else:
        logger.info("Loading existing index")
        indexer = load_index()

    # ── Retriever ─────────────────────────────────────────────
    retriever = create_retriever(indexer)

    # ── Generator ─────────────────────────────────────────────
    hf_model_name = KAGGLE_MODELS.get(llm_model, llm_model)
    generator = KaggleGenerator(model_name=hf_model_name)

    # ── Кеш ───────────────────────────────────────────────────
    cache = AnswerCache(cache_path)

    # ── Вопросы ───────────────────────────────────────────────
    logger.info("Loading questions from %s", QUESTIONS_CSV)
    questions_df = pd.read_csv(QUESTIONS_CSV)
    total = len(questions_df)

    # ── Цикл генерации ────────────────────────────────────────
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

        # Шаг 1: Проверяем кеш
        cached_answer = cache.get(query, hf_model_name)
        if cached_answer is not None:
            results.append({"q_id": q_id, "answer": cached_answer})
            stats["cached"] += 1
            continue

        # Шаг 2: Retrieval + Generation
        context = None
        try:
            context = retriever.get_context(query)
            answer = generator.generate(query, context)
        except Exception as e:
            logger.error("Failed to process q_id=%s: %s", q_id, e, exc_info=True)
            # Use already-retrieved context if available, otherwise empty string
            # (don't call get_context again - it will fail with same error)
            answer = extract_answer_from_context(query, context or "")
            stats["failed"] += 1

        # Шаг 3: Валидация
        if validate_answers and answer:
            if not validate_answer(query, answer, min_overlap):
                logger.warning(
                    "Invalid answer for q_id=%s | query='%s' | answer='%s'",
                    q_id,
                    query[:60],
                    answer[:60],
                )
                stats["invalid"] += 1

        # Шаг 4: Кешируем и сохраняем
        if answer:
            cache.set(query, hf_model_name, q_id, answer)

        results.append({"q_id": q_id, "answer": answer})
        stats["generated"] += 1

        # Чекпоинт
        if (idx + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = SUBMISSION_CSV.parent / f"submission_checkpoint_{idx + 1}.csv"
            pd.DataFrame(results).to_csv(checkpoint_path, index=False)
            logger.info("Checkpoint saved: %d answers", idx + 1)

    # ── Итоги ─────────────────────────────────────────────────
    logger.info(
        "Pipeline done | total=%d | cached=%d | generated=%d | failed=%d | invalid=%d",
        total,
        stats["cached"],
        stats["generated"],
        stats["failed"],
        stats["invalid"],
    )

    # ── Сохранение ────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    SUBMISSION_CSV.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(SUBMISSION_CSV, index=False)
    logger.info("Results saved to %s (%d rows)", SUBMISSION_CSV, len(results_df))


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kaggle RAG Pipeline")

    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build index from scratch",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-7b",
        choices=list(KAGGLE_MODELS.keys()),
        help="LLM model (open-source, Hugging Face)",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("data/answer_cache.json"),
        help="Path to answer cache",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Disable answer validation",
    )
    parser.add_argument(
        "--min-overlap",
        type=int,
        default=1,
        help="Minimum word overlap for validation",
    )

    args = parser.parse_args()

    run_pipeline(
        build_index=args.build_index,
        llm_model=args.model,
        cache_path=args.cache_path,
        validate_answers=not args.no_validate,
        min_overlap=args.min_overlap,
    )