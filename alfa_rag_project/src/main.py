"""
Main orchestrator for RAG pipeline.
Coordinates chunking, indexing, retrieval, and generation.
"""

import hashlib
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from tqdm import tqdm

from config import (
    INDEX_PATH,
    QUESTIONS_CSV,
    SUBMISSION_CSV,
    WEBSITES_CSV,
)
from chunker import chunk_all_websites
from generator import create_generator
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
# Кеш ответов
# ─────────────────────────────────────────────

class AnswerCache:
    """
    Персистентный кеш ответов на диске (JSON).

    Зачем:
        LLM-генерация — самая дорогая часть pipeline.
        При повторном запуске (с теми же вопросами и моделью)
        нет смысла пересчитывать уже готовые ответы.

    Структура файла:
        {
            "<cache_key>": {
                "q_id": "q_001",
                "query": "Как открыть вклад?",
                "answer": "Для открытия вклада...",
                "model": "qwen2.5:7b"
            }
        }

    Cache key:
        SHA-256(query + "|" + model_name)[:16]
        Включаем model_name — разные модели дают разные ответы.
    """

    def __init__(self, cache_path: Path):
        """
        Args:
            cache_path: Путь к JSON файлу кеша.
        """
        self._path = cache_path
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        """Загружает кеш с диска если файл существует."""
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    self._data = json.load(f)
                logger.info("Cache loaded: %d entries from %s", len(self._data), self._path)
            except (json.JSONDecodeError, OSError) as e:
                # Битый кеш не должен останавливать pipeline
                logger.warning("Cache file corrupted, starting fresh: %s", e)
                self._data = {}
        else:
            logger.info("No cache file found at %s, starting fresh", self._path)

    def _save(self) -> None:
        """Сохраняет кеш на диск."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _make_key(query: str, model: str) -> str:
        """
        Строит ключ кеша.

        Args:
            query: Вопрос пользователя.
            model: Имя модели генерации.

        Returns:
            Hex-строка 16 символов.
        """
        raw = f"{query.strip()}|{model}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def get(self, query: str, model: str) -> str | None:
        """
        Возвращает кешированный ответ или None.

        Args:
            query: Вопрос пользователя.
            model: Имя модели.

        Returns:
            Ответ из кеша или None если не найден.
        """
        key = self._make_key(query, model)
        entry = self._data.get(key)
        return entry["answer"] if entry else None

    def set(self, query: str, model: str, q_id: str, answer: str) -> None:
        """
        Сохраняет ответ в кеш.

        Args:
            query: Вопрос пользователя.
            model: Имя модели.
            q_id: ID вопроса (для трейсинга).
            answer: Сгенерированный ответ.
        """
        key = self._make_key(query, model)
        self._data[key] = {
            "q_id": q_id,
            "query": query,
            "answer": answer,
            "model": model,
        }
        # Сохраняем сразу — защита от прерывания pipeline на середине
        self._save()

    def __len__(self) -> int:
        return len(self._data)


# ─────────────────────────────────────────────
# Валидация ответов
# ─────────────────────────────────────────────

# Стоп-слова: слишком общие, не несут смысла для валидации
_STOP_WORDS = frozenset([
    "как", "что", "где", "когда", "почему", "зачем", "кто",
    "можно", "нужно", "надо", "это", "есть", "для", "при",
    "или", "если", "чтобы", "который", "которая", "которое",
    "в", "на", "с", "по", "из", "от", "до", "за", "к", "у",
    "я", "мы", "вы", "он", "она", "они",
])


def _extract_meaningful_words(text: str) -> set[str]:
    """
    Извлекает значимые слова из текста.

    Убирает стоп-слова и слова короче 3 символов.

    Args:
        text: Произвольный текст.

    Returns:
        Множество значимых слов в нижнем регистре.
    """
    words = text.lower().split()
    return {
        w.strip(".,!?;:\"'()[]")
        for w in words
        if len(w) > 2 and w not in _STOP_WORDS
    }


def validate_answer(query: str, answer: str, min_overlap: int = 1) -> bool:
    """
    Проверяет релевантность ответа вопросу.

    Логика:
        Ответ релевантен если хотя бы min_overlap значимых слов
        из вопроса присутствует в ответе.

    Ограничения:
        Это лексическая проверка, не семантическая.
        "открыть счёт" и "завести вклад" — разные слова,
        даже если по смыслу близко. Для полноценной валидации
        нужна отдельная модель, но для базового фильтра достаточно.

    Args:
        query: Вопрос пользователя.
        answer: Сгенерированный ответ.
        min_overlap: Минимальное количество общих слов.

    Returns:
        True если ответ считается релевантным.

    Examples:
        >>> validate_answer("Как открыть вклад?", "Для открытия вклада...")
        True
        >>> validate_answer("Как открыть вклад?", "Позвоните 8-800-200-00-00")
        False
    """
    if not answer or not answer.strip():
        return False

    query_words = _extract_meaningful_words(query)
    answer_words = _extract_meaningful_words(answer)

    if not query_words:
        # Вопрос из одних стоп-слов — не можем валидировать, пропускаем
        return True

    overlap = query_words & answer_words
    is_valid = len(overlap) >= min_overlap

    if not is_valid:
        logger.debug(
            "Answer validation failed | query_words=%s | answer_words_sample=%s",
            query_words,
            list(answer_words)[:5],
        )

    return is_valid


# ─────────────────────────────────────────────
# Основной pipeline
# ─────────────────────────────────────────────

def run_pipeline(
    build_index: bool = False,
    llm_model: str = "qwen2.5:7b",
    cache_path: Path = Path("data/answer_cache.json"),
    validate_answers: bool = True,
    min_overlap: int = 1,
) -> None:
    """
    Запускает полный RAG pipeline.

    Args:
        build_index: Строить индекс с нуля (True) или загрузить (False).
        llm_model: Имя модели генерации.
        cache_path: Путь к файлу кеша ответов.
        validate_answers: Включить валидацию ответов.
        min_overlap: Минимальный overlap слов для валидации.
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

    # ── Retriever и Generator ─────────────────────────────────
    retriever = create_retriever(indexer)
    generator = create_generator(model=llm_model)

    # ── Кеш ───────────────────────────────────────────────────
    cache = AnswerCache(cache_path)

    # ── Вопросы ───────────────────────────────────────────────
    logger.info("Loading questions from %s", QUESTIONS_CSV)
    questions_df = pd.read_csv(QUESTIONS_CSV)
    total = len(questions_df)

    # ── Цикл генерации ────────────────────────────────────────
    results = []
    stats = {"cached": 0, "generated": 0, "failed": 0, "invalid": 0}

    for _, row in tqdm(questions_df.iterrows(), total=total, desc="Generating"):
        q_id = str(row["q_id"])
        query = str(row["query"]).strip()

        # Шаг 1: Проверяем кеш
        cached_answer = cache.get(query, llm_model)
        if cached_answer is not None:
            results.append({"q_id": q_id, "answer": cached_answer})
            stats["cached"] += 1
            continue

        # Шаг 2: Retrieval + Generation
        try:
            context = retriever.get_context(query)
            answer = generator.generate(query, context)
        except Exception as e:
            # Один сломанный вопрос не роняет весь pipeline
            logger.error("Failed to process q_id=%s: %s", q_id, e, exc_info=True)
            answer = ""
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
                # Не заменяем на пустую строку — нерелевантный ответ
                # лучше чем пустота для downstream метрик.
                # Логируем для анализа.

        # Шаг 4: Кешируем и сохраняем результат
        if answer:
            cache.set(query, llm_model, q_id, answer)

        results.append({"q_id": q_id, "answer": answer})
        stats["generated"] += 1

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

    parser = argparse.ArgumentParser(description="RAG Pipeline")

    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build index from scratch (default: load existing if available)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5:7b",
        help="LLM model name for generation",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("data/answer_cache.json"),
        help="Path to answer cache JSON file",
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
        help="Minimum word overlap for answer validation (default: 1)",
    )

    args = parser.parse_args()

    run_pipeline(
        build_index=args.build_index,
        llm_model=args.model,
        cache_path=args.cache_path,
        validate_answers=not args.no_validate,
        min_overlap=args.min_overlap,
    )
