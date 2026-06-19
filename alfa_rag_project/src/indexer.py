"""
Indexing module for RAG pipeline.
Handles embedding generation and FAISS index management.
"""

import gc
import hashlib
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import (
    CHUNK_MAPPING_PATH,
    EMBEDDER_MODEL,
    INDEX_PATH,
)
from chunker import Chunk

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Нормализация текста перед эмбеддингом
# ─────────────────────────────────────────────

# Таблица замены ё → е для str.translate()
# Используем translate() вместо replace() — быстрее для одного символа
_YO_TABLE = str.maketrans("ёЁ", "еЕ")


def normalize_for_embedding(text: str) -> str:
    """
    Нормализует текст перед подачей в модель эмбеддингов.

    Проблема:
        BGE-M3 и аналогичные модели обучены на корпусах,
        где "ё" и "е" используются непоследовательно.
        "счёт" и "счет" могут получить разные эмбеддинги,
        что приводит к пропущенным результатам при поиске.

    Что делаем:
        1. ё → е (самая частая причина расхождений в русских текстах)
        2. Unicode NFC нормализация (составные символы → канонические)
        3. Схлопывание множественных пробелов

    Чего не делаем:
        - Не приводим к lowercase: модель чувствительна к регистру
          для имён собственных (Сбербанк ≠ сбербанк)
        - Не стеммим: это задача для keyword-поиска, не эмбеддингов

    Args:
        text: Исходный текст чанка или запроса.

    Returns:
        Нормализованный текст для encode().

    Examples:
        >>> normalize_for_embedding("Открыть счёт в Сбербанке")
        "Открыть счет в Сбербанке"
        >>> normalize_for_embedding("cafe\\u0301")  # café с combining accent
        "café"  # NFC: канонический символ
    """
    if not text:
        return text

    # 1. ё → е
    text = text.translate(_YO_TABLE)

    # 2. Unicode NFC: "е\\u0301" (е + combining accent) → "é"
    text = unicodedata.normalize("NFC", text)

    # 3. Множественные пробелы (на случай если текст пришёл не из clean_text)
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


# ─────────────────────────────────────────────
# Дедупликация
# ─────────────────────────────────────────────

def _compute_chunk_hash(text: str) -> str:
    """
    Вычисляет хеш текста чанка для дедупликации.

    Используем SHA-256 первые 16 байт (достаточно для коллизионной
    устойчивости при типичных размерах корпуса < 1М чанков).

    Хешируем нормализованный текст, чтобы "счёт" и "счет"
    считались дубликатами.

    Args:
        text: Текст чанка.

    Returns:
        Hex-строка хеша (32 символа).
    """
    normalized = normalize_for_embedding(text).lower().strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


def deduplicate_chunks(chunks: list[Chunk]) -> tuple[list[Chunk], int]:
    """
    Удаляет дублирующиеся чанки перед индексированием.

    Дубликаты возникают когда:
        - Один FAQ-ответ спаршен с нескольких URL
        - Документ переиндексирован с небольшими правками
        - Шаблонные блоки ("Подробнее на сайте") встречаются на каждой странице

    Args:
        chunks: Исходный список чанков (может содержать дубли).

    Returns:
        Tuple (уникальные чанки, количество удалённых дублей).

    Examples:
        >>> c1 = Chunk(chunk_id=1, text="Счет открыт.", web_id="url1")
        >>> c2 = Chunk(chunk_id=2, text="Счёт открыт.", web_id="url2")  # дубль
        >>> unique, dropped = deduplicate_chunks([c1, c2])
        >>> len(unique), dropped
        (1, 1)
    """
    seen_hashes: set[str] = set()
    unique_chunks: list[Chunk] = []
    duplicates_count = 0

    for chunk in chunks:
        chunk_hash = _compute_chunk_hash(chunk.text)

        if chunk_hash in seen_hashes:
            duplicates_count += 1
            logger.debug(
                "Duplicate chunk dropped: id=%s, web_id=%s, text_prefix='%s'",
                chunk.chunk_id,
                chunk.web_id,
                chunk.text[:50],
            )
            continue

        seen_hashes.add(chunk_hash)
        unique_chunks.append(chunk)

    return unique_chunks, duplicates_count


# ─────────────────────────────────────────────
# Индексер
# ─────────────────────────────────────────────

class Indexer:
    """
    Manages FAISS index and chunk-to-text mapping.
    """

    def __init__(self, model_name: str = EMBEDDER_MODEL):
        """
        Initialize the indexer with embedding model.

        Args:
            model_name: Name of the SentenceTransformer model.
        """
        self.model = SentenceTransformer(model_name)
        self.index: Optional[faiss.IndexFlatIP] = None

        # chunk_id (int) → {"web_id": str, "text": str}
        # Ключи — int, не str. JSON-загрузка требует явного приведения.
        self.chunk_mapping: dict[int, dict[str, Any]] = {}

    def _get_embedding_dim(self) -> int:
        """Get the embedding dimension from the model."""
        return self.model.get_sentence_embedding_dimension()

    def build_index(self, chunks: list[Chunk]) -> None:
        """
        Build FAISS index from chunks.

        Pipeline:
            1. Дедупликация чанков
            2. Нормализация текстов (ё→е, NFC)
            3. Генерация эмбеддингов
            4. Построение FAISS IndexFlatIP
            5. Сборка chunk_mapping

        Args:
            chunks: List of Chunk objects to index.

        Raises:
            ValueError: If chunks list is empty or all chunks are duplicates.
        """
        if not chunks:
            raise ValueError("Cannot build index from empty chunks list")

        # ── Шаг 1: Дедупликация ───────────────────────────────
        unique_chunks, dropped = deduplicate_chunks(chunks)

        if dropped > 0:
            logger.info(
                "Deduplication: %d chunks removed, %d unique remain",
                dropped,
                len(unique_chunks),
            )

        if not unique_chunks:
            raise ValueError(
                f"All {len(chunks)} chunks were duplicates. Nothing to index."
            )

        # ── Шаг 2: Нормализация текстов ───────────────────────
        texts_for_embedding = [
            normalize_for_embedding(chunk.text)
            for chunk in unique_chunks
        ]

        # ── Шаг 3: Генерация эмбеддингов (по батчам для экономии памяти) ──────────────────────
        logger.info("Generating embeddings for %d chunks", len(unique_chunks))

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = 4  # 4 — агрессивно мало для CUDA OOM на 2xT4 (было 32 — убивало память)

        all_embeddings = []
        # CUDA OOM mitigation: отключаем градиенты (не нужны для инференса)
        with torch.no_grad():
            with tqdm(total=len(texts_for_embedding), desc="Embedding") as pbar:
                for i in range(0, len(texts_for_embedding), batch_size):
                    batch = texts_for_embedding[i:i + batch_size]
                    batch_emb = self.model.encode(
                        batch,
                        batch_size=len(batch),
                        show_progress_bar=False,
                        normalize_embeddings=True,
                        device=device,
                    )
                    all_embeddings.append(batch_emb)
                    pbar.update(len(batch))
                    # Очистка кэша GPU раз в 10 батчей
                    if device == "cuda" and (i // batch_size) % 10 == 0:
                        torch.cuda.empty_cache()

        embeddings = np.vstack(all_embeddings).astype(np.float32)

        # Очистка памяти после эмбеддингов
        del all_embeddings
        gc.collect()
        torch.cuda.empty_cache()

        # ── Шаг 4: FAISS индекс ───────────────────────────────
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        logger.info(
            "FAISS index built: dim=%d, vectors=%d",
            dim,
            self.index.ntotal,
        )

        # ── Шаг 5: chunk_mapping (int ключи) ──────────────────
        self.chunk_mapping = {
            faiss_id: {
                "web_id": chunk.web_id,
                "text": chunk.text,
                "chunk_id": chunk.chunk_id,
                "parent_id": chunk.parent_id,
                "parent_text": chunk.parent_text,
                "parent_start": chunk.parent_start,
                "parent_end": chunk.parent_end,
            }
            for faiss_id, chunk in enumerate(unique_chunks)
        }

        # Сохраняем индекс сразу после построения (защита от падения)
        self.save()
        logger.info("Index auto-saved after build")

    def save(self) -> None:
        """
        Save FAISS index and chunk mapping to disk.

        Raises:
            RuntimeError: If index is not built.
        """
        if self.index is None:
            raise RuntimeError("Index not built yet. Call build_index() first.")

        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(INDEX_PATH))

        # JSON не поддерживает int ключи — сохраняем как str,
        # при загрузке явно приводим обратно к int.
        serializable_mapping = {
            str(faiss_id): data
            for faiss_id, data in self.chunk_mapping.items()
        }
        with open(CHUNK_MAPPING_PATH, "w", encoding="utf-8") as f:
            json.dump(serializable_mapping, f, ensure_ascii=False, indent=2)

        logger.info(
            "Index saved: %s (%d vectors)",
            INDEX_PATH,
            self.index.ntotal,
        )

    def load(self) -> None:
        """
        Load FAISS index and chunk mapping from disk.

        Raises:
            FileNotFoundError: If index or mapping files don't exist.
        """
        if not INDEX_PATH.exists():
            raise FileNotFoundError(f"Index not found at {INDEX_PATH}")

        if not CHUNK_MAPPING_PATH.exists():
            raise FileNotFoundError(
                f"Chunk mapping not found at {CHUNK_MAPPING_PATH}"
            )

        self.index = faiss.read_index(str(INDEX_PATH))

        with open(CHUNK_MAPPING_PATH, "r", encoding="utf-8") as f:
            raw_mapping = json.load(f)

        # JSON ключи — строки. Приводим к int для единообразия.
        self.chunk_mapping = {
            int(faiss_id): data
            for faiss_id, data in raw_mapping.items()
        }

        logger.info(
            "Index loaded: %s (%d vectors, %d chunks)",
            INDEX_PATH,
            self.index.ntotal,
            len(self.chunk_mapping),
        )

    def is_built(self) -> bool:
        """Check if index is loaded and non-empty."""
        return self.index is not None and len(self.chunk_mapping) > 0

    def get_chunk_by_id(self, chunk_id: int) -> Optional[dict[str, Any]]:
        """
        Get chunk metadata by FAISS ID.

        Args:
            chunk_id: FAISS vector ID (int, от 0 до ntotal-1).

        Returns:
            Dict с ключами web_id, text, chunk_id или None если не найден.
        """
        # int ключи — явное приведение защищает от случайной подачи str
        return self.chunk_mapping.get(int(chunk_id))

    def get_parent_by_child_id(self, child_id: int) -> Optional[dict[str, Any]]:
        """Return parent metadata for a child chunk, or the child itself in legacy mode."""
        chunk_data = self.get_chunk_by_id(child_id)
        if chunk_data is None:
            return None

        parent_id = chunk_data.get("parent_id")
        if parent_id is None:
            return chunk_data

        for faiss_id, data in self.chunk_mapping.items():
            if data.get("parent_id") == parent_id and data.get("parent_text"):
                return data

        return chunk_data

    def get_parent_texts(self, child_ids: Iterable[int]) -> dict[int, str]:
        """Return unique parent texts for child chunk IDs."""
        parent_texts: dict[int, str] = {}
        for child_id in child_ids:
            parent = self.get_parent_by_child_id(child_id)
            if parent is None:
                continue

            raw_parent_id = parent.get("parent_id")
            parent_id = int(raw_parent_id if raw_parent_id is not None else parent.get("chunk_id", child_id))
            parent_text = parent.get("parent_text") or parent.get("text") or ""
            if parent_text:
                parent_texts[parent_id] = parent_text

        return parent_texts

    def get_all_texts(self) -> list[str]:
        """
        Get all chunk texts in FAISS order.

        Returns:
            List of chunk texts ordered by FAISS ID.
        """
        return [
            self.chunk_mapping[i]["text"]
            for i in range(len(self.chunk_mapping))
        ]


# ─────────────────────────────────────────────
# Фабричные функции
# ─────────────────────────────────────────────

def build_and_save_index(chunks: list[Chunk]) -> Indexer:
    """
    Convenience function: build index and save to disk.

    Args:
        chunks: List of chunks to index.

    Returns:
        Indexer instance with built index.
    """
    indexer = Indexer()
    indexer.build_index(chunks)
    indexer.save()
    return indexer


def load_index() -> Indexer:
    """
    Convenience function: load existing index from disk.

    Returns:
        Indexer instance with loaded index.
    """
    indexer = Indexer()
    indexer.load()
    return indexer