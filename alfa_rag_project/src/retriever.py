"""
Retrieval module for RAG pipeline.
Performs hybrid search (BM25 + FAISS), cross-encoder reranking, and context formatting.
"""

import re
import html
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple, Set

import numpy as np
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

import torch

from config import TOP_K_RETRIEVAL, TOP_K_RERANK, RERANKER_MODEL, TOP_K_BM25, RERANKER_BATCH_SIZE
from indexer import Indexer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Конфигурация очистки
# ─────────────────────────────────────────────

@dataclass
class CleanerConfig:
    """Настройки очистки текста чанка."""
    
    # Минимальная длина текста после очистки (символы)
    min_length: int = 20
    
    # Обрезать незавершённые предложения в конце чанка
    trim_incomplete_sentences: bool = True


# ─────────────────────────────────────────────
# Паттерны очистки
# ─────────────────────────────────────────────

# HTML-теги: <b>, </p>, <br/> и т.д.
_HTML_TAG_RE = re.compile(r"<[^>]{1,100}>", re.UNICODE)

# HTML-сущности: &nbsp; & &#160; и т.д.
# html.unescape() покрывает большинство, но на случай битых сущностей:
_BROKEN_HTML_ENTITY_RE = re.compile(r"&[a-zA-Z]{2,8};?|&#\d{1,5};?", re.UNICODE)

# Chunk ID в начале строки: [38644], [0], [123456]
_CHUNK_ID_PREFIX_RE = re.compile(r"^\s*\[\d+\]\s*", re.UNICODE)

# Неразрывные и управляющие пробелы: \xa0, \u200b, \t и подобные
_WHITESPACE_VARIANTS_RE = re.compile(
    r"[\xa0\u00a0\u200b\u200c\u200d\u2060\ufeff\t]+",
    re.UNICODE,
)

# Повторяющиеся пробелы (после замены спецсимволов)
_MULTI_SPACE_RE = re.compile(r" {2,}", re.UNICODE)

# Повторяющиеся переносы строк (больше двух подряд)
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}", re.UNICODE)

# Мусорные символы: вертикальная черта, звёздочки, подчёркивания как разделители
_DECORATIVE_RE = re.compile(r"[|*]{2,}|_{3,}", re.UNICODE)


# ─────────────────────────────────────────────
# Функции очистки
# ─────────────────────────────────────────────

def _remove_chunk_id_prefix(text: str) -> str:
    """
    Удаляет префикс chunk ID из начала строки.
    
    Examples:
        "[38644] Корреспондентский счёт..." → "Корреспондентский счёт..."
        "[0] Текст"                         → "Текст"
    """
    return _CHUNK_ID_PREFIX_RE.sub("", text)


def _strip_html(text: str) -> str:
    """
    Убирает HTML-теги и декодирует HTML-сущности.
    
    Порядок важен: сначала unescape (чтобы <b> стало <b>),
    затем strip тегов.
    
    Examples:
        "<b>Счёт</b>"      → "Счёт"
        "&nbsp;текст&" → " текст&"  → после trim → "текст"
        "&#160;данные"     → "\xa0данные" → заменяется далее
    """
    # 1. Декодируем HTML-сущности (&nbsp; → \xa0, & → &)
    text = html.unescape(text)
    
    # 2. Убираем теги
    text = _HTML_TAG_RE.sub(" ", text)
    
    # 3. На случай битых сущностей которые html.unescape не поймал
    text = _BROKEN_HTML_ENTITY_RE.sub(" ", text)
    
    return text


def _normalize_whitespace(text: str) -> str:
    """
    Нормализует все виды пробельных символов.
    
    Examples:
        "текст\\xa0\\xa0данные" → "текст данные"
        "слово\\t\\tслово"      → "слово слово"
    """
    # Заменяем спецпробелы на обычный
    text = _WHITESPACE_VARIANTS_RE.sub(" ", text)
    
    # Схлопываем многократные пробелы
    text = _MULTI_SPACE_RE.sub(" ", text)
    
    # Схлопываем многократные переносы строк
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    
    return text


def _remove_decorative(text: str) -> str:
    """
    Убирает декоративные символы-разделители.
    
    Examples:
        "Заголовок\n***\nТекст"    → "Заголовок\n\nТекст"
        "Раздел\n---|---\nДанные"  → без изменений (не попадает)
    """
    return _DECORATIVE_RE.sub("", text)


def _trim_incomplete_sentence(text: str) -> str:
    """
    Обрезает незавершённое предложение в конце чанка.
    
    Чанки часто обрываются посередине предложения из-за
    разбивки документа по фиксированному размеру.
    
    Logic:
        Если текст не заканчивается на [.!?»")],
        находим последнее полное предложение и обрезаем до него.
    
    Examples:
        "Счёт открывается. Для этого нужно" → "Счёт открывается."
        "Позвоните по номеру 8-800."         → без изменений (полное)
    """
    stripped = text.rstrip()
    
    # Проверяем: текст заканчивается на знак конца предложения?
    if re.search(r"[.!?»\"\)]\s*$", stripped, re.UNICODE):
        return stripped  # Уже полное
    
    # Ищем последнюю точку/восклицательный/вопросительный знак
    last_end = max(
        stripped.rfind("."),
        stripped.rfind("!"),
        stripped.rfind("?"),
        stripped.rfind("»"),
    )
    
    if last_end == -1:
        # Нет ни одного конца предложения — возвращаем как есть
        # (лучше неполный текст, чем пустота)
        return stripped
    
    return stripped[:last_end + 1]


def clean_chunk_text(
    text: str,
    config: Optional[CleanerConfig] = None,
) -> str:
    """
    Полная очистка текста чанка для подачи в LLM/fallback.
    
    Pipeline:
        1. Убрать chunk ID префикс
        2. Убрать HTML
        3. Нормализовать пробелы
        4. Убрать декоративные символы
        5. Обрезать незавершённые предложения
        6. Финальный trim
    
    Args:
        text: Сырой текст чанка из индекса.
        config: Настройки очистки.
        
    Returns:
        Очищенный текст или пустая строка если текст стал слишком коротким.
        
    Examples:
        >>> clean_chunk_text("[38644] <b>Счёт</b>&nbsp;открывается в банке. Для")
        "Счёт открывается в банке."
    """
    if config is None:
        config = CleanerConfig()
    
    if not text or not text.strip():
        return ""
    
    # Шаг 1: chunk ID
    text = _remove_chunk_id_prefix(text)
    
    # Шаг 2: HTML
    text = _strip_html(text)
    
    # Шаг 3: пробелы
    text = _normalize_whitespace(text)
    
    # Шаг 4: декоративные символы
    text = _remove_decorative(text)
    
    # Шаг 5: незавершённые предложения
    if config.trim_incomplete_sentences:
        text = _trim_incomplete_sentence(text)
    
    # Шаг 6: финальный trim
    text = text.strip()
    
    # Проверка минимальной длины
    if len(text) < config.min_length:
        logger.debug("Chunk too short after cleaning (%d chars), skipping", len(text))
        return ""
    
    return text


# ─────────────────────────────────────────────
# BM25 Tokenize
# ─────────────────────────────────────────────

def _tokenize_for_bm25(text: str) -> list[str]:
    """
    Tokenize text for BM25 search.
    
    Simple whitespace + punctuation tokenization for Russian text.
    
    Args:
        text: Input text to tokenize.
        
    Returns:
        List of tokens.
    """
    # Simple tokenization: split on whitespace and remove punctuation
    text = re.sub(r"[^\w\sа-яёa-z]", " ", text.lower(), flags=re.UNICODE)
    return [t for t in text.split() if t and len(t) > 1]


# ─────────────────────────────────────────────
# Retriever
# ─────────────────────────────────────────────

class Retriever:
    """
    Handles hybrid retrieval with FAISS and BM25, plus cross-encoder reranking.
    """
    
    def __init__(
        self,
        indexer: Indexer,
        reranker_model: str = RERANKER_MODEL,
        cleaner_config: Optional[CleanerConfig] = None,
    ):
        """
        Initialize retriever with indexer, reranker, and BM25 index.
        
        Args:
            indexer: Indexer instance with loaded FAISS index.
            reranker_model: Name of the cross-encoder model.
            cleaner_config: Settings for chunk text cleaning.
        """
        self.indexer = indexer
        self.reranker = CrossEncoder(reranker_model)
        self.cleaner_config = cleaner_config or CleanerConfig()
        
        # Build BM25 index from all chunk texts
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_texts: list[str] = []
        self._bm25_ids: list[int] = []
        self._build_bm25_index()
    
    def _build_bm25_index(self) -> None:
        """Build BM25 index from all chunks in the indexer."""
        if not self.indexer.is_built():
            return
        
        all_texts = self.indexer.get_all_texts()
        self._bm25_texts = all_texts
        self._bm25_ids = list(range(len(all_texts)))
        
        # Tokenize all texts for BM25
        tokenized_texts = [_tokenize_for_bm25(text) for text in all_texts]
        self._bm25_index = BM25Okapi(tokenized_texts)
        
        logger.info(
            "BM25 index built: %d documents",
            len(all_texts),
        )
    
    def _bm25_search(self, query: str) -> List[Tuple[int, str, float]]:
        """
        Search using BM25 lexical index.
        
        Args:
            query: Search query.
            
        Returns:
            List of (chunk_id, text, score) tuples from BM25.
        """
        if self._bm25_index is None:
            return []
        
        tokenized_query = _tokenize_for_bm25(query)
        if not tokenized_query:
            return []
        
        bm25_scores = self._bm25_index.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(bm25_scores)[::-1][:TOP_K_BM25]
        
        results = []
        for idx in top_indices:
            if bm25_scores[idx] > 0:
                chunk_id = self._bm25_ids[idx]
                chunk_data = self.indexer.get_chunk_by_id(chunk_id)
                if chunk_data is not None:
                    results.append((chunk_id, chunk_data["text"], float(bm25_scores[idx])))
        
        return results
    
    def retrieve(self, query: str) -> List[Tuple[int, str, float]]:
        """
        Retrieve top-k chunks for a query using hybrid search.
        
        Two-stage retrieval pipeline:
            1. Get Top-15 chunks from FAISS (semantic search)
            2. Get Top-15 chunks from BM25 (exact keyword match)
            3. Merge and deduplicate (resulting in ~20-25 unique chunks)
            4. Pass to cross-encoder for reranking
        
        Args:
            query: Search query.
            
        Returns:
            List of (chunk_id, text, score) tuples, top-k after reranking.
            Text здесь — сырой, очистка происходит в get_context().
            
        Raises:
            RuntimeError: If indexer is not built or loaded.
        """
        if not self.indexer.is_built():
            raise RuntimeError("Indexer not built or loaded")
        
        # ── Stage 1: FAISS semantic search ───────────────────────────────
        query_embedding = self.indexer.model.encode(
            [query],
            normalize_embeddings=True,
        )
        query_embedding = query_embedding.astype(np.float32)
        
        scores, indices = self.indexer.index.search(
            query_embedding,
            TOP_K_RETRIEVAL,
        )
        
        faiss_candidates: List[Tuple[int, str]] = []
        for cid in indices[0].tolist():
            chunk_data = self.indexer.get_chunk_by_id(cid)
            if chunk_data is not None:
                faiss_candidates.append((cid, chunk_data["text"]))
        
        # ── Stage 2: BM25 lexical search ───────────────────────────────
        bm25_candidates = self._bm25_search(query)
        
        # ── Stage 3: Merge and deduplicate ───────────────────────────────
        seen_ids: Set[int] = set()
        merged_candidates: List[Tuple[int, str]] = []
        
        # Add FAISS candidates first
        for chunk_id, text in faiss_candidates:
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                merged_candidates.append((chunk_id, text))
        
        # Add BM25 candidates (deduplicated)
        for chunk_id, text, _ in bm25_candidates:
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                merged_candidates.append((chunk_id, text))
        
        if not merged_candidates:
            return []

        # ── Stage 4: Cross-encoder reranking (batched for memory efficiency) ───────────────────────────────
        # Process in small batches to avoid CUDA OOM with large reranker model
        all_pairs = [(query, text) for _, text in merged_candidates]
        rerank_scores = []
        
        for i in range(0, len(all_pairs), RERANKER_BATCH_SIZE):
            batch = all_pairs[i:i + RERANKER_BATCH_SIZE]
            batch_scores = self.reranker.predict(batch)
            rerank_scores.extend(batch_scores.tolist() if hasattr(batch_scores, 'tolist') else list(batch_scores))
            
            # Clear GPU cache after each batch to prevent memory fragmentation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        top_indices = np.argsort(rerank_scores)[::-1][:TOP_K_RERANK]
        
        results = []
        for idx in top_indices:
            chunk_id, text = merged_candidates[idx]
            score = float(rerank_scores[idx])
            results.append((chunk_id, text, score))
        
        return results
    
    def get_context(
        self,
        query: str,
        cleaner_config: Optional[CleanerConfig] = None,
    ) -> str:
        """
        Get formatted context for LLM / fallback generation.
        
        Changes vs original:
            - Chunk ID убран из текста контекста
            - Каждый чанк проходит clean_chunk_text()
            - Пустые чанки после очистки пропускаются
            - Разделитель между чанками — чистый \\n\\n
            - Чанки реверсированы (Lost in the Middle fix)
        
        "Lost in the Middle" fix:
            LLMs tend to focus on the beginning and end of context,
            missing important information in the middle. By reversing
            the order of chunks, we ensure the most relevant chunks
            (from reranking) appear at both start and end.
        
        Args:
            query: Search query.
            cleaner_config: Override cleaner settings for this call.
            
        Returns:
            Clean context string ready for LLM, or "" if nothing retrieved.
        """
        results = self.retrieve(query)
        
        if not results:
            return ""
        
        config = cleaner_config or self.cleaner_config
        
        context_parts = []
        
        for chunk_id, raw_text, score in results:
            cleaned = clean_chunk_text(raw_text, config)
            
            if not cleaned:
                logger.debug(
                    "Chunk %d dropped after cleaning (score=%.3f)",
                    chunk_id,
                    score,
                )
                continue
            
            context_parts.append(cleaned)
        
        # "Lost in the Middle" fix: reverse the order of chunks
        # This ensures the most relevant chunks appear at both start and end
        context_parts = context_parts[::-1]
        
        return "\n\n".join(context_parts)


# ─────────────────────────────────────────────
# Фабрика
# ─────────────────────────────────────────────

def create_retriever(indexer: Indexer) -> Retriever:
    """
    Convenience function to create retriever.
    
    Args:
        indexer: Indexer instance.
        
    Returns:
        Retriever instance with default settings.
    """
    return Retriever(indexer)