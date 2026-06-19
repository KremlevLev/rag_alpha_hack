"""
Parent-child chunking for retrieval.

The indexer stores small child chunks because they are easier to match
semantically and lexically. The retriever expands matched children back to
their parent chunks before reranking and LLM context formatting. This keeps
embedding precision high while giving the generator enough surrounding text
to answer without saying "Нет ответа".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from chunker import Chunk, Chunker, ChunkerConfig, clean_text
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    PARENT_CHILD_ENABLED,
    PARENT_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParentChildConfig:
    """Settings for parent-child chunk creation."""

    parent_chunk_size: int = PARENT_CHUNK_SIZE
    parent_chunk_overlap: int = PARENT_CHUNK_OVERLAP
    child_chunk_size: int = CHUNK_SIZE
    child_chunk_overlap: int = CHUNK_OVERLAP
    min_parent_length: int = 160
    min_child_length: int = 40


def _split_with_sentence_boundaries(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    min_length: int,
) -> list[str]:
    """Split cleaned text into sentence-aware chunks."""
    chunker = Chunker(
        ChunkerConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_length=min_length,
            keep_sentence_boundary=True,
        )
    )
    return chunker.chunk_text(text)


def build_parent_child_chunks(
    websites_data: Iterable[Tuple[int, str]],
    config: ParentChildConfig | None = None,
) -> list[Chunk]:
    """
    Build child chunks with parent metadata.

    Parent chunks are the text passed to the LLM. Child chunks are indexed for
    retrieval. Each child keeps a pointer to its parent so `Retriever` can
    expand narrow child hits into useful context.
    """
    cfg = config or ParentChildConfig()
    all_chunks: list[Chunk] = []
    next_child_id = 0
    next_parent_id = 0

    websites_list = list(websites_data)
    website_count = len(websites_list)

    for web_id, raw_text in websites_list:
        cleaned = clean_text(raw_text)
        if not cleaned:
            continue

        parent_texts = _split_with_sentence_boundaries(
            cleaned,
            cfg.parent_chunk_size,
            cfg.parent_chunk_overlap,
            cfg.min_parent_length,
        )

        for parent_text in parent_texts:
            child_texts = _split_with_sentence_boundaries(
                parent_text,
                cfg.child_chunk_size,
                cfg.child_chunk_overlap,
                cfg.min_child_length,
            )

            if not child_texts:
                next_parent_id += 1
                continue

            for child_text in child_texts:
                if len(child_text) < cfg.min_child_length:
                    continue

                start = parent_text.find(child_text)
                all_chunks.append(
                    Chunk(
                        chunk_id=next_child_id,
                        web_id=web_id,
                        text=child_text,
                        parent_id=next_parent_id,
                        parent_text=parent_text,
                        parent_start=start if start >= 0 else None,
                        parent_end=(start + len(child_text)) if start >= 0 else None,
                    )
                )
                next_child_id += 1

            next_parent_id += 1

    logger.info(
        "Built %d child chunks with parent-child metadata from %d websites",
        len(all_chunks),
        website_count,
    )
    return all_chunks


def build_chunks(
    websites_data: Iterable[Tuple[int, str]],
    use_parent_child: bool = PARENT_CHILD_ENABLED,
) -> list[Chunk]:
    """Build chunks using parent-child mode or the legacy single-level mode."""
    if use_parent_child:
        return build_parent_child_chunks(websites_data)

    from chunker import chunk_all_websites

    return chunk_all_websites(websites_data)
