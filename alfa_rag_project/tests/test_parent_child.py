"""
Tests for parent-child chunking and retrieval metadata.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from indexer import Indexer
from parent_child import build_chunks, build_parent_child_chunks
from retriever import _expand_child_candidates_to_parents


class FakeIndexer:
    """Minimal indexer double for parent expansion tests."""

    def __init__(self) -> None:
        self.mapping: dict[int, dict[str, object]] = {
            0: {
                "chunk_id": 0,
                "parent_id": 10,
                "parent_text": "Parent text with card lock, PIN, and fraud protection details.",
            },
            1: {
                "chunk_id": 1,
                "parent_id": 10,
                "parent_text": "Parent text with card lock, PIN, and fraud protection details.",
            },
            2: {
                "chunk_id": 2,
                "parent_id": None,
                "text": "Legacy single-level chunk text.",
            },
        }

    def get_parent_by_child_id(self, child_id: int) -> dict[str, object] | None:
        return self.mapping.get(child_id)


def test_build_parent_child_chunks_indexes_children_with_parent_metadata() -> None:
    website_text = " ".join(
        [
            "Карта заблокирована для операций в интернете.",
            "Для разблокировки откройте приложение Альфа-Банка.",
            "Перейдите в раздел «Карты», выберите карту и нажмите «Разблокировать».",
            "Если разблокировка недоступна, перевыпустите карту в том же разделе.",
            "Новая карта будет доставлена курьером или в офис банка.",
            "После получения активируйте её через приложение.",
        ]
    )

    chunks = build_parent_child_chunks([(1, website_text)])

    assert chunks
    assert all(chunk.parent_id is not None for chunk in chunks)
    assert all(chunk.parent_text for chunk in chunks)
    assert all(chunk.parent_start is not None for chunk in chunks)
    assert all(chunk.parent_end is not None for chunk in chunks)
    assert all(chunk.text in chunk.parent_text for chunk in chunks)
    assert all(len(chunk.parent_text) >= len(chunk.text) for chunk in chunks)


def test_build_chunks_legacy_mode_returns_single_level_chunks() -> None:
    text = "Карта заблокирована для операций в интернете. Для разблокировки откройте приложение Альфа-Банка. Перейдите в раздел «Карты», выберите карту и нажмите «Разблокировать»."
    chunks = build_chunks([(1, text)], use_parent_child=False)

    assert chunks
    assert all(chunk.parent_id is None for chunk in chunks)
    assert all(chunk.parent_text is None for chunk in chunks)


def test_expand_child_candidates_deduplicates_parent_context() -> None:
    expanded = _expand_child_candidates_to_parents(
        FakeIndexer(),
        [(0, "child one"), (1, "child two")],
    )

    assert expanded == [
        (
            10,
            "Parent text with card lock, PIN, and fraud protection details.",
        )
    ]


def test_expand_child_candidates_keeps_legacy_chunks_as_themselves() -> None:
    expanded = _expand_child_candidates_to_parents(FakeIndexer(), [(2, "legacy child")])

    assert expanded == [(2, "legacy child")]


def test_indexer_get_parent_texts_handles_legacy_chunks() -> None:
    indexer = Indexer.__new__(Indexer)
    indexer.chunk_mapping = {
        0: {
            "chunk_id": 0,
            "parent_id": 10,
            "parent_text": "Parent text for child zero.",
        },
        1: {
            "chunk_id": 1,
            "parent_id": None,
            "text": "Legacy chunk text.",
        },
    }

    assert indexer.get_parent_texts([0, 1]) == {
        10: "Parent text for child zero.",
        1: "Legacy chunk text.",
    }


def test_indexer_get_parent_texts_skips_missing_children() -> None:
    indexer = Indexer.__new__(Indexer)
    indexer.chunk_mapping = {}

    assert indexer.get_parent_texts([0, 1, 2]) == {}
