"""
Tests for the enterprise ETL loader.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncio

from loader import DocumentLoadError, _make_doc_id, load_document


def test_load_document_raises_on_missing_file() -> None:
    async def _test() -> None:
        with pytest.raises(DocumentLoadError, match="File not found"):
            await load_document("/nonexistent/file.pdf")

    asyncio.run(_test())


def test_make_doc_id_is_deterministic() -> None:
    p1 = Path("/tmp/test_file.pdf")
    p2 = Path("/tmp/test_file.pdf")
    assert _make_doc_id(p1) == _make_doc_id(p2)


def test_make_doc_id_unique_per_path() -> None:
    p1 = Path("/tmp/doc_a.pdf")
    p2 = Path("/tmp/doc_b.pdf")
    assert _make_doc_id(p1) != _make_doc_id(p2)