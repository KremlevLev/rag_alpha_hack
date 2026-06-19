"""
Tests for the enterprise reranker.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reranker import CohereReranker, set_cohere_key


def test_cohere_reranker_raises_without_key() -> None:
    set_cohere_key("")
    with pytest.raises(ValueError, match="API key not set"):
        CohereReranker()