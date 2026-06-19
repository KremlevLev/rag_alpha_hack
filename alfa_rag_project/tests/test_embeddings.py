"""
Tests for the enterprise embedding service.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embeddings import create_embedder, LocalEmbedder
from embeddings.embedder import set_voyage_key, set_cohere_key


def test_create_local_embedder() -> None:
    embedder = create_embedder("local")
    assert isinstance(embedder, LocalEmbedder)
    assert embedder.dimension > 0


def test_voyage_embedder_raises_without_key() -> None:
    set_voyage_key("")
    from embeddings.embedder import VoyageEmbedder

    with pytest.raises(ValueError, match="API key not set"):
        VoyageEmbedder()


def test_cohere_embedder_raises_without_key() -> None:
    set_cohere_key("")
    from embeddings.embedder import CohereEmbedder

    with pytest.raises(ValueError, match="API key not set"):
        CohereEmbedder()