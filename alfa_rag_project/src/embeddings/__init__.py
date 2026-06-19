"""Enterprise embedding service — Voyage AI / Cohere Embed v3 / local fallback."""

from .embedder import EmbeddingProvider, create_embedder, VoyageEmbedder, CohereEmbedder, LocalEmbedder

__all__ = ["EmbeddingProvider", "create_embedder", "VoyageEmbedder", "CohereEmbedder", "LocalEmbedder"]