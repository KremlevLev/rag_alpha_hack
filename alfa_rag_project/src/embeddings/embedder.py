"""
Enterprise embedding service.

Supports Voyage AI, Cohere Embed v3, and local sentence-transformers
as a fallback. All embedders are async and use tenacity for retries.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import httpx
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# ── Environment-based API keys (set from outside) ─────────────
VOYAGE_API_KEY: Optional[str] = None
COHERE_API_KEY: Optional[str] = None


def set_voyage_key(key: str) -> None:
    global VOYAGE_API_KEY
    VOYAGE_API_KEY = key


def set_cohere_key(key: str) -> None:
    global COHERE_API_KEY
    COHERE_API_KEY = key


# ── Abstract protocol ──────────────────────────────────────────


class EmbeddingProvider(ABC):
    """Abstract embedding service."""

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        ...

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...


# ── Voyage AI ──────────────────────────────────────────────────


class VoyageEmbedder(EmbeddingProvider):
    """Voyage AI embedding client (model: voyage-3)."""

    MODEL = "voyage-3"
    BASE_URL = "https://api.voyageai.com/v1/embeddings"

    def __init__(self, api_key: Optional[str] = None, batch_size: int = 32) -> None:
        self._api_key = api_key or VOYAGE_API_KEY
        if not self._api_key:
            raise ValueError("Voyage API key not set. Call set_voyage_key() or pass api_key.")
        self._batch_size = batch_size
        self._dim = 1024

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts: List[str]) -> List[List[float]]:
        return await self._call(texts, input_type="document")

    async def embed_query(self, text: str) -> List[float]:
        results = await self._call([text], input_type="query")
        return results[0]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30), reraise=True)
    async def _call(self, texts: List[str], input_type: str) -> List[List[float]]:
        all_embeddings: List[List[float]] = []
        async with httpx.AsyncClient(timeout=60.0) as client:
            for i in range(0, len(texts), self._batch_size):
                batch = texts[i : i + self._batch_size]
                resp = await client.post(
                    self.BASE_URL,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={"model": self.MODEL, "input": batch, "input_type": input_type},
                )
                if resp.status_code != 200:
                    logger.error("Voyage API error %d: %s", resp.status_code, resp.text[:200])
                    resp.raise_for_status()

                data = resp.json()
                for item in data.get("data", []):
                    all_embeddings.append(item["embedding"])
        return all_embeddings


# ── Cohere Embed v3 ────────────────────────────────────────────


class CohereEmbedder(EmbeddingProvider):
    """Cohere Embed v3 client (model: embed-english-v3.0)."""

    MODEL = "embed-english-v3.0"
    BASE_URL = "https://api.cohere.com/v1/embed"

    def __init__(self, api_key: Optional[str] = None, batch_size: int = 32) -> None:
        self._api_key = api_key or COHERE_API_KEY
        if not self._api_key:
            raise ValueError("Cohere API key not set. Call set_cohere_key() or pass api_key.")
        self._batch_size = batch_size
        self._dim = 1024

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts: List[str]) -> List[List[float]]:
        return await self._call(texts, input_type="search_document")

    async def embed_query(self, text: str) -> List[float]:
        results = await self._call([text], input_type="search_query")
        return results[0]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30), reraise=True)
    async def _call(self, texts: List[str], input_type: str) -> List[List[float]]:
        all_embeddings: List[List[float]] = []
        async with httpx.AsyncClient(timeout=60.0) as client:
            for i in range(0, len(texts), self._batch_size):
                batch = texts[i : i + self._batch_size]
                resp = await client.post(
                    self.BASE_URL,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.MODEL,
                        "texts": batch,
                        "input_type": input_type,
                        "embedding_types": ["float"],
                    },
                )
                if resp.status_code != 200:
                    logger.error("Cohere API error %d: %s", resp.status_code, resp.text[:200])
                    resp.raise_for_status()

                data = resp.json()
                all_embeddings.extend(data.get("embeddings", []))
        return all_embeddings


# ── Local sentence-transformers fallback ───────────────────────


class LocalEmbedder(EmbeddingProvider):
    """Local sentence-transformers fallback embedder.

    Model defaults to BAAI/bge-m3 for compatibility with existing indexer.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        self._model_name = model_name
        self._model = None
        self._dim = 1024 if "bge-m3" in model_name else 768

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts: List[str]) -> List[List[float]]:
        model = await self._get_model()
        emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return emb.tolist()

    async def embed_query(self, text: str) -> List[float]:
        model = await self._get_model()
        emb = model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        return emb[0].tolist()

    async def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading local embedder: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model


# ── Factory ─────────────────────────────────────────────────────


def create_embedder(provider: str = "local", **kwargs) -> EmbeddingProvider:
    """Factory: create embedder by provider name.

    Args:
        provider: One of 'voyage', 'cohere', 'local'.
        **kwargs: Passed to the embedder constructor (api_key, model_name, etc.)

    Returns:
        EmbeddingProvider instance.
    """
    if provider == "voyage":
        return VoyageEmbedder(**kwargs)
    elif provider == "cohere":
        return CohereEmbedder(**kwargs)
    else:
        return LocalEmbedder(**kwargs)