"""
Enterprise ETL document loader using Unstructured.io (API or local).

Async parsing with tenacity retries. Supports PDF, DOCX, PPTX, HTML.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config import DATA_DIR

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx", ".pptx", ".html", ".htm", ".txt", ".md"})

ENTERPRISE_API_KEY: Optional[str] = None  # Set via enterprise_settings or env


class DocumentLoadError(Exception):
    """Raised when document parsing fails."""


class Document:
    """A parsed document ready for chunking."""

    def __init__(self, doc_id: str, source: str, content: str, metadata: Optional[dict] = None) -> None:
        self.doc_id = doc_id
        self.source = source
        self.content = content
        self.metadata = metadata or {}


def set_api_key(key: str) -> None:
    global ENTERPRISE_API_KEY
    ENTERPRISE_API_KEY = key


async def load_document(source: str) -> Document:
    """
    Parse a single document file.

    Uses Unstructured API if ENTERPRISE_API_KEY is set,
    otherwise falls back to local partition.
    """
    path = Path(source)
    if not path.exists():
        raise DocumentLoadError(f"File not found: {source}")
    if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        raise DocumentLoadError(f"Unsupported extension: {path.suffix}")

    doc_id = _make_doc_id(path)
    content = await _parse(path)
    return Document(
        doc_id=doc_id,
        source=str(path.resolve()),
        content=content,
        metadata={"filename": path.name, "extension": path.suffix},
    )


async def load_documents(sources: list[str]) -> list[Document]:
    """Parse multiple documents concurrently."""
    import asyncio

    tasks = [load_document(src) for src in sources]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    docs: list[Document] = []
    for source, result in zip(sources, results):
        if isinstance(result, Exception):
            logger.error("Failed to load %s: %s", source, result)
            continue
        docs.append(result)
    return docs


def _make_doc_id(path: Path) -> str:
    """Generate a deterministic document ID from the file path."""
    raw = str(path.resolve()).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


async def _parse(path: Path) -> str:
    """Parse with API or local fallback."""
    if ENTERPRISE_API_KEY:
        return await _parse_via_api(path)
    return await _parse_via_partition(path)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30), reraise=True)
async def _parse_via_api(path: Path) -> str:
    """Parse via Unstructured API."""
    url = "https://api.unstructuredapp.io/general/v0/general"
    async with httpx.AsyncClient(timeout=120.0) as client:
        with open(path, "rb") as f:
            files = {"files": (path.name, f)}
            headers = {"api-key": ENTERPRISE_API_KEY or ""}
            resp = await client.post(url, files=files, headers=headers)
        if resp.status_code != 200:
            raise DocumentLoadError(f"Unstructured API {resp.status_code}: {resp.text[:200]}")
        elements = resp.json()
    return "\n\n".join(
        e.get("text", "") for e in elements
        if e.get("type") in ("NarrativeText", "ListItem", "Title", "Table")
    )


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
async def _parse_via_partition(path: Path) -> str:
    """Local fallback via unstructured.partition.auto."""
    try:
        from unstructured.partition.auto import partition
    except ImportError as exc:
        raise DocumentLoadError("Install unstructured: pip install unstructured[pdf,docx,pptx]") from exc

    elements = partition(filename=str(path), strategy="auto")
    return "\n\n".join(
        el.text for el in elements
        if el.text and el.category in ("NarrativeText", "ListItem", "Title", "Table")
    )