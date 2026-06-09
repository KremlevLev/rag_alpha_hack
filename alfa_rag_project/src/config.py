"""
Configuration module for RAG pipeline.
Contains all paths, model names, and hyperparameters.
"""

from pathlib import Path
from typing import Final

# Project root directory
PROJECT_ROOT: Final[Path] = Path(__file__).parent.parent

# Data paths
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
WEBSITES_CSV: Final[Path] = DATA_DIR / "websites.csv"
QUESTIONS_CSV: Final[Path] = DATA_DIR / "questions.csv"
SUBMISSION_CSV: Final[Path] = DATA_DIR / "submission.csv"

# FAISS index and metadata paths
INDEX_PATH: Final[Path] = DATA_DIR / "faiss_index.bin"
CHUNK_MAPPING_PATH: Final[Path] = DATA_DIR / "chunk_mapping.json"

# Model names
EMBEDDER_MODEL: Final[str] = "BAAI/bge-m3"
RERANKER_MODEL: Final[str] = "BAAI/bge-reranker-v2-m3"

# LLM configuration (Ollama/vLLM)
LLM_BASE_URL: Final[str] = "http://localhost:11434/v1"
LLM_MODEL: Final[str] = "qwen2.5:7b"  # Default model, can be changed

# Chunking parameters - optimized for context density
CHUNK_SIZE: Final[int] = 450  # Target chunk size in characters (3-5 sentences)
CHUNK_OVERLAP: Final[int] = 100  # Overlap between chunks in characters

# Retrieval parameters - optimized for quality
TOP_K_RETRIEVAL: Final[int] = 30  # Number of candidates from FAISS
TOP_K_BM25: Final[int] = 25  # Number of candidates from BM25
TOP_K_RERANK: Final[int] = 10  # Number of final results after reranking

# Reranker batch size for memory efficiency (prevents CUDA OOM)
RERANKER_BATCH_SIZE: Final[int] = 15  # Process 15 pairs at a time (Vikhr-1B is smaller, more memory available)

# Generation parameters - optimized for BERT-Recall-L
MAX_SENTENCES: Final[int] = 2  # Maximum sentences in answer (primary limit)
MAX_RESPONSE_WORDS: Final[int] = 30  # Soft word limit for density
MAX_RESPONSE_CHARS: Final[int] = 150  # Hard safety limit (3x reference length)
TEMPERATURE: Final[float] = 0.1  # Low temperature for deterministic output

# API timeout
LLM_TIMEOUT: Final[int] = 30  # Timeout for LLM API calls in seconds