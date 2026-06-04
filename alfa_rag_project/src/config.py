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

# Chunking parameters
CHUNK_SIZE: Final[int] = 200  # Target chunk size in characters (reduced for shorter answers)
CHUNK_OVERLAP: Final[int] = 30  # Overlap between chunks in characters

# Retrieval parameters
TOP_K_RETRIEVAL: Final[int] = 15  # Number of candidates from FAISS
TOP_K_RERANK: Final[int] = 3  # Number of final results after reranking

# Generation parameters - CRITICAL FOR BERT-RECALL-L
MAX_RESPONSE_WORDS: Final[int] = 15  # Hard limit for response length (reduced from 30)
MAX_RESPONSE_CHARS: Final[int] = 150  # Hard character limit
TEMPERATURE: Final[float] = 0.1  # Low temperature for deterministic output