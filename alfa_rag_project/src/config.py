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
CHUNK_SIZE: Final[int] = 500  # Target chunk size in characters (3-5 sentences)
CHUNK_OVERLAP: Final[int] = 120  # Overlap between chunks in characters

# Retrieval parameters - optimized for quality
TOP_K_RETRIEVAL: Final[int] = 40  # Number of candidates from FAISS
TOP_K_BM25: Final[int] = 15  # Number of candidates from BM25
TOP_K_RERANK: Final[int] = 15  # Number of final results after reranking

# Reranker batch size for memory efficiency (prevents CUDA OOM)
RERANKER_BATCH_SIZE: Final[int] = 4  # 4 — агрессивно мало для 2xT4 (было 15 — OOM)

# Минимальный score reranker'а для формирования ответа
# Если лучший чанк набрал score < MIN_RERANK_SCORE, вопрос нерелевантен контексту.
# Отдаём "Нет ответа." — это совпадает с эталоном (q_id=7,13) и сохраняет BERTScore.
MIN_RERANK_SCORE: Final[float] = 0.05

# Generation parameters - optimized for BERT-Recall-L
# Эталоны (sample_submission.csv) имеют медиану ~200-280 символов, max 700+.
# Порог без штрафа = 1.5 * Lr. Цель: покрыть recall, не уходя в 3x.
# 550 символов сидит между 1.5xLr (~350-400) и 3xLr (~700-800).
MAX_SENTENCES: Final[int] = 5          # было 2 — душило recall на длинных эталонах
MAX_RESPONSE_WORDS: Final[int] = 80    # было 30
MAX_RESPONSE_CHARS: Final[int] = 550   # было 150 — главная утечка скора
TEMPERATURE: Final[float] = 0.1  # Low temperature for deterministic output

# API timeout
LLM_TIMEOUT: Final[int] = 30  # Timeout for LLM API calls in seconds