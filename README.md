# Alfa-Bank RAG Hackathon Pipeline

Pure Python RAG system built for the Alfa-Bank MIPT hackathon.  
**Final pipeline score:** BERTScore-Recall-L optimized.

## Quick Start (Kaggle 2xT4)

```python
!git clone https://github.com/KremlevLev/RAG-Pipeline-for-High-Load-Banking-Data.git /kaggle/working/rag_alpha_hack
!pip install -r /kaggle/working/rag_alpha_hack/requirements.txt
```

```bash
cd /kaggle/working/rag_alpha_hack && python src/kaggle_main.py --build-index --fast-quality --no-validate
```

Output: `data/submission.csv` (~7000 answers, ~2–4 hours).

## Debug (first 100 questions)

```bash
python src/kaggle_main.py --build-index --fast-quality --no-validate --limit 100
```

## CLI Options

| Flag | Description |
|------|-------------|
| `--build-index` | Rebuild FAISS index from scratch |
| `--model MODEL` | LLM model key (default: qwen2.5-3b with `--fast-quality`) |
| `--fast-quality` | **Production mode:** vLLM + Qwen2.5-3B + reduced candidate pools |
| `--legacy-chunking` | Disable parent-child retrieval (fallback to single-level chunks) |
| `--limit N` | Generate only first N questions |
| `--no-validate` | Disable answer validation (recommended for speed) |
| `--vllm` | Explicit vLLM mode (auto-enabled by `--fast-quality`) |
| `--vllm-batch-size N` | vLLM batch size (default: 16 with `--fast-quality`) |

## Architecture

### GPU Layout (2xT4, `--fast-quality`)

| GPU | Process |
|-----|---------|
| **cuda:0** | vLLM (Qwen2.5-3B, `tensor_parallel_size=1`) |
| **cuda:1** | Cross-Encoder Reranker (BAAI/bge-reranker-v2-m3) |

Qwen2.5-3B fits in ~6 GiB on one T4 (14.56 GiB). The second GPU is reserved for the reranker — avoids CPU fallback which would make generation >6.5s/question and exceed Kaggle's 12-hour session.

### Pipeline Flow

```text
websites.csv → Parent-Child Chunker → FAISS + BM25 → RRF Fusion
    → Parent Expansion → Cross-Encoder Rerank → LLM → Post-process → submission.csv
```

1. **Parent-Child Chunking** — large parent chunks (1100 chars) split into smaller child chunks (500 chars). Children are indexed for precise matching; parents are used for LLM context.
2. **FAISS Semantic Search** — `BAAI/bge-m3` embeddings, top-30 child candidates (fast-quality mode).
3. **BM25 Lexical Search** — exact term matching, top-15 child candidates.
4. **RRF Fusion** — Reciprocal Rank Fusion merges results, expands child hits back to parent chunks.
5. **Cross-Encoder Reranking** — `BAAI/bge-reranker-v2-m3` on cuda:1, top-8 parent chunks to LLM.
6. **LLM Generation** — Qwen2.5-3B via vLLM (continuous batching).
7. **Post-processing** — garbage detection, reference rescue, adaptive truncation.

### Speed Optimizations (fast-quality mode)

| Parameter | Normal | fast-quality |
|-----------|--------|-------------|
| `TOP_K_RETRIEVAL` | 40 | **30** |
| `TOP_K_BM25` | 20 | **15** |
| `TOP_K_RERANK` | 10 | **8** |
| `tensor_parallel_size` | 1 | **1** (leaves cuda:1 for reranker) |
| `gpu_memory_utilization` | 0.55 | **0.80** |
| Speed guard | Disabled | Disabled |

## Project Structure

```text
├── README.md
├── requirements.txt
├── docker-compose.yml          # Qdrant + Phoenix (enterprise)
├── .env.example                # API keys template
├── src/
│   ├── kaggle_main.py          # **Main pipeline entry point**
│   ├── config.py               # All hyperparameters
│   ├── chunker.py              # Sentence-aware chunking
│   ├── parent_child.py         # Parent-child chunking (default)
│   ├── indexer.py              # FAISS index + BGE-M3 embeddings
│   ├── retriever.py            # Hybrid search + BM25 + reranker
│   ├── generator.py            # LLM generation + fallback extraction
│   ├── loader.py               # Enterprise: async Unstructured ETL
│   ├── reranker.py             # Enterprise: Cohere Rerank v3 client
│   ├── embeddings/             # Enterprise: Voyage/Cohere/BGE embedders
│   ├── storage/                # Enterprise: Qdrant hybrid store
│   ├── orchestrator/           # Enterprise: LangGraph Self-RAG
│   ├── generation/             # Enterprise: Claude/OpenRouter clients
│   └── observability/          # Enterprise: Phoenix/LangSmith tracing
└── tests/
    ├── test_kaggle_cleanup.py  # Garbage detection tests
    ├── test_parent_child.py    # Parent-child retrieval tests
    ├── test_loader.py          # ETL loader tests
    ├── test_embeddings.py      # Embedder tests
    ├── test_reranker.py        # Reranker tests
    ├── test_orchestrator.py    # LangGraph tests
    ├── test_generation.py      # LLM client tests
    └── test_observability.py   # Tracing tests
```

## Key Features

- **Parent-child retrieval** — index small chunks for precision, expand to parents for LLM context
- **GPU-aware scheduling** — vLLM on cuda:0, reranker on cuda:1 (no contention)
- **Hybrid search** — FAISS + BM25 + Reciprocal Rank Fusion
- **Cross-encoder reranking** — BAAI/bge-reranker-v2-m3 on dedicated GPU
- **Reference answer injection** — dominant hint from `sample_submission.csv`
- **Garbage detection** — mixed-script, numeric-only, question-only, repetition patterns
- **Reference rescue** — fallback to cleaned reference answer when LLM output is garbage
- **Checkpoint resume** — saves every 2000 answers, resumes from last checkpoint
- **Persistent cache** — JSON-based answer cache to avoid regenerating identical queries
- **BERTScore-aware truncation** — MAX_RESPONSE_CHARS=550, MAX_SENTENCES=5

## Configuration (`src/config.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `PARENT_CHUNK_SIZE` | 1100 | Parent chunk (LLM context unit) |
| `PARENT_CHUNK_OVERLAP` | 220 | Overlap between parent chunks |
| `CHUNK_SIZE` | 500 | Child chunk (indexed unit) |
| `CHUNK_OVERLAP` | 120 | Overlap between child chunks |
| `TOP_K_RETRIEVAL` | 40 | FAISS candidates (30 in fast-quality) |
| `TOP_K_BM25` | 20 | BM25 candidates (15 in fast-quality) |
| `TOP_K_RERANK` | 10 | Reranked results (8 in fast-quality) |
| `TOP_K_CONTEXT` | 8 | Chunks passed to LLM |
| `MAX_SENTENCES` | 5 | Max sentences in answer |
| `MAX_RESPONSE_CHARS` | 550 | Safety limit for BERTScore |

## Known Issues

1. **vLLM + merged model** — fine-tuned Vikhr-1B has broken tokenizer config for vLLM; auto-switches to base Vikhr tokenizer.
2. **Index incompatibility** — parent-child index differs from legacy index. Use `--legacy-chunking` if loading a legacy index.
3. **First run slow** — model downloads + index building take ~1 hour on Kaggle.

## Enterprise Modules (Optional)

The following modules provide production-grade replacements for the local pipeline components.
They require API keys (see `.env.example`) and external services (Qdrant, Cohere, etc.).

| Module | Replaces | Purpose |
|--------|----------|---------|
| `loader.py` | — | Async Unstructured.io document parsing |
| `embeddings/` | Indexer embedder | Voyage AI / Cohere Embed v3 |
| `storage/qdrant_store.py` | FAISS + BM25 | Qdrant hybrid search (dense + sparse) |
| `reranker.py` | BGE reranker | Cohere Rerank v3 |
| `orchestrator/` | kaggle_main loop | LangGraph Self-RAG with query rewrite |
| `generation/` | KaggleGenerator | Claude / GPT-4o / OpenRouter |
| `observability/` | — | Arize Phoenix / LangSmith tracing |

## Metric

**BERTScore-Recall-L** with relative length penalty:

- No penalty if `L <= 1.5 × reference_length`
- Linear penalty from 1.5× to 3×
- Zero score if `L >= 3× reference_length`

Reference answers (250–450+ chars) are from `data/sample_submission.csv` (allowed leakage).
