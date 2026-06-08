# RAG Pipeline for Alfa-Bank MIPT Hackathon

Pure Python RAG system without heavy frameworks (LangChain, Haystack).

## Project Structure

```
alfa_rag_project/
├── requirements.txt     # Dependencies
├── README.md           # This file
├── data/
│   ├── websites.csv    # Input: web_id, website, text
│   ├── questions.csv   # Input: q_id, query
│   ├── submission.csv  # Output: predictions
│   ├── faiss_index.bin # FAISS index (generated)
│   └── chunk_mapping.json  # Chunk metadata (generated)
└── src/
    ├── config.py       # Configuration and constants
    ├── chunker.py      # Text chunking with razdel
    ├── indexer.py      # FAISS index management
    ├── retriever.py    # Vector search + cross-encoder reranking
    ├── generator.py    # LLM generation with brevity constraints
    ├── main.py         # Pipeline orchestrator (Ollama/local)
    ├── kaggle_main.py  # Kaggle-optimized pipeline (Hugging Face)
    ├── OR_main.py      # OpenRouter API pipeline
    ├── finetuning.py   # LoRA/QLoRA fine-tuning module
    ├── __main__.py     # Entry point for `python -m main`
    └── __init__.py     # Package exports
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Local/Ollama (main.py)
```bash
cd alfa_rag_project/src
python main.py --build-index --model qwen2.5:7b
```

### Kaggle (kaggle_main.py) - Open-source models with 2x T4 GPU
```bash
cd alfa_rag_project/src
python kaggle_main.py --build-index --model vikhr-1b
```

### Available Kaggle models
- `vikhr-1b-finetuned` - lirex111/vikhrllama1B_AlfaBank (fine-tuned for Alfa-Bank, RECOMMENDED)
- `vikhr-1b` - Vikhrmodels/Vikhr-Llama-3.2-1B-instruct (fast, base model)
- `qwen2.5-7b` - Qwen/Qwen2.5-7B-Instruct (recommended for Russian)
- `qwen2-7b` - Qwen/Qwen2-7B-Instruct
- `mistral-7b` - Mistral-7B-Instruct-v0.3
- `llama3-8b` - Meta-Llama-3-8B-Instruct

### Fine-tuning with LoRA/QLoRA

Create a fine-tuning dataset from sample_submission.csv (75.8 score baseline):

```bash
python finetuning.py --create-dataset
```

Run LoRA/QLoRA fine-tuning:

```bash
# QLoRA (4-bit quantization, memory efficient)
python finetuning.py --train --model vikhr-1b --epochs 3 --batch-size 4

# Regular LoRA (full precision)
python finetuning.py --train --model vikhr-1b --epochs 3 --batch-size 4 --no-qlora
```

The fine-tuned model will be saved to `data/finetuned_models/` and can be used for inference.

### OpenRouter (OR_main.py) - API-based inference
```bash
# Set API key
export OPENROUTER_API_KEY="sk-or-..."

# Run
python OR_main.py --model qwen2.5-7b
```

No model downloads - uses OpenRouter API for open-source models.

## Architecture

### Pipeline Flow

```
questions.csv → Retriever → Generator → submission.csv
                   ↓
             FAISS + BM25 + Reranker
```

### Two-Stage Retrieval

1. **FAISS Semantic Search** - BGE-M3 embeddings, finds semantically similar chunks
2. **BM25 Lexical Search** - Exact keyword matching, finds exact term matches
3. **Cross-Encoder Reranking** - BGE-reranker-v2-m3 scores and re-ranks merged candidates

### Answer Generation

1. **Context Retrieval** - Top-k chunks cleaned and formatted
2. **LLM Generation** - Qwen/Qwen2.5-7B-Instruct with few-shot prompting
3. **Post-processing** - Truncation to 2 sentences + 150 chars (BERT-Recall-L compliance)

## Key Features

- **Chunking**: Sentence-aware splitting using `razdel.sentenize` (no broken sentences)
- **Embeddings**: BGE-M3 (1024-dim) with normalized vectors
- **Reranking**: BGE-reranker-v2-m3 cross-encoder (batched for memory efficiency)
- **Brevity**: System prompt + post-processing to limit response to ~30 words
- **FAISS**: Inner Product similarity for cosine with normalized vectors
- **Kaggle Support**: Hugging Face models with automatic 2x T4 GPU detection
- **Hybrid Search**: Combines semantic (FAISS) and lexical (BM25) search
- **Lost in the Middle Fix**: Reverses chunk order to improve LLM attention

## Configuration

Key parameters in `config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `CHUNK_SIZE` | 450 | Target chunk size in characters (3-5 sentences) |
| `CHUNK_OVERLAP` | 100 | Overlap between chunks in characters |
| `TOP_K_RETRIEVAL` | 15 | FAISS candidates |
| `TOP_K_BM25` | 15 | BM25 candidates |
| `TOP_K_RERANK` | 10 | Final results after reranking |
| `RERANKER_BATCH_SIZE` | 15 | Batch size for memory efficiency (Vikhr-1B is smaller) |
| `MAX_SENTENCES` | 2 | Maximum sentences in answer |
| `MAX_RESPONSE_CHARS` | 150 | Hard safety limit (3x reference length) |

## Kaggle Deployment

1. Clone repository in Kaggle notebook
2. Install dependencies: `!pip install -r requirements.txt`
3. Run: `python kaggle_main.py --build-index --model vikhr-1b`
4. Results saved to `data/submission.csv`

**Note**: First run will download models (~1-2GB). Use pre-built index if available.

## Checkpoints

Pipeline saves checkpoints every 2000 answers:
- `data/submission_checkpoint_2000.csv`
- `data/submission_checkpoint_4000.csv`
- `data/submission_checkpoint_6000.csv`

**Auto-resume:** Pipeline automatically resumes from the last checkpoint on restart.

## Memory Optimization

For Kaggle 2x T4 (14.56 GiB VRAM):
- `RERANKER_BATCH_SIZE=15` - Process 15 pairs at a time (Vikhr-1B is smaller, more memory available)
- Batched reranking prevents CUDA OOM
- `TOP_K_RETRIEVAL=15` and `TOP_K_BM25=15` for quality (can be reduced to 5 for speed)

## Module Details

### config.py
Centralized configuration with all paths, model names, and hyperparameters.

### chunker.py
- `Chunker` class with configurable parameters
- `clean_text()` - removes HTML, service phrases, normalizes whitespace
- `chunk_text()` - sentence-aware splitting with overlap

### indexer.py
- `build_and_save_index()` - creates FAISS index with BGE-M3 embeddings
- `load_index()` - loads existing index
- `normalize_for_embedding()` - ё→е translation, NFC normalization
- Deduplication via SHA-256 hashing

### retriever.py
- `Retriever` class with hybrid search
- `retrieve()` - FAISS + BM25 + reranking pipeline
- `get_context()` - cleaned context for LLM
- `clean_chunk_text()` - removes chunk ID, HTML, decorative chars

### generator.py
- `Generator` class for LLM calls
- `extract_answer_from_context()` - fallback extraction with TF-IDF scoring
- `truncate_to_sentences()` - primary truncation (BERT-Recall-L)
- `truncate_to_chars()` - safety truncation

### finetuning.py
- `create_finetuning_dataset()` - creates QA dataset from questions + sample_submission
- `load_model_for_finetuning()` - loads model with LoRA/QLoRA
- `train()` - runs supervised fine-tuning
- Supports QLoRA (4-bit) and regular LoRA

### main.py / kaggle_main.py / OR_main.py
- Three entry points for different environments
- `AnswerCache` - persistent JSON cache
- `validate_answer()` - word overlap validation
- Auto-resume from checkpoints