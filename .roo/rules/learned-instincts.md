# Learned Instincts

This file acts as a local memory bank for project-specific patterns, conventions, and gotchas discovered during development.

---

## Template

```markdown
## [Date] - [Task Description]

### Patterns Discovered
- [Pattern 1 with file reference]
- [Pattern 2 with file reference]

### Conventions
- [Convention 1]
- [Convention 2]

### Gotchas
- [Gotcha 1 and solution]
- [Gotcha 2 and solution]
```

---

## 2026-06-01 - RAG Pipeline for Alfa-Bank MIPT Hackathon

### Patterns Discovered
- Sentence-aware chunking with `razdel.sentenize` to avoid breaking Russian text mid-sentence
- FAISS `IndexFlatIP` with normalized embeddings for cosine similarity
- Cross-encoder reranking after initial vector retrieval
- Post-processing truncation to enforce BERT-Recall-L metric constraints
- Fallback extraction from context when LLM unavailable

### Conventions
- All modules use type hints and dataclasses
- Configuration centralized in `config.py` with Final constants
- Chunk IDs are sequential integers across all documents
- JSON for chunk mapping (human-readable), FAISS binary for index

### Gotchas
- Ollama requires `api_key="ollama"` even though it's not used
- BGE-M3 produces 1024-dimensional embeddings
- Need to handle empty context gracefully in generator
- Russian morphology: "счёта" != "счет" - need normalization
- CSV column is `text`, not `web` in websites.csv

---

## 2026-06-03 - Integrated answer extraction methods into generator.py

### Patterns Discovered
- TF-IDF scoring with position weighting for sentence ranking in `extract_answer_from_context`
- Junk sentence filtering using regex patterns for marketing phrases and short fragments
- Duplicate query detection to avoid returning question paraphrases
- Informative bonus scoring for sentences with action verbs and procedural language
- Multi-sentence answer assembly with position-based ordering

### Conventions
- `ExtractorConfig` dataclass for configurable extraction parameters
- Private helper functions (`_is_junk_sentence`, `_is_duplicate_of_query`, `_compute_tfidf_score`, `_informative_bonus`)
- `_ScoredSentence` internal dataclass for scoring pipeline
- `extract_answer_from_context` returns empty string when no answer found (not "Недостаточно информации")

### Gotchas
- Test assertions must account for original text preservation (ё stays as ё in output)
- Short sentences (<4 words) are filtered by default, adjust `min_sentence_words` in config if needed
- Junk patterns include: "узнайте больше", "подробнее на сайте", "свяжитесь с нами", etc.

---

## 2026-06-03 - Integrated chunk cleaning methods into retriever.py

### Patterns Discovered
- HTML tag stripping with `html.unescape()` for entity decoding
- Whitespace normalization for non-breaking spaces, tabs, and special characters
- Decorative character removal (stars, pipes, underscores as separators)
- Incomplete sentence trimming to avoid mid-sentence chunk boundaries
- Chunk ID prefix removal for clean context output

### Conventions
- `CleanerConfig` dataclass for configurable cleaning parameters
- Private helper functions (`_remove_chunk_id_prefix`, `_strip_html`, `_normalize_whitespace`, `_remove_decorative`, `_trim_incomplete_sentence`)
- `clean_chunk_text()` returns empty string when text is too short after cleaning
- `get_context()` now uses cleaning pipeline before returning context

### Gotchas
- HTML entities must be decoded before tag removal (order matters)
- Non-breaking spaces (\xa0) and zero-width chars (\u200b) need explicit handling
- Incomplete sentence trimming requires checking for [.!?»")], endings
- Minimum length check (20 chars) prevents empty/whitespace-only chunks

---

## 2026-06-03 - Integrated text cleaning methods into chunker.py

### Patterns Discovered
- HTML entity decoding with `html.unescape()` before tag removal
- Service phrase removal for bank FAQ boilerplate text
- Whitespace normalization for special characters
- `Chunker` class with configurable parameters
- `clean_text()` function for pre-chunking text preparation

### Conventions
- `ChunkerConfig` dataclass for configurable chunking parameters
- Private helper functions for cleaning pipeline
- `clean_text()` removes service phrases before chunking
- `Chunker.chunk_text()` provides full pipeline: clean → split → build → filter

### Gotchas
- Service phrases must be ordered from specific to general to avoid over-matching
- One-word sentences are filtered as artifacts in `_split_into_sentences`
- Long sentences (> chunk_size) are emitted as standalone chunks

---

## 2026-06-03 - Integrated deduplication and normalization into indexer.py

### Patterns Discovered
- `normalize_for_embedding()` with ё→е translation and NFC Unicode normalization
- SHA-256 hashing for chunk deduplication (first 16 bytes)
- FAISS ID mapping separate from original chunk IDs
- JSON serialization with string keys, int keys on load

### Conventions
- `_YO_TABLE` for fast ё→е translation using str.translate()
- `_compute_chunk_hash()` normalizes text before hashing
- `deduplicate_chunks()` returns tuple of (unique_chunks, dropped_count)
- `build_index()` pipeline: deduplicate → normalize → embed → index → map

### Gotchas
- FAISS assigns sequential IDs (0, 1, 2...) independent of chunk.chunk_id
- JSON doesn't support int keys, so convert to str on save and back on load
- Original text preserved in mapping (with ё), normalized only for embedding

---

## 2026-06-04 - Created Obsidian REST API client module

### Patterns Discovered
- Context manager pattern for HTTP session lifecycle management
- Factory function `create_obsidian_client()` for convenient instantiation
- Dataclass-based configuration with frozen=True for immutability

### Conventions
- `ObsidianConfig` dataclass for configurable parameters (base_url, timeout, api_key)
- `ObsidianClient` class with private `_session` attribute
- All methods use `raise_for_status()` for explicit error handling
- Type hints on all function signatures

### Gotchas
- Default port 27777 is standard for Obsidian REST API plugin
- API key is optional - some configurations don't require authentication
- `append_to_file` reads existing content first, then updates

---

## 2026-06-04 - Fixed BERT-Recall-L Length Penalty Issue

### Patterns Discovered
- Character-based truncation (`truncate_to_chars`) is PRIMARY for BERT-Recall-L compliance
- Word-based truncation (`truncate_to_words`) is SECONDARY, applied before char truncation
- Dual-limit strategy: MAX_RESPONSE_WORDS=15 + MAX_RESPONSE_CHARS=150 ensures answers stay under 3x reference length
- Retriever's `clean_chunk_text()` already removes `[chunk_id]` prefixes, so context is clean

### Conventions
- `truncate_to_chars()` tries to end at sentence boundary (., !, ?, ») before hard cut
- Both `Generator.generate()` and `extract_answer_from_context()` apply char truncation
- `MAX_RESPONSE_CHARS` imported in generator.py for direct use

### Gotchas
- BERT-Recall-L metric: Length Coefficient = 0 if answer >= 3x reference length
- Need to truncate BEFORE returning answer, not after
- Character limit (150) is more restrictive than word limit (15 words ≈ 100-120 chars)
- Test `test_char_truncation_applied` verifies extraction respects char limit

---

## 2026-06-05 - Fixed extraction to be greedy (score 0.23 issue)

### Patterns Discovered
- Sentence-based truncation (`truncate_to_sentences`) is PRIMARY limit, not character-based
- Greedy extraction: always return something from context, never empty string
- Lowered `min_sentence_words` from 4 to 2 to allow shorter informative sentences
- Lowered `min_score` from 0.1 to 0.01 to include more candidates

### Conventions
- `extract_answer_from_context` scores ALL sentences, not just filtered ones
- Fallback returns first non-junk sentence if no scored candidates
- MAX_RESPONSE_CHARS=450 (not 150) allows 3x reference length without penalty

### Gotchas
- "Недостаточно информации" gives 0 points - extraction must return context content
- Short sentences (2-3 words) can be informative in banking context
- Need to balance between precision and recall in extraction

---

## 2026-06-06 - Code Review: Low Score (0.22) Root Cause Analysis

### Patterns Discovered
- SYSTEM_PROMPT explicitly told LLM to return "Недостаточно информации" - this kills BERT-Recall-L score
- All generators (OR_main, kaggle_main, main) returned "Недостаточно информации" on errors instead of fallback extraction
- `extract_answer_from_context` had multiple return paths for "Недостаточно информации" instead of greedy mode

### Conventions
- Changed SYSTEM_PROMPT: "Если в контексте нет прямого ответа, выбери самое релевантное предложение из контекста"
- All generators now use `extract_answer_from_context` as fallback instead of "Недостаточно информации"
- `extract_answer_from_context` is now greedy: always returns first sentence if no scored candidates

### Gotchas
- "Недостаточно информации" gives 0 points on leaderboard - extraction must return context content
- Empty context in generator should still call fallback to get something from retriever
- Need to handle edge case: `if not scored` with empty `raw_sentences` - added safety check

### Code Review Findings

#### main.py
- ✅ Good: AnswerCache with persistent JSON cache
- ✅ Good: validate_answer with word overlap check
- ⚠️ Fixed: Exception handler now uses `extract_answer_from_context(query, retriever.get_context(query))` instead of empty string

#### OR_main.py
- ✅ Good: OpenRouter API integration with timeout
- ⚠️ Fixed: `generate()` now uses fallback for empty context
- ⚠️ Fixed: Exception handler now uses `extract_answer_from_context` instead of "Недостаточно информации"

#### kaggle_main.py
- ✅ Good: Multi-GPU support for 2x T4
- ⚠️ Fixed: `generate()` now uses fallback for empty context
- ⚠️ Fixed: Exception handler now uses `extract_answer_from_context` instead of "Недостаточно информации"

#### generator.py
- ⚠️ Fixed: SYSTEM_PROMPT no longer explicitly requests "Недостаточно информации"
- ⚠️ Fixed: `extract_answer_from_context` is now greedy - always returns something
- ⚠️ Fixed: `Generator.generate()` uses fallback for empty context

---

## 2026-06-06 - Implemented 5 Architectural Improvements

### Patterns Discovered
- Hybrid search (BM25 + FAISS) increases recall by combining semantic and lexical search
- "Lost in the Middle" fix: reversing chunk order ensures relevant info appears at both start and end
- Few-shot prompting improves answer quality and consistency
- MAX_RESPONSE_CHARS=450 allows 3x reference length without penalty
- TOP_K_RERANK=10 provides more candidates for cross-encoder reranking

### Conventions
- `TOP_K_BM25` config constant for BM25 retrieval size (15)
- `_tokenize_for_bm25()` helper for Russian text tokenization
- `_bm25_search()` method in Retriever for lexical search
- `retrieve()` now merges FAISS and BM25 candidates before reranking
- `get_context()` reverses chunk order to combat "Lost in the Middle"
- SYSTEM_PROMPT includes 3 few-shot examples for better instruction following

### Gotchas
- BM25 requires rank_bm25 package: `pip install rank-bm25`
- BM25 tokenization must handle Russian characters (а-яё)
- Reversing chunks may affect context coherence - but improves LLM attention
- Few-shot examples should be in Russian to match model training
- Need to ensure TOP_K_RERANK >= 4 to get enough candidates after merge

---

## 2026-06-07 - Fixed CUDA OOM in Cross-Encoder Reranking

### Patterns Discovered
- Cross-encoder reranker (BGE-reranker-v2-m3) requires ~7 GiB for batch of 20-30 pairs
- Kaggle T4 GPU (14.56 GiB) runs out of memory when processing all candidates at once
- Memory fragmentation from PyTorch causes "reserved but unallocated" memory waste
- Exception handlers calling `get_context()` again cause double OOM on same query

### Conventions
- `RERANKER_BATCH_SIZE=4` in config.py for memory-efficient processing
- Batched reranking in `retrieve()` with `torch.cuda.empty_cache()` after each batch
- Exception handlers now use pre-fetched context instead of calling `get_context()` again

### Gotchas
- `sentence_transformers.CrossEncoder.predict()` supports `batch_size` parameter
- Need to import `torch` in retriever.py for `empty_cache()` call
- Error handling must avoid re-calling the failing function
- GPU memory fragmentation is worse than actual usage - clear cache proactively

---

## 2026-06-07 - Performance Optimization for Kaggle 12-Hour Limit

### Patterns Discovered
- 6-7 seconds per question = ~12+ hours for 6977 questions (exceeds Kaggle limit)
- LLM generation (Qwen-7B) takes ~4-5s, retrieval takes ~1-2s
- int8 quantization can reduce LLM inference time by ~30-50%
- Reducing retrieval candidates (10→5) saves ~0.5s per question

### Conventions
- `USE_INT8=True` in kaggle_main.py for int8 quantization
- `BitsAndBytesConfig(load_in_8bit=True)` for proper int8 quantization
- `TOP_K_RETRIEVAL=5` and `TOP_K_BM25=5` in config.py
- Expected time: 6-7s → 3-4s per question (under 12 hours total)

### Gotchas
- int8 quantization requires bitsandbytes package
- `load_in_8bit` must be passed via `quantization_config` parameter, not directly
- May slightly reduce answer quality but acceptable for speed
- Need to ensure enough candidates after merge (5+5=10, still enough for reranking)
- Qwen recommended for Russian banking context due to better instruction following

---

## 2026-06-07 - Reverted to Quality-Optimized Parameters

### Patterns Discovered
- Reduced retrieval parameters (15→5) degraded answer quality significantly
- int8 quantization on Kaggle T4 caused 10-12s per question (slower than expected)
- OR_main.py (OpenRouter API) provides better quality with acceptable speed
- Quality is more important than speed for leaderboard score

### Conventions
- `TOP_K_RETRIEVAL=15` and `TOP_K_BM25=15` restored for better recall
- `TOP_K_RERANK=10` for more candidates after merge
- Using OR_main.py for production inference via OpenRouter API
- kaggle_main.py kept as fallback for offline scenarios

### Gotchas
- int8 quantization may not work well with all model architectures
- Qwen2.5-7B on Kaggle T4 with int8 was slower than expected
- API-based inference (OR_main) more reliable for time-constrained competitions
- Keep both implementations for different deployment scenarios
