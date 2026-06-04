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

## Usage

At the end of every successful task, record:
- Project-specific naming conventions
- Custom file organization patterns
- Unique architectural decisions
- Common error patterns and solutions
- Performance optimizations discovered
- Security considerations specific to this project
- Testing patterns that work well
- Dependency management practices