# Learned Instincts

This file acts as a local memory bank for project-specific patterns, conventions, and gotchas discovered during development.

---

## 2026-06-17 - LLM Output Cleanup for submission (18).csv

### Patterns Discovered
- `submission (18).csv` has heavy prompt/context leakage: `Вопрос:`, `Контекст для ответа на вопрос:`, `PASSWORDPASSWORD`, repeated HTML/navigation fragments, and `Нет ответа. --- Контекст`.
- The retriever context header `Контекст для ответа на вопрос:` is copied by weak models, so context should be passed to the LLM without that header.
- Fallback extraction must skip prompt metadata sentences before scoring; otherwise `extract_answer_from_context()` can return leaked headers.

### Conventions
- Keep user prompt label-free: pass `query + context` rather than `Вопрос:` / `Контекст:` / `Ответь кратко на основе контекста`.
- Strip service phrases from fallback answers: `Согласно фрагменту`, `В предоставленных фрагментах`, `Контекст для ответа`, prompt labels, and `Нет ответа. --- Контекст`.
- Treat repeated garbage tokens (`PASSWORD`, `Berkeley`, `correo`, `Raven reality`, `Cre1READ1 reality`, `totalitarian`, `icrobialatedi`) as garbage and fall back to extraction.

### Gotchas
- In this studio there is no GPU, so `kaggle_main.py --fast-quality` cannot be run end-to-end; it fails because vLLM is unavailable.
- The script path is `src/kaggle_main.py`; running `python kaggle_main.py` from the project root fails.
- Quick validation in this environment should use `py_compile`, pytest, and local cleanup simulations rather than full inference.
---

## 2026-06-17 - submission (19).csv garbage cleanup

### Patterns Discovered
- `submission (19).csv` contains 100 rows and still has repeated/corrupted tokens: `POLITICOfgets`, `bütün`, `ement.5`, `Lie`, Hangul `특`, `S.Appно`, `Сlderно`, `undergoingётного`, `estemitted`, `benefitedте`, `Qменить`, `Satисов`, `mustíBlend`, `CEPASSWORDS`, and repeated `SFSF...`.
- The strongest signal is mixed Latin/Cyrillic tokens inside one word, plus random all-Latin tokens that are not known banking/service terms.
- Reference answers are allowed leakage in this project (`README.md` says `sample_submission.csv` is allowed), so garbage LLM outputs should be rescued with the cleaned reference answer before falling back to context extraction.
- Reference preambles like `Согласно Фрагменту N,` and `В Фрагменте N:` should be stripped before using references as hints/fallbacks.

### Conventions
- `PIPELINE_VERSION` must change when prompt/context/post-processing changes; use `v3-reference-rescue-cleanup` for this fix so stale answer cache is not reused.
- `finalize_generated_answer()` should post-process LLM output, detect garbage, then use `clean_reference_answer(reference_answer)` if available, otherwise use `extract_answer_from_context()`.
- Keep known Latin banking/service tokens allowed: `SMS`, `IBAN`, `QR`, `POS`, `PIN`, `BIC`, `3DS`, `CashBack`, `HOLD`, URLs, and major payment/app names.

### Gotchas
- Do not run `kaggle_main.py` in this studio; it has no GPU/vLLM.
- Use `py_compile`, pytest, and in-memory cleanup simulations over existing submissions instead of full inference here.
- Local cleanup simulation over `data/submission (19).csv` reduced detected garbage from 46/100 to 4/100; the remaining 4 are reference-style answers containing `В предоставленных фрагментах`, not the previous corruption tokens.
---

## 2026-06-17 - submission (20).csv mixed-script cleanup

### Patterns Discovered
- `submission (20).csv` contains 100 rows; the new garbage is mostly mixed-script fragments rather than the previous all-Latin corruption tokens.
- Representative bad fragments include `*baseи移到`, `_Imageли`, `enroll-уведомление`, `интернет-б_song`, and replacement-character corruption like `от�тировать`.
- The previous mixed Latin/Cyrillic detector missed one-character transitions and hyphen/underscore joins, so `QR-код`, `3-D Secure`, `SMS`, `IBAN`, and `push-уведомления` must remain allowed while random mixed tokens are rejected.
- Reference fallback can still contain prompt-style phrases such as `Согласно предоставленным фрагментам`, `в предоставленных фрагментах`, `в Фрагменте N указан...`, and `Ответ: ...`; these should be stripped before using references as rescue answers.

### Conventions
- Bump `PIPELINE_VERSION` when post-processing changes; use `v4-mixed-script-cleanup` for this fix.
- Keep `finalize_generated_answer()` as the final gate: post-process LLM output, detect garbage, rescue with `clean_reference_answer(reference_answer)`, then fall back to `extract_answer_from_context()`.
- Add focused cleanup tests for CJK/mixed-script garbage and reference cleanup, not just generic repeated-token garbage.

### Gotchas
- Do not run `kaggle_main.py` in this studio; it has no GPU/vLLM.
- Local cleanup simulation over `data/submission (20).csv` reduced detected garbage from 5/100 to 0/100 without invoking the full pipeline.
- Validation here should stay on `py_compile`, pytest, and in-memory cleanup simulations over existing submissions.

---

## 2026-06-17 - submission (22).csv repeated/duplicated-answer cleanup

### Patterns Discovered
- `submission (22).csv` contains 100 rows; the remaining bad outputs include repeated punctuation (`* * * ...`), short-token repetition (`_perper...`), duplicated question text before a partial answer, prompt-like chat repetition, incomplete sentence fragments, and residual phrases such as `Где мои деньги`.
- The previous cleanup missed these because it focused on mixed-script corruption and repeated word blocks, but weak models can also emit punctuation-only output, repeat 2–4 character tokens, or copy the question into the answer before answering.
- Reference fallback remains the safest rescue path when an LLM answer is detected as garbage and `sample_submission.csv` is available.

### Conventions
- Bump `PIPELINE_VERSION` when post-processing changes; use `v5-submission22-cleanup` for this fix.
- Keep `finalize_generated_answer()` as the final gate: post-process, detect garbage, rescue with `clean_reference_answer(reference_answer)`, then fall back to `extract_answer_from_context()`.
- Add focused tests for repeated punctuation, short-token repetition, duplicated question text, incomplete answers, and rescue fallback.
### Gotchas
- Do not run `kaggle_main.py` in this studio; it has no GPU/vLLM.
- Local cleanup simulation over `data/submission (22).csv` reduced detected garbage from 12/100 to 0/100 without invoking the full pipeline.
- Validation here should stay on `py_compile`, pytest, and in-memory cleanup simulations over existing submissions.

---

## 2026-06-18 - submission (24).csv no-answer, numeric-only, and question-only cleanup

### Patterns Discovered
- `submission (24).csv` still has three high-signal bad-output families: explicit no-answer refusals (`Нет ответа`), numeric-only lists like `1. 2. 3. ... 32.`, and question-only outputs that repeat the user question without answering.
- Weak models copy the old instruction `Если ответа нет — напиши: "Нет ответа."`, so prompts must explicitly forbid that phrase and request a useful clarification or best-effort answer instead.
- Reference rescue remains the safest fallback for garbage outputs when `sample_submission.csv` is available, but reference cleaning still strips `В предоставленных фрагментах` because that is the established cleanup convention.

### Conventions
- Bump `PIPELINE_VERSION` when prompt/post-processing changes; current version is `v6-submission24-cleanup`.
- Keep `finalize_generated_answer()` as the final gate: post-process, detect garbage, rescue with `clean_reference_answer(reference_answer)`, then fall back to `extract_answer_from_context()`.
- Keep the question-only detector bounded (`len <= 220`, safe character-class regex) to avoid catastrophic regex backtracking on long generated answers.

### Gotchas
- Do not remove `Нет ответа` inside `clean_llm_answer()`; doing so can turn `За последние 0 месяца Нет ответа.` into a non-garbage partial answer and skip reference rescue.
- In this studio, validate with `py_compile`, pytest, and in-memory cleanup simulations over existing submissions rather than full inference.
- Local cleanup simulation over `data/submission (24).csv` showed `before_garbage=1407`; after reference rescue, remaining garbage was `1307`, with many remaining `нет ответа` phrases coming from reference answers rather than generated garbage.

---

## 2026-06-19 - Parent-Child Retrieval Upgrade

### Patterns Discovered
- Indexing small child chunks (~500 chars) and expanding hits to parent chunks (~1100 chars) improves both precision and context quality
- Parent-child build splits each parent into overlapping children using the same sentence-aware `Chunker`
- `get_parent_by_child_id()` uses `parent_id` as global key; in legacy mode (no `parent_id`) it returns the child itself
- `_expand_child_candidates_to_parents()` deduplicates via `setdefault()`, ensuring each parent appears once in reranking
- Reranker scores parent chunks; `TOP_K_CONTEXT=8` prevents flooding the model window
- `get_context()` now slices results to `TOP_K_CONTEXT` before checking `MIN_RERANK_SCORE`

### Conventions
- `build_chunks(websites_data, use_parent_child=True)` is the default entry point in `kaggle_main.py`
- `--legacy-chunking` CLI flag disables parent-child for backward compatibility
- Child chunks carry `parent_id`, `parent_text`, `parent_start`, `parent_end` metadata
- Indexer serialises all parent-child fields into `chunk_mapping.json`
- Tests use `FakeIndexer` (lightweight double) to avoid loading the full `Indexer` model

### Gotchas
- `parent_id` can be `None` in legacy mode → `get_parent_texts` must handle `int(None)` safely; fixed with explicit `raw_parent_id if raw_parent_id is not None else ...`
- `build_chunks(use_parent_child=False)` delegates to `chunk_all_websites` which may filter short text → test must provide realistically long input
- Expanding child hits to parents before reranking means `RERANKER_BATCH_SIZE` governs parent pairs, not child pairs — but counts are smaller (rerank ≤15 vs merge ~80)
- String escaping in `_QUESTION_ONLY_CHAR_RE` regex with raw string and curly quotes → must use `\u2019` for the apostrophe

