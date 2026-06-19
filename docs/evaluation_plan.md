# Evaluation Notebook Plan

Goal: build a Jupyter-based workflow that explains where each architecture is helping, hurting, or failing, using `data/reference/reference_answers.csv` as the current high-confidence reference answer file.

Important wording: the reference file is not true hidden gold. It scored 91.58%, so notebook metrics should be called **agreement with reference**, not final accuracy.

## Inputs

Current required inputs:

- `data/public-test_1780368312.json`: original questions, options, contexts, and metadata that can be parsed.
- `data/reference/reference_answers.csv`: manual/reference answers with columns `qid,answer`.
- `data/submissions/*.csv`: model submission files with columns `qid,answer`.

Future optional inputs:

- `data/traces/trace_<version>.jsonl`: per-question solver traces for route/path/margin/votes/S5/RAG diagnostics.
- `docs/version_results.md`: leaderboard scores and runtime notes for cross-reference.

## Notebook Sections

### 1. Load And Normalize Inputs

- Read the reference answer file.
- Read every submission file from `data/submissions/`.
- Normalize answer letters to uppercase.
- Validate duplicate qids, missing qids, invalid answer labels, and extra qids.
- Parse the public-test questions to attach question metadata.

Primary output:

- list of discovered versions
- validation warnings table

### 2. Per-Version Scorer

Compare every submission against `reference_answers.csv`.

Metrics:

- number of questions
- agreement with reference
- disagreement count
- missing count
- invalid count
- extra qid count

Primary table:

`version | n | agrees_reference | disagrees_reference | missing | invalid | agreement_rate`

### 3. Score By Question Type

Use parsed question metadata and/or route predictions to group agreement by:

- `reading`
- `stem`
- `knowledge`
- `safety`

This should show which route family is leaking accuracy and which router/prompt path deserves attention.

If traces exist, prefer the actual final route from the trace. If traces do not exist, use the deterministic parser/router output as an approximation.

Primary table:

`version | question_type | n | agreement_rate | disagree_count`

### 4. Delta Analyzer

Compare retained final-compliant version transitions, for example:

- `v01_baseline -> v02_alpha`
- `v02_alpha -> v02_beta`

For each transition, compute:

- unchanged and agrees with reference
- unchanged and disagrees with reference
- changed toward reference
- changed away from reference
- changed but both disagree with reference
- net reference gain

Primary output:

- transition summary table
- fixed question list
- broken question list
- unchanged-wrong question list

This helps decide whether a redesign generalized or merely moved errors around.

### 5. Persistent Failure Set

Find questions that disagree with the reference across all or most versions.

These are high-value architecture targets because they likely need a new capability rather than a small prompt tweak.

Useful columns:

- qid
- reference answer
- all version answers
- question text
- options
- question type

### 6. Regression Set

Find questions that were aligned with the reference in an earlier version but became misaligned in a later version.

Important examples:

- `v02_alpha` agrees, `v02_beta` disagrees
- a future final-compliant version agrees less than `v02_beta`

These rows tell us what a new module broke.

### 7. Confidence And Vote Analysis

Requires trace files.

Group by:

- margin bucket
- vote pattern
- unanimous vs split self-consistency
- direct answer vs final answer

Questions to answer:

- Are high-margin wrong answers common?
- Is unanimous self-consistency actually reliable?
- Does low-margin self-consistency fix more than it breaks?

### 8. S5 And RAG Impact Panels

Requires trace files.

For S5:

- route override count
- overrides that move toward reference
- overrides that move away from reference
- source route -> target route pairs

For RAG:

- RAG used count
- RAG changed answer count
- RAG moved toward reference
- RAG moved away from reference
- top retrieval/rerank score buckets, if available

### 9. Question Browser

Interactive section for manual inspection:

- choose a `qid`
- show question/context/options
- show reference answer
- show every version answer
- show route/path/margin/votes/S5/RAG trace if available

This is the debugging workspace for concrete fixes.

Note: S5/RAG panels are for offline analysis only under the current organizer
constraints. Final retained runners should use one LLM only and leave S5/RAG
trace fields null/false.

## Recommended Outputs

The notebook should write reusable artifacts:

- `reports/eval/version_summary.csv`
- `reports/eval/question_type_summary.csv`
- `reports/eval/per_question_matrix.csv`
- `reports/eval/version_deltas.csv`
- `reports/eval/persistent_failures.csv`
- `reports/eval/regressions.csv`
- `reports/eval/audit_queue.csv`

## Preparation Needed

The current submission CSVs are enough for agreement scoring and version deltas.

To diagnose why answers changed, future runs should also save trace files with at least:

- `qid`
- `answer`
- `route`
- `path`
- `margin`
- `first_answer`
- `votes`
- `layer1_route`
- `semantic_route`
- `route_override`
- `override_blockers`
- `rag_used`
- `rag_context_used`
- `rag_top_score`
- `error`
- `elapsed_seconds`

Without traces, the notebook can say **what** changed. With traces, it can say **why** it changed.
