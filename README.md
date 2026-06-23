# Team Cow - HackAIthon 2026 Bang C Innovator

Offline Vietnamese multiple-choice question answering system for the HackAIthon
2026 Bang C submission track.

This repository is organized around a single final submission path:
`src/v03_gamma.py`.

## Submission Snapshot

| Item | Value |
| --- | --- |
| Team name | `Cow` |
| Team members | `[Fill in full names here]` |
| School / organization | `[Fill in school or organization]` |
| GitHub repository | `[Fill in repository URL]` |
| Docker Hub image | |
| Final runner | `src/v03_gamma.py` |
| Primary model | `Qwen/Qwen3.5-4B` |
| Inference mode | Offline, one-model only |
| Target hardware | 16 GB VRAM |

## What This Submission Does

The system reads a multiple-choice test file, routes each question into a small
set of reasoning modes, runs the model with constrained answer extraction, and
writes a valid submission file with one answer letter per question.

The retained final branch is `v03_gamma`, chosen because it gave the best
speed/reliability tradeoff for a judge-like 16 GB VRAM environment. Later
branches reached higher public accuracy, but they were much slower and remained
OOM-prone on longer runs.

## Competition I/O Contract

The shipped container behavior is:

- Read `/data/private_test.csv` if present
- Otherwise read `/data/public_test.csv`
- If CSV is absent, fall back to `/data/private_test.json` or
  `/data/public_test.json`
- Write `/output/pred.csv`
- Output format: exactly two columns, `qid,answer`

Each `answer` is a single valid letter for that question (`A`, `B`, `C`, ...).

The entrypoint also accepts manual overrides:

```bash
./run.sh <input.json|csv> <output.csv> [trace.jsonl]
```

## Compliance Summary

This repository is aligned to the current Bang C constraints:

- one open LLM only
- offline inference only
- no external APIs or internet access at runtime
- no embedding model
- no reranker model
- no RAG / retrieval pipeline
- no second LLM
- model size within the competition limit

The final submission path uses only `Qwen/Qwen3.5-4B`.

## Final Architecture

`v03_gamma` is a wave-batched, route-aware pipeline:

1. Load and normalize questions from JSON or CSV
2. Parse question structure and choices
3. Route each item deterministically into one of:
   `READING`, `STEM`, `KNOWLEDGE`, `SAFETY`
4. Run a first-pass reasoning stage
5. Extract a valid answer letter with constrained choice decoding
6. Escalate selected questions with self-consistency
7. Merge answers and always write a complete submission file

In practice:

- `STEM` always gets self-consistency
- `READING` gets reread/self-consistency for detail and reason/purpose patterns
- `KNOWLEDGE` gets extra compute for harder cases
- `SAFETY` can force a refusal option when the prompt is harmful and a refusal
  answer exists

The final path also includes:

- wave batching for throughput
- option shuffling across SC samples
- duplicate-option-safe vote remapping
- length-safe extraction prompts
- checkpointing plus always-emit emergency write behavior

## Why `v03_gamma` Is the Final Branch

We evaluated multiple versions of the system.

`v03_delta` reached a stronger public score, but it depended on heavier
continuation-scored margin logic that increased runtime by roughly 3.5x to 4x
and still showed OOM risk on 16 GB judge-like hardware.

`v03_gamma` was selected as the final operating point because it is:

- more likely to complete a long private-set run
- materially faster
- simpler to ship safely
- still strong on the public benchmark

This is an efficiency-first final branch, not a fallback branch.

## Reported Results

Public-set results tracked in this repository:

| Version | Score | Runtime |
| --- | --- | --- |
| `v02_gamma` | 85.31% | 12.77 s/question |
| `v03_alpha` | 84.23% | 3.87 s/question |
| `v03_gamma` | **85.96%** | 7.98 s/question |
| `v03_delta` | 87.04% | 27.53 s/question |

For the final submission, we prefer `v03_gamma` over `v03_delta` because judge
reliability matters more than squeezing out a small public-set gain with a much
heavier branch.

Full history is documented in [docs/version_results.md](docs/version_results.md).

## Repository Layout

```text
configs/
  pipeline_config.yaml      Runtime settings and judge-safe defaults

data/
  reference/                Reference answers for local evaluation
  submissions/              Generated submission CSVs
  traces/                   Per-question trace JSONL files

docs/
  status.md                 Current working status and design notes
  planning_v3.md            Final build plan
  version_results.md        Version-by-version score log

notebooks/
  evaluation.ipynb          Local scoring and error analysis

src/
  v03_gamma.py              Final submission runner
  v02_gamma.py              Compatibility shim
  wave_solver.py            Wave 1 / Wave 2 batching logic
  batch_extract.py          Batched constrained answer extraction
  sc_policy.py              Route-aware self-consistency policy
  parser.py                 Question parsing
  router.py                 Deterministic route assignment
  reasoning_agent.py        Model-facing inference wrapper
  data_loader.py            JSON/CSV loading and submission writing
  config.py                 YAML-backed shared config loader

tests/
  Focused tests for routing, parsing, extraction, runner entrypoints, and SC
```

## Running Locally

Create an environment:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run the final pipeline directly:

```bash
python src/v03_gamma.py \
  --input data/public-test_1780368312.json \
  --output data/submissions/submission_v03_gamma.csv \
  --trace-output data/traces/trace_v03_gamma.jsonl \
  --safe-mode
```

Quick smoke run:

```bash
python src/v03_gamma.py \
  --input data/public-test_1780368312.json \
  --limit 5 \
  --safe-mode
```

## Running As a Container

The intended container workflow is:

```bash
docker pull <fill-in-dockerhub-image>
mkdir -p data output
cp private_test.csv data/
docker run --rm --gpus all \
  -v "$PWD/data:/data" \
  -v "$PWD/output:/output" \
  <fill-in-dockerhub-image>
```

Expected output:

```text
output/pred.csv
```

with header:

```csv
qid,answer
```

If you are testing from the repository without a published image yet:

```bash
mkdir -p output
./run.sh data/private_test.csv output/pred.csv
```

## Input Format Notes

The loader supports:

- organizer-style JSON
- CSV with separate choice columns like `A,B,C,D`
- CSV where choices are embedded in the question text

The official container contract should still be treated as
`/data/public_test.csv` or `/data/private_test.csv`.

## Output Guarantees

The final runner is designed to avoid invalid submissions:

- output is always written in `qid,answer` format
- fallback answers are prefilled before inference starts
- answers are normalized to valid choice labels
- checkpointing reduces the chance of losing progress
- atomic output writing helps avoid partial-file corruption
- signal and exception handlers emit a best-effort complete submission on failure

This does not mean the run is mathematically impossible to fail, but the code
is intentionally shaped around completion reliability.

## Configuration

The runtime source of truth is:

```text
configs/pipeline_config.yaml
```

That file controls:

- model selection
- vLLM settings
- safe-mode limits
- self-consistency policy
- token budgets
- fallback answer behavior

Python modules consume the same config through
`src/config.py`, so YAML and runtime constants stay aligned.

## Tests

Run the focused suite with:

```bash
pytest -q
```

The most relevant final-runner checks cover:

- parser behavior
- routing behavior
- constrained guided-choice extraction
- adaptive/self-consistency policy
- gamma entrypoints
- output format and I/O contract

## Files to Read First

If you are reviewing or extending this submission, start here:

- [docs/status.md](docs/status.md)
- [docs/version_results.md](docs/version_results.md)
- [src/v03_gamma.py](src/v03_gamma.py)
- [src/wave_solver.py](src/wave_solver.py)
- [configs/pipeline_config.yaml](configs/pipeline_config.yaml)

## Known Limits

- `v03_gamma` uses a lightweight confidence signal rather than the heavier
  exact-margin logic explored later
- the final branch favors judge-safe deployment over peak public-set accuracy
- later experiments exist in the repo history but are not the shipped path

## Fill-In Before Submission

Please replace these blanks before final handoff:

- Team members in the submission table
- School / organization
- Public repository URL
- Docker Hub image name and tag
- Any final slide or write-up links you want to expose

## Contact

- Team: `Cow`
- Representative: `[Fill in contact name]`
- Email: `[Fill in contact email]`
