# HackAIthon 2026 — Bảng C (Innovator)

An offline Vietnamese multiple-choice question-answering agent for the Vietnamese
Student HackAIthon 2026. Given a JSON/CSV of questions (each with 2–11 answer
choices), it picks the best option letter for each and writes a complete
`submission.csv`.

**v3 architecture — one model, no retrieval.** The system uses a single LLM
(`Qwen/Qwen3.5-4B`) served by vLLM. There is **no RAG, no embedding model, and no
reranker** — those were removed in v3 because the competition rules tightened to
one model (≤5B params, 16 GB VRAM, no embedding/rerank) *and* our own measurements
showed retrieval was hurting accuracy on a small model. Everything runs offline:
the model is baked into the Docker image at build time.

## How It Works

```
                      Input (JSON / CSV)
                            │
                            ▼
        ┌───────────────────────────────────────────┐
        │  Rule router  — "how hard to think?"        │
        │  READING · STEM · SAFETY · KNOWLEDGE        │
        └───────────────────────┬─────────────────────┘
                                ▼
        ┌───────────────────────────────────────────┐
        │  Two-pass guided choice                     │
        │   1) reason freely (think mode)             │
        │   2) constrain to a valid letter + margin   │
        └───────────────────────┬─────────────────────┘
                                ▼
        ┌───────────────────────────────────────────┐
        │  Escalation ladder (correctness net)        │
        │   STEM      → always self-consistency vote  │
        │   READING   → reason/purpose items vote     │
        │   KNOWLEDGE → vote if margin is low         │
        │   SAFETY    → force refusal if harmful      │
        └───────────────────────┬─────────────────────┘
                                ▼
        ┌───────────────────────────────────────────┐
        │  Never-crash runner (run.py)                │
        │  pre-fills FALLBACK, checkpoints, atomic    │
        │  write, always emits a complete CSV         │
        └───────────────────────┬─────────────────────┘
                                ▼
                          submission.csv
```

- **Router decides effort, escalation decides trust.** A routing mistake only
  costs a little time; the escalation ladder still catches the hard question, so
  a misroute never forces a wrong answer.
- **Guided choice** constrains the model's output to a real option letter (A–Z),
  so there is no fragile regex answer parsing.
- **Logprob margin** (`prob(top1) − prob(top2)`) is the confidence signal that
  decides when to escalate to self-consistency voting.
- **The runner can never score 0.** Every `qid` is pre-filled with the fallback
  letter, progress is checkpointed, and a complete CSV is written on normal exit,
  on a crash, and on `SIGTERM`/`SIGINT`.

## Models

| Component   | Model              | Notes                                          |
| ----------- | ------------------ | ---------------------------------------------- |
| Primary LLM | `Qwen/Qwen3.5-4B`  | Single model, ≤5B params, think/no-think modes |
| Serving     | vLLM (≥0.17)       | Batched decoding + guided choice + prefix cache |

No embedding, reranker, or second model is used in v3.

## Project Structure

```
├── run.py                       ★ v3 entrypoint — never-crash runner (S7)
├── main.py                        S0 model-free fallback runner (emits FALLBACK CSV)
├── src/
│   ├── config.py                  Central constants (model id, thresholds, FALLBACK)
│   ├── data_loader.py             JSON/CSV loader + writer (UTF-8, A–Z labels)
│   ├── io_utils.py                S0 I/O surface
│   ├── parser.py                  Splits passage/context, derives route hints
│   ├── router.py                  Layer-1 rule router (route_l1 / route_question)
│   ├── extract.py                 Guided-choice extraction + logprob margin
│   ├── reasoning_agent.py         vLLM / HuggingFace inference wrapper
│   ├── solve.py                   S4 route policy + escalation + self-consistency
│   └── models.py                  Model loading (vLLM primary / HF fallback)
│       (v3 is a single ≤5B model — no RAG, embedder, reranker, or 2nd model)
│
├── configs/
│   ├── pipeline_config.yaml       Active model id + vLLM + inference settings
│   └── prompts.yaml               Vietnamese route prompt templates
│
├── tests/                         Unit tests (incl. tests/test_run_s7.py for the runner)
├── data/                          Input files + submission output
├── Dockerfile                     v3 offline image (Qwen3.5-4B baked in)
├── run.sh                         Container entrypoint → python run.py
├── docker-compose.yml             GPU run with data volume mounted
└── requirements.txt               Python dependencies (vLLM installed separately)
```

## Quick Start — Google Colab (recommended)

A free **T4 GPU** Colab runtime is enough: `Qwen/Qwen3.5-4B` in fp16 is ~8 GB and
fits in 16 GB. Use **Runtime → Change runtime type → T4 GPU** (or A100), then run
the cells below.

**Cell 1 — confirm a GPU is attached**

```python
!nvidia-smi
```

**Cell 2 — clone the repo**

```python
!git clone https://github.com/lqminhhh/hackaithon-innovator.git
%cd hackaithon-innovator
```

**Cell 3 — install dependencies (vLLM is installed separately)**

```python
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"   # avoids a known download breakage

!pip install -q -r requirements.txt
!pip install -q "vllm>=0.17.0"                   # required for the Qwen3.5 architecture
```

> If you hit `KeyError: 'qwen3_5'`, your `transformers`/`vllm` is too old —
> re-run the `vllm` install above (it pulls a compatible `transformers`), then
> **restart the Colab runtime** and re-run from Cell 2.

**Cell 4 — run the agent (no RAG, no embedder, no reranker)**

```python
!python run.py \
    --input data/public-test_1780368312.json \
    --output data/submission.csv
```

The first run downloads the model (~8 GB) into the Colab session, then solves the
questions. The runner prints per-question progress and checkpoints to
`data/checkpoint.json`, so if the Colab session drops you can re-run the same
command and it **resumes** instead of starting over.

**Cell 5 — download the submission**

```python
from google.colab import files
files.download("data/submission.csv")
```

### Useful flags

```python
# Smoke test on the first 15 questions
!python run.py --input data/smoke-test-15.json --output data/sub_smoke.csv

# Cap the number of questions (quick sanity check)
!python run.py --input data/public-test_1780368312.json --output data/sub.csv --limit 20

# Use a specific model id (defaults to Qwen/Qwen3.5-4B)
!python run.py --input data/public-test_1780368312.json --output data/sub.csv \
    --model-id Qwen/Qwen3.5-4B

# Ignore any existing checkpoint and start fresh
!python run.py --input data/public-test_1780368312.json --output data/sub.csv --no-resume
```

## Quick Start — Local

```bash
git clone https://github.com/lqminhhh/hackaithon-innovator.git
cd hackaithon-innovator
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install "vllm>=0.17.0"      # CUDA GPU required for vLLM

python run.py \
    --input data/public-test_1780368312.json \
    --output data/submission.csv
```

Without a CUDA GPU, vLLM is unavailable and the agent falls back to the slower
HuggingFace backend automatically. For a model-free smoke test of the I/O
contract, use `python main.py --input ... --output ...` (emits an all-fallback
CSV).

## Docker (offline production image)

```bash
# Build — bakes Qwen/Qwen3.5-4B into the image (no internet needed at run time)
docker build -t bangc:v3 .

# Run with GPU, mounting ./data in and out
docker run --gpus all -v $(pwd)/data:/data bangc:v3 \
    /data/public-test_1780368312.json /data/submission.csv

# Dress rehearsal: confirm it works fully offline (the real pre-submit test)
docker run --gpus all --network none -v $(pwd)/data:/data bangc:v3 \
    /data/public-test_1780368312.json /data/submission.csv
```

## Configuration

Core constants live in `src/config.py`:

| Parameter      | Default            | Controls                                        |
| -------------- | ------------------ | ----------------------------------------------- |
| `LLM_MODEL`    | `Qwen/Qwen3.5-4B`  | The single model used for every question        |
| `MARGIN_LOW`   | per-route dict     | Escalate below this margin (per route: R .10 / S .15 / K .20 / Sa .05) |
| `SC_N`         | `5`                | Self-consistency samples for non-STEM escalation |
| `SC_N_STEM`    | `{high:3, low:7}`  | Adaptive STEM vote depth by first-pass margin   |
| `SC_TEMP`      | `0.6`              | Sampling temperature for voting (Qwen guidance) |
| `TOK`          | per-route caps     | Max tokens per route (STEM gets the most)       |
| `FALLBACK`     | `"A"`              | Safe default letter for any failed question     |

vLLM/runtime settings (gpu memory, max_model_len, prefix caching) and the
active model id live in `configs/pipeline_config.yaml`.

## Input / Output Format

**Input** — JSON array or CSV (auto-detected):

```json
[
  {"qid": "test_0001", "question": "...", "choices": ["...", "...", "...", "..."]}
]
```

**Output** — CSV with exactly two columns, one row per input qid, UTF-8:

```csv
qid,answer
test_0001,A
test_0002,C
```

## Competition

- **Event:** Vietnamese Student HackAIthon 2026
- **Track:** Bảng C — Innovator
- **Site:** [http://hackaithon.vsds.vn](http://hackaithon.vsds.vn)
- **Deadline:** June 23, 2026
