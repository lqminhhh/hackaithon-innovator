# HackAIthon 2026 — Bảng C (Innovator)

A multi-agent Vietnamese multiple-choice question answering system built for the Vietnamese Student HackAIthon 2026. Given a set of questions with four answer choices, the system retrieves relevant knowledge, reasons step-by-step, and outputs the best answer for each question.

The entire system runs offline inside a Docker container with no internet access at inference time. All models, embeddings, and the knowledge base are baked into the image at build time.

## How It Works

```
                         Input (JSON/CSV)
                              │
                              ▼
               ┌─────────────────────────────────┐
               │     Parallel execution          │
               │                                 │
               │  Retrieval Agent    Reasoning   │
               │  BM25 + FAISS      Agent        │
               │  hybrid search     (CoT, no     │
               │       │            context)     │
               │       ▼                │        │
               │  Relevance Gate        │        │
               │  cosine ≥ 0.65?        │        │
               │  YES → inject context  │        │
               │  NO  → drop            │        │
               └──────────┬─────────────┘        │
                          │                      │
                          ▼                      ▼
                    CoT Pass 2 (with context)
                          │
                          ▼
                   Confidence Gate
              ┌──────────┼──────────────┐
              ▼          ▼               ▼
          ≥ 0.85     0.55–0.84        < 0.55
         Fast exit   Consistency     Ensemble
         (done)      sampling        Qwen + Gemma
                     N=2→5→7         dual-model
              └──────────┼──────────────┘
                          │
                          ▼
                  Answer Normaliser
                  (robust A/B/C/D extraction)
                          │
                          ▼
                   submission.csv
```

**Step 1 — Parallel execution:** For each question, two things happen simultaneously. The retrieval agent searches the knowledge base (Vietnamese Wikipedia + pre-generated reasoning chains) using hybrid BM25 + dense search. At the same time, the reasoning agent runs a first chain-of-thought pass using only the question and answer choices.

**Step 2 — Context injection:** Retrieved chunks are checked against a relevance threshold (cosine ≥ 0.65). If relevant context is found, the model runs a second CoT pass with that context injected. If nothing relevant was found, the first pass answer is used as-is. This prevents the model from anchoring on irrelevant information.

**Step 3 — Confidence routing:** The model reports its own confidence score. High-confidence answers (≥ 0.85) are emitted immediately. Medium-confidence answers go through adaptive consistency sampling — the model is sampled 2 to 7 times at higher temperature and the majority vote wins. Low-confidence answers (< 0.55) trigger a dual-model ensemble where both Qwen and Gemma vote independently.

**Step 4 — Answer normalisation:** Model outputs are parsed through a multi-layer regex extractor that handles various output formats (Vietnamese, English, parenthesised, bare letters) and always produces a valid A/B/C/D answer.

## Data Layout & Leakage Prevention

```
data/
├── reference/                      ← Reference questions for CoT generation
│   ├── public_test.csv             ← Initial 3-question sample
│   ├── round1_test.json            ← Add past test files here over time
│   └── ...
│
├── public-test_1780368312.json     ← Active test file (inference only)
├── cot_chains.jsonl                ← Generated CoT chains (built from reference/)
├── faiss.index                     ← FAISS vector index
└── chunks.jsonl                    ← Chunk metadata
```

### Adding new reference files

As you receive more test files, move completed ones into `data/reference/` and re-run:

```bash
python scripts/generate_cot_chains.py --input data/reference/
python scripts/build_index.py
```

The script **appends** new chains and **skips** questions already processed, so you never redo work.

### How data leakage is prevented

CoT and inference use **strictly separate files**:

- **Build time:** `generate_cot_chains.py` scans `data/reference/` (past sample/test files). The resulting worked examples are embedded into the FAISS index alongside Vietnamese Wikipedia.
- **Inference time:** `pipeline.py` runs on the active test file. The model never sees its own prior answers — CoT chains in the index come from the reference set, not the active test file.

As a safety net, the retrieval agent also supports same-QID decontamination (`exclude_qid` parameter) which silently skips any CoT chunk whose `qid` matches the question currently being answered.

## Tech Stack

| Component | Model / Library | Why |
|---|---|---|
| Primary LLM | Qwen/Qwen2.5-7B-Instruct | Best Vietnamese + multilingual reasoning at 7B parameters |
| Secondary LLM | google/gemma-2-9b-it | Different model family reduces correlated errors in ensemble |
| Embeddings | keepitreal/vietnamese-sbert | Vietnamese-tuned, accurate token boundaries for Vietnamese text |
| Vector search | FAISS (IndexFlatIP) | Fast offline cosine similarity search, no server needed |
| Keyword search | rank_bm25 | Catches exact term matches that dense embeddings miss |
| Quantisation | bitsandbytes 4-bit | Fits both models in ≤24 GB VRAM on a single GPU |
| Orchestration | Python asyncio | Runs retrieval and reasoning in parallel per question |

## Project Structure

```
├── src/
│   ├── pipeline.py              Main orchestrator — wires all modules together
│   ├── data_loader.py           Universal JSON/CSV loader with choice normalisation
│   ├── retrieval_agent.py       BM25 + dense hybrid search with relevance gating
│   ├── reasoning_agent.py       Qwen2.5-7B chain-of-thought inference wrapper
│   ├── confidence_gate.py       Routes questions by confidence score
│   ├── consistency_sampler.py   Adaptive N=2→5→7 majority vote sampling
│   ├── ensemble_agent.py        Dual-model Qwen + Gemma fusion
│   ├── normaliser.py            Robust A/B/C/D extraction from any output format
│   └── models.py                Model loading (4-bit CUDA / float16 MPS / float32 CPU)
│
├── scripts/
│   ├── download_wiki.sh         Downloads + extracts Vietnamese Wikipedia dump
│   ├── extract_wiki.py          Custom wiki XML parser (replaces broken wikiextractor)
│   ├── build_index.py           Builds FAISS index from wiki + CoT chains + domain texts
│   ├── generate_cot_chains.py   Pre-generates reasoning traces for retrieval
│   └── topic_map.py             Analyses question topics to identify knowledge gaps
│
├── configs/
│   ├── pipeline_config.yaml     All thresholds, model paths, chunk sizes
│   └── prompts.yaml             Vietnamese prompt templates
│
├── tests/                       Unit and integration tests
├── notebooks/
│   └── colab_full_run.ipynb     Complete Colab notebook (clone → run → submit)
├── data/
│   └── reference/               Reference question files for CoT generation
├── Dockerfile                   Production container with baked-in models
├── run.sh                       Container entrypoint
├── docker-compose.yml           Local dev with GPU volume mounts
└── requirements.txt             Pinned Python dependencies
```

## Quick Start

### Local development

```bash
git clone https://github.com/lqminhhh/hackaithon-innovator.git
cd hackaithon-innovator
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Build the knowledge base (one-time, requires GPU)

```bash
# 1. Download and extract Vietnamese Wikipedia (~20 min)
pip install mwparserfromhell
bash scripts/download_wiki.sh

# 2. Generate CoT reasoning chains from reference questions
#    Scans data/reference/ — add more files there as you get them
python scripts/generate_cot_chains.py --input data/reference/

# 3. Build the FAISS index (~30-60 min)
#    Merges Wikipedia + CoT chains + any domain texts into a single index
python scripts/build_index.py
```

### Run inference (on actual test file)

```bash
python src/pipeline.py \
    --input data/public-test_1780368312.json \
    --output data/submission.csv
```

### Docker (production)

```bash
# Build (bakes models + FAISS index into image)
docker build -t bangc:v1 .

# Run
docker run --gpus all -v $(pwd)/data:/data bangc:v1

# Verify offline operation
docker run --gpus all --network none -v $(pwd)/data:/data bangc:v1
```

### Google Colab

Open `notebooks/colab_full_run.ipynb` in Colab with a T4/A100 GPU runtime. The notebook handles everything from cloning to submission file download.

## Configuration

All tunable parameters are in `configs/pipeline_config.yaml`:

| Parameter | Default | What it controls |
|---|---|---|
| `retrieval.relevance_threshold` | 0.65 | Minimum cosine similarity to inject context |
| `retrieval.top_k` | 5 | Number of chunks to retrieve |
| `confidence_gate.fast_exit_threshold` | 0.85 | Skip further processing above this confidence |
| `confidence_gate.ensemble_threshold` | 0.55 | Trigger dual-model ensemble below this |
| `consistency_sampler.n_max` | 7 | Maximum samples for majority vote |
| `inference.temperature_sampling` | 0.7 | Temperature for consistency sampling |
| `quantisation.load_in_4bit` | true | Enable 4-bit quantisation (requires CUDA) |

## Input / Output Format

**Input** — JSON array or CSV. The system auto-detects the format.

```json
[
  {"qid": "test_0001", "question": "...", "choices": ["...", "...", "...", "..."]},
  ...
]
```

**Output** — CSV with exactly two columns:

```csv
id,answer
test_0001,A
test_0002,C
```

## Competition

- **Event:** Vietnamese Student HackAIthon 2026
- **Track:** Bảng C — Innovator
- **Site:** http://hackaithon.vsds.vn
- **Deadline:** June 23, 2026
