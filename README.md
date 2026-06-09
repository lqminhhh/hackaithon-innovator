# HackAIthon 2026 — Bảng C (Innovator)

Multi-agent Vietnamese multiple-choice QA system with hybrid retrieval, chain-of-thought reasoning, adaptive consistency sampling, and dual-model ensemble.

## Architecture

```
public_test.csv → Classifier → [Retrieval + CoT Pass 1 in parallel]
                                        ↓
                              Confidence Gate
                    ┌───────────┼────────────────┐
                  ≥0.85      0.55–0.84         <0.55
                 fast exit   consistency     dual-model
                              N=2→7          Qwen + Gemma
                    └───────────┼────────────────┘
                                ↓
                        Answer Normaliser → submission.csv
```

## Quick Start

### 1. Setup

```bash
git clone <your-repo>
cd hackaithon-bangc
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Build knowledge base

```bash
bash scripts/download_wiki.sh
python scripts/generate_cot_chains.py --input data/public_test.csv
python scripts/build_index.py
```

### 3. Run inference

```bash
python src/pipeline.py --input data/public_test.csv --output data/submission.csv
```

### 4. Docker

```bash
docker build -t bangc:v1 .
docker run --gpus all -v $(pwd)/data:/data bangc:v1
```

Offline verification (no internet):
```bash
docker run --gpus all --network none -v $(pwd)/data:/data bangc:v1
```

## Tech Stack

| Component | Choice |
|---|---|
| Primary LLM | Qwen/Qwen2.5-7B-Instruct (4-bit) |
| Secondary LLM | google/gemma-2-9b-it (4-bit) |
| Embeddings | keepitreal/vietnamese-sbert |
| Vector search | FAISS (offline) |
| Keyword search | rank_bm25 |
| Quantisation | bitsandbytes 4-bit |

## Project Structure

```
├── src/                 # Core pipeline modules
├── scripts/             # Data preparation and index building
├── configs/             # YAML configuration (thresholds, models, prompts)
├── tests/               # Unit and integration tests
├── notebooks/           # EDA and calibration notebooks
├── data/                # Input data, FAISS index, chunks
├── Dockerfile           # Production container
├── run.sh               # Container entrypoint
└── docker-compose.yml   # Local dev with GPU
```

## Competition

- **Event:** Vietnamese Student HackAIthon 2026
- **Track:** Bảng C — Innovator
- **Site:** http://hackaithon.vsds.vn
