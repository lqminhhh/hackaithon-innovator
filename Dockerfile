FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    git wget curl build-essential python3.11 python3.11-dev python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# ── Python base deps (no vLLM yet, avoids torch version conflict) ─────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── vLLM — install from PINNED main-branch commit that supports
#    Qwen3.5 hybrid Gated DeltaNet attention AND Gemma 4 E4B.
#    Verify the commit on Day 1 by running the compatibility spike.
#    Replace <VLLM_COMMIT> with the actual confirmed hash.
# ─────────────────────────────────────────────────────────────────────
# RUN pip install --no-cache-dir \
#     "git+https://github.com/vllm-project/vllm.git@<VLLM_COMMIT>"
#
# Temporary: install latest vLLM release until commit is pinned
RUN pip install --no-cache-dir vllm

# ── Source and configs ────────────────────────────────────────────────
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY run.sh .
RUN chmod +x run.sh

# ── Pre-built data artifacts (FAISS index, chunk metadata) ───────────
# These are built by scripts/build_index.py and committed/mounted
COPY data/faiss_narrow.index ./data/faiss_narrow.index
COPY data/chunks_narrow.jsonl ./data/chunks_narrow.jsonl

# ── Bake in model weights at build time (no internet at inference) ────
# Requires: HF_TOKEN env var set, or weights pre-cached in /root/.cache
ENV HF_HUB_OFFLINE=0
ENV TRANSFORMERS_OFFLINE=0

# Primary: Qwen3.5-9B (AWQ 4-bit preferred; falls back to BF16)
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('Qwen/Qwen3.5-9B', ignore_patterns=['*.msgpack', '*.h5'])"

# Secondary: Gemma 4 E4B (google/gemma-4-E4B-it)
# NOTE: Requires accepting Google's licence on HuggingFace before building.
#   huggingface-cli login   (or set HF_TOKEN env var)
# If access is unavailable, set models.secondary to '' in pipeline_config.yaml.
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('google/gemma-4-E4B-it', ignore_patterns=['*.msgpack', '*.h5'])" \
    || echo "WARNING: Gemma 4 E4B download failed — secondary juror disabled"

# Embedder: BGE-M3
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-m3')"

# Disable all outbound network after weights are baked
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

ENTRYPOINT ["./run.sh"]
