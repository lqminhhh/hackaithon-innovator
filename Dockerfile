# Submission image: Qwen3.5-4B on vLLM + never-crash runner.
# - Single model only; no RAG / embeddings / reranker.
# - Entry: run.sh -> python -m src.run (checkpoint, fault isolation, always-emit).
# - Model baked at build time; runtime is offline (no HuggingFace downloads).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV HF_HUB_ENABLE_HF_TRANSFER=0 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Python 3.11 (deadsnakes on Ubuntu 22.04).
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl git build-essential \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-distutils \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Inference deps (vllm==0.23.0 pinned in requirements.txt for Qwen3.5).
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# Application code + configs (prompts.yaml, pipeline_config.yaml).
COPY src/ ./src/
COPY configs/ ./configs/
COPY run.sh ./
RUN chmod +x run.sh

# Bake the v3 model into the image so inference needs no network.
ARG MODEL_ID=Qwen/Qwen3.5-4B
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_ID}')"

# Offline at runtime - model is already in the image cache.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# run.sh: ./run.sh <input.json|csv> <output.csv>
ENTRYPOINT ["./run.sh"]
