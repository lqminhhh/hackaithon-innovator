# v3 image: single model (Qwen3.5-4B) on vLLM, offline at run time.
# No RAG / embedding / reranker models — those were removed in v3 (illegal + measured to hurt).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# HF_HUB_ENABLE_HF_TRANSFER=0: hf_transfer breaks downloads unless installed (see handoff notes).
ENV HF_HUB_ENABLE_HF_TRANSFER=0 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Python 3.11 + system deps (deadsnakes for 3.11 on Ubuntu 22.04).
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl git build-essential \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-distutils \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Python deps. vLLM >= 0.17 is required for Qwen3.5 (the qwen3_5 architecture);
# it is intentionally not pinned in requirements.txt (optional for CPU dev), so
# install it explicitly here for the GPU image.
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir "vllm>=0.17.0"

# Source, configs, and entrypoint. The v3 inference path is run.py (S7 runner);
# src/pipeline.py is legacy and is deliberately NOT the entrypoint.
COPY src/ ./src/
COPY configs/ ./configs/
COPY run.py main.py run.sh ./
RUN chmod +x run.sh

# Bake the single v3 model into the image so inference needs no internet.
# (Switch to an AWQ repo here if you decide to ship 4-bit for a small/unknown card.)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-4B')"

# run.sh forwards: ./run.sh <input.json|csv> <output.csv>
ENTRYPOINT ["./run.sh"]
