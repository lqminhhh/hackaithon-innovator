# v3 image: single model (Qwen3.5-4B) on vLLM, offline at run time.
# No RAG / embedding / reranker models — those were removed in v3.
# Final image uses the original CUDA stack tested for this project.
FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

# HF_HUB_ENABLE_HF_TRANSFER=0: hf_transfer breaks downloads unless installed (see handoff notes).
ENV HF_HUB_ENABLE_HF_TRANSFER=0 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /code

# Python 3.11 + system deps (deadsnakes for 3.11 on Ubuntu 22.04).
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl git build-essential \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-distutils \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Python deps. Keep the container and local environment aligned by installing
# the exact pinned GPU stack from requirements.txt, including vLLM.
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Source, configs, and entrypoint. The final submission path is the wave-batched
# `src.v03_gamma` runner; `src/run.py` stays in the repo as a fallback utility.
COPY src/ ./src/
COPY configs/ ./configs/
COPY predict.py ./
COPY inference.sh ./
COPY run.sh ./
RUN chmod +x run.sh inference.sh

# Bake the single v3 model into the image so inference needs no internet.
# The image uses the original CUDA 12.9.1 base; host machines need a
# compatible NVIDIA driver.
# (Switch to an AWQ repo here if you decide to ship 4-bit for a small/unknown card.)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-4B')"

# inference.sh defaults to the BTC contract:
# - input: /code/private_test.json
# - output: /code/submission.csv and /code/submission_time.csv
CMD ["bash", "inference.sh"]
