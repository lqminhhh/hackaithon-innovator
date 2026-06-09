FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    git wget curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and configs
COPY src/ ./src/
COPY configs/ ./configs/

# Copy pre-built data artifacts (FAISS index, chunk metadata)
COPY data/faiss.index ./data/faiss.index
COPY data/chunks.jsonl ./data/chunks.jsonl

# Pre-download model weights into the image (no internet at eval time)
RUN python -c "\
from transformers import AutoModelForCausalLM, AutoTokenizer; \
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True); \
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)"

RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('keepitreal/vietnamese-sbert')"

# Optionally bake in secondary model (uncomment when ready)
# RUN python -c "\
# from transformers import AutoModelForCausalLM, AutoTokenizer; \
# AutoModelForCausalLM.from_pretrained('google/gemma-2-9b-it', trust_remote_code=True); \
# AutoTokenizer.from_pretrained('google/gemma-2-9b-it', trust_remote_code=True)"

COPY run.sh .
RUN chmod +x run.sh

ENTRYPOINT ["./run.sh"]
