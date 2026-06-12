#!/bin/bash
# Download model weights for local development (GGUF) and Docker build (HF).
#
# For local MacBook dev (llama.cpp, Metal):
#   bash scripts/download_weights.sh gguf
#
# For Docker build time (Hugging Face snapshot):
#   bash scripts/download_weights.sh hf
#
# GGUF repos (confirmed correct as of June 2026):
#   Qwen3.5-9B  → unsloth/Qwen3.5-9B-GGUF    (~5.68 GB Q4_K_M)
#   Gemma 4 E4B → bartowski/gemma-4-4b-it-GGUF (~3.0  GB Q4_K_M)
#
# After download, set the gguf_paths in configs/pipeline_config.yaml.

set -euo pipefail

MODE=${1:-gguf}
WEIGHTS_DIR=${WEIGHTS_DIR:-./weights}
mkdir -p "$WEIGHTS_DIR"

# Use 'hf' if available (new cli), fall back to 'huggingface-cli'
HF_CMD="hf"
if ! command -v hf &>/dev/null; then
    HF_CMD="huggingface-cli"
fi

if [ "$MODE" = "gguf" ]; then
    echo "=== Downloading GGUF weights for llama-cpp-python ==="
    echo "Destination: $(realpath $WEIGHTS_DIR)"
    echo ""

    # ── Qwen3.5-9B Q4_K_M (primary juror, ~5.68 GB) ─────────────────
    echo "Downloading Qwen3.5-9B Q4_K_M..."
    $HF_CMD download \
        "unsloth/Qwen3.5-9B-GGUF" \
        "Qwen3.5-9B-Q4_K_M.gguf" \
        --local-dir "$WEIGHTS_DIR"

    QWEN_PATH="$(realpath $WEIGHTS_DIR)/Qwen3.5-9B-Q4_K_M.gguf"
    echo "  → $QWEN_PATH"

    # ── Gemma 4 E4B Q4_K_M (secondary juror, ~3.0 GB) ───────────────
    # Gemma 4 E4B = google/gemma-4-E4B-it; bartowski repo name: gemma-4-4b-it-GGUF
    echo ""
    echo "Downloading Gemma 4 E4B Q4_K_M..."
    $HF_CMD download \
        "bartowski/gemma-4-4b-it-GGUF" \
        "gemma-4-4b-it-Q4_K_M.gguf" \
        --local-dir "$WEIGHTS_DIR" \
        || {
            echo "  bartowski repo failed, trying mradermacher fallback..."
            $HF_CMD download \
                "mradermacher/gemma-4-E4B-GGUF" \
                "gemma-4-E4B.Q4_K_M.gguf" \
                --local-dir "$WEIGHTS_DIR" \
                || echo "  WARNING: Gemma 4 E4B download failed. Secondary juror disabled."
        }

    GEMMA_FILE="$WEIGHTS_DIR/gemma-4-4b-it-Q4_K_M.gguf"
    if [ ! -f "$GEMMA_FILE" ]; then
        GEMMA_FILE="$WEIGHTS_DIR/gemma-4-E4B.Q4_K_M.gguf"
    fi
    GEMMA_PATH="$(realpath $GEMMA_FILE 2>/dev/null || echo '')"

    echo ""
    echo "=== Done. Update configs/pipeline_config.yaml: ==="
    echo ""
    echo "gguf_paths:"
    echo "  primary:   \"$QWEN_PATH\""
    if [ -n "$GEMMA_PATH" ] && [ -f "$GEMMA_PATH" ]; then
        echo "  secondary: \"$GEMMA_PATH\""
    else
        echo "  secondary: \"\"  # Gemma download failed — leave empty to disable juror"
    fi
    echo ""
    echo "Note: Gemma 4 requires --chat-template gemma in llama.cpp."
    echo "      LlamaCppBackend passes the Gemma chat format automatically."

elif [ "$MODE" = "hf" ]; then
    echo "=== Pre-downloading HuggingFace weights (for Docker build) ==="
    python3 -c "
from huggingface_hub import snapshot_download
print('Downloading Qwen3.5-9B...')
snapshot_download('Qwen/Qwen3.5-9B', ignore_patterns=['*.msgpack', '*.h5'])
print('Downloading Gemma 4 E4B...')
try:
    snapshot_download('google/gemma-4-E4B-it', ignore_patterns=['*.msgpack', '*.h5'])
    print('  Gemma 4 E4B downloaded.')
except Exception as e:
    print(f'  WARNING: Gemma download failed: {e}')
    print('  Set models.secondary to empty string in pipeline_config.yaml to disable.')
print('Downloading BGE-M3...')
from sentence_transformers import SentenceTransformer
SentenceTransformer('BAAI/bge-m3')
print('All done.')
"
else
    echo "Usage: $0 [gguf|hf]"
    echo ""
    echo "  gguf  — download GGUF files for local llama-cpp-python (MacBook)"
    echo "  hf    — pre-download HuggingFace snapshots (Docker build time)"
    exit 1
fi
