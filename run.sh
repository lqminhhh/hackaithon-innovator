#!/bin/bash
# v3 container entrypoint: thin wrapper around the current wave runner.
#   ./run.sh <input.json|csv> <output.csv>
set -euo pipefail

INPUT=${1:-/data/public-test_1780368312.json}
OUTPUT=${2:-/data/submission.csv}
TRACE=${3:-/data/trace.jsonl}

exec python src/v02_gamma.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --trace-output "$TRACE" \
  --safe-mode
