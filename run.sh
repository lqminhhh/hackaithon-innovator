#!/bin/bash
# Final container entrypoint: never-crash runner (src/run.py).
#   ./run.sh <input.json|csv> <output.csv>
set -euo pipefail

INPUT=${1:-/data/public-test_1780368312.json}
OUTPUT=${2:-/data/submission.csv}

exec python -m src.run --input "$INPUT" --output "$OUTPUT"
