#!/bin/bash
# v3 container entrypoint: thin wrapper around the S7 never-crash runner.
#   ./run.sh <input.json|csv> <output.csv>
set -euo pipefail

INPUT=${1:-/data/public-test_1780368312.json}
OUTPUT=${2:-/data/submission.csv}

exec python run.py --input "$INPUT" --output "$OUTPUT"
