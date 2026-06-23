#!/bin/bash
# Competition container entrypoint:
# - read /data/public_test.csv or /data/private_test.csv by default
# - write /output/pred.csv with columns: qid,answer
# Usage: ./run.sh [input.json|csv] [output.csv] [trace.jsonl]
set -euo pipefail

if [[ $# -ge 1 ]]; then
  INPUT=$1
elif [[ -f /data/private_test.csv ]]; then
  INPUT=/data/private_test.csv
elif [[ -f /data/public_test.csv ]]; then
  INPUT=/data/public_test.csv
elif [[ -f /data/private_test.json ]]; then
  INPUT=/data/private_test.json
elif [[ -f /data/public_test.json ]]; then
  INPUT=/data/public_test.json
else
  echo "No input file found. Expected /data/private_test.csv or /data/public_test.csv." >&2
  exit 1
fi

OUTPUT=${2:-/output/pred.csv}
TRACE_OUTPUT=${3:-/output/trace_v03_gamma.jsonl}

mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$TRACE_OUTPUT")"

exec python -m src.v03_gamma \
  --safe-mode \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --trace-output "$TRACE_OUTPUT"
