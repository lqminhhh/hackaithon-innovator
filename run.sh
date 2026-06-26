#!/bin/bash
# Compatibility entrypoint for local runs.
# BTC entrypoint is inference.sh -> predict.py.
# Usage: ./run.sh [input.json|csv] [output.csv] [trace.jsonl] [submission_time.csv]
set -euo pipefail

if [[ $# -ge 1 ]]; then
  INPUT=$1
elif [[ -f /code/private_test.json ]]; then
  INPUT=/code/private_test.json
elif [[ -f /code/private_test.csv ]]; then
  INPUT=/code/private_test.csv
elif [[ -f /data/private_test.csv ]]; then
  INPUT=/data/private_test.csv
elif [[ -f /data/public_test.csv ]]; then
  INPUT=/data/public_test.csv
elif [[ -f /data/private_test.json ]]; then
  INPUT=/data/private_test.json
elif [[ -f /data/public_test.json ]]; then
  INPUT=/data/public_test.json
else
  echo "No input file found. Expected /code/private_test.json or a supported /data input." >&2
  exit 1
fi

OUTPUT=${2:-/code/submission.csv}
TRACE_OUTPUT=${3:-/tmp/trace_v03_gamma.jsonl}
TIME_OUTPUT=${4:-/code/submission_time.csv}

mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$TRACE_OUTPUT")" "$(dirname "$TIME_OUTPUT")"

exec python predict.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --trace-output "$TRACE_OUTPUT" \
  --time-output "$TIME_OUTPUT"
