#!/bin/bash
set -euo pipefail

INPUT=${1:-/data/public_test.json}
OUTPUT=${2:-/data/submission.csv}
AUDIT=${3:-/data/submission_audit.json}

echo "=== Entropy-Gated Jury ==="
echo "Input:  $INPUT"
echo "Output: $OUTPUT"
echo "Audit:  $AUDIT"
echo "Started: $(date)"

python src/pipeline.py \
    --input  "$INPUT" \
    --output "$OUTPUT" \
    --audit  "$AUDIT"

echo "Finished: $(date)"
echo "=== Done ==="
