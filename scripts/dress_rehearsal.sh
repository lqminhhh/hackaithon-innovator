#!/bin/bash
# Dress rehearsal — full offline timed run in Docker
# Usage: bash scripts/dress_rehearsal.sh [input_file] [image_tag]
#
# Verifies:
#   1. Docker image runs with --network none (no internet at eval time)
#   2. submission.csv is produced with the correct format
#   3. Wall-clock reported
#   4. CSV passes the invariant check (correct columns, no nulls)
#
# Run this at least 3 times before submission. All three must pass.

set -euo pipefail

INPUT=${1:-data/public-test_1780368312.json}
IMAGE=${2:-bangc:egj}
OUTDIR=$(mktemp -d)

echo "=== Dress Rehearsal ==="
echo "Image:  $IMAGE"
echo "Input:  $INPUT"
echo "Output: $OUTDIR"
echo ""

# Build image if needed
if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "[build] Building image $IMAGE ..."
    docker build -t "$IMAGE" .
fi

echo "[run] Starting container (--network none)..."
START_TS=$(date +%s)

docker run --rm \
    --gpus all \
    --network none \
    -v "$(pwd)/$INPUT":/data/input.json:ro \
    -v "$OUTDIR":/data \
    "$IMAGE" \
    /data/input.json /data/submission.csv /data/audit.json

END_TS=$(date +%s)
ELAPSED=$(( END_TS - START_TS ))

echo ""
echo "[validate] Checking output..."

python3 - <<EOF
import sys, pandas as pd
df = pd.read_csv("$OUTDIR/submission.csv")
assert list(df.columns) == ["id", "answer"], f"Wrong columns: {list(df.columns)}"
assert df["id"].notna().all(), "Null ids found"
assert df["answer"].notna().all(), "Null answers found"
assert (df["answer"].str.len() == 1).all(), "Non-single-letter answers found"
print(f"  Rows:    {len(df)}")
print(f"  Columns: {list(df.columns)}")
print(f"  Answers: {df['answer'].value_counts().to_dict()}")
EOF

echo ""
echo "=== PASS ==="
echo "Wall-clock: ${ELAPSED}s"
echo "Output: $OUTDIR/submission.csv"
echo ""

if [ "$ELAPSED" -gt 900 ]; then
    echo "WARNING: ${ELAPSED}s > 15 min budget. Review thinking budgets and n_consistency."
fi
