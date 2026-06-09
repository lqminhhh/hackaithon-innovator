#!/bin/bash
INPUT=${1:-/data/public_test.json}
OUTPUT=${2:-/data/submission.csv}
python src/pipeline.py --input "$INPUT" --output "$OUTPUT"
