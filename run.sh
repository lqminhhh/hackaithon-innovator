#!/bin/bash
INPUT_CSV=${1:-/data/public_test.csv}
OUTPUT_CSV=${2:-/data/submission.csv}
python src/pipeline.py --input "$INPUT_CSV" --output "$OUTPUT_CSV"
