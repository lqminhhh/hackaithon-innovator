#!/bin/bash
# BTC-compatible container entrypoint.
# Reads /code/private_test.json by default and writes:
# - /code/submission.csv
# - /code/submission_time.csv
set -euo pipefail

exec python predict.py "$@"
