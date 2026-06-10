#!/bin/bash
# Download and extract Vietnamese Wikipedia dump.
#
# Usage:
#   bash scripts/download_wiki.sh
#
# Output:
#   data/wiki_text/  — folder of JSON files (one article per line)

set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/../data" && pwd)"
DUMP_URL="https://dumps.wikimedia.org/viwiki/latest/viwiki-latest-pages-articles.xml.bz2"
DUMP_FILE="$DATA_DIR/viwiki-latest-pages-articles.xml.bz2"
OUTPUT_DIR="$DATA_DIR/wiki_text"

echo "=== Vietnamese Wikipedia Download & Extraction ==="

# Step 1: Download
if [ -f "$DUMP_FILE" ]; then
    echo "Dump already downloaded: $DUMP_FILE"
else
    echo "Downloading from: $DUMP_URL"
    echo "(This is ~1.5 GB, may take a while...)"
    wget -c -O "$DUMP_FILE" "$DUMP_URL"
fi

# Step 2: Extract with custom script (wikiextractor is broken on Python 3.11+)
if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ]; then
    echo "Wiki text already extracted: $OUTPUT_DIR"
else
    echo "Installing mwparserfromhell (for wikitext cleanup)..."
    pip install mwparserfromhell 2>/dev/null || pip3 install mwparserfromhell

    echo "Extracting articles (this takes 15-30 minutes)..."
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    python "$SCRIPT_DIR/extract_wiki.py" \
        --input "$DUMP_FILE" \
        --output "$OUTPUT_DIR"

    echo "Extraction complete."
fi

# Summary
ARTICLE_COUNT=$(find "$OUTPUT_DIR" -type f | xargs wc -l 2>/dev/null | tail -1 | awk '{print $1}')
echo ""
echo "=== Done ==="
echo "Output directory: $OUTPUT_DIR"
echo "Approx lines (articles): $ARTICLE_COUNT"
echo ""
echo "Next step: run 'python scripts/build_index.py' to build the FAISS index."
