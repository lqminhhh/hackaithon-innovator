#!/usr/bin/env python3
"""Extract clean text from a Vietnamese Wikipedia XML dump.

Replaces wikiextractor (broken on Python 3.11+) with a simple
iterative XML parser + mwparserfromhell for wikitext stripping.

Usage:
    python scripts/extract_wiki.py \
        --input data/viwiki-latest-pages-articles.xml.bz2 \
        --output data/wiki_text
"""

from __future__ import annotations

import argparse
import bz2
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

# Namespace prefix in the MediaWiki XML dump
MW_NS = "{http://www.mediawiki.org/xml/export-0.11/}"

# Minimal wikitext cleanup when mwparserfromhell is unavailable
_WIKI_MARKUP = re.compile(
    r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]"  # [[link|display]] → display
    r"|'''?"                             # bold / italic markers
    r"|\{\{[^}]*\}\}"                   # templates
    r"|<ref[^>]*>.*?</ref>"             # references
    r"|<[^>]+>"                          # HTML tags
    r"|\[https?://[^\]]*\]"             # external links
    r"|={2,}.*?={2,}",                  # headings
    re.DOTALL,
)


def strip_wikitext(text: str) -> str:
    """Strip wikitext markup to plain text."""
    try:
        import mwparserfromhell
        return mwparserfromhell.parse(text).strip_code()
    except ImportError:
        pass
    text = _WIKI_MARKUP.sub(r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_articles(dump_path: str, output_dir: str, articles_per_file: int = 1000):
    """Parse the bz2-compressed XML dump and write JSON files."""
    os.makedirs(output_dir, exist_ok=True)

    file_idx = 0
    article_count = 0
    buffer: list[str] = []

    def flush():
        nonlocal file_idx, buffer
        if not buffer:
            return
        subdir = os.path.join(output_dir, "AA")
        os.makedirs(subdir, exist_ok=True)
        out_path = os.path.join(subdir, f"wiki_{file_idx:05d}")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(buffer) + "\n")
        file_idx += 1
        buffer = []

    print(f"Reading {dump_path}...")
    opener = bz2.open if dump_path.endswith(".bz2") else open

    with opener(dump_path, "rt", encoding="utf-8", errors="replace") as f:
        for event, elem in ET.iterparse(f, events=("end",)):
            if elem.tag != f"{MW_NS}page":
                continue

            ns_elem = elem.find(f"{MW_NS}ns")
            if ns_elem is not None and ns_elem.text != "0":
                elem.clear()
                continue

            title_elem = elem.find(f"{MW_NS}title")
            title = title_elem.text if title_elem is not None else ""

            id_elem = elem.find(f"{MW_NS}id")
            page_id = id_elem.text if id_elem is not None else ""

            rev = elem.find(f"{MW_NS}revision")
            text_elem = rev.find(f"{MW_NS}text") if rev is not None else None
            raw_text = text_elem.text if text_elem is not None and text_elem.text else ""

            if raw_text.lower().startswith("#redirect") or raw_text.lower().startswith("#đổi"):
                elem.clear()
                continue

            clean = strip_wikitext(raw_text)
            if len(clean) < 50:
                elem.clear()
                continue

            record = json.dumps(
                {"id": page_id, "title": title, "text": clean},
                ensure_ascii=False,
            )
            buffer.append(record)
            article_count += 1

            if len(buffer) >= articles_per_file:
                flush()

            if article_count % 10000 == 0:
                print(f"  {article_count} articles extracted...")

            elem.clear()

    flush()
    print(f"Done: {article_count} articles → {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract Vietnamese Wikipedia dump to JSON")
    parser.add_argument("--input", required=True, help="Path to .xml.bz2 dump")
    parser.add_argument("--output", required=True, help="Output directory for JSON files")
    args = parser.parse_args()
    extract_articles(args.input, args.output)


if __name__ == "__main__":
    main()
