#!/usr/bin/env python3
"""Analyse public_test.csv to identify topic distribution and gaps.

Scans all questions, classifies them by topic using keyword heuristics,
and prints a summary table.  Use this to decide which domain texts to
add to the knowledge base.

Usage:
    python scripts/topic_map.py --input data/public_test.csv
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "Lịch sử": [
        "lịch sử", "triều đại", "thế kỷ", "chiến tranh", "cách mạng",
        "năm nào", "thời kỳ", "vua", "đế quốc", "phong trào",
    ],
    "Địa lý": [
        "địa lý", "quốc gia", "thủ đô", "sông", "núi", "biển",
        "châu lục", "dân số", "diện tích", "khí hậu",
    ],
    "Khoa học tự nhiên": [
        "hóa học", "vật lý", "sinh học", "nguyên tố", "phản ứng",
        "năng lượng", "tế bào", "gen", "ADN", "phân tử",
    ],
    "Toán học": [
        "tính", "phương trình", "hàm số", "đạo hàm", "tích phân",
        "xác suất", "thống kê", "hình học", "đại số",
    ],
    "Văn học": [
        "tác phẩm", "tác giả", "nhà thơ", "nhà văn", "truyện",
        "thơ", "văn học", "bài thơ", "tiểu thuyết",
    ],
    "Pháp luật": [
        "luật", "hiến pháp", "điều", "nghị định", "quy định",
        "quyền", "pháp luật", "hình sự", "dân sự",
    ],
    "Kinh tế": [
        "kinh tế", "GDP", "lạm phát", "thương mại", "tài chính",
        "ngân hàng", "chứng khoán", "thuế", "doanh nghiệp",
    ],
    "Công nghệ": [
        "máy tính", "internet", "phần mềm", "AI", "trí tuệ nhân tạo",
        "lập trình", "dữ liệu", "mạng", "công nghệ",
    ],
}


def detect_topic(text: str) -> str:
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in text_lower)
        if score > 0:
            scores[topic] = score
    if not scores:
        return "Khác"
    return max(scores, key=scores.get)


def main():
    parser = argparse.ArgumentParser(description="Analyse topic distribution")
    parser.add_argument("--input", default="data/public_test.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Total questions: {len(df)}\n")

    topic_counter = Counter()
    for _, row in df.iterrows():
        text = f"{row['question']} {row['A']} {row['B']} {row['C']} {row['D']}"
        topic = detect_topic(text)
        topic_counter[topic] += 1

    print(f"{'Topic':<25} {'Count':>6} {'%':>7}")
    print("-" * 40)
    for topic, count in topic_counter.most_common():
        pct = 100 * count / len(df)
        print(f"{topic:<25} {count:>6} {pct:>6.1f}%")

    if "Khác" in topic_counter:
        print(
            f"\n⚠  {topic_counter['Khác']} questions ({100*topic_counter['Khác']/len(df):.1f}%) "
            f"are unclassified — consider adding domain texts for these topics."
        )


if __name__ == "__main__":
    main()
