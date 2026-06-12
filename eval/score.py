"""Evaluation scorer for the Entropy-Gated Jury.

Usage
-----
python eval/score.py \\
    --pred  data/submission.csv \\
    --gold  eval/dev_set.jsonl \\
    --audit data/submission_audit.json   # optional, for tier / tool stats

Outputs
-------
  Overall accuracy, per-type accuracy, per-choice-count accuracy,
  escalation rate, tool usage, wall-clock (if --elapsed <seconds> given).
  Appends one row to docs/version_results.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_gold(path: str) -> dict[str, dict]:
    """Load ground-truth from dev_set.jsonl.

    Each line: {"qid": ..., "answer": "B", "type": "quantitative", "n_choices": 4}
    Returns {qid: {answer, type, n_choices}}.
    """
    gold: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qid = str(obj["qid"])
            gold[qid] = {
                "answer": str(obj["answer"]).upper(),
                "type": obj.get("type", "unknown"),
                "n_choices": int(obj.get("n_choices", 4)),
            }
    return gold


def load_pred(path: str) -> dict[str, str]:
    """Load predictions from submission.csv → {qid: letter}."""
    df = pd.read_csv(path)
    id_col = "id" if "id" in df.columns else "qid"
    return {str(row[id_col]): str(row["answer"]).upper() for _, row in df.iterrows()}


def load_audit(path: str) -> dict[str, dict]:
    """Load audit_log.json → {qid: entry}."""
    if not path or not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)
    return {str(e["qid"]): e for e in entries}


def score(pred: dict[str, str], gold: dict[str, dict], audit: dict[str, dict]) -> dict:
    qids = sorted(gold.keys())
    missing = [qid for qid in qids if qid not in pred]
    if missing:
        print(f"WARNING: {len(missing)} qids missing from predictions", file=sys.stderr)

    correct_total = 0
    by_type: dict[str, list[bool]] = defaultdict(list)
    by_n_choices: dict[int, list[bool]] = defaultdict(list)
    tier_counts = defaultdict(int)
    tool_counts = defaultdict(int)

    for qid in qids:
        gold_ans = gold[qid]["answer"]
        pred_ans = pred.get(qid, "")
        ok = pred_ans == gold_ans
        correct_total += int(ok)
        by_type[gold[qid]["type"]].append(ok)
        by_n_choices[gold[qid]["n_choices"]].append(ok)

        if qid in audit:
            ae = audit[qid]
            tier_counts[ae.get("tier", "?")] += 1
            jury = ae.get("jury", {})
            tool = jury.get("tool_answer")
            if tool:
                tool_counts["tool_hit"] += 1
            if "jury" in ae:
                tool_counts["jury_total"] += 1

    n = len(qids)
    result: dict = {
        "n": n,
        "overall_acc": correct_total / n if n else 0.0,
        "by_type": {k: sum(v) / len(v) for k, v in sorted(by_type.items())},
        "by_n_choices": {k: sum(v) / len(v) for k, v in sorted(by_n_choices.items())},
        "tier1_count": tier_counts.get(1, 0),
        "tier2_count": tier_counts.get(2, 0),
        "tool_hits": tool_counts.get("tool_hit", 0),
        "jury_total": tool_counts.get("jury_total", 0),
    }
    return result


def print_report(result: dict, elapsed: float | None = None) -> None:
    print(f"\n{'=' * 55}")
    print(f"  Accuracy:  {result['overall_acc']:.4f}  ({result['n']} questions)")
    if elapsed is not None:
        print(f"  Wall-clock: {elapsed:.1f}s  ({elapsed / result['n']:.2f}s/q)")

    print("\n  Per type:")
    for t, acc in result["by_type"].items():
        print(f"    {t:<25} {acc:.4f}")

    print("\n  Per choice count:")
    for n, acc in result["by_n_choices"].items():
        print(f"    n={n:<3}  {acc:.4f}")

    if result["tier1_count"] or result["tier2_count"]:
        total = result["tier1_count"] + result["tier2_count"]
        print(
            f"\n  Tier 1: {result['tier1_count']} ({result['tier1_count']/total:.0%})  "
            f"Tier 2: {result['tier2_count']} ({result['tier2_count']/total:.0%})"
        )
    if result["jury_total"]:
        print(
            f"  Tool hits: {result['tool_hits']} / {result['jury_total']} jury questions"
            f" ({result['tool_hits']/result['jury_total']:.0%})"
        )
    print(f"{'=' * 55}\n")


def append_version_results(result: dict, tag: str, elapsed: float | None) -> None:
    vr_path = ROOT / "docs" / "version_results.md"
    if not vr_path.exists():
        return
    wall = f"{elapsed:.0f}s" if elapsed else "TBD"
    row = (
        f"| {tag} | - | EGJ refactor | - | - | "
        f"Qwen3.5-9B + Gemma4-E4B | BGE-M3 narrow | "
        f"{result['overall_acc']:.4f} | {wall} | TBD | - |"
    )
    with open(vr_path, "a", encoding="utf-8") as f:
        f.write(row + "\n")


def main():
    parser = argparse.ArgumentParser(description="Score EGJ predictions")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--audit", default=None)
    parser.add_argument("--elapsed", type=float, default=None, help="Wall-clock seconds")
    parser.add_argument("--tag", default="dev", help="Version tag for version_results.md")
    args = parser.parse_args()

    gold = load_gold(args.gold)
    pred = load_pred(args.pred)
    audit = load_audit(args.audit) if args.audit else {}
    result = score(pred, gold, audit)
    print_report(result, args.elapsed)
    append_version_results(result, args.tag, args.elapsed)


if __name__ == "__main__":
    main()
