"""Generate creativity-component charts from an audit_log.json.

Produces three figures required by planning doc §5:
  1. Confidence-distribution histogram (tier-1 logprob margins, all 463 questions)
  2. Escalation set breakdown (tier-1 vs tier-2 questions, by type)
  3. Agreement matrix (Qwen vote × Gemma verdict for all jury questions)

Usage
-----
python eval/charts.py --audit data/submission_audit.json --outdir docs/charts/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_audit(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def print_confidence_histogram(entries: list[dict], bins: int = 20) -> None:
    """ASCII histogram of tier-1 logprob margins."""
    margins = [
        e["tier1"]["margin"]
        for e in entries
        if "tier1" in e and "margin" in e["tier1"]
    ]
    if not margins:
        print("No margin data found.")
        return

    min_m = min(margins)
    max_m = max(margins)
    width = (max_m - min_m) / bins if max_m > min_m else 1.0
    counts = [0] * bins
    for m in margins:
        idx = min(int((m - min_m) / width), bins - 1)
        counts[idx] += 1

    max_count = max(counts, default=1)
    bar_width = 40

    print("\n=== Tier-1 Logprob Margin Distribution ===")
    for i, count in enumerate(counts):
        lo = min_m + i * width
        hi = lo + width
        bar = "█" * int(bar_width * count / max_count)
        print(f"  {lo:6.2f}–{hi:5.2f} | {bar:<{bar_width}} {count}")
    print(f"\n  n={len(margins)}, mean={sum(margins)/len(margins):.2f}, "
          f"min={min_m:.2f}, max={max_m:.2f}")


def print_escalation_breakdown(entries: list[dict]) -> None:
    """Tier-1 vs tier-2 breakdown by question type flag."""
    tier1 = [e for e in entries if e.get("tier") == 1]
    tier2 = [e for e in entries if e.get("tier") == 2]

    print(f"\n=== Escalation Breakdown ===")
    print(f"  Tier 1 (fast-exit): {len(tier1)} ({len(tier1)/len(entries):.0%})")
    print(f"  Tier 2 (jury):      {len(tier2)} ({len(tier2)/len(entries):.0%})")

    flag_keys = ["has_context", "is_quantitative", "has_refusal_choice", "is_legal"]
    print("\n  Escalation rate by flag:")
    for flag in flag_keys:
        flagged = [e for e in entries if e.get("flags", {}).get(flag)]
        esc = [e for e in flagged if e.get("tier") == 2]
        if flagged:
            print(f"    {flag:<25}: {len(esc)}/{len(flagged)} escalated ({len(esc)/len(flagged):.0%})")

    by_n = {}
    for e in entries:
        n = e.get("flags", {}).get("n_choices", 4)
        if n not in by_n:
            by_n[n] = [0, 0]
        by_n[n][0] += 1
        if e.get("tier") == 2:
            by_n[n][1] += 1
    print("\n  Escalation rate by n_choices:")
    for n in sorted(by_n):
        total, esc = by_n[n]
        print(f"    n={n:<3}: {esc}/{total} ({esc/total:.0%})")


def print_agreement_matrix(entries: list[dict]) -> None:
    """Qwen vote × Gemma verdict agreement matrix for jury questions."""
    jury_entries = [e for e in entries if "jury" in e]
    if not jury_entries:
        print("\nNo jury entries found.")
        return

    agree = sum(
        1 for e in jury_entries
        if e["jury"].get("qwen_vote") == e["jury"].get("gemma_vote")
        and e["jury"].get("gemma_vote") is not None
    )
    disagree = sum(
        1 for e in jury_entries
        if e["jury"].get("gemma_vote") is not None
        and e["jury"].get("qwen_vote") != e["jury"].get("gemma_vote")
    )
    no_gemma = sum(1 for e in jury_entries if e["jury"].get("gemma_vote") is None)

    tool_breaks = sum(
        1 for e in jury_entries
        if e["jury"].get("tool_answer") is not None
    )

    print(f"\n=== Jury Agreement Matrix ({len(jury_entries)} questions) ===")
    print(f"  Qwen == Gemma (agree):  {agree} ({agree/len(jury_entries):.0%})")
    print(f"  Qwen != Gemma (disagree): {disagree} ({disagree/len(jury_entries):.0%})")
    print(f"  Gemma unavailable:      {no_gemma}")
    print(f"  Tool answer provided:   {tool_breaks}")

    resolutions = {}
    for e in jury_entries:
        rule = e["jury"].get("resolution", "unknown")
        resolutions[rule] = resolutions.get(rule, 0) + 1
    print("\n  Resolution rules used:")
    for rule, count in sorted(resolutions.items(), key=lambda x: -x[1]):
        print(f"    {rule:<40}: {count}")


def save_json_summary(entries: list[dict], outdir: Path) -> None:
    margins = [
        e["tier1"]["margin"]
        for e in entries
        if "tier1" in e and "margin" in e["tier1"]
    ]
    jury_entries = [e for e in entries if "jury" in e]
    agree = sum(
        1 for e in jury_entries
        if e["jury"].get("qwen_vote") == e["jury"].get("gemma_vote")
        and e["jury"].get("gemma_vote") is not None
    )
    summary = {
        "n_questions": len(entries),
        "tier1_count": sum(1 for e in entries if e.get("tier") == 1),
        "tier2_count": sum(1 for e in entries if e.get("tier") == 2),
        "mean_margin": sum(margins) / len(margins) if margins else None,
        "jury_agree": agree,
        "jury_disagree": len(jury_entries) - agree,
    }
    out = outdir / "audit_summary.json"
    outdir.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {out}")


def main():
    parser = argparse.ArgumentParser(description="EGJ audit-log charts")
    parser.add_argument("--audit", required=True)
    parser.add_argument("--outdir", default="docs/charts")
    args = parser.parse_args()

    entries = load_audit(args.audit)
    print(f"Loaded {len(entries)} audit entries from {args.audit}")

    print_confidence_histogram(entries)
    print_escalation_breakdown(entries)
    print_agreement_matrix(entries)
    save_json_summary(entries, Path(args.outdir))


if __name__ == "__main__":
    main()
