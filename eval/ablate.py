"""Ablation runner — toggles components via config and diffs the score report.

Usage
-----
python eval/ablate.py \\
    --input  data/public-test_1780368312.json \\
    --gold   eval/dev_set.jsonl \\
    --output data/ablation_results.json

For each ablation defined in ABLATIONS, it:
  1. Writes a temporary pipeline_config.yaml with the component disabled
  2. Runs the pipeline end-to-end
  3. Scores the output
  4. Records accuracy delta vs the baseline run

Rule from planning doc: no change merges without a dev-set delta.
Negative or flat ablations (disabling a component → no loss) → ship disabled.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_CFG_PATH = ROOT / "configs" / "pipeline_config.yaml"

# Each ablation: (name, dict of config overrides to apply)
ABLATIONS: list[tuple[str, dict]] = [
    ("baseline_all_on", {}),
    ("no_jury",       {"ablation": {"use_jury": False}}),
    ("no_code_exec",  {"ablation": {"use_code_exec": False}}),
    ("no_rag",        {"ablation": {"use_rag": False}}),
    ("no_gemma",      {"ablation": {"use_gemma": False}}),
    ("no_thinking",   {"ablation": {"use_thinking": False}}),
    ("tau_high",      {"gate": {"tau": 3.0}}),
    ("tau_low",       {"gate": {"tau": 0.5}}),
]


def _deep_merge(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def run_ablation(
    name: str,
    overrides: dict,
    input_path: str,
    gold_path: str,
    base_cfg: dict,
) -> dict:
    merged = _deep_merge(base_cfg, overrides)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=ROOT / "configs", encoding="utf-8"
    ) as tmp_cfg_file:
        yaml.dump(merged, tmp_cfg_file, allow_unicode=True)
        tmp_cfg_path = tmp_cfg_file.name

    with tempfile.NamedTemporaryFile(
        suffix=".csv", delete=False, dir=ROOT / "data"
    ) as tmp_out:
        out_path = tmp_out.name

    audit_path = out_path.replace(".csv", "_audit.json")

    t_start = time.time()
    result = {"name": name, "overrides": overrides}

    try:
        proc = subprocess.run(
            [
                sys.executable, str(ROOT / "src" / "pipeline.py"),
                "--input", input_path,
                "--output", out_path,
                "--audit", audit_path,
            ],
            capture_output=True,
            text=True,
            timeout=1200,
            env={**__import__("os").environ, "EGJ_CONFIG": tmp_cfg_path},
        )
        elapsed = time.time() - t_start
        result["elapsed_s"] = round(elapsed, 1)

        if proc.returncode != 0:
            result["error"] = proc.stderr[-500:]
            return result

        # Score
        score_proc = subprocess.run(
            [
                sys.executable, str(ROOT / "eval" / "score.py"),
                "--pred", out_path,
                "--gold", gold_path,
                "--audit", audit_path,
                "--elapsed", str(elapsed),
                "--tag", name,
            ],
            capture_output=True, text=True,
        )
        result["score_stdout"] = score_proc.stdout

        # Parse accuracy from stdout
        for line in score_proc.stdout.splitlines():
            if "Accuracy" in line:
                try:
                    result["accuracy"] = float(line.split()[1])
                except (IndexError, ValueError):
                    pass

    except subprocess.TimeoutExpired:
        result["error"] = "timeout"
    except Exception as e:
        result["error"] = str(e)
    finally:
        Path(tmp_cfg_path).unlink(missing_ok=True)
        Path(out_path).unlink(missing_ok=True)
        Path(audit_path).unlink(missing_ok=True)

    return result


def main():
    parser = argparse.ArgumentParser(description="EGJ ablation runner")
    parser.add_argument("--input", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--output", default="data/ablation_results.json")
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=None,
        help="Names of specific ablations to run (default: all)",
    )
    args = parser.parse_args()

    with open(_CFG_PATH) as f:
        base_cfg = yaml.safe_load(f)

    ablations_to_run = [
        (name, overrides)
        for name, overrides in ABLATIONS
        if args.ablations is None or name in args.ablations
    ]

    all_results: list[dict] = []
    baseline_acc: float | None = None

    for name, overrides in ablations_to_run:
        print(f"\n{'─' * 50}")
        print(f"Running ablation: {name}")
        print(f"Overrides: {overrides or '(none)'}")
        result = run_ablation(name, overrides, args.input, args.gold, base_cfg)

        if name == "baseline_all_on":
            baseline_acc = result.get("accuracy")

        if baseline_acc is not None and "accuracy" in result:
            result["delta_vs_baseline"] = round(result["accuracy"] - baseline_acc, 4)

        all_results.append(result)
        print(f"  accuracy={result.get('accuracy', 'n/a')}  delta={result.get('delta_vs_baseline', 'n/a')}")

    # Summary table
    print(f"\n{'=' * 55}")
    print(f"{'Name':<25} {'Acc':>8} {'Delta':>8} {'Time':>8}")
    print(f"{'─' * 55}")
    for r in all_results:
        print(
            f"{r['name']:<25} "
            f"{r.get('accuracy', 'ERR'):>8} "
            f"{r.get('delta_vs_baseline', ''):>8} "
            f"{r.get('elapsed_s', ''):>7}s"
        )
    print(f"{'=' * 55}\n")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
