"""Assembler — merges tier-1 accepts and tier-2 verdicts into the final submission.

Hard invariants before writing (planning doc §3.5):
  - Exactly N rows (N = number of input questions)
  - Every qid present exactly once
  - Every answer in that question's legal letter set
  - No nulls

Behaviour on violation:
  - Dev mode (strict=True)  → raise AssemblyError immediately
  - Prod mode (strict=False) → log warning, use safe fallback (tier-1 or "A")

Outputs:
  submission.csv       id,answer
  audit_log.json       per-question detail: flags, tier, jury votes, margins, rationale
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.jury import JuryVerdict
from src.inference import InferenceResult
from src.parsing import ParsedQuestion

logger = logging.getLogger(__name__)


class AssemblyError(RuntimeError):
    pass


def assemble(
    questions: list[ParsedQuestion],
    tier1_results: list[InferenceResult],
    tier1_accepted: list[bool],
    jury_verdicts: dict[str, JuryVerdict],   # qid → verdict
    output_csv: str | Path,
    output_audit: str | Path,
    strict: bool = False,
) -> pd.DataFrame:
    """Assemble and validate the submission, then write both output files.

    Parameters
    ----------
    questions        all parsed questions (must be complete)
    tier1_results    tier-1 InferenceResult for every question
    tier1_accepted   whether each question passed the confidence gate
    jury_verdicts    JuryVerdicts indexed by qid (only escalated questions)
    output_csv       path for submission.csv
    output_audit     path for audit_log.json
    strict           if True, raises on invariant violations
    """
    output_csv = Path(output_csv)
    output_audit = Path(output_audit)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_audit.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    audit_rows: list[dict] = []
    seen_qids: set[str] = set()
    errors: list[str] = []

    for q, t1_result, accepted in zip(questions, tier1_results, tier1_accepted):
        qid = q.qid

        if qid in seen_qids:
            msg = f"Duplicate qid: {qid}"
            errors.append(msg)
            if strict:
                raise AssemblyError(msg)
            logger.warning(msg)
            continue
        seen_qids.add(qid)

        # Choose answer source
        if accepted:
            answer = t1_result.letter
            tier = 1
            verdict = None
        else:
            verdict = jury_verdicts.get(qid)
            if verdict is None:
                msg = f"Missing jury verdict for escalated qid: {qid}"
                errors.append(msg)
                if strict:
                    raise AssemblyError(msg)
                logger.warning(msg + " — using tier-1 fallback")
                answer = t1_result.letter
                tier = 1
            else:
                answer = verdict.answer
                tier = 2

        # Validate answer is in legal letter set
        valid_letters = q.valid_letters
        if answer not in valid_letters:
            msg = f"qid={qid}: answer={answer!r} not in {valid_letters}"
            errors.append(msg)
            if strict:
                raise AssemblyError(msg)
            logger.warning(msg + " — falling back to tier-1 or 'A'")
            answer = t1_result.letter if t1_result.letter in valid_letters else valid_letters[0]

        if not answer:
            msg = f"qid={qid}: null answer"
            errors.append(msg)
            if strict:
                raise AssemblyError(msg)
            logger.warning(msg + " — using 'A'")
            answer = valid_letters[0]

        rows.append({"id": qid, "answer": answer})

        # Audit entry
        audit_entry: dict = {
            "qid": qid,
            "answer": answer,
            "tier": tier,
            "flags": {
                "has_context": q.has_context,
                "is_quantitative": q.is_quantitative,
                "has_refusal_choice": q.has_refusal_choice,
                "is_legal": q.is_legal,
                "n_choices": q.n_choices,
            },
            "tier1": {
                "letter": t1_result.letter,
                "margin": t1_result.margin,
                "logprob_dist": t1_result.logprob_dist,
            },
        }
        if verdict is not None:
            audit_entry["jury"] = {
                "qwen_vote": verdict.qwen_vote,
                "qwen_margin": verdict.qwen_margin,
                "gemma_vote": verdict.gemma_vote,
                "tool_answer": verdict.tool_answer,
                "resolution": verdict.resolution_rule,
                "detail": verdict.audit,
            }
        audit_rows.append(audit_entry)

    # Final row-count check
    n_input = len(questions)
    n_output = len(rows)
    if n_output != n_input:
        msg = f"Row count mismatch: input={n_input}, output={n_output}"
        errors.append(msg)
        if strict:
            raise AssemblyError(msg)
        logger.warning(msg)

    if errors:
        logger.warning("Assembly completed with %d errors: %s", len(errors), errors[:5])

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    logger.info("Wrote %d rows to %s", len(df), output_csv)

    with open(output_audit, "w", encoding="utf-8") as f:
        json.dump(audit_rows, f, ensure_ascii=False, indent=2)
    logger.info("Wrote audit log to %s", output_audit)

    return df
