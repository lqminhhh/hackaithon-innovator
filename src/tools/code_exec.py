"""Sandboxed Python code execution for quantitative questions.

Workflow
--------
1. The orchestrator calls build_code_prompt() to ask Qwen3.5 to emit a
   standalone Python script ending in ``print(answer_value)``.
2. execute_code() runs the script in a subprocess with:
     - 5-second timeout
     - whitelisted imports only (math, fractions, datetime, itertools,
       statistics, numpy, sympy)
     - no network access (--network none in Docker; restricted on host)
     - stdout capped at 1 kB
3. match_to_choice() fuzzy-matches the printed numeric value against the
   choice texts with a 1% relative tolerance, handling Vietnamese number
   formats (decimal comma, thousand dot, units).
4. Returns the matching choice letter, or None on failure (crash,
   timeout, no match) so the jury can fall back to signals 1+2.
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

# Imports available inside sandboxed scripts
ALLOWED_IMPORTS = frozenset(
    ["math", "fractions", "datetime", "itertools", "statistics", "numpy", "sympy",
     "decimal", "cmath", "functools", "collections", "operator"]
)

_EXEC_TIMEOUT = 5          # seconds
_MAX_STDOUT = 1024         # bytes
_REL_TOL = 0.01            # 1% relative tolerance for numeric match
_ABS_TOL = 1e-9            # absolute tolerance for near-zero values

# ── Vietnamese number normaliser ─────────────────────────────────────

_UNIT_STRIP = re.compile(
    r"\s*(?:đồng|VNĐ|USD|EUR|km|km/h|m/s|m²|m³|cm|mm|kg|g|mg|l|L|ml|"
    r"kW|W|V|A|Ω|Hz|J|Pa|K|°C|%)\b",
    re.IGNORECASE,
)
_THOUSAND_SEP = re.compile(r"(\d)\.(\d{3})(?!\d)")   # 1.000 → 1000
_DECIMAL_COMMA = re.compile(r"(\d),(\d)")             # 1,5 → 1.5


def normalise_number(text: str) -> Optional[float]:
    """Parse a Vietnamese-format number string to float, or return None."""
    s = text.strip()

    # Detect scale words BEFORE stripping (so they are not removed prematurely)
    scale = 1.0
    if re.search(r"\btỷ\b", s, re.IGNORECASE):
        scale = 1e9
        s = re.sub(r"\btỷ\b", "", s, flags=re.IGNORECASE).strip()
    elif re.search(r"\btriệu\b", s, re.IGNORECASE):
        scale = 1e6
        s = re.sub(r"\btriệu\b", "", s, flags=re.IGNORECASE).strip()
    elif re.search(r"\bnghìn\b|\bngàn\b", s, re.IGNORECASE):
        scale = 1e3
        s = re.sub(r"\bnghìn\b|\bngàn\b", "", s, flags=re.IGNORECASE).strip()

    # Strip units (after scale detection)
    s = _UNIT_STRIP.sub("", s).strip()

    # Remove thousand separators (dots before 3-digit groups)
    s = _THOUSAND_SEP.sub(r"\1\2", s)
    # Convert decimal comma to decimal point
    s = _DECIMAL_COMMA.sub(r"\1.\2", s)

    # Extract the first numeric token
    m = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group()) * scale
    except ValueError:
        return None


def _numbers_close(a: float, b: float) -> bool:
    if a == b:
        return True
    if math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b):
        return False
    return math.isclose(a, b, rel_tol=_REL_TOL, abs_tol=_ABS_TOL)


# ── import whitelist enforcement ───────────────────────────────────────

_IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+(\w+)", re.MULTILINE)


def _check_imports(code: str) -> Optional[str]:
    """Return an error string if the code imports disallowed modules."""
    for m in _IMPORT_RE.finditer(code):
        module = m.group(1)
        if module not in ALLOWED_IMPORTS:
            return f"Disallowed import: {module}"
    return None


# ── code execution ─────────────────────────────────────────────────────


def execute_code(code: str) -> tuple[Optional[str], Optional[str]]:
    """Run code in a subprocess; return (stdout_stripped, error_or_None).

    stdout is capped at _MAX_STDOUT bytes. Timeouts and crashes
    return (None, error_description).
    """
    import_err = _check_imports(code)
    if import_err:
        return None, import_err

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        script_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            timeout=_EXEC_TIMEOUT,
            text=True,
            env={
                "PATH": "/usr/bin:/bin",
                "PYTHONPATH": "",
                "HOME": "/tmp",
            },
        )
        stdout = proc.stdout[:_MAX_STDOUT].strip()
        if proc.returncode != 0:
            return None, f"Exit {proc.returncode}: {proc.stderr[:200]}"
        return stdout or None, None
    except subprocess.TimeoutExpired:
        return None, f"Timeout ({_EXEC_TIMEOUT}s)"
    except Exception as e:
        return None, str(e)
    finally:
        try:
            Path(script_path).unlink(missing_ok=True)
        except Exception:
            pass


# ── match output to choice ─────────────────────────────────────────────


def match_to_choice(
    stdout: str, options: dict[str, str]
) -> Optional[str]:
    """Fuzzy-match the printed value against the choice texts.

    Returns the matching letter, or None if no choice is close enough.
    """
    exec_val = normalise_number(stdout)
    if exec_val is None:
        # Try exact string match (e.g. the script prints a letter directly)
        cleaned = stdout.strip().upper()
        if cleaned in options:
            return cleaned
        return None

    for letter, choice_text in options.items():
        choice_val = normalise_number(choice_text)
        if choice_val is not None and _numbers_close(exec_val, choice_val):
            return letter

    return None


# ── prompt builder ────────────────────────────────────────────────────


def build_code_prompt(query: str, options: dict[str, str]) -> str:
    """Build the user message asking the model to emit runnable Python.

    The resulting code is expected to end with print(answer_value).
    """
    labels = sorted(options.keys())
    options_block = "\n".join(f"{l}. {options[l]}" for l in labels)
    return (
        f"Bài toán: {query}\n\n"
        f"Các lựa chọn:\n{options_block}\n\n"
        "Viết chương trình Python ngắn để tính kết quả. "
        "Dòng cuối cùng phải là: print(answer) "
        "trong đó answer là giá trị số của đáp án đúng. "
        "Chỉ dùng các module: math, fractions, numpy, sympy, statistics, itertools. "
        "Không import gì khác. Chỉ viết code, không giải thích."
    )
