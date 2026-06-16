"""S0 I/O compatibility module.

The project already had ``data_loader.py`` before the v2 plan was written.
This module provides the planned S0 import surface while keeping the existing
implementation in one place.
"""

from src.data_loader import letters, load_questions, write_submission

__all__ = ["letters", "load_questions", "write_submission"]
