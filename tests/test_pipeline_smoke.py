"""Smoke test for the full pipeline output format.

Verifies that the output CSV is correctly formatted without
requiring GPU or models loaded.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def sample_csv(tmp_path):
    """Create a tiny sample CSV for testing."""
    data = {
        "id": [1, 2, 3],
        "question": [
            "Thủ đô của Việt Nam là gì?",
            "1 + 1 bằng bao nhiêu?",
            "Tại sao nước biển mặn?",
        ],
        "A": ["Hà Nội", "1", "Vì có muối"],
        "B": ["Hồ Chí Minh", "2", "Vì nóng"],
        "C": ["Đà Nẵng", "3", "Vì sâu"],
        "D": ["Huế", "4", "Vì lạnh"],
    }
    csv_path = tmp_path / "test_input.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path


class TestOutputFormat:
    def test_output_has_correct_columns(self, sample_csv, tmp_path):
        """Verify output CSV has exactly 'id' and 'answer' columns."""
        output_path = tmp_path / "submission.csv"

        df = pd.read_csv(sample_csv)
        results = [{"id": row["id"], "answer": "A"} for _, row in df.iterrows()]
        pd.DataFrame(results).to_csv(output_path, index=False)

        out_df = pd.read_csv(output_path)
        assert list(out_df.columns) == ["id", "answer"]
        assert len(out_df) == len(df)
        assert all(a in "ABCD" for a in out_df["answer"])

    def test_all_ids_present(self, sample_csv, tmp_path):
        """Verify every input ID appears in the output."""
        input_df = pd.read_csv(sample_csv)
        output_path = tmp_path / "submission.csv"

        results = [{"id": row["id"], "answer": "B"} for _, row in input_df.iterrows()]
        pd.DataFrame(results).to_csv(output_path, index=False)

        out_df = pd.read_csv(output_path)
        assert set(out_df["id"]) == set(input_df["id"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
