from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mlx_qwen_judge import row_indices


class RowSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "id": [10, 11, 12, 13],
                "has_context": [True, False, True, False],
                "category": ["qa", "qa", "numeric_or_date", "qa"],
            }
        )

    def test_id_file_intersects_context_slice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ids.csv"
            pd.DataFrame({"id": [10, 11, 13]}).to_csv(path, index=False)
            self.assertEqual(row_indices(self.frame, "context", str(path)).tolist(), [0])

    def test_router_trace_uncertainty_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(
                json.dumps({"uncertainty": [{"id": 11}, {"id": 12}]}),
                encoding="utf-8",
            )
            self.assertEqual(row_indices(self.frame, "no_context", str(path)).tolist(), [1])


if __name__ == "__main__":
    unittest.main()
