from __future__ import annotations

import unittest

from audit_transitive_answer_equivalence import calibrate_threshold, summarize_judge


class CalibrationTests(unittest.TestCase):
    def test_rejects_single_lucky_positive(self) -> None:
        result = calibrate_threshold(
            [0.95, 0.93, 0.92, 0.90],
            [1, 0, 1, 1],
            min_positive_controls=3,
        )
        self.assertIsNone(result.threshold)
        self.assertEqual(result.reason, "insufficient_zero_false_positive_controls")

    def test_selects_broadest_zero_false_positive_threshold(self) -> None:
        result = calibrate_threshold(
            [0.99, 0.97, 0.95, 0.80, 0.70],
            [1, 1, 1, 0, 1],
            min_positive_controls=3,
        )
        self.assertEqual(result.threshold, 0.95)
        self.assertEqual(result.selected, 3)
        self.assertEqual(result.precision, 1.0)

    def test_empty_controls_abstain(self) -> None:
        result = calibrate_threshold([], [], min_positive_controls=1)
        self.assertIsNone(result.threshold)
        self.assertEqual(result.reason, "no_controls")

    def test_sample_judge_uses_zero_based_row_index(self) -> None:
        records = [
            {"row_index": 7, "id": 8, "truth": 1},
            {"row_index": 8, "id": 9, "truth": 0},
        ]
        summary = summarize_judge(
            records,
            {7: {"pred": 1, "confidence": 0.9}, 8: {"pred": 1, "confidence": 0.8}},
            sample=True,
        )
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(len(summary["wrong"]), 1)

    def test_test_judge_reports_both_flip_directions(self) -> None:
        records = [
            {"row_index": 0, "id": 10},
            {"row_index": 1, "id": 11},
        ]
        summary = summarize_judge(
            records,
            {10: {"pred": 1}, 11: {"pred": 0}},
            sample=False,
            base_labels={10: 0, 11: 1},
        )
        self.assertEqual(summary["disagreements"], 2)
        self.assertEqual(summary["zero_to_one"], 1)
        self.assertEqual(summary["one_to_zero"], 1)


if __name__ == "__main__":
    unittest.main()
