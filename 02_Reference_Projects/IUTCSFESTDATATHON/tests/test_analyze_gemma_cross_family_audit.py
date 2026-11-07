from __future__ import annotations

import unittest

from analyze_gemma_cross_family_audit import analyze


def record(
    split: str,
    row_id: int,
    consensus: int | None,
    *,
    has_context: bool = False,
) -> dict:
    return {
        "split": split,
        "id": row_id,
        "consensus": consensus,
        "has_context": has_context,
        "prompt_bn": "প্রশ্ন",
        "response_bn": "উত্তর",
    }


class AnalyzeGemmaCrossFamilyAuditTests(unittest.TestCase):
    def test_harmful_control_disagreements_close_gate(self) -> None:
        trace = [
            record("sample", row_id, 0) for row_id in range(5)
        ] + [record("test", 10, 0)]
        report, rows = analyze(
            trace,
            {row_id: 1 for row_id in range(5)},
            {row_id: 1 for row_id in range(5)},
            {10: 1},
        )
        self.assertEqual(report["sample_fixes"], 0)
        self.assertFalse(rows[0]["calibration_gate"])

    def test_five_perfect_corrections_open_direction_gate(self) -> None:
        trace = [
            record("sample", row_id, 0) for row_id in range(5)
        ] + [record("test", 10, 0)]
        report, rows = analyze(
            trace,
            {row_id: 0 for row_id in range(5)},
            {row_id: 1 for row_id in range(5)},
            {10: 1},
        )
        self.assertEqual(report["sample_fixes"], 5)
        self.assertTrue(rows[0]["calibration_gate"])

    def test_consensus_agreement_is_not_a_test_candidate(self) -> None:
        trace = [record("sample", 0, 1), record("test", 10, 1)]
        report, rows = analyze(trace, {0: 1}, {0: 1}, {10: 1})
        self.assertEqual(report["test_disagreements"], 0)
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
