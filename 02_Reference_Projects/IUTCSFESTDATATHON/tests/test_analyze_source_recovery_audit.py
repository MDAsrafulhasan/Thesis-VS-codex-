from __future__ import annotations

import unittest

from analyze_source_recovery_audit import analyze


def row(
    split: str,
    row_id: int,
    verdict: int,
    *,
    label: int | None = None,
    confidence: float = 0.99,
    relation: str = "exact_origin",
    has_context: bool = False,
) -> dict:
    return {
        "split": split,
        "id": row_id,
        "label": label,
        "verdict": verdict,
        "confidence": confidence,
        "source_relation": relation,
        "has_context": has_context,
        "hit_count": 1,
        "hit_domains": ["example.org"],
        "prompt_bn": "প্রশ্ন",
        "response_bn": "উত্তর",
    }


class AnalyzeSourceRecoveryAuditTests(unittest.TestCase):
    def test_perfect_supported_cohort_is_review_eligible(self) -> None:
        rows = [
            row("sample", index, index % 2, label=index % 2)
            for index in range(1, 6)
        ]
        rows.append(row("test", 10, 0))
        report, disagreements = analyze(rows, {10: 1})
        self.assertEqual(report["calibration_gated_ids"], [10])
        self.assertTrue(disagreements[0]["calibration_gate"])

    def test_single_control_error_closes_gate(self) -> None:
        rows = [
            row("sample", index, index % 2, label=index % 2)
            for index in range(1, 6)
        ]
        rows[0]["verdict"] = 1 - rows[0]["label"]
        rows.append(row("test", 10, 0))
        report, disagreements = analyze(rows, {10: 1})
        self.assertEqual(report["calibration_gated_ids"], [])
        self.assertFalse(disagreements[0]["calibration_gate"])

    def test_agreement_with_base_is_not_a_candidate(self) -> None:
        rows = [
            row("sample", index, index % 2, label=index % 2)
            for index in range(1, 6)
        ]
        rows.append(row("test", 10, 1))
        report, disagreements = analyze(rows, {10: 1})
        self.assertEqual(report["test_disagreements"], 0)
        self.assertEqual(disagreements, [])


if __name__ == "__main__":
    unittest.main()
