import unittest

import pandas as pd

from audit_benhallu_raw_lineage import audit_raw_lineage


class BenHalluRawLineageAuditTests(unittest.TestCase):
    def test_separates_raw_exact_from_normalized_only(self):
        source = [
            ("প্রসঙ্গ", "প্রশ্ন?", "উত্তর", 1, "gold", "g1"),
            ("প্রসঙ্গ", "প্রশ্ন?", "ভুল", 0, "factualness", "h1"),
        ]
        frame = pd.DataFrame(
            [
                {
                    "id": 1,
                    "context": "প্রসঙ্গ",
                    "prompt_bn": "প্রশ্ন?",
                    "response_bn": "উত্তর",
                    "label": 1,
                },
                {
                    "id": 2,
                    "context": "প্রসঙ্গ",
                    "prompt_bn": "প্রশ্ন ?",
                    "response_bn": "ভুল",
                    "label": 0,
                },
            ]
        )
        records, summary = audit_raw_lineage(
            frame,
            source,
            base_labels={1: 1, 2: 0},
        )
        self.assertEqual(len(records), 2)
        self.assertTrue(records[0]["raw_exact"])
        self.assertFalse(records[1]["raw_exact"])
        self.assertEqual(summary["normalized_only_ids"], [2])
        self.assertEqual(summary["correct"], 2)
        self.assertEqual(summary["base_disagreements"], 0)

    def test_excludes_normalized_label_conflicts(self):
        source = [
            ("প্রসঙ্গ", "প্রশ্ন?", "উত্তর", 1, "gold", "g1"),
            ("প্রসঙ্গ", "প্রশ্ন ?", "উত্তর", 0, "factualness", "h1"),
        ]
        frame = pd.DataFrame(
            [
                {
                    "id": 1,
                    "context": "প্রসঙ্গ",
                    "prompt_bn": "প্রশ্ন?",
                    "response_bn": "উত্তর",
                    "label": 1,
                }
            ]
        )
        records, summary = audit_raw_lineage(frame, source)
        self.assertEqual(records, [])
        self.assertEqual(summary["normalized_label_conflicts"], 1)


if __name__ == "__main__":
    unittest.main()
