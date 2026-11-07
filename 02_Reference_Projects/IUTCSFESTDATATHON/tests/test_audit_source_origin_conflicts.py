import unittest

import pandas as pd

from audit_source_origin_conflicts import (
    audit_frame,
    exact_origin_entries,
    raw_signature,
)
from source_answer_bank import BankEntry


def entry(question: str, answer: str, source: str) -> BankEntry:
    return BankEntry(
        question=question,
        answers=[answer],
        choices=[answer, "ভুল"],
        correct_index=0,
        source_repo=source,
        source_file=f"{source}.json",
        license="test",
        source_kind="mcq",
    )


class SourceOriginConflictAuditTests(unittest.TestCase):
    def test_raw_signature_preserves_punctuation(self):
        self.assertNotEqual(raw_signature("কোথায়?"), raw_signature("কোথায় -"))

    def test_exact_raw_origin_selects_matching_source(self):
        entries = [
            entry("সঠিক উত্তর কী?", "ক", "new"),
            entry("সঠিক উত্তর কী ?", "খ", "old"),
        ]
        matches = exact_origin_entries("সঠিক উত্তর কী ?", entries)
        self.assertEqual({item.source_repo for item in matches}, {"old"})

    def test_conflict_audit_is_perfect_when_origin_matches(self):
        entries = [
            entry("আসন কত?", "৫০", "new"),
            entry("আসন কত ?", "৩০", "old"),
        ]
        frame = pd.DataFrame([
            {
                "id": 7,
                "context": "[NULL]",
                "prompt_bn": "আসন কত ?",
                "response_bn": "৩০",
                "label": 1,
            }
        ])
        records, summary = audit_frame(frame, entries, base_labels={7: 0})
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["label"], 1)
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["base_disagreement_ids"], [7])


if __name__ == "__main__":
    unittest.main()
