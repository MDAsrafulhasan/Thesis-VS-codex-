import unittest

import pandas as pd

from audit_source_conflict_votes import (
    audit_conflict_votes,
    repository_votes,
    strict_majority,
)
from source_answer_bank import BankEntry


def entry(question: str, answer: str, choices: list[str], source: str) -> BankEntry:
    return BankEntry(
        question=question,
        answers=[answer],
        choices=choices,
        correct_index=choices.index(answer),
        source_repo=source,
        source_file=f"{source}.json",
        license="test",
        source_kind="mcq",
    )


class SourceConflictVoteAuditTests(unittest.TestCase):
    def test_repository_duplicates_count_once(self):
        entries = [
            entry("প্রশ্ন?", "সঠিক", ["সঠিক", "ভুল"], "a"),
            entry("প্রশ্ন?", "সঠিক", ["সঠিক", "ভুল"], "a"),
            entry("প্রশ্ন?", "ভুল", ["সঠিক", "ভুল"], "b"),
        ]
        votes = repository_votes("সঠিক", entries)
        self.assertEqual(votes, {"a": 1, "b": 0})
        self.assertIsNone(strict_majority(votes))

    def test_strict_repository_majority(self):
        self.assertEqual(strict_majority({"a": 1, "b": 1, "c": 0}), 1)
        self.assertIsNone(strict_majority({"a": 1, "b": 0}))

    def test_audit_uses_exact_origin_before_voting(self):
        choices = ["সঠিক", "ভুল"]
        entries = [
            entry("প্রশ্ন?", "সঠিক", choices, "a"),
            entry("প্রশ্ন?", "সঠিক", choices, "b"),
            entry("প্রশ্ন?", "ভুল", choices, "c"),
            entry("প্রশ্ন ?", "ভুল", choices, "d"),
        ]
        frame = pd.DataFrame(
            [
                {
                    "id": 9,
                    "context": "[NULL]",
                    "prompt_bn": "প্রশ্ন?",
                    "response_bn": "সঠিক",
                    "label": 1,
                }
            ]
        )
        records, summary = audit_conflict_votes(frame, entries, base_labels={9: 0})
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["base_disagreement_ids"], [9])
        self.assertEqual(records[0]["repository_votes"], {"a": 1, "b": 1, "c": 0})


if __name__ == "__main__":
    unittest.main()
