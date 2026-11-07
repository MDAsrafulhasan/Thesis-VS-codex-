import unittest

import pandas as pd

from build_phase1_audit_ledger import build_ledger, classify_route


class Phase1AuditLedgerTests(unittest.TestCase):
    def test_routes_specialized_lexical_and_reasoning_rows(self) -> None:
        self.assertEqual(classify_route("শব্দটির বিপরীত শব্দ কী?", "উত্তর", "", "antonym"), "lexical_antonym")
        self.assertEqual(classify_route("বাগধারাটির অর্থ কী?", "উত্তর", "", "qa"), "lexical_idiom")
        self.assertEqual(classify_route("৩:২ অনুপাতে মোট কত?", "৫", "", "numeric_or_date"), "structured_reasoning")

    def test_new_records_cannot_be_submitted_without_adjudication(self) -> None:
        unresolved = pd.DataFrame([{"id": 7, "row_index": 0, "block": 1, "category": "qa", "base_label": 0}])
        test = pd.DataFrame([{"id": 7, "prompt_bn": "প্রশ্ন?", "response_bn": "উত্তর", "context": "[NULL]"}])
        record = build_ledger(unresolved, test)[0]
        self.assertEqual(record["confidence_tier"], "U")
        self.assertIsNone(record["proposed_label"])
        self.assertFalse(record["submission_eligible"])


if __name__ == "__main__":
    unittest.main()
