import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from audit_azmine_bcs import audit, load_azmine_file


class AzmineBcsAuditTests(unittest.TestCase):
    def test_one_based_answer_index_is_parsed(self):
        rows = [{
            "question": "বাংলাদেশের রাজধানী কী?",
            "options": ["চট্টগ্রাম", "ঢাকা", "খুলনা", "রাজশাহী"],
            "answer": "2",
            "file_name": "test.pdf",
        }]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bcs.json"
            path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            entries = load_azmine_file(path)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].answers, ["ঢাকা"])
        self.assertEqual(entries[0].correct_index, 1)

    def test_exact_audit_compares_base(self):
        rows = [{
            "question": "বাংলাদেশের রাজধানী কী?",
            "options": ["চট্টগ্রাম", "ঢাকা"],
            "answer": "2",
            "file_name": "test.pdf",
        }]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bcs.json"
            path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            entries = load_azmine_file(path)
        frame = pd.DataFrame([{
            "id": 5,
            "context": "[NULL]",
            "prompt_bn": "বাংলাদেশের রাজধানী কী?",
            "response_bn": "ঢাকা",
            "label": 1,
        }])
        payload = audit(frame, entries, base_labels={5: 0})
        self.assertEqual(payload["summary"]["exact_correct"], 1)
        self.assertEqual(payload["summary"]["exact_base_disagreement_ids"], [5])


if __name__ == "__main__":
    unittest.main()
