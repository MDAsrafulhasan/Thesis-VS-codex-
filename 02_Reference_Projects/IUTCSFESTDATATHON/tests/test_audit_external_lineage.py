from pathlib import Path
import tempfile
import unittest

import pandas as pd

from audit_external_lineage import response_matches, tydiqa_matches


class ExternalLineageAuditTests(unittest.TestCase):
    def test_exact_and_contained_answers(self):
        self.assertTrue(response_matches("নব্যপ্রস্তর", ["নব্যপ্রস্তর যুগে"]))
        self.assertTrue(response_matches("১৯১৫", ["১৯১৫ সালের পরে"]))

    def test_different_answer(self):
        self.assertFalse(response_matches("মধ্যপ্রস্তর", ["নব্যপ্রস্তর যুগে"]))

    def test_tydiqa_requires_exact_normalized_gold(self):
        source = pd.DataFrame(
            [
                {
                    "id": "bengali-source",
                    "question": "প্রতিষ্ঠাতা কে?",
                    "answers": {"text": ["সম্রাট ওয়েন"], "answer_start": [0]},
                }
            ]
        )
        frame = pd.DataFrame(
            [
                {"id": 1, "prompt_bn": "প্রতিষ্ঠাতা কে?", "response_bn": "সম্রাট ওয়েন"},
                {"id": 2, "prompt_bn": "প্রতিষ্ঠাতা কে?", "response_bn": "ইয়াং সম্রাট ওয়েন"},
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.parquet"
            source.to_parquet(path)
            records = tydiqa_matches(frame, [path])

        self.assertEqual([record["id"] for record in records], [1])


if __name__ == "__main__":
    unittest.main()
