import unittest

from hf_phase2_data import evidence_record, nli_record


class HfPhase2DataTests(unittest.TestCase):
    def test_nli_labels_map_to_hallucination_binary(self):
        entailment = nli_record({"premise": "ঢাকা বাংলাদেশের রাজধানী।", "hypothesis": "বাংলাদেশের রাজধানী ঢাকা।", "label": 0}, "repo", "file", "cc0", "train")
        contradiction = nli_record({"premise": "ঢাকা বাংলাদেশের রাজধানী।", "hypothesis": "চট্টগ্রাম বাংলাদেশের রাজধানী।", "label": 2}, "repo", "file", "cc0", "train")
        neutral = nli_record({"premise": "ঢাকা বাংলাদেশের রাজধানী।", "hypothesis": "ঢাকায় আজ বৃষ্টি।", "label": 1}, "repo", "file", "cc0", "train")
        self.assertEqual(entailment["label"], 1)
        self.assertEqual(contradiction["label"], 0)
        self.assertEqual(neutral["label"], 0)

    def test_evidence_schema(self):
        row = evidence_record({"question": "প্রশ্ন?", "answer": "উত্তর", "context": "প্রমাণ", "date": "2025"}, "repo", "file", "law", "cc-by")
        self.assertEqual(row["domain"], "law")
        self.assertEqual(row["document_date"], "2025")


if __name__ == "__main__":
    unittest.main()
