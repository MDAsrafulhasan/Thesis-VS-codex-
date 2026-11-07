import json
from pathlib import Path
import tempfile
import unittest

from typed_evidence import infer_intent, load_typed_evidence, typed_evidence_decision


class TypedEvidenceTests(unittest.TestCase):
    def test_infers_distinct_senses_and_relations(self) -> None:
        self.assertEqual(infer_intent("'তালকানা' বাগধারাটির অর্থ কী?"), "figurative_meaning")
        self.assertEqual(infer_intent("'তালকানা' শব্দের শাব্দিক অর্থ কী?"), "literal_meaning")
        self.assertEqual(
            infer_intent("উপন্যাসটির ধারাবাহিক প্রকাশ কত সালে শুরু হয়?"),
            "serialization_start",
        )
        self.assertEqual(infer_intent("ব্যক্তিটি কত সালে মৃত্যুবরণ করেন?"), "death_date")

    def test_accepts_and_rejects_only_explicit_answers(self) -> None:
        index = load_typed_evidence([Path("resources/typed_evidence_v1.json")])
        prompt = "এখানে 'তালকানা' বাগধারা/প্রবাদটির অর্থ কী?"
        accepted, accepted_evidence = typed_evidence_decision(prompt, "কাণ্ডজ্ঞানহীন", "", index)
        rejected, rejected_evidence = typed_evidence_decision(
            prompt, "ছন্দ বা তাল বুঝতে না পারা, বেসুরা", "", index
        )
        abstained, _ = typed_evidence_decision(prompt, "অজানা অন্য উত্তর", "", index)
        self.assertEqual(accepted, 1)
        self.assertEqual(rejected, 0)
        self.assertIsNone(abstained)
        self.assertEqual(accepted_evidence["intent"], "figurative_meaning")
        self.assertEqual(rejected_evidence["rule"], "typed_exact_evidence")

    def test_rejects_mismatched_declared_intent(self) -> None:
        rows = [
            {
                "question": "'তালকানা' শব্দের শাব্দিক অর্থ কী?",
                "intent": "figurative_meaning",
                "accepted_answers": ["তালহীন"],
                "sources": ["https://one.example", "https://two.example"],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "declared intent"):
                load_typed_evidence([path])

    def test_rejects_overlapping_answer_sets(self) -> None:
        rows = [
            {
                "question": "'তালকানা' শব্দের শাব্দিক অর্থ কী?",
                "intent": "literal_meaning",
                "accepted_answers": ["তালহীন"],
                "rejected_answers": ["তালহীন"],
                "sources": ["https://one.example", "https://two.example"],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_typed_evidence([path])


if __name__ == "__main__":
    unittest.main()
