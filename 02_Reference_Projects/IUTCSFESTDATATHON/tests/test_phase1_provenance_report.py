import json
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from phase1_provenance_report import choose_tier, lexical_decision, load_bagdhara, load_evidence_trace


class Phase1ProvenanceReportTests(unittest.TestCase):
    def test_tier_priority(self):
        families = [
            {"tier": "B_deterministic"},
            {"tier": "A_released_provenance"},
            {"tier": "A_exact_answer_bank"},
        ]
        self.assertEqual(choose_tier(families), "A_released_provenance")
        self.assertEqual(choose_tier([]), "C_unresolved")

    def test_loads_multiple_trace_shapes(self):
        with TemporaryDirectory() as directory:
            structured = Path(directory) / "structured_math_trace.json"
            structured.write_text(json.dumps([{"id": 7, "new": 0, "rule": "binary"}]), encoding="utf-8")
            self.assertEqual(load_evidence_trace(structured)[7]["label"], 0)
            self.assertEqual(load_evidence_trace(structured)[7]["family"], "structured")

            verified = Path(directory) / "verified_fact_trace.json"
            verified.write_text(json.dumps({"records": [{"id": 9, "prediction": 1}]}), encoding="utf-8")
            self.assertEqual(load_evidence_trace(verified)[9]["tier"], "A_verified_evidence")

            tydiqa = Path(directory) / "tydiqa_gold_trace.json"
            tydiqa.write_text(json.dumps({"records": [{"id": 11, "prediction": 1}]}), encoding="utf-8")
            evidence = load_evidence_trace(tydiqa)[11]
            self.assertEqual(evidence["family"], "tydiqa_gold_exact")
            self.assertEqual(evidence["tier"], "A_exact_answer_bank")

    def test_exact_task_bagdhara(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bagdhara.jsonl"
            item = {"idiom": "আকাশ কুসুম", "alternative_idioms": [], "literal_meaning": "আকাশের ফুল", "figurative_meaning_bn": "অবাস্তব কল্পনা"}
            path.write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")
            bank = load_bagdhara(path)
            label, score = lexical_decision('"আকাশ কুসুম" এর ভাবার্থ কী?', "অবাস্তব কল্পনা", bank, 0.4)
            self.assertEqual(label, 1)
            self.assertEqual(score, 1.0)
            wrong_task, _ = lexical_decision('"আকাশ কুসুম" এর শাব্দিক অর্থ কী?', "অবাস্তব কল্পনা", bank, 0.4)
            self.assertEqual(wrong_task, 0)

            wrapped, _ = lexical_decision(
                "এখানে 'আকাশ কুসুম' প্রবাদ/বাগধারাটির অর্থ কী?",
                "অবাস্তব কল্পনা",
                bank,
                0.4,
                positive_only=True,
            )
            self.assertEqual(wrapped, 1)
            abstained, _ = lexical_decision(
                "এখানে 'আকাশ কুসুম' প্রবাদ/বাগধারাটির অর্থ কী?",
                "খুব বাস্তব ঘটনা",
                bank,
                0.4,
                positive_only=True,
            )
            self.assertIsNone(abstained)


if __name__ == "__main__":
    unittest.main()
