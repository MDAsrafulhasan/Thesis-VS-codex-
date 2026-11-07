import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from idiom_semantic_overrides import (
    apply_corrections,
    correction_for,
    extract_idiom,
    load_corrections,
    load_idiom_dataset,
)


class IdiomSemanticOverrideTests(unittest.TestCase):
    def test_extracts_both_prompt_orders(self) -> None:
        self.assertEqual(
            extract_idiom("এখানে 'তালকানা' বাগধারা/প্রবাদটির অর্থ কী?"),
            "তালকানা",
        )
        self.assertEqual(
            extract_idiom("এখানে 'মাকাল ফল' প্রবাদ/বাগধারাটির অর্থ কী?"),
            "মাকাল ফল",
        )
        self.assertIsNone(extract_idiom("'তালকানা' শব্দের আক্ষরিক অর্থ কী?"))

    def test_content_keyed_correction_does_not_depend_on_row_id(self) -> None:
        source = [
            {
                "idiom": "তালকানা",
                "literal_meaning": "তালহীন ছন্দ",
                "figurative_meaning_bn": "কাণ্ডজ্ঞানহীন",
            }
        ]
        corrections = [
            {
                "idiom": "তালকানা",
                "rejected_responses": ["বেসুরা"],
                "correct_label": 0,
                "figurative_meaning_bn": "কাণ্ডজ্ঞানহীন",
                "reason": "literal sense",
                "sources": ["https://source-one.example", "https://source-two.example"],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.json"
            correction_path = root / "corrections.json"
            source_path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
            correction_path.write_text(json.dumps(corrections, ensure_ascii=False), encoding="utf-8")
            loaded = load_corrections(correction_path, load_idiom_dataset(source_path))

        self.assertIsNotNone(
            correction_for("এখানে 'তালকানা' বাগধারা/প্রবাদটির অর্থ কী?", "বেসুরা", loaded)
        )
        self.assertIsNone(
            correction_for("এখানে 'তালকানা' বাগধারা/প্রবাদটির অর্থ কী?", "কাণ্ডজ্ঞানহীন", loaded)
        )

        frame = pd.DataFrame(
            [
                {
                    "id": 991337,
                    "context": "তার কাজকর্ম তালকানা।",
                    "prompt_bn": "এখানে 'তালকানা' বাগধারা/প্রবাদটির অর্থ কী?",
                    "response_bn": "বেসুরা",
                }
            ]
        )
        base = pd.DataFrame([{"id": 991337, "label": 1}])
        output, trace = apply_corrections(frame, base, loaded)
        self.assertEqual(output.to_dict("records"), [{"id": 991337, "label": 0}])
        self.assertEqual(trace[0]["id"], 991337)
        self.assertNotIn("991337", Path(__file__).parents[1].joinpath(
            "scripts/idiom_semantic_overrides.py"
        ).read_text(encoding="utf-8"))

    def test_rejects_correction_that_disagrees_with_dataset(self) -> None:
        source = [{"idiom": "তালকানা", "figurative_meaning_bn": "কাণ্ডজ্ঞানহীন"}]
        corrections = [
            {
                "idiom": "তালকানা",
                "rejected_responses": ["বেসুরা"],
                "correct_label": 0,
                "figurative_meaning_bn": "সুরহীন",
                "sources": ["a", "b"],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.json"
            correction_path = root / "corrections.json"
            source_path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
            correction_path.write_text(json.dumps(corrections, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "figurative meaning disagrees"):
                load_corrections(correction_path, load_idiom_dataset(source_path))


if __name__ == "__main__":
    unittest.main()
