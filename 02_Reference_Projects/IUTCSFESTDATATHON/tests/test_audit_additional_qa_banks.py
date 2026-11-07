import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from audit_additional_qa_banks import load_json_squad, load_pair_json


class AdditionalQaBankAuditTests(unittest.TestCase):
    def test_loads_squad_and_banglarqa_shapes(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "qa.json"
            path.write_text(
                json.dumps(
                    {
                        "data": [
                            {
                                "paragraphs": [
                                    {
                                        "qas": [
                                            {"question": "রাজধানী কী?", "answers": [{"text": "ঢাকা"}]}
                                        ]
                                    }
                                ]
                            },
                            {
                                "qas": [
                                    {
                                        "question_text": "উত্তর কী?",
                                        "answers": {"answer_text": ["হ্যাঁ", "হ্যাঁ "]},
                                    }
                                ]
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            bank = load_json_squad([path])
        self.assertEqual(bank["রাজধানী কী"], {"ঢাকা"})
        self.assertEqual(bank["উত্তর কী"], {"হ্যাঁ"})

    def test_loads_input_output_pair_shape(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.json"
            path.write_text(
                json.dumps([{"input": " প্রশ্ন? ", "output": " উত্তর "}], ensure_ascii=False),
                encoding="utf-8",
            )
            bank = load_pair_json(path)
        self.assertEqual(bank["প্রশ্ন"], {"উত্তর"})


if __name__ == "__main__":
    unittest.main()
