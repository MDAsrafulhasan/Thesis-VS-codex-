import json
import unittest

from hf_phase2_data import iter_legal_acts


class NormalizeLegalActsTests(unittest.TestCase):
    def test_iter_legal_acts_nested_payload(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            from pathlib import Path
            path = Path(directory) / "acts.json"
            path.write_text(json.dumps({"dataset_info": {}, "acts": [{
                "act_title": "আইন", "source_url": "https://example.test", "sections": [
                    {"section_title": "ধারা ১", "section_content": "এই আইন প্রযোজ্য।"}
                ]
            }]}, ensure_ascii=False), encoding="utf-8")
            rows = list(iter_legal_acts(path))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["question"], "আইন: ধারা ১")


if __name__ == "__main__":
    unittest.main()
