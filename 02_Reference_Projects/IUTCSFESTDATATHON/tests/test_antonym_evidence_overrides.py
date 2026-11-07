import unittest

from antonym_evidence_overrides import (
    canonical,
    headword_from_prompt,
    parse_list_source,
    parse_wiktionary_antonyms,
    source_consensus,
)


class AntonymEvidenceTests(unittest.TestCase):
    def test_parse_wiktionary_antonym_section(self):
        text = """=== বিশেষণ ===
==== বিপরীতার্থক শব্দ ====
স্থির, শান্ত, [[ধীর]]
=== অনুবাদ ===
ইংরেজি: restless
"""
        self.assertEqual(parse_wiktionary_antonyms(text), {"স্থির", "শান্ত", "ধীর"})

    def test_parse_wiktionary_templates(self):
        text = """=== বিপরীত ===
* {{l|bn|পরাধীন}}
=== আরও দেখুন ===
স্বাধীনতা
"""
        self.assertEqual(parse_wiktionary_antonyms(text), {"পরাধীন"})

    def test_parses_numbered_table(self):
        body = "<table><tr><th>ক্রম</th><th>শব্দ</th><th>বিপরীত</th></tr><tr><td>১</td><td>ভূত</td><td>ভবিষ্যৎ</td></tr></table>"
        self.assertEqual(parse_list_source(body), [(canonical("ভূত"), {canonical("ভবিষ্যৎ")})])

    def test_two_sources_support(self):
        records = [
            {"source": "a", "url": "a", "headword": canonical("তীক্ষ্ণ"), "answers": [canonical("ভোঁতা")]},
            {"source": "b", "url": "b", "headword": canonical("তীক্ষ্ণ"), "answers": [canonical("ভোঁতা")]},
        ]
        pred, _ = source_consensus(canonical("তীক্ষ্ণ"), "ভোঁতা", records, 2)
        self.assertEqual(pred, 1)

    def test_two_sources_contradict(self):
        records = [
            {"source": "a", "url": "a", "headword": canonical("ভূত"), "answers": [canonical("ভবিষ্যৎ")]},
            {"source": "b", "url": "b", "headword": canonical("ভূত"), "answers": [canonical("ভবিষ্যত")]},
        ]
        pred, _ = source_consensus(canonical("ভূত"), "বর্তমান", records, 2, reject_alternatives=True)
        self.assertEqual(pred, 0)

    def test_alternative_antonym_does_not_imply_negative(self):
        records = [
            {"source": "a", "url": "a", "headword": canonical("ক্ষণস্থায়ী"), "answers": [canonical("দীর্ঘস্থায়ী")]},
            {"source": "b", "url": "b", "headword": canonical("ক্ষণস্থায়ী"), "answers": [canonical("দীর্ঘস্থায়ী")]},
        ]
        pred, _ = source_consensus(canonical("ক্ষণস্থায়ী"), "চিরস্থায়ী", records, 2)
        self.assertIsNone(pred)

    def test_one_source_abstains(self):
        records = [{"source": "a", "url": "a", "headword": canonical("ভূত"), "answers": [canonical("ভবিষ্যৎ")]}]
        pred, _ = source_consensus(canonical("ভূত"), "বর্তমান", records, 2)
        self.assertIsNone(pred)

    def test_parses_closing_quote_only_prompt(self):
        self.assertEqual(headword_from_prompt("সমীপ' শব্দের শুদ্ধ বিপরীত শব্দ কোনটি?"), canonical("সমীপ"))


if __name__ == "__main__":
    unittest.main()
