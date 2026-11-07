from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from law_evidence_overrides import (
    Expected,
    child_marriage_expected,
    compare_expected,
    dna_expected,
    extract_maximum,
    load_law_corpus,
    section_text,
)


class LawParserTest(unittest.TestCase):
    def test_extract_maximum_imprisonment(self) -> None:
        text = "তিনি অনধিক ৫ (পাঁচ) বৎসর কারাদণ্ড, তবে ৩ (তিন) বৎসরের নিম্নে নহে"
        self.assertEqual(extract_maximum(text, "imprisonment_years"), (5, None))

    def test_extract_maximum_fine(self) -> None:
        text = "বা অনধিক ৩ (তিন) লক্ষ টাকা অর্থদণ্ড, বা উভয় দণ্ডে দণ্ডিত"
        self.assertEqual(extract_maximum(text, "fine"), (3, "লক্ষ"))

    def test_bengali_word_quantity_comparison(self) -> None:
        expected = Expected(5, None, "৫ বছর", "test", "test.mdx", "evidence")
        self.assertEqual(compare_expected("পাঁচ বছর", expected), 1)
        self.assertEqual(compare_expected("দশ বছর", expected), 0)

    def test_scaled_fine_comparison(self) -> None:
        expected = Expected(3, "লক্ষ", "৩ লক্ষ টাকা", "test", "test.mdx", "evidence")
        self.assertEqual(compare_expected("তিন লক্ষ টাকা", expected), 1)
        self.assertEqual(compare_expected("তিন হাজার টাকা", expected), 0)

    def test_section_extraction(self) -> None:
        text = "### প্রথম\n২৮। alpha\n### দ্বিতীয়\n২৯। beta"
        self.assertEqual(section_text(text, 28), "alpha")

    def test_normalized_corpus_can_supply_an_act(self) -> None:
        corpus = {
            "1151": "২৮। তিনি অনধিক ৫ (পাঁচ) বৎসর কারাদণ্ড, তবে ৩ (তিন) বৎসরের নিম্নে নহে।"
        }
        expected = dna_expected(
            "ডিএনএ আইন, ২০১৪ এর ২৮ ধারা অনুযায়ী সর্বোচ্চ কত বছর কারাদণ্ড হতে পারে?",
            None,
            corpus,
        )
        self.assertIsNotNone(expected)
        self.assertEqual((expected.value, expected.rule), (5, "dna_max_imprisonment"))

    def test_packaged_legacy_act_resolves_child_age(self) -> None:
        corpus = load_law_corpus(Path("resources/legacy_law_acts.json"), {"149"})
        expected = child_marriage_expected(
            "বাল্যবিবাহ নিরোধ আইন, ১৯২৯ অনুযায়ী পুরুষদের ক্ষেত্রে শিশু বলতে কত বছরের কম?",
            None,
            corpus,
        )
        self.assertIsNotNone(expected)
        self.assertEqual((expected.value, expected.rule), (21, "child_age_male"))
