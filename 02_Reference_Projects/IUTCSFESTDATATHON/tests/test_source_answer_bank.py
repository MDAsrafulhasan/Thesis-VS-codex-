from __future__ import annotations

import unittest

from datathon_utils import normalize_bn
from source_answer_bank import (
    BankEntry,
    canonical_answer,
    compare_mcq,
    compare_open_qa,
    number_signature,
    source_answers_conflict,
    strip_inline_choices,
)


def mcq(gold: int, choices: list[str]) -> BankEntry:
    return BankEntry(
        question="প্রশ্ন",
        answers=[choices[gold]],
        choices=choices,
        correct_index=gold,
        source_repo="test",
        source_file="test.json",
        license="test",
        source_kind="mcq",
    )


def open_qa(answer: str) -> BankEntry:
    return BankEntry(
        question="প্রশ্ন",
        answers=[answer],
        choices=[],
        correct_index=None,
        source_repo="test",
        source_file="test.json",
        license="test",
        source_kind="open_qa",
    )


class CanonicalAnswerTest(unittest.TestCase):
    def test_bengali_unicode_variants_normalize(self) -> None:
        self.assertEqual(normalize_bn("কোথায় হয়"), normalize_bn("কোথায় হয়"))

    def test_bengali_digits_and_year_suffix(self) -> None:
        self.assertEqual(canonical_answer("১৯১৩ সালে"), "1913")
        self.assertEqual(number_signature("৬.৪৫"), ("6.45",))
        self.assertEqual(number_signature("১২.৫০%"), ("12.5",))

    def test_html_and_count_suffix(self) -> None:
        self.assertEqual(canonical_answer("∠AOD &gt; ∠BOC"), "aod>boc")
        self.assertEqual(canonical_answer("১৯৩টি"), "193")
        self.assertEqual(number_signature("২রা নভেম্বর, ১৮৮৬"), ("2", "1886"))
        self.assertEqual(canonical_answer("A.A’=1"), canonical_answer("A.A' = 1"))

    def test_inline_bengali_mcq_choices_are_removed(self) -> None:
        question = "মুক্তিযুদ্ধভিত্তিক উপন্যাস কোনটি? ক) কাঁটাতারে প্রজাপতি খ) অন্যটি"
        self.assertEqual(strip_inline_choices(question), "মুক্তিযুদ্ধভিত্তিক উপন্যাস কোনটি?")
        self.assertIsNone(strip_inline_choices("বাংলাদেশের রাজধানী কী?"))


class ComparatorTest(unittest.TestCase):
    def test_exact_wrong_option_is_hallucinated(self) -> None:
        entry = mcq(1, ["ঢাকা", "চট্টগ্রাম", "খুলনা", "রাজশাহী"])
        self.assertEqual(compare_mcq("খুলনা", entry).label, 0)

    def test_numeric_format_variant_is_faithful(self) -> None:
        entry = mcq(2, ["0.0929", "7.32", "6.45", "64.5"])
        decision = compare_mcq("৬.৪৫", entry)
        self.assertEqual((decision.label, decision.rule), (1, "mcq_exact_option"))

    def test_negative_endpoint_does_not_collapse_into_positive_option(self) -> None:
        entry = mcq(
            0,
            [
                "-(3/2) < x < -1",
                "-(3/2) < x < 1",
                "-(3/2) ≤ x ≤ 1",
                "-(3/2) < x ≤ 1",
            ],
        )
        decision = compare_mcq("-(3/2) < x < 1", entry)
        self.assertEqual((decision.label, decision.rule), (0, "mcq_exact_option"))

    def test_open_and_closed_interval_options_remain_distinct(self) -> None:
        entry = mcq(1, ["[1, ∞)", "(1, ∞)", "(12, ∞)", "(-1, ∞)"])
        self.assertEqual(compare_mcq("[1,∞)", entry).label, 0)
        self.assertEqual(compare_mcq("(1,∞)", entry).label, 1)

    def test_gold_substring_inside_explicit_distractor_is_rejected(self) -> None:
        entry = mcq(
            1,
            [
                "Physical address",
                "Logical address",
                "Both physical and logical address",
                "উপরের কোনটি নয়",
            ],
        )
        decision = compare_mcq("Both physical and logical addresses", entry)
        self.assertEqual((decision.label, decision.rule), (0, "mcq_fuzzy_distractor"))

    def test_short_entity_variant_is_faithful(self) -> None:
        entry = mcq(0, ["বঙ্গবন্ধু শেখ মুজিবুর রহমান", "শেখ হাসিনা", "ভাসানী", "ফজলুল হক"])
        self.assertEqual(compare_mcq("শেখ মুজিবুর রহমান", entry).label, 1)

    def test_scaled_open_qa_number_abstains(self) -> None:
        entry = open_qa("প্রায় ১০ মিলিয়ন")
        self.assertIsNone(compare_open_qa("প্রায় ১ কোটি", entry))

    def test_cross_script_translation_abstains(self) -> None:
        entry = open_qa("Weapons of Mass Destruction")
        self.assertIsNone(compare_open_qa("গণবিধ্বংসী অস্ত্র", entry))

    def test_disagreeing_source_keys_conflict(self) -> None:
        first = mcq(0, ["প্রধানমন্ত্রী", "রাষ্ট্রপতি", "মন্ত্রী", "সচিব"])
        second = BankEntry(
            question=first.question,
            answers=["রাষ্ট্রপতি"],
            choices=first.choices,
            correct_index=1,
            source_repo="second-source",
            source_file="second.json",
            license="test",
            source_kind="mcq",
        )
        self.assertTrue(source_answers_conflict([first, second]))

    def test_malformed_source_number_abstains(self) -> None:
        entry = open_qa("১৯১৫১৯১৫ সালের পরে")
        self.assertIsNone(compare_open_qa("১৯১৫", entry))


if __name__ == "__main__":
    unittest.main()
