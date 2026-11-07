from pathlib import Path
import unittest

from typed_idiom_matcher import TypedIdiomMatcher, extract_typed_idiom


DATASET = Path(
    "outputs/source_cache/bangla_bagdhara_lrec/"
    "bagdhara_export_done_2026-03-01_010356.json"
)


@unittest.skipUnless(DATASET.exists(), "public Bangla Bagdhara audit dataset is not cached")
class TypedIdiomMatcherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.matcher = TypedIdiomMatcher(DATASET)

    def test_extracts_quoted_and_dash_prompt_forms(self) -> None:
        self.assertEqual(
            extract_typed_idiom("এখানে 'তালকানা' বাগধারা/প্রবাদটির অর্থ কী?"),
            "তালকানা",
        )
        self.assertEqual(
            extract_typed_idiom("শিব রাত্রির সলতে – বাগধারটির অর্থ কী?"),
            "শিব রাত্রির সলতে",
        )

    def test_distinguishes_talkana_literal_and_figurative_senses(self) -> None:
        prompt = "এখানে 'তালকানা' বাগধারা/প্রবাদটির অর্থ কী?"
        literal, evidence = self.matcher.decide(prompt, "তালহীন বা বিশৃঙ্খল ছন্দ")
        figurative, _ = self.matcher.decide(prompt, "কাণ্ডজ্ঞানহীন, বিবেচনাবিহীন")
        self.assertEqual(literal, 0)
        self.assertEqual(figurative, 1)
        self.assertEqual(evidence["rule"], "typed_idiom_opposite_sense")
        self.assertEqual(evidence["match_kind"], "primary")

    def test_exact_headword_beats_an_unrelated_alias_record(self) -> None:
        label, evidence = self.matcher.decide(
            '"হেরফের" এর শাব্দিক অর্থ কী?',
            "ছোটখাটো পরিবর্তন বা ওঠানামা",
        )
        self.assertEqual(label, 1)
        self.assertEqual(evidence["match_kind"], "primary")
        self.assertTrue(
            all(record["source_idiom"] == "হেরফের" for record in evidence["records"])
        )

    def test_conflicting_duplicate_headword_records_abstain(self) -> None:
        label, _ = self.matcher.decide(
            '"খেটেখুটে" এর শাব্দিক অর্থ কী?',
            "পরিশ্রম করে",
        )
        self.assertIsNone(label)

    def test_shared_friend_token_does_not_accept_an_opposite_meaning(self) -> None:
        label, _ = self.matcher.decide(
            "এখানে 'দুধের মাছি' বাগধারা/প্রবাদটির অর্থ কী?",
            "দুঃসময়ের বন্ধু",
        )
        self.assertIsNone(label)

    def test_low_overlap_response_abstains(self) -> None:
        label, _ = self.matcher.decide(
            "এখানে 'হাতের পাঁচ' বাগধারা/প্রবাদটির অর্থ কী?",
            "সবচেয়ে অপ্রয়োজনীয় বস্তু",
        )
        self.assertIsNone(label)


if __name__ == "__main__":
    unittest.main()
