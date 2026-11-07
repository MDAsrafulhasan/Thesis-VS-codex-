import unittest

from fuzzy_source_overrides import MatchFeatures, eligible, match_features, relation_compatible


class FuzzySourceGateTest(unittest.TestCase):
    def test_matching_numeric_template_requires_same_numbers(self) -> None:
        features = match_features(
            "একটি কাজ ৮ দিনে হলে মোট কত দিন লাগবে?",
            "একটি কাজ ১২ দিনে হলে মোট কত দিন লাগবে?",
            0.95,
            0.60,
        )
        self.assertFalse(features.same_numbers)
        self.assertFalse(eligible(features))

    def test_equivalent_spelling_variant_is_eligible(self) -> None:
        features = match_features(
            "বাংলাদেশের লোকশিল্প জাদুঘর কোথায় অবস্থিত?",
            "বাংলাদেশের লোকশিল্প জাদুঘর কোথায় অবস্থিত?",
            0.94,
            0.45,
        )
        self.assertTrue(features.same_numbers)
        self.assertTrue(eligible(features))

    def test_weak_semantic_overlap_abstains(self) -> None:
        features = MatchFeatures(
            cosine=0.91,
            margin=0.02,
            token_jaccard=0.40,
            token_coverage=0.60,
            sequence_ratio=0.85,
            same_numbers=True,
            same_relation=True,
        )
        self.assertFalse(eligible(features))

    def test_start_end_relation_swap_abstains(self) -> None:
        query = "শরিফ হুসাইন বিন আলী কর্তৃক সূচিত আরব বিদ্রোহ কবে শেষ হয়?"
        candidate = "শরিফ হুসাইন বিন আলী কর্তৃক সূচিত আরব বিদ্রোহ কবে শুরু হয়?"
        self.assertFalse(relation_compatible(query, candidate))
        features = match_features(query, candidate, 0.98, 0.70)
        self.assertTrue(features.same_numbers)
        self.assertFalse(features.same_relation)
        self.assertFalse(eligible(features))

    def test_surface_paraphrase_preserves_relation(self) -> None:
        query = "উপন্যাসটি কত সালে প্রথম প্রকাশিত হয়?"
        candidate = "উপন্যাসটি প্রথম কবে প্রকাশ পায়?"
        self.assertTrue(relation_compatible(query, candidate))
