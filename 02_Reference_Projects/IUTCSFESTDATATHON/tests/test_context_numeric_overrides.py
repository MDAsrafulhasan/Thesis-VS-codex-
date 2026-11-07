import unittest

from context_numeric_overrides import decide, numeric_equivalent, word_numeric_components


class ContextNumericTests(unittest.TestCase):
    def test_reads_bengali_number_words(self):
        self.assertEqual(word_numeric_components("অষ্টাশি থেকে একশত চারটি"), [104.0, 88.0])
        self.assertEqual(word_numeric_components("সাত হাজারেরও বেশি মানুষ"), [7000.0])

    def test_rejects_conflicting_numeric_range(self):
        label, evidence = decide(
            "কীবোর্ডে সাধারণত মোট কতগুলি বোতাম থাকে?",
            "কীবোর্ডে সাধারণত অষ্টাশি থেকে একশত চারটি বোতাম থাকে।",
            "কী-বোর্ডে ৮৪ থেকে ১০১টি বা কোন কোন কী-বোর্ডে ১০২টি কী আছে।",
            0.4,
        )
        self.assertEqual(label, 0)
        self.assertEqual(evidence["rule"], "question_conditioned_numeric_range_contradiction")

    def test_when_question_ignores_casualty_numbers(self):
        label, _ = decide(
            "প্রথম ইঙ্গ-আফগান যুদ্ধ কবে হয়েছিল?", "১৮৩৯ থেকে ১৮৪২ সালের মধ্যে",
            "প্রথম যুদ্ধ ১৮৩৯ থেকে ১৮৪২ সালের মধ্যে হয়। এতে ৪৫০০ সৈনিক এবং ১২০০০ অনুসারী নিহত হয়।",
            0.2,
        )
        self.assertNotEqual(label, 0)

    def test_bengali_words_match_digits(self):
        self.assertTrue(numeric_equivalent("দুই লক্ষ তেরো হাজার দুই শত জন", "আলেপ্পোর জনসংখ্যা ২১৩২০০।"))

    def test_colloquial_year_matches(self):
        self.assertTrue(numeric_equivalent("আঠারশ' বত্রিশ সালে", "প্রথম ডায়নামো নির্মিত হয় ১৮৩২ সালে।"))
        self.assertEqual(word_numeric_components("আঠারশ' বত্রিশ সালে"), [1832.0])

    def test_indian_and_western_scales_match(self):
        self.assertTrue(numeric_equivalent("প্রায় ১ কোটি", "প্রায় ১০ মিলিয়ন মানুষ মারা যায়।"))

    def test_date_component_mismatch_rejected(self):
        self.assertFalse(numeric_equivalent("২৭ অক্টোবর, ১৮৬৭", "জন্ম ২৮ অক্টোবর, ১৮৬৭।"))

    def test_relation_mismatch_abstains(self):
        pred, _ = decide(
            "বাংলাদেশে বর্তমানে দৈনিক গ্যাসের চাহিদা কত?",
            "৭ দশমিক ৬৩ ট্রিলিয়ন ঘনফুট",
            "দেশে বর্তমানে উত্তোলনযোগ্য গ্যাসের মজুত ৭ দশমিক ৬৩ ট্রিলিয়ন ঘনফুট।",
            0.4,
        )
        self.assertIsNone(pred)

    def test_foundation_stone_is_not_establishment(self):
        pred, _ = decide(
            "রাজশাহী ক্যাডেট কলেজ কবে প্রতিষ্ঠিত হয়?",
            "১৯৬৪ সালের ৬ই নভেম্বর",
            "১৯৬৪ সালের ৬ই নভেম্বর কলেজটির ভিত্তিপ্রস্তর স্থাপন করেন।",
            0.4,
        )
        self.assertIsNone(pred)

    def test_same_surface_date_is_not_an_override(self):
        pred, _ = decide(
            "ব্যক্তিটি কবে জন্মগ্রহণ করেন?",
            "১২ জুন ১৯৭৭",
            "ব্যক্তিটি জন্মগ্রহণ করেন ১২ জুলাই ১৯৭৭।",
            0.4,
        )
        self.assertIsNone(pred)

    def test_multiple_numeric_fields_abstain(self):
        pred, _ = decide(
            "সর্বোচ্চ কত বছর কারাদণ্ড হতে পারে?",
            "দুই বছর",
            "অনধিক ৩ বছর, তবে ২ বছরের নিম্নে নয়।",
            0.4,
        )
        self.assertIsNone(pred)

    def test_single_numeric_fact_with_approximation_is_supported(self):
        pred, _ = decide(
            "কান্তনগর মন্দিরে মোট কতটি টেরাকোটা টালি রয়েছে বলে জানা যায়?",
            "প্রায় ১৫,০০০টি",
            "পুরো কান্তনগর মন্দিরে প্রায় ১৫,০০০-এর মতো টেরাকোটা টালি রয়েছে।",
            0.4,
        )
        self.assertEqual(pred, 1)

    def test_semicolon_separates_independent_year_claims(self):
        pred, evidence = decide(
            "আলেকজান্ডার কানিংহাম কত সালে পাহাড়পুর বৌদ্ধ বিহার আবিষ্কার করেন?",
            "১৮৭৯ সাল",
            "১৮৭৯ খ্রিস্টাব্দে স্যার আলেকজান্ডার কানিংহাম পাহাড়পুরের এই বিশাল স্থাপনা আবিষ্কার করেন; পরবর্তীতে ১৯৮৫ খ্রিস্টাব্দে ইউনেস্কো স্বীকৃতি দেয়।",
            0.4,
        )
        self.assertEqual(pred, 1)
        self.assertNotIn("১৯৮৫", evidence["evidence_sentence"])

    def test_exact_numeric_range_is_supported(self):
        pred, evidence = decide(
            "বর্ষাকালে জলাবন কত ফুট পানির নিচে থাকে?",
            "২০-৩০ ফুট",
            "বর্ষাকালে জলাবন ২০-৩০ ফুট পানির নিচে থাকে।",
            0.4,
        )
        self.assertEqual(pred, 1)
        self.assertEqual(evidence["rule"], "relation_bound_exact_numeric_signature")

    def test_single_relation_bound_number_mismatch_is_rejected(self):
        pred, evidence = decide(
            "জাফলং শহর থেকে কত কিলোমিটার দূরে অবস্থিত?",
            "৩০ কিলোমিটার",
            "জাফলং শহর থেকে ৬২ কিলোমিটার দূরে অবস্থিত।",
            0.4,
        )
        self.assertEqual(pred, 0)
        self.assertEqual(evidence["rule"], "relation_bound_single_numeric_contradiction")

    def test_kobe_does_not_bypass_establishment_relation(self):
        pred, _ = decide(
            "কলেজটি কবে প্রতিষ্ঠিত হয়?",
            "১৯৬৪",
            "১৯৬৪ সালে কলেজটির ভিত্তিপ্রস্তর স্থাপন করা হয়।",
            0.4,
        )
        self.assertIsNone(pred)

    def test_superlative_claim_does_not_accept_one_matching_member(self):
        pred, _ = decide(
            "পৃথিবীর দীর্ঘতম হিমবাহটির দৈর্ঘ্য কত?",
            "৫৮ কিলোমিটার",
            "এটি পৃথিবীর দীর্ঘতম হিমবাহগুলোর একটি। এর দৈর্ঘ্য ৫৮ কিলোমিটার।",
            0.4,
        )
        self.assertIsNone(pred)

    def test_conjunction_splits_earlier_and_current_quantities(self):
        pred, evidence = decide(
            "দ্বীপে বর্তমানে কতজন মানুষ বাস করে?",
            "সাত হাজারেরও বেশি",
            "দ্বীপে ১০০ থেকে ১২৫ বছর আগে বসতি শুরু হয় এবং বর্তমানে "
            "সেখানে সাত হাজারেরও বেশি মানুষ বাস করে।",
            0.4,
        )
        self.assertEqual(pred, 1)
        self.assertNotIn("১২৫", evidence["evidence_sentence"])


if __name__ == "__main__":
    unittest.main()
