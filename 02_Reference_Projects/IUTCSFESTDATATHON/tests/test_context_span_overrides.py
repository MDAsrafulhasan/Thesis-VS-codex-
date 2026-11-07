import unittest

from context_span_overrides import context_span_decision


class ContextSpanOverrideTests(unittest.TestCase):
    def test_confirms_answer_in_relation_anchored_clause(self) -> None:
        label, evidence = context_span_decision(
            "তুরস্কের রাজধানী কোথায়?",
            "আঙ্কারা",
            "তুরস্কের রাজধানী আঙ্কারা দেশটির বৃহত্তম শহর নয়।",
        )
        self.assertEqual(label, 1)
        self.assertEqual(evidence["rule"], "relation_guarded_exact_context_span")

    def test_semicolon_keeps_discovery_year_from_unesco_relation(self) -> None:
        label, _ = context_span_decision(
            "পাহাড়পুর কত সালে ইউনেস্কো বিশ্ব ঐতিহ্যের স্বীকৃতি লাভ করে?",
            "১৮৭৯ সাল",
            "১৮৭৯ সালে পাহাড়পুর আবিষ্কৃত হয়; ১৯৮৫ সালে ইউনেস্কো "
            "বিশ্ব ঐতিহ্যের স্বীকৃতি দেয়।",
        )
        self.assertIsNone(label)

    def test_numeric_question_rejects_copied_entity_answer(self) -> None:
        label, _ = context_span_decision(
            "তারেক মাসুদের সর্বশেষ চলচ্চিত্রটি কত সালে মুক্তি পায়?",
            "রানওয়ে",
            "তারেক মাসুদের সর্বশেষ চলচ্চিত্র রানওয়ে ২০১০ সালে মুক্তি পায়।",
        )
        self.assertIsNone(label)

    def test_numeric_question_remains_with_numeric_relation_parser(self) -> None:
        label, _ = context_span_decision(
            "বাঁকুড়া জেলা কত সালে প্রতিষ্ঠিত হয়?",
            "১৭৬৫",
            "১৭৬৫ সালে রাজ্যটি অধিকার করা হয় এবং ১৮৮১ সালে বাঁকুড়া "
            "জেলা প্রতিষ্ঠিত হয়।",
        )
        self.assertIsNone(label)

    def test_start_relation_does_not_accept_completion_clause(self) -> None:
        label, _ = context_span_decision(
            "ভবনটির নির্মাণ কাজ কত সালে শুরু হয়?",
            "১৯৮২",
            "ভবনটির নির্মাণ ১৯৬১ সালে শুরু হয়; নির্মাণ কাজ ১৯৮২ সালে সমাপ্ত হয়।",
        )
        self.assertIsNone(label)

    def test_single_subject_token_is_not_enough(self) -> None:
        label, _ = context_span_decision(
            "লালনের আখড়া কোথায় ছিল?",
            "চট্টগ্রাম",
            "চট্টগ্রামে লালনের অনেক শিষ্য ছিলেন।",
        )
        self.assertIsNone(label)


if __name__ == "__main__":
    unittest.main()
