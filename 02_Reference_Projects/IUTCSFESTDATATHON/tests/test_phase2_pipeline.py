import unittest

from phase2_pipeline import context_decision, historical_alias_decision, verified_decision


class Phase2PipelineTests(unittest.TestCase):
    def test_verified_fact_requires_exact_answer(self):
        facts = {"বাংলাদেশের রাজধানী কী": {"answers": ["ঢাকা"], "sources": ["a", "b"]}}
        self.assertEqual(verified_decision("বাংলাদেশের রাজধানী কী", "ঢাকা", facts)[0], 1)
        self.assertEqual(verified_decision("বাংলাদেশের রাজধানী কী", "চট্টগ্রাম", facts)[0], 0)

    def test_context_birth_date_router(self):
        label, evidence = context_decision(
            "কাজী নজরুল ইসলাম কত সালে জন্মগ্রহণ করেন?", "১৮৯৯ সালে",
            "কাজী নজরুল ইসলাম (জন্ম: ২৪ মে ১৮৯৯) একজন কবি।", 0.4,
        )
        self.assertEqual(label, 1)
        self.assertEqual(evidence["rule"], "birth_date")

    def test_historical_capital_does_not_accept_modern_alias(self):
        context = "কুতারাজা ছিল এর রাজধানী যা বর্তমান বান্দা আচেহ।"
        self.assertEqual(
            historical_alias_decision("সালতানাতের রাজধানীর নাম কী ছিল?", "বান্দা আচেহ নগরী।", context)[0],
            0,
        )
        self.assertEqual(
            historical_alias_decision("সালতানাতের রাজধানীর নাম কী ছিল?", "কুতারাজা", context)[0],
            1,
        )


if __name__ == "__main__":
    unittest.main()
