import unittest

from audit_context_rule_safety import compare_strict_date, strict_parse_date
from birth_date_overrides import DateValue


class ContextRuleSafetyAuditTests(unittest.TestCase):
    def test_la_suffix_preserves_four_digit_year(self):
        self.assertEqual(
            strict_parse_date("১লা ডিসেম্বর ১৯৮৫"),
            DateValue(year=1985, month=12, day=1),
        )

    def test_month_year_is_not_split_into_day_and_short_year(self):
        self.assertEqual(
            strict_parse_date("ডিসেম্বর ১৯৮৫"),
            DateValue(year=1985),
        )

    def test_strict_comparison_preserves_correct_label(self):
        expected = strict_parse_date("জন্ম: ১লা ডিসেম্বর ১৯৮৫")
        self.assertIsNotNone(expected)
        self.assertEqual(
            compare_strict_date("তিনি কবে জন্মগ্রহণ করেন?", "১লা ডিসেম্বর ১৯৮৫", expected),
            1,
        )


if __name__ == "__main__":
    unittest.main()
