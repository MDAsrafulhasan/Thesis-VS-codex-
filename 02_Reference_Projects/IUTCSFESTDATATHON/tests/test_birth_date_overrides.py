import unittest

from birth_date_overrides import DateValue, compare_date, extract_birth_date, parse_date


class BirthDateTests(unittest.TestCase):
    def test_day_month_year(self):
        self.assertEqual(parse_date("৭ই এপ্রিল, ১৯২০"), DateValue(1920, 4, 7))

    def test_month_day_year(self):
        self.assertEqual(parse_date("ফেব্রুয়ারি ১, ১৯৩১"), DateValue(1931, 2, 1))

    def test_year_first_date(self):
        self.assertEqual(parse_date("১৯০৫ খ্রিস্টাব্দের ২৯শে ডিসেম্বর"), DateValue(1905, 12, 29))

    def test_birth_marker_beats_death(self):
        evidence = extract_birth_date("করিম (জন্ম: ৭ এপ্রিল ১৯২০ - মৃত্যু: ১১ ডিসেম্বর ২০১২) একজন শিল্পী।")
        self.assertEqual(evidence.date, DateValue(1920, 4, 7))

    def test_lead_date_range(self):
        evidence = extract_birth_date("গিরিশচন্দ্র সেন (১৮৩৪ - ১৯১০) একজন অনুবাদক।")
        self.assertEqual(evidence.date, DateValue(1834))

    def test_birth_name_then_date(self):
        evidence = extract_birth_date("নিবেদিতা (জন্ম মার্গারেট নোবেল; ২৮ অক্টোবর ১৮৬৭ – ১৩ অক্টোবর ১৯১১) ছিলেন লেখিকা।")
        self.assertEqual(evidence.date, DateValue(1867, 10, 28))

    def test_birth_verb_is_not_birth_field(self):
        evidence = extract_birth_date("আব্দুল জব্বার ১০ ফেব্রুয়ারি ১৯৩৮ সালে জন্মগ্রহণ করেন। ১৯৫৬ সালে তিনি উত্তীর্ণ হন।")
        self.assertEqual(evidence.date, DateValue(1938, 2, 10))

    def test_earliest_birth_field_date_wins(self):
        evidence = extract_birth_date("কৃষ্ণান (জন্ম: ১৯৪৪) ৯ জুন ১৯৯৭ থেকে মন্ত্রী ছিলেন।")
        self.assertEqual(evidence.date, DateValue(1944))

    def test_lead_range_beats_place_birth_verb(self):
        evidence = extract_birth_date("ক্লডিয়া পটার (১৮৮১ - ১৯৭০) টেক্সাসের কাছে জন্মগ্রহণ করেন।")
        self.assertEqual(evidence.date, DateValue(1881))

    def test_wrong_month_rejected(self):
        self.assertEqual(compare_date("তিনি কবে জন্মগ্রহণ করেন?", "৮ এপ্রিল ১৯২০", DateValue(1920, 4, 7)), 0)

    def test_year_question_accepts_year(self):
        self.assertEqual(compare_date("তিনি কত সালে জন্মগ্রহণ করেন?", "১৯২০", DateValue(1920, 4, 7)), 1)


if __name__ == "__main__":
    unittest.main()
