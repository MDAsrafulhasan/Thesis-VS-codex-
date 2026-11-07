import unittest

from build_phase2_training import corrupt_number


class Phase2TrainingTests(unittest.TestCase):
    def test_corrupts_bengali_year_after_normalization(self):
        self.assertEqual(corrupt_number("১৯৭১ সালে"), "1978 সালে")

    def test_abstains_without_number(self):
        self.assertIsNone(corrupt_number("ঢাকা"))


if __name__ == "__main__":
    unittest.main()
