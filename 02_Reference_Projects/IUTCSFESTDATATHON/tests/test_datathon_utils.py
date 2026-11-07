import unittest

from datathon_utils import normalize_bn


class DatathonUtilsTests(unittest.TestCase):
    def test_normalizes_bengali_digits(self):
        self.assertEqual(normalize_bn("১৯২২"), "1922")

    def test_normalizes_arabic_digit_variants(self):
        self.assertEqual(normalize_bn("٢٠٢٦ ۲۰۲۶"), "2026 2026")


if __name__ == "__main__":
    unittest.main()
