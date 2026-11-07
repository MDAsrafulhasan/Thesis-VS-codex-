from __future__ import annotations

import unittest

from bn_num2words import num2words


class BengaliNum2WordsTest(unittest.TestCase):
    def test_router_range_matches_reference_examples(self) -> None:
        expected = {
            2: "দুই",
            88: "আটাশি",
            101: "একশত এক",
            1987: "এক হাজার নয়শত সাতাশি",
            100_000: "এক লাখ",
            10_000_000: "এক কোটি",
        }
        self.assertEqual({number: num2words(number) for number in expected}, expected)


if __name__ == "__main__":
    unittest.main()
