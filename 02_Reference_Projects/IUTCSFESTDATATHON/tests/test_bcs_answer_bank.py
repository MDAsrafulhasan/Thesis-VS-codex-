from __future__ import annotations

import unittest

from bcs_answer_bank import clean_option, clean_question


class BcsCleaningTest(unittest.TestCase):
    def test_question_number_prefix(self) -> None:
        self.assertEqual(clean_question("45. বাংলাদেশের অন্তর্গত সুন্দরবন"), "বাংলাদেশের অন্তর্গত সুন্দরবন")

    def test_latin_option_prefix(self) -> None:
        self.assertEqual(clean_option("B. নিউইয়র্কে"), "নিউইয়র্কে")

    def test_bengali_option_prefix(self) -> None:
        self.assertEqual(clean_option("গ) নিতান্ত অলস"), "নিতান্ত অলস")
