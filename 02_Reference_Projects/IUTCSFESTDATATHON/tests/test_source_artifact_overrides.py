import unittest

from source_artifact_overrides import compatible, dedupe_prefix


class SourceArtifactTests(unittest.TestCase):
    def test_collapsed_prefix_with_suffix(self):
        self.assertEqual(dedupe_prefix("নব্যপ্রস্তরনব্যপ্রস্তর যুগে"), "নব্যপ্রস্তর যুগে")
        self.assertTrue(compatible("নব্যপ্রস্তর", "নব্যপ্রস্তরনব্যপ্রস্তর যুগে"))

    def test_person_full_name_expands_short_answer(self):
        self.assertTrue(compatible("খাদিজা বিনতে খুওয়াইলিদ", "খাদিজা"))

    def test_orthographic_variant(self):
        self.assertTrue(compatible("গোহাটি", "গৌহাটী"))

    def test_different_numbers_rejected(self):
        self.assertFalse(compatible("১৯৯৫ সালে", "১৯৯৩ সালে"))

    def test_inflected_copy_followed_by_clean_copy(self):
        self.assertTrue(compatible("মুক্তমনা", "মুক্তমনায়মুক্তমনা"))

    def test_full_name_followed_by_token_prefix_copy(self):
        self.assertTrue(
            compatible("শফিক উল্লা খান পাঠান", "শফিক উল্লা খান পাঠানশফিক উল্লা খান")
        )


if __name__ == "__main__":
    unittest.main()
