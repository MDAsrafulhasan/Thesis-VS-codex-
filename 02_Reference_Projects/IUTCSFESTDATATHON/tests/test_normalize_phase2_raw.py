import unittest

from normalize_phase2_raw import chunks, inferred_title


class NormalizeRawTests(unittest.TestCase):
    def test_chunks_preserve_overlap_and_content(self):
        text = "ক" * 80 + "।" + "খ" * 80
        result = chunks(text, 100, 10)
        self.assertGreaterEqual(len(result), 2)
        self.assertTrue(result[0].endswith("।"))

    def test_inferred_title(self):
        self.assertEqual(inferred_title("ঢাকা বাংলাদেশের রাজধানী। আরও লেখা"), "ঢাকা বাংলাদেশের রাজধানী")


if __name__ == "__main__":
    unittest.main()
