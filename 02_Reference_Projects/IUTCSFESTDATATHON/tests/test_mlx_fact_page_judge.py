import unittest

from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_fact_page_judge import build_page_retriever, parse_output, split_chunks


class FactPageJudgeTests(unittest.TestCase):
    def test_parse_supported(self):
        raw = '{"verdict":"supported","expected_answer":"১৯১৭","confidence":0.96,"evidence_rank":2}'
        self.assertEqual(parse_output(raw), ("supported", 0.96, "1917", 2, None))

    def test_parse_bengali_digits(self):
        raw = '{"verdict":"contradicted","expected_answer":"১৯৭৫","confidence":০.৯১,"evidence_rank":১}'
        verdict, confidence, expected, rank, error = parse_output(raw)
        self.assertEqual((verdict, expected, rank, error), ("contradicted", "1975", 1, None))
        self.assertAlmostEqual(confidence, 0.91)

    def test_bad_json_abstains(self):
        verdict, confidence, _, _, error = parse_output("not json")
        self.assertEqual(verdict, "insufficient")
        self.assertEqual(confidence, 0.0)
        self.assertIsNotNone(error)

    def test_chunks_are_local(self):
        chunks = split_chunks({"query": "x", "title": "x", "extract": "প্রথম বাক্যটি যথেষ্ট লম্বা। দ্বিতীয় বাক্যটিও যথেষ্ট লম্বা। তৃতীয় বাক্যটিও যথেষ্ট লম্বা।"})
        self.assertEqual(len(chunks), 3)
        self.assertNotIn("তৃতীয়", chunks[0]["text"])

    def test_page_retriever_does_not_index_body_for_selection(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "pages.json"
            path.write_text(
                '[{"query":"পদ্মা সেতু","title":"পদ্মা সেতু","extract":"সুন্দরবন শব্দটি এখানে incidental"},'
                '{"query":"সুন্দরবন","title":"সুন্দরবন","extract":"প্রাসঙ্গিক লেখা"}]',
                encoding="utf-8",
            )
            pages, vectorizer, matrix = build_page_retriever(path, 1000)
            from mlx_fact_page_judge import retrieve

            result = retrieve("সুন্দরবন কত সালে", pages, vectorizer, matrix, 1)
            self.assertEqual(result[0]["title"], "সুন্দরবন")


if __name__ == "__main__":
    unittest.main()
