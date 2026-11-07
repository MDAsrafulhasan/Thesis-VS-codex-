import unittest

from build_fact_page_html_corpus import ParagraphParser


class ParagraphParserTests(unittest.TestCase):
    def test_extracts_article_paragraphs(self):
        parser = ParagraphParser()
        parser.feed("<nav>skip</nav><p>প্রথম <b>অনুচ্ছেদ</b>।</p><p>দ্বিতীয়<br>লাইন।</p>")
        self.assertEqual(parser.paragraphs, ["প্রথম অনুচ্ছেদ।", "দ্বিতীয় লাইন।"])


if __name__ == "__main__":
    unittest.main()
