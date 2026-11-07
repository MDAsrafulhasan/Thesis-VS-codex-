import unittest
from types import SimpleNamespace

from selective_llm_judge import binary_token_ids, consensus_label, make_prompt


class SelectiveJudgeTests(unittest.TestCase):
    def test_prompt_contains_claim_context_and_evidence(self):
        row = SimpleNamespace(prompt_bn="প্রশ্ন", response_bn="উত্তর", context="প্রসঙ্গ")
        prompt = make_prompt(row, [{"context": "প্রমাণ"}])
        self.assertTrue(all(value in prompt for value in ("প্রশ্ন", "উত্তর", "প্রসঙ্গ", "প্রমাণ")))

    def test_no_context_consensus_requires_three_positive_views(self):
        positive = {"noctx_fact": 1, "exam_strict": 1, "zero_bias": 1}
        self.assertEqual(consensus_label(positive, has_context=False), 1)
        positive["zero_bias"] = 0
        self.assertEqual(consensus_label(positive, has_context=False), 0)

    def test_context_uses_context_view_only(self):
        self.assertEqual(consensus_label({"context_nli": 1}, has_context=True), 1)

    def test_binary_labels_require_distinct_single_tokens(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return {"0": [10], "1": [11], " 0": [20], " 1": [21]}[text]

        self.assertEqual(binary_token_ids(Tokenizer()), {0: 10, 1: 11})

    def test_binary_labels_reject_multitoken_only_encoding(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return [1, 2]

        with self.assertRaises(ValueError):
            binary_token_ids(Tokenizer())


if __name__ == "__main__":
    unittest.main()
