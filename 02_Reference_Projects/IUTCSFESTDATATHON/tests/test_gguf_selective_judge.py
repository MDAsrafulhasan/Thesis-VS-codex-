from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gguf_selective_judge import judge_row, make_prompt, parse_binary, stable_seed


class GgufSelectiveJudgeTest(unittest.TestCase):
    def test_stable_seed_is_repeatable_and_view_specific(self) -> None:
        self.assertEqual(stable_seed(42, "exam_strict"), stable_seed(42, "exam_strict"))
        self.assertNotEqual(stable_seed(42, "exam_strict"), stable_seed(42, "zero_bias"))

    def test_parse_binary(self) -> None:
        self.assertEqual(parse_binary("0"), 0)
        self.assertEqual(parse_binary(" 1\n"), 1)
        with self.assertRaises(ValueError):
            parse_binary("unknown")

    def test_prompt_contains_row(self) -> None:
        row = SimpleNamespace(context="তথ্য", prompt_bn="প্রশ্ন", response_bn="উত্তর")
        prompt = make_prompt(row, "context_nli")
        self.assertIn("প্রসঙ্গ: তথ্য", prompt)
        self.assertIn("প্রশ্ন: প্রশ্ন", prompt)
        self.assertTrue(prompt.endswith("লেবেল:"))

    @patch("gguf_selective_judge.call_server")
    def test_unanimous_policy_stops_after_first_zero(self, call_server) -> None:
        call_server.side_effect = [(1, "1"), (0, "0")]
        row = SimpleNamespace(id=7, context="[NULL]", prompt_bn="প্রশ্ন", response_bn="উত্তর")
        result = judge_row(row, "http://server", 10, 0)
        self.assertEqual(result["new"], 0)
        self.assertEqual(list(result["views"]), ["exam_strict", "zero_bias"])
        self.assertEqual(call_server.call_count, 2)

    @patch("gguf_selective_judge.call_server")
    def test_unanimous_policy_runs_all_views_for_positive(self, call_server) -> None:
        call_server.side_effect = [(1, "1"), (1, "1"), (1, "1")]
        row = SimpleNamespace(id=8, context="", prompt_bn="প্রশ্ন", response_bn="উত্তর")
        result = judge_row(row, "http://server", 10, 0)
        self.assertEqual(result["new"], 1)
        self.assertEqual(
            list(result["views"]),
            ["exam_strict", "zero_bias", "noctx_fact"],
        )
        self.assertEqual(call_server.call_count, 3)


if __name__ == "__main__":
    unittest.main()
