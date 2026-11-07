from __future__ import annotations

import unittest
from fractions import Fraction

from structured_overrides import answer_matches, solve_math, solve_sandhi


class MathSolverTest(unittest.TestCase):
    def assert_solution(self, prompt: str, response: str, expected_rule: str, expected_label: bool = True) -> None:
        solved = solve_math(prompt)
        self.assertIsNotNone(solved)
        self.assertEqual(solved.rule, expected_rule)
        self.assertEqual(answer_matches(response, solved), expected_label)

    def test_construction_worker_rate(self) -> None:
        self.assert_solution(
            "একটি নির্মাণ প্রকল্প ক একাই ৬ দিনে সমাপ্ত করতে পারে, আবার খ একাই তা ৫৬ দিনে সমাপ্ত করতে পারে। উভয়ে একত্রে কাজ শুরু করলে প্রকল্পটি কত দিনে সমাপ্ত হবে?",
            "৫.৪২ দিন",
            "construction_worker_rate",
        )

    def test_invalid_zero_duration_is_hallucinated(self) -> None:
        self.assert_solution(
            "একটি নির্মাণ প্রকল্প ক একাই ০ দিনে সমাপ্ত করতে পারে, আবার খ একাই তা ৫০ দিনে সমাপ্ত করতে পারে। উভয়ে একত্রে কাজ শুরু করলে প্রকল্পটি কত দিনে সমাপ্ত হবে?",
            "২৫ দিন",
            "construction_worker_rate_invalid",
            expected_label=False,
        )

    def test_two_worker_rate(self) -> None:
        self.assert_solution(
            "রহিম একা একটি কাজ ৪ দিনে সম্পন্ন করতে সক্ষম, করিম একা সেই একই কাজ ৪০ দিনে সম্পন্ন করতে সক্ষম। দুজনে যৌথভাবে কাজ করলে কাজটি শেষ হতে কত দিন সময় নেবে?",
            "৩.৬৩৬ দিন",
            "two_worker_rate",
        )

    def test_ratio(self) -> None:
        self.assert_solution(
            "১৬৮ টাকা তিন ব্যবসায়িক অংশীদারের মধ্যে ৬:৫:১ অনুপাতে বণ্টন করা হলো। দ্বিতীয় অংশীদার কত টাকা পাবেন?",
            "৭০ টাকা",
            "three_way_ratio",
        )

    def test_simple_interest(self) -> None:
        self.assert_solution(
            "৫০০০ টাকা মূলধনে বার্ষিক ৮% সরল সুদের হারে ৩ বছরে সুদ কত?",
            "১২০০ টাকা",
            "simple_interest",
        )

    def test_lcm(self) -> None:
        self.assert_solution(
            "তিনটি বাস স্টপেজ থেকে বাস যথাক্রমে প্রতি ১১ মিনিট, প্রতি ৪ মিনিট ও প্রতি ৮ মিনিট অন্তর ছাড়ে। আবার একই সময়ে কত মিনিট পর ছাড়বে?",
            "৮৮ মিনিট",
            "lcm_departure",
        )

    def test_weekday(self) -> None:
        self.assert_solution(
            "আজ শনিবার হলে ১০ দিন পর কী বার হবে?",
            "মঙ্গলবার",
            "weekday_offset",
        )

    def test_log_product_quotient(self) -> None:
        solved = solve_math("log ax=1, logay=2, logaz=3 হলে, loga(x3y2/z) এর মান কত?")
        self.assertEqual(solved.expected, Fraction(4))
        self.assertFalse(answer_matches("1", solved))

    def test_conjugate_radical_cube_sum(self) -> None:
        solved = solve_math("x = √4 + √3 হলে, x 3 + 1/x3 এর মান কত?")
        self.assertEqual(solved.expected, Fraction(52))
        self.assertFalse(answer_matches("5√3", solved))

    def test_polynomial_parameter_at_root(self) -> None:
        solved = solve_math("f(x) = x³ + kx² – 6x – 9; k এর মান কত হলে f(3) = 0 হবে?")
        self.assertEqual(solved.expected, Fraction(0))
        self.assertFalse(answer_matches("-1", solved))

    def test_binary_addition(self) -> None:
        solved = solve_math("(১০১১)২ + (০১০১)২=?")
        self.assertTrue(answer_matches("(১০০০০)২", solved))
        self.assertFalse(answer_matches("(১১০০০)২", solved))

    def test_exponential_substitution_roots(self) -> None:
        solved = solve_math("2^x+2^(1-x)=3 হলে, x = কত?")
        self.assertTrue(answer_matches("(0,1)", solved))
        self.assertFalse(answer_matches("2", solved))

    def test_quadratic_equal_roots(self) -> None:
        solved = solve_math("যদি x² + px + 6 = 0 এর মূল দুটি সমান হয় এবং p>0, তবে p এর মান কত?")
        self.assertTrue(answer_matches("√24", solved))
        self.assertFalse(answer_matches("24", solved))

    def test_quadratic_inequality_open_interval(self) -> None:
        solved = solve_math("2x²+5x+3 < 0 এর সমাধান কোনটি?")
        self.assertEqual(solved.rule, "quadratic_inequality")
        self.assertTrue(answer_matches("-(3/2) < x < -1", solved))
        self.assertTrue(answer_matches("(-3/2, -1)", solved))
        self.assertFalse(answer_matches("-(3/2) < x < 1", solved))

    def test_quadratic_inequality_closed_interval(self) -> None:
        solved = solve_math("x 2 – 7x + 12 ≤ 0 এর সমাধান সেট –")
        self.assertEqual(solved.rule, "quadratic_inequality")
        self.assertTrue(answer_matches("[3, 4]", solved))
        self.assertTrue(answer_matches("3 ≤ x ≤ 4", solved))
        self.assertFalse(answer_matches("(-∞, 3]", solved))

    def test_grouped_linear_equation_preserves_negative_solution(self) -> None:
        solved = solve_math("(2 + x) + 3 = 3(x + 2) হলে x এর মান কত?")
        self.assertEqual((solved.expected, solved.rule), (Fraction(-1, 2), "grouped_linear_equation"))
        self.assertTrue(answer_matches("-1/2", solved))
        self.assertFalse(answer_matches("1/2", solved))

    def test_inscribed_angle(self) -> None:
        solved = solve_math(
            "O কেন্দ্র বিশিষ্ট বৃত্তে x কোণের মান কত? (যেখানে এক্স একটি বৃত্তস্থ কোণ, আর একই জ্যায়ের উপরে স্থিত কেন্দ্রস্থ কোণ ৭০ ডিগ্রি।)"
        )
        self.assertEqual(solved.expected, Fraction(35))
        self.assertTrue(answer_matches("৩৫ ডিগ্রি", solved))

    def test_polynomial_factor_lcm(self) -> None:
        solved = solve_math("x³+x²y, x²y +xy² এর ল. সা.গু কোনটি?")
        self.assertEqual(solved.rule, "polynomial_factor_lcm")
        self.assertTrue(answer_matches("x²y (x +y)", solved))
        self.assertFalse(answer_matches("xy (x +y)", solved))


class SandhiSolverTest(unittest.TestCase):
    def test_vowel_sandhi(self) -> None:
        solved = solve_sandhi("বিদ্যা' ও 'আলয়' শব্দ দুটির সন্ধিতে কোন শব্দ গঠিত হয়?")
        self.assertEqual(solved.expected, "বিদ্যালয়")
        self.assertFalse(answer_matches("বিদ্যাআলয়", solved))

    def test_consonant_sandhi(self) -> None:
        solved = solve_sandhi("উৎ' ও 'ঘাটন' শব্দ দুটির সন্ধিতে কোন শব্দ গঠিত হয়?")
        self.assertEqual(solved.expected, "উদ্ঘাটন")
        self.assertTrue(answer_matches("উদ্ঘাটন", solved))

    def test_visarga_sandhi(self) -> None:
        solved = solve_sandhi("পুনঃ' ও 'মিলন' শব্দ দুটির সন্ধিতে কোন শব্দ গঠিত হয়?")
        self.assertEqual(solved.expected, "পুনর্মিলন")
        self.assertFalse(answer_matches("পুনঃমিলন", solved))


if __name__ == "__main__":
    unittest.main()
