import unittest

import pandas as pd

from audit_benhallueval_fiveway import audit


class BenHalluEvalFiveWayAuditTests(unittest.TestCase):
    def test_exact_gold_and_hallucination_matches(self):
        test = pd.DataFrame(
            [
                {"id": 1, "context": "প্রসঙ্গ", "prompt_bn": "প্রশ্ন?", "response_bn": "সঠিক"},
                {"id": 2, "context": "প্রসঙ্গ", "prompt_bn": "প্রশ্ন?", "response_bn": "ভুল"},
            ]
        )
        submission = pd.DataFrame([{"id": 1, "label": 1}, {"id": 2, "label": 1}])
        gold = pd.DataFrame(
            [{"id": "source", "context": "প্রসঙ্গ", "question": "প্রশ্ন?", "correct_answer": "সঠিক"}]
        )
        hallucinations = pd.DataFrame(
            [{"source_id": "source", "hallucinated_answer": "ভুল"}]
        )

        result = audit(test, submission, gold, hallucinations)

        self.assertEqual(result["evidence_label"].tolist(), [1, 0])
        self.assertEqual(result["agrees"].tolist(), [True, False])

    def test_does_not_cross_questions_with_shared_context(self):
        test = pd.DataFrame(
            [{"id": 1, "context": "একই প্রসঙ্গ", "prompt_bn": "মায়ের নাম কী?", "response_bn": "মা"}]
        )
        submission = pd.DataFrame([{"id": 1, "label": 1}])
        gold = pd.DataFrame(
            [{"id": "source", "context": "একই প্রসঙ্গ", "question": "বাবার নাম কী?", "correct_answer": "বাবা"}]
        )
        hallucinations = pd.DataFrame(
            [{"source_id": "source", "hallucinated_answer": "মা"}]
        )

        result = audit(test, submission, gold, hallucinations)

        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
