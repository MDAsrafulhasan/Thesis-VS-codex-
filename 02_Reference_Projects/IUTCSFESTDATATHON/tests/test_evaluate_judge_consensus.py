import unittest

import numpy as np

from evaluate_judge_consensus import agreement_report


class JudgeConsensusEvaluationTests(unittest.TestCase):
    def test_scores_only_cross_family_agreements(self) -> None:
        report = agreement_report(np.asarray([0, 1, 1]), {0: 0, 1: 0, 2: 1}, {0: 0, 1: 1, 2: 1})
        self.assertEqual(report["agreement_rows"], 2)
        self.assertEqual(report["metrics_on_agreement"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
