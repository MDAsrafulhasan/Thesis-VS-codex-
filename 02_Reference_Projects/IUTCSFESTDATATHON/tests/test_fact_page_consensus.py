import unittest

from fact_page_consensus import consensus_prediction


class FactPageConsensusTests(unittest.TestCase):
    def test_two_agree(self):
        pred, _ = consensus_prediction(
            [{"pred": 0, "confidence": 1.0}, {"pred": 0, "confidence": 0.9}, {"pred": None, "confidence": 0.0}],
            0.85,
            2,
        )
        self.assertEqual(pred, 0)

    def test_conflict_abstains(self):
        pred, _ = consensus_prediction(
            [{"pred": 0, "confidence": 1.0}, {"pred": 1, "confidence": 1.0}, {"pred": 0, "confidence": 0.95}],
            0.85,
            2,
        )
        self.assertIsNone(pred)

    def test_single_vote_abstains(self):
        pred, _ = consensus_prediction([{"pred": 1, "confidence": 1.0}], 0.85, 2)
        self.assertIsNone(pred)


if __name__ == "__main__":
    unittest.main()
