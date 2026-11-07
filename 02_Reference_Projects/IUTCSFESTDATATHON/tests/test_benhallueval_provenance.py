import unittest

from benhallueval_provenance import decide


class BenHalluEvalProvenanceTests(unittest.TestCase):
    def test_exact_positive_and_negative(self):
        correct = {"প্রশ্ন": {"সঠিক"}}
        hallucinated = {"প্রশ্ন": {"ভুল"}}
        self.assertEqual(decide("প্রশ্ন", "সঠিক", correct, hallucinated)[0], 1)
        self.assertEqual(decide("প্রশ্ন", "ভুল", correct, hallucinated)[0], 0)

    def test_preserves_devanagari_distinction(self):
        correct = {"मनोविज्ञान": {"सिगमुंड फ्रयेड"}}
        hallucinated = {"मनोविज्ञान": {"कार्ल गुस्ताव युंग"}}
        self.assertEqual(decide("मनोविज्ञान", "कार्ल गुस्ताव युंग", correct, hallucinated)[0], 0)


if __name__ == "__main__":
    unittest.main()
