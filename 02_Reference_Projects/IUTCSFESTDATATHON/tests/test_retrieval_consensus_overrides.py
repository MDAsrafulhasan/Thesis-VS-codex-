import unittest

from retrieval_consensus_overrides import decide


class RetrievalConsensusTests(unittest.TestCase):
    def evidence(self, answer, source, relation_match=None):
        item = {"answer": answer, "source_repo": source, "score": 0.9}
        if relation_match is not None:
            item["relation_match"] = relation_match
        return item

    def test_two_sources_support(self):
        evidence = [self.evidence("ঢাকা", "a"), self.evidence("ঢাকা", "b")]
        self.assertEqual(decide("ঢাকা", evidence, 0.65, 2)[0], 1)

    def test_two_sources_numeric_contradiction(self):
        evidence = [self.evidence("১৯৭১", "a"), self.evidence("১৯৭১ সালে", "b")]
        self.assertEqual(decide("১৯৭২", evidence, 0.65, 2)[0], 0)

    def test_same_source_does_not_count_twice(self):
        evidence = [self.evidence("ঢাকা", "a"), self.evidence("ঢাকা", "a")]
        self.assertIsNone(decide("ঢাকা", evidence, 0.65, 2)[0])

    def test_explicit_relation_mismatches_do_not_vote(self):
        evidence = [
            self.evidence("১৯২০", "a", False),
            self.evidence("১৯২০", "b", False),
            self.evidence("১৯৭৫", "c", True),
        ]
        self.assertIsNone(decide("১৯২০", evidence, 0.65, 2)[0])


if __name__ == "__main__":
    unittest.main()
