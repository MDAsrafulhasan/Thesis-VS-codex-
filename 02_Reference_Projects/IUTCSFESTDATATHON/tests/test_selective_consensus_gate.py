import json
import tempfile
import unittest
from pathlib import Path

from selective_consensus_gate import consensus_decisions, load_votes


def vote(label: int, trace: str) -> dict:
    return {"label": label, "trace": trace, "record": {}}


class SelectiveConsensusGateTests(unittest.TestCase):
    def test_loads_jsonl_judge_traces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "judge.jsonl"
            path.write_text(json.dumps({"id": 7, "pred": 1}) + "\n", encoding="utf-8")
            self.assertEqual(load_votes(path)[7]["label"], 1)

    def test_requires_two_judges_and_corroborating_evidence(self) -> None:
        a = {1: vote(0, "a"), 2: vote(1, "a"), 3: vote(0, "a")}
        b = {1: vote(0, "b"), 2: vote(0, "b"), 3: vote(0, "b")}
        evidence = {1: vote(0, "e"), 3: vote(1, "e")}
        self.assertEqual([item["id"] for item in consensus_decisions(a, b, evidence)], [1])

    def test_uncorroborated_mode_still_requires_cross_family_agreement(self) -> None:
        a = {1: vote(1, "a"), 2: vote(0, "a")}
        b = {1: vote(1, "b"), 2: vote(1, "b")}
        self.assertEqual([item["id"] for item in consensus_decisions(a, b, {}, require_evidence=False)], [1])


if __name__ == "__main__":
    unittest.main()
