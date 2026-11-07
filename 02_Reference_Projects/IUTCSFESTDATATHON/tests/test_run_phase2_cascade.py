from pathlib import Path
from types import SimpleNamespace
import unittest

from run_phase2_cascade import judge_command, router_command


class Phase2CascadeCommandTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            input=Path("input.csv"), output=Path("final.csv"),
            verified_facts=Path("facts.json"), base_submission=Path("base.csv"),
            typed_evidence=[Path("typed.json")],
            idiom_dataset=Path("idioms.json"),
            law_corpus=[Path("laws.jsonl"), Path("legacy-laws.json")],
            verifier=Path("verifier.joblib"), bank=[Path("a.jsonl"), Path("b.jsonl")],
            fuzzy_bank=[Path("fuzzy.jsonl")],
            lexical_bank=Path("lexical.jsonl"),
            benhallu_gold=Path("gold.csv"), benhallu_hallucinations=Path("hall.csv"),
            judge_model="model", evidence=Path("evidence.jsonl"), batch_size=4,
        )

    def test_router_command_contains_all_artifacts(self):
        command = router_command(self.args, Path("router.csv"), Path("router.json"))
        for value in ("base.csv", "verifier.joblib", "a.jsonl", "b.jsonl", "fuzzy.jsonl", "lexical.jsonl", "gold.csv", "hall.csv", "facts.json", "typed.json", "idioms.json", "laws.jsonl", "legacy-laws.json"):
            self.assertIn(value, command)

    def test_judge_command_contains_handoff_paths(self):
        command = judge_command(self.args, Path("router.csv"), Path("router.json"), Path("judge.json"))
        for value in ("router.csv", "router.json", "final.csv", "judge.json", "evidence.jsonl"):
            self.assertIn(value, command)


if __name__ == "__main__":
    unittest.main()
