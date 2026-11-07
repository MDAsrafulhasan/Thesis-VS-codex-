from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle_notebooks" / "relation_swap_verifier_training"


class KaggleRelationSwapTrainingTests(unittest.TestCase):
    def test_training_runs_on_kaggle_t4x2(self) -> None:
        metadata = json.loads(
            (KERNEL / "kernel-metadata.json").read_text(encoding="utf-8")
        )
        self.assertTrue(metadata["enable_gpu"])
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
        self.assertTrue(metadata["enable_internet"])
        self.assertIn("bengali-hallucination", metadata["competition_sources"])

    def test_external_model_and_data_revisions_are_pinned(self) -> None:
        source = (KERNEL / "relation_swap_verifier_training.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'MODEL_REVISION = "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"',
            source,
        )
        self.assertIn(
            'DATASET_REVISION = "84e2d15ee0bb19f6e72aabef0dc72bff3609138b"',
            source,
        )
        self.assertIn("same_context_relation_swap", source)

    def test_arrow_answer_arrays_are_not_boolean_coerced(self) -> None:
        source = (KERNEL / "relation_swap_verifier_training.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("values is not None and len(values) > 0", source)
        self.assertNotIn('value.get("text") or []', source)

    def test_training_kernel_cannot_submit(self) -> None:
        source = (KERNEL / "relation_swap_verifier_training.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("submission.csv", source)
        self.assertNotIn("competitions submit", source)
        self.assertIn("relation_test_predictions.csv", source)

    def test_checkpoint_requires_both_precision_gates(self) -> None:
        source = (KERNEL / "relation_swap_verifier_training.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("threshold[\"precision\"] >= EXTERNAL_TARGET_PRECISION", source)
        self.assertIn('official["precision_0"] == 1.0', source)
        self.assertIn("if gate_passed:", source)


if __name__ == "__main__":
    unittest.main()
