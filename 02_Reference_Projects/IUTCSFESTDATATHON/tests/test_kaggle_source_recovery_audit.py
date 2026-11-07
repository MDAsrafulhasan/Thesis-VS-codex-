from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kaggle_notebooks" / "source_recovery_audit"


class KaggleSourceRecoveryAuditTests(unittest.TestCase):
    def test_kernel_uses_t4x2_research_inputs(self) -> None:
        metadata = json.loads(
            (KERNEL_DIR / "kernel-metadata.json").read_text(encoding="utf-8")
        )
        self.assertTrue(metadata["enable_gpu"])
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
        self.assertTrue(metadata["enable_internet"])
        self.assertIn("bengali-hallucination", metadata["competition_sources"])
        self.assertEqual(len(metadata["model_sources"]), 1)

    def test_script_is_diagnostic_only(self) -> None:
        source = (KERNEL_DIR / "source_recovery_audit.py").read_text(encoding="utf-8")
        self.assertIn(
            'TEST_PATH = Path("/kaggle/input/competitions/bengali-hallucination/test set.csv")',
            source,
        )
        self.assertIn("source_recovery_candidates.csv", source)
        self.assertNotIn("submission.csv", source)
        self.assertNotIn("competitions submit", source)

    def test_frozen_compliance_notebook_is_separate(self) -> None:
        metadata = json.loads(
            (KERNEL_DIR / "kernel-metadata.json").read_text(encoding="utf-8")
        )
        self.assertNotEqual(
            metadata["id"],
            "seyamalam/bengali-hallucination-v85-compliance-inference",
        )


if __name__ == "__main__":
    unittest.main()
