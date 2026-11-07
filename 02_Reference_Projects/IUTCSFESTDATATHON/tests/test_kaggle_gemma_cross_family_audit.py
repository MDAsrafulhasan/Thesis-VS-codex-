from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle_notebooks" / "gemma_cross_family_audit"


class KaggleGemmaCrossFamilyAuditTests(unittest.TestCase):
    def test_kernel_is_offline_t4x2_and_pins_gemma(self) -> None:
        metadata = json.loads(
            (KERNEL / "kernel-metadata.json").read_text(encoding="utf-8")
        )
        self.assertTrue(metadata["enable_gpu"])
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
        self.assertFalse(metadata["enable_internet"])
        self.assertEqual(len(metadata["model_sources"]), 1)
        self.assertIn("gemma-4-12b", metadata["model_sources"][0])

    def test_kernel_is_diagnostic_only(self) -> None:
        source = (KERNEL / "gemma_cross_family_audit.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("gemma_cross_family_trace.jsonl", source)
        self.assertIn("root ::= [01]", source)
        self.assertNotIn("submission.csv", source)
        self.assertNotIn("competitions submit", source)

    def test_frozen_compliance_kernel_is_not_modified_or_reused(self) -> None:
        source = (KERNEL / "gemma_cross_family_audit.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("phase2_v85_inference", source)
        self.assertNotIn("submission_latest_lineage_consensus", source)


if __name__ == "__main__":
    unittest.main()
