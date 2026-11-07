from __future__ import annotations

import json
import unittest
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle_notebooks" / "phase2_v91_inference"
EXPECTED_TEST_PATH = (
    'TEST_PATH = Path("/kaggle/input/competitions/bengali-hallucination/test set.csv")'
)


class Phase2V91KaggleInferenceTest(unittest.TestCase):
    def test_kernel_is_public_offline_t4x2_with_pinned_inputs(self) -> None:
        metadata = json.loads((KERNEL / "kernel-metadata.json").read_text(encoding="utf-8"))
        self.assertFalse(metadata["is_private"])
        self.assertFalse(metadata["enable_internet"])
        self.assertTrue(metadata["enable_gpu"])
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
        self.assertEqual(
            set(metadata["dataset_sources"]),
            {
                "seyamalam/bengali-hallucination-phase2-runtime-v91",
                "seyamalam/bengali-hallucination-llama-cpp-cuda-b9637-public",
            },
        )
        self.assertEqual(
            metadata["model_sources"],
            ["snowrab/qwen3-6-27b-q4-k-p/pyTorch/default/1"],
        )
        self.assertEqual(metadata["competition_sources"], ["bengali-hallucination"])

    def test_raw_input_and_v91_typed_assets_are_explicit(self) -> None:
        source = (KERNEL / "phase2_v91_inference.py").read_text(encoding="utf-8")
        self.assertIn(EXPECTED_TEST_PATH, source)
        self.assertIn('"--typed-evidence"', source)
        self.assertIn('"--idiom-dataset"', source)
        self.assertIn("bengali-hallucination-phase2-runtime-v91", source)
        self.assertNotIn("submission_v91", source)
        self.assertNotIn("submission_v85", source)
        self.assertNotIn("rglob", source)

    def test_notebook_matches_compliant_source_requirements(self) -> None:
        notebook = nbformat.read(KERNEL / "phase2_v91_inference.ipynb", as_version=4)
        source = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        self.assertIn(EXPECTED_TEST_PATH, source)
        self.assertIn('"--typed-evidence"', source)
        self.assertIn('"--idiom-dataset"', source)
        self.assertNotIn("submission_v91", source)
        self.assertNotIn("rglob", source)
        self.assertEqual(
            notebook.metadata["huntrix_release"]["reference_output_sha256"],
            "0fec5d610f4e7b94ab6c46e2094d58e940b0f36781e9fa98d5466d2135474c28",
        )


if __name__ == "__main__":
    unittest.main()
