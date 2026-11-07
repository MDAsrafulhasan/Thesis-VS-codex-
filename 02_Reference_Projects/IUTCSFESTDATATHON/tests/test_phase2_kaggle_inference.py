from __future__ import annotations

import json
import unittest
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle_notebooks" / "phase2_v85_inference"


class Phase2KaggleInferenceTest(unittest.TestCase):
    def test_kernel_is_offline_and_has_required_inputs(self) -> None:
        metadata = json.loads((KERNEL / "kernel-metadata.json").read_text(encoding="utf-8"))
        self.assertFalse(metadata["enable_internet"])
        self.assertTrue(metadata["enable_gpu"])
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
        self.assertIn("bengali-hallucination", metadata["competition_sources"])
        self.assertEqual(len(metadata["model_sources"]), 1)
        self.assertEqual(
            set(metadata["dataset_sources"]),
            {
                "seyamalam/bengali-hallucination-phase2-runtime-v85",
                "seyamalam/bengali-hallucination-llama-cpp-cuda-b9637",
            },
        )
        self.assertNotIn("kernel_sources", metadata)
        self.assertEqual(metadata["code_file"], "phase2_v85_inference.ipynb")
        self.assertEqual(metadata["kernel_type"], "notebook")

    def test_raw_inference_does_not_reference_saved_phase1_predictions(self) -> None:
        source = (KERNEL / "phase2_v85_inference.py").read_text(encoding="utf-8")
        self.assertNotIn("submission_latest_lineage_consensus", source)
        self.assertNotIn("submission_v85", source)
        self.assertIn('"--input", str(test_path)', source)
        self.assertIn('WORK / "submission.csv"', source)
        self.assertIn(
            'TEST_PATH = Path("/kaggle/input/competitions/bengali-hallucination/test set.csv")',
            source,
        )
        self.assertIn("pd.read_csv(TEST_PATH)", source)
        self.assertNotIn("rglob", source)
        self.assertNotIn("test_candidates", source)

    def test_notebook_contains_the_same_compliant_inference_source(self) -> None:
        notebook = nbformat.read(KERNEL / "phase2_v85_inference.ipynb", as_version=4)
        self.assertEqual(notebook.metadata.kernelspec.name, "python3")
        source = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        self.assertIn(
            'TEST_PATH = Path("/kaggle/input/competitions/bengali-hallucination/test set.csv")',
            source,
        )
        self.assertIn("pd.read_csv(TEST_PATH)", source)
        self.assertNotIn("rglob", source)
        self.assertNotIn("submission_latest_lineage_consensus", source)
        self.assertNotIn("submission_v85", source)


if __name__ == "__main__":
    unittest.main()
