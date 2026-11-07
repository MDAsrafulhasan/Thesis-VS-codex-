#!/usr/bin/env python3
"""Standalone verification checks for the thesis evaluation pipeline."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval import compute_bhs, compute_lins_ccc, parse_judge_output
from make_splits import create_dialect_split


ROOT = Path(__file__).resolve().parent


class PipelineVerification(unittest.TestCase):
    def test_mock_generation_and_stratified_split_sizes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="benhallu_pipeline_") as tmp:
            output_dir = Path(tmp)
            command = [
                sys.executable,
                str(ROOT / "make_splits.py"),
                "--generate_mock_data",
                "--output_dir",
                str(output_dir),
                "--mode",
                "benhallu",
                "--seed",
                "42",
            ]
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)

            mock_primary = pd.read_csv(output_dir / "mock_data" / "benhallu_mock_16k.csv")
            self.assertEqual(len(mock_primary), 16_000)
            self.assertEqual(
                list(mock_primary.columns),
                ["id", "task", "track", "hallucination_type", "question", "context", "candidate_answer", "ground_truth_label"],
            )
            mock_dialect = pd.read_csv(output_dir / "mock_data" / "dialectal_bias_mock_4k.csv")
            self.assertEqual(len(mock_dialect), 4_000)
            self.assertEqual(
                list(mock_dialect.columns),
                ["id", "dialect", "question_type", "question", "response", "human_or_primary_score"],
            )
            self.assertTrue(mock_dialect.human_or_primary_score.between(0, 10).all())
            expected = {
                "benhallu_micro_debug_120.csv": (120, 30, 90),
                "benhallu_stratified_10pct.csv": (1_600, 400, 1_200),
                "benhallu_full_16k.csv": (16_000, 4_000, 12_000),
            }
            for filename, (total, track_a, track_b) in expected.items():
                split = pd.read_csv(output_dir / filename)
                self.assertEqual(len(split), total)
                self.assertEqual(int((split.track == "Track_A").sum()), track_a)
                self.assertEqual(int((split.track == "Track_B").sum()), track_b)
                self.assertEqual(split.id.nunique(), total)

            eval_dir = output_dir / "eval_smoke"
            eval_command = [
                sys.executable,
                str(ROOT / "eval.py"),
                "--input_csv",
                str(output_dir / "benhallu_micro_debug_120.csv"),
                "--output_dir",
                str(eval_dir),
                "--backend",
                "mock",
                "--models",
                "resume-check",
                "--prompt_modes",
                "zero_shot",
                "cot",
            ]
            eval_result = subprocess.run(eval_command, cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(eval_result.returncode, 0, msg=eval_result.stdout + eval_result.stderr)
            checkpoint = eval_dir / "outputs" / "resume-check_zero_shot_benhallu_micro_debug_120.jsonl"
            required_fields = {
                "id", "task", "track", "hallucination_type", "raw_prompt", "raw_response",
                "parsed_verdict", "is_collapsed", "latency_ms", "prompt_tokens", "completion_tokens",
            }
            first_record = json.loads(checkpoint.read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(required_fields.issubset(first_record))
            self.assertEqual(len(checkpoint.read_text(encoding="utf-8").splitlines()), 120)
            with checkpoint.open("ab") as handle:
                handle.write(b'{"id":"torn')
            resume_result = subprocess.run(eval_command, cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(resume_result.returncode, 0, msg=resume_result.stdout + resume_result.stderr)
            self.assertEqual(len(checkpoint.read_text(encoding="utf-8").splitlines()), 120)
            self.assertTrue(checkpoint.read_bytes().endswith(b"\n"))
            for name in (
                "table1_overall_and_per_task_bhs.csv",
                "table2_granular_hallucination_types.csv",
                "table3_cot_delta_ablation.csv",
                "table4_weight_sensitivity.csv",
            ):
                self.assertTrue((eval_dir / "results" / name).is_file())

            dialect_command = [
                sys.executable,
                str(ROOT / "make_splits.py"),
                "--generate_mock_data",
                "--output_dir",
                str(output_dir),
                "--mode",
                "dialect",
                "--seed",
                "42",
            ]
            dialect_result = subprocess.run(dialect_command, cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(dialect_result.returncode, 0, msg=dialect_result.stdout + dialect_result.stderr)
            dialect_subset = pd.read_csv(output_dir / "dialect_ccc_subset_300.csv")
            self.assertEqual(len(dialect_subset), 300)
            self.assertEqual(
                dialect_subset.dialect.value_counts().to_dict(),
                {"Standard_Bengali": 100, "Tangail": 100, "Chittagong": 100},
            )
            exact_100 = pd.concat(
                [group.head(100) for _, group in mock_dialect.groupby("dialect") if len(group.head(100)) == 100],
                ignore_index=True,
            )
            exact_100 = exact_100[exact_100.dialect.isin(("Standard_Bengali", "Tangail", "Chittagong"))]
            edge_output = output_dir / "dialect_exact_100"
            create_dialect_split(exact_100, edge_output, seed=42)
            self.assertEqual(len(pd.read_csv(edge_output / "dialect_ccc_subset_300.csv")), 300)

    def test_always_judges_each_have_bhs_50_percent(self) -> None:
        always_yes = compute_bhs(preds_track_a=np.ones(10, dtype=int), preds_track_b=np.ones(10, dtype=int))
        always_no = compute_bhs(preds_track_a=np.zeros(10, dtype=int), preds_track_b=np.zeros(10, dtype=int))
        self.assertEqual(always_yes["BHS"], 50.0)
        self.assertEqual(always_no["BHS"], 50.0)

    def test_lins_ccc_formulation(self) -> None:
        x = np.asarray([1.0, 2.0, 4.0, 7.0])
        self.assertEqual(compute_lins_ccc(x, x), 1.0)
        self.assertLess(compute_lins_ccc(x, x + 2.0), 1.0)
        orthogonal_x = np.asarray([-1.0, 0.0, 1.0, 0.0])
        orthogonal_y = np.asarray([0.0, 1.0, 0.0, -1.0])
        self.assertEqual(compute_lins_ccc(orthogonal_x, orthogonal_y), 0.0)

    def test_repetition_collapse_in_english_and_bengali(self) -> None:
        banglallama_13b_english_loop = "the answer is unsupported by the context " * 4
        banglallama_13b_bengali_loop = "এই উত্তরটি নথির তথ্যের সঙ্গে মেলে না " * 4
        for text in (banglallama_13b_english_loop, banglallama_13b_bengali_loop):
            result = parse_judge_output(text)
            self.assertTrue(result["is_collapsed"])
            self.assertTrue(result["parse_failure"])
        truncated = parse_judge_output("Reasoning continued without a verdict", completion_tokens=1024)
        self.assertTrue(truncated["is_collapsed"])

    def test_bilingual_and_structured_verdict_parsing(self) -> None:
        self.assertEqual(parse_judge_output("<verdict>হ্যাঁ</verdict>")["parsed_verdict"], 1)
        self.assertEqual(parse_judge_output("Final Verdict: No")["parsed_verdict"], 0)
        self.assertEqual(parse_judge_output('{"hallucinated": true}') ["parsed_verdict"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
