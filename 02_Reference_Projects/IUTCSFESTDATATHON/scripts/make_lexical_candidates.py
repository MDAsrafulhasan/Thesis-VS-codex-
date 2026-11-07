from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from datathon_utils import OUTPUT_DIR, ROOT


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def validate_submission(path: Path) -> None:
    df = pd.read_csv(path)
    if list(df.columns) != ["id", "label"]:
        raise ValueError(f"{path} has columns {df.columns.tolist()}, expected ['id', 'label']")
    if len(df) != 2516:
        raise ValueError(f"{path} has {len(df)} rows, expected 2516")
    bad = set(df["label"].tolist()) - {0, 1}
    if bad:
        raise ValueError(f"{path} has invalid labels: {bad}")
    if int(df["id"].min()) != 1 or int(df["id"].max()) != 2516:
        raise ValueError(f"{path} id range is {(df['id'].min(), df['id'].max())}, expected 1..2516")
    print("validated", path, df["label"].value_counts().sort_index().to_dict())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default=str(OUTPUT_DIR / "evidence_index" / "lexical_char_tfidf"))
    parser.add_argument("--lexical-evidence", default=str(OUTPUT_DIR / "evidence_test_noctx_lexical_exact.jsonl"))
    parser.add_argument("--base-submission", default=str(OUTPUT_DIR / "submission_mlx_qwen36_27b_noctx_fact_no_context.csv"))
    parser.add_argument("--evidence025-submission", default=str(OUTPUT_DIR / "submission_mlx_qwen36_27b_noctx_fact_evidence025.csv"))
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--skip-retrieve", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    evidence_path = Path(args.lexical_evidence)
    if not args.skip_retrieve:
        run(
            [
                py,
                "scripts/evidence_index.py",
                "retrieve",
                "--index-dir",
                args.index_dir,
                "--split",
                "test",
                "--output",
                str(evidence_path),
                "--top-k",
                "5",
                "--batch-size",
                "64",
                "--query-mode",
                "prompt",
                "--inject-exact-question",
            ]
        )
    if not evidence_path.exists():
        raise FileNotFoundError(evidence_path)

    safe_output = OUTPUT_DIR / "submission_mlx_qwen36_27b_noctx_fact_lexical_override.csv"
    safe_trace = OUTPUT_DIR / "lexical_overrides_test.json"
    run(
        [
            py,
            "scripts/apply_lexical_overrides.py",
            "--split",
            "test",
            "--evidence-jsonl",
            str(evidence_path),
            "--base-submission",
            args.base_submission,
            "--threshold",
            str(args.threshold),
            "--output",
            str(safe_output),
            "--trace-output",
            str(safe_trace),
        ]
    )
    validate_submission(safe_output)

    if Path(args.evidence025_submission).exists():
        aggressive_output = OUTPUT_DIR / "submission_mlx_qwen36_27b_evidence025_lexical_override.csv"
        aggressive_trace = OUTPUT_DIR / "lexical_overrides_test_on_evidence025.json"
        run(
            [
                py,
                "scripts/apply_lexical_overrides.py",
                "--split",
                "test",
                "--evidence-jsonl",
                str(evidence_path),
                "--base-submission",
                args.evidence025_submission,
                "--threshold",
                str(args.threshold),
                "--output",
                str(aggressive_output),
                "--trace-output",
                str(aggressive_trace),
            ]
        )
        validate_submission(aggressive_output)
    else:
        print("skipping aggressive candidate; missing", args.evidence025_submission)


if __name__ == "__main__":
    main()
