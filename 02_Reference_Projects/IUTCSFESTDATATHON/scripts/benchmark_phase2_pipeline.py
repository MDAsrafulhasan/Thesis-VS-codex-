from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd


def total_bytes(paths: list[Path]) -> int:
    return sum(path.stat().st_size for path in paths if path.exists() and path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the offline router and extrapolate to the 5k Phase 2 set.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--base-submission", type=Path)
    parser.add_argument("--bank", type=Path, action="append", default=[])
    parser.add_argument("--verified-facts", type=Path, default=Path("resources/verified_fact_answers.json"))
    parser.add_argument("--target-rows", type=int, default=5000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    repeats = max(1, (args.target_rows + len(frame) - 1) // len(frame))
    synthetic = pd.concat([frame] * repeats, ignore_index=True).iloc[: args.target_rows].copy()
    synthetic["id"] = range(len(synthetic))
    synthetic_base = None
    if args.base_submission:
        base = pd.read_csv(args.base_submission)
        if len(base) != len(frame):
            raise ValueError("base submission length does not match input")
        synthetic_base = pd.concat([base] * repeats, ignore_index=True).iloc[: args.target_rows].copy()
        synthetic_base["id"] = synthetic["id"]
    with tempfile.TemporaryDirectory() as raw_dir:
        temp = Path(raw_dir)
        input_path, output_path, trace_path = temp / "input.csv", temp / "submission.csv", temp / "trace.json"
        synthetic.to_csv(input_path, index=False)
        command = [
            sys.executable, "scripts/phase2_pipeline.py", "--input", str(input_path),
            "--output", str(output_path), "--trace", str(trace_path),
            "--verified-facts", str(args.verified_facts),
        ]
        if synthetic_base is not None:
            base_path = temp / "base_submission.csv"
            synthetic_base.to_csv(base_path, index=False)
            command += ["--base-submission", str(base_path)]
        if args.model:
            command += ["--model", str(args.model)]
        for bank in args.bank:
            command += ["--bank", str(bank)]
        started = time.perf_counter()
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        seconds = time.perf_counter() - started
        reproduction_differences = None
        if synthetic_base is not None:
            reproduced = pd.read_csv(output_path)
            reproduction_differences = int((reproduced["label"] != synthetic_base["label"]).sum())
        artifacts = [path for path in [args.model, args.verified_facts, *args.bank] if path is not None]
        report = {
            "target_rows": args.target_rows,
            "wall_seconds": seconds,
            "rows_per_second": args.target_rows / max(seconds, 1e-9),
            "artifact_bytes": total_bytes(artifacts),
            "artifact_gib": total_bytes(artifacts) / 1024**3,
            "within_9_hours": seconds < 9 * 3600,
            "within_50_gib": total_bytes(artifacts) < 50 * 1024**3,
            "base_reproduction_differences": reproduction_differences,
            "command": command,
            "pipeline_stdout": result.stdout,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
