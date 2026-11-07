from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def router_command(args, submission: Path, trace: Path) -> list[str]:
    command = [
        sys.executable,
        "scripts/phase2_pipeline.py",
        "--input", str(args.input),
        "--output", str(submission),
        "--trace", str(trace),
        "--verified-facts", str(args.verified_facts),
    ]
    for path in getattr(args, "typed_evidence", []):
        command.extend(["--typed-evidence", str(path)])
    if getattr(args, "idiom_dataset", None):
        command.extend(["--idiom-dataset", str(args.idiom_dataset)])
    for corpus in args.law_corpus:
        command.extend(["--law-corpus", str(corpus)])
    if args.base_submission:
        command += ["--base-submission", str(args.base_submission)]
    if args.verifier:
        command += ["--model", str(args.verifier)]
    for bank in args.bank:
        command += ["--bank", str(bank)]
    for bank in args.fuzzy_bank:
        command += ["--fuzzy-bank", str(bank)]
    if args.lexical_bank:
        command += ["--lexical-bank", str(args.lexical_bank)]
    if args.benhallu_gold:
        command += ["--benhallu-gold", str(args.benhallu_gold)]
    if args.benhallu_hallucinations:
        command += ["--benhallu-hallucinations", str(args.benhallu_hallucinations)]
    return command


def judge_command(args, base: Path, router_trace: Path, trace: Path) -> list[str]:
    command = [
        sys.executable,
        "scripts/selective_llm_judge.py",
        "--input", str(args.input),
        "--base-submission", str(base),
        "--router-trace", str(router_trace),
        "--model", str(args.judge_model),
        "--output", str(args.output),
        "--trace", str(trace),
        "--batch-size", str(args.batch_size),
    ]
    if args.evidence:
        command += ["--evidence", str(args.evidence)]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline deterministic router and selective 4-bit judge.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--bank", type=Path, action="append", default=[])
    parser.add_argument("--fuzzy-bank", type=Path, action="append", default=[])
    parser.add_argument("--lexical-bank", type=Path)
    parser.add_argument("--benhallu-gold", type=Path)
    parser.add_argument("--benhallu-hallucinations", type=Path)
    parser.add_argument("--verified-facts", type=Path, default=Path("resources/verified_fact_answers.json"))
    parser.add_argument("--typed-evidence", type=Path, action="append", default=[])
    parser.add_argument("--idiom-dataset", type=Path)
    parser.add_argument("--law-corpus", type=Path, action="append", default=[])
    parser.add_argument("--base-submission", type=Path)
    parser.add_argument("--verifier", type=Path)
    parser.add_argument("--judge-model")
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()
    if not args.skip_llm and not args.judge_model:
        parser.error("--judge-model is required unless --skip-llm is set")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    router_submission = args.work_dir / "router_submission.csv"
    router_trace = args.work_dir / "router_trace.json"
    judge_trace = args.work_dir / "judge_trace.json"
    subprocess.run(router_command(args, router_submission, router_trace), check=True)
    if args.skip_llm:
        shutil.copyfile(router_submission, args.output)
        print("wrote", args.output, "router-only")
        return
    subprocess.run(judge_command(args, router_submission, router_trace, judge_trace), check=True)
    print("wrote", args.output, "router+selective-judge")


if __name__ == "__main__":
    main()
