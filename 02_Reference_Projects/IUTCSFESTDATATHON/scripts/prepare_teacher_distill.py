from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datathon_utils import OUTPUT_DIR


def keep_row(row: dict[str, Any], min_confidence: float, require_parseable: bool) -> bool:
    if require_parseable and row.get("teacher_parse_error"):
        return False
    try:
        label = int(row["teacher_label"])
        confidence = float(row.get("teacher_confidence", 0.0))
    except (KeyError, TypeError, ValueError):
        return False
    return label in {0, 1} and confidence >= min_confidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MLX teacher labels into a student-training JSONL file.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "teacher_distill_qwen36_27b.jsonl")
    parser.add_argument("--min-confidence", type=float, default=0.85)
    parser.add_argument("--allow-parse-errors", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped = 0
    agreement = 0
    labels = {0: 0, 1: 0}
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if not keep_row(row, args.min_confidence, require_parseable=not args.allow_parse_errors):
                skipped += 1
                continue
            teacher_label = int(row["teacher_label"])
            gold_label = row.get("gold_label", row.get("label"))
            out = dict(row)
            out["original_label"] = gold_label
            out["label"] = teacher_label
            out["source"] = f"teacher_{row.get('teacher_model', 'unknown')}_{out.get('source', 'unknown')}"
            target.write(json.dumps(out, ensure_ascii=False) + "\n")
            kept += 1
            labels[teacher_label] += 1
            try:
                agreement += int(int(gold_label) == teacher_label)
            except (TypeError, ValueError):
                pass

    summary = {
        "kept": kept,
        "skipped": skipped,
        "labels": labels,
        "teacher_original_agreement": agreement / kept if kept else 0.0,
        "min_confidence": args.min_confidence,
    }
    print("wrote", args.output, json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
