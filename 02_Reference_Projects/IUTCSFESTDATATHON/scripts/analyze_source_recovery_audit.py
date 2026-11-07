from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


THRESHOLDS = (0.80, 0.90, 0.95, 0.99)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def read_submission(path: Path) -> dict[int, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "label"]:
            raise ValueError("base submission must contain exactly id,label")
        result = {int(row["id"]): int(row["label"]) for row in reader}
    if any(label not in {0, 1} for label in result.values()):
        raise ValueError("base submission labels must be binary")
    return result


def binary_f1(truth: list[int], pred: list[int], positive: int) -> float:
    tp = sum(t == positive and p == positive for t, p in zip(truth, pred))
    fp = sum(t != positive and p == positive for t, p in zip(truth, pred))
    fn = sum(t == positive and p != positive for t, p in zip(truth, pred))
    denominator = 2 * tp + fp + fn
    return (2 * tp / denominator) if denominator else 0.0


def metrics(rows: list[dict], threshold: float) -> dict:
    decided = [
        row
        for row in rows
        if row.get("verdict") in {0, 1}
        and float(row.get("confidence", 0.0)) >= threshold
        and row.get("label") in {0, 1}
    ]
    truth = [int(row["label"]) for row in decided]
    pred = [int(row["verdict"]) for row in decided]
    correct = sum(t == p for t, p in zip(truth, pred))
    f1_0 = binary_f1(truth, pred, 0) if truth else None
    f1_1 = binary_f1(truth, pred, 1) if truth else None
    return {
        "threshold": threshold,
        "covered": len(decided),
        "correct": correct,
        "accuracy": correct / len(decided) if decided else None,
        "f1_0": f1_0,
        "f1_1": f1_1,
        "macro_f1": (f1_0 + f1_1) / 2 if truth else None,
    }


def cohort_key(row: dict) -> str:
    context = "context" if bool(row.get("has_context")) else "no_context"
    return f"{context}:{row.get('source_relation', 'parse_error')}"


def sample_calibration(sample_rows: list[dict]) -> dict[str, dict]:
    cohorts: dict[str, list[dict]] = defaultdict(list)
    cohorts["all"].extend(sample_rows)
    for row in sample_rows:
        context = "context" if bool(row.get("has_context")) else "no_context"
        relation = str(row.get("source_relation", "parse_error"))
        cohorts[context].append(row)
        cohorts[f"relation:{relation}"].append(row)
        cohorts[cohort_key(row)].append(row)
    return {
        name: {
            "rows": len(rows),
            "thresholds": [metrics(rows, threshold) for threshold in THRESHOLDS],
        }
        for name, rows in sorted(cohorts.items())
    }


def metric_at(
    calibration: dict[str, dict], key: str, threshold: float
) -> dict | None:
    cohort = calibration.get(key)
    if not cohort:
        return None
    for item in cohort["thresholds"]:
        if item["threshold"] == threshold:
            return item
    return None


def analyze(
    judgment_rows: list[dict],
    base_labels: dict[int, int],
    *,
    confidence_threshold: float = 0.95,
    min_control_support: int = 5,
) -> tuple[dict, list[dict]]:
    sample_rows = [row for row in judgment_rows if row.get("split") == "sample"]
    test_rows = [row for row in judgment_rows if row.get("split") == "test"]
    if not sample_rows or not test_rows:
        raise ValueError("judgment trace must contain sample and test rows")

    calibration = sample_calibration(sample_rows)
    disagreements = []
    safe_review = []
    for row in test_rows:
        row_id = int(row["id"])
        if row_id not in base_labels:
            raise ValueError(f"test id {row_id} is absent from base submission")
        verdict = row.get("verdict")
        base_label = base_labels[row_id]
        if verdict not in {0, 1} or int(verdict) == base_label:
            continue
        key = cohort_key(row)
        control = metric_at(calibration, key, confidence_threshold)
        supported = bool(
            control
            and control["covered"] >= min_control_support
            and control["accuracy"] == 1.0
        )
        candidate = {
            "id": row_id,
            "base_label": base_label,
            "proposed_label": int(verdict),
            "confidence": float(row.get("confidence", 0.0)),
            "source_relation": row.get("source_relation", "parse_error"),
            "has_context": bool(row.get("has_context")),
            "cohort": key,
            "control_covered": control["covered"] if control else 0,
            "control_accuracy": control["accuracy"] if control else None,
            "calibration_gate": bool(
                supported
                and float(row.get("confidence", 0.0)) >= confidence_threshold
                and row.get("source_relation")
                in {"exact_origin", "independent_evidence"}
            ),
            "hit_count": int(row.get("hit_count", 0)),
            "hit_domains": row.get("hit_domains", []),
            "prompt_bn": row.get("prompt_bn", ""),
            "response_bn": row.get("response_bn", ""),
            "correct_answer": row.get("correct_answer", ""),
            "reason": row.get("reason", ""),
        }
        disagreements.append(candidate)
        if candidate["calibration_gate"]:
            safe_review.append(candidate)

    disagreements.sort(
        key=lambda row: (
            not row["calibration_gate"],
            -row["confidence"],
            row["id"],
        )
    )
    report = {
        "policy": (
            "diagnostic only; calibration_gate means eligible for manual source review, "
            "not an automatic label change"
        ),
        "confidence_threshold": confidence_threshold,
        "min_control_support": min_control_support,
        "sample_rows": len(sample_rows),
        "test_rows": len(test_rows),
        "parse_errors": sum("parse_error" in row for row in judgment_rows),
        "sample_calibration": calibration,
        "test_disagreements": len(disagreements),
        "calibration_gated_for_review": len(safe_review),
        "calibration_gated_ids": [row["id"] for row in safe_review],
    }
    return report, disagreements


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "id",
        "base_label",
        "proposed_label",
        "confidence",
        "source_relation",
        "has_context",
        "cohort",
        "control_covered",
        "control_accuracy",
        "calibration_gate",
        "hit_count",
        "hit_domains",
        "prompt_bn",
        "response_bn",
        "correct_answer",
        "reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["hit_domains"] = json.dumps(
                output["hit_domains"], ensure_ascii=False
            )
            writer.writerow(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate the Kaggle source-recovery audit against released labels."
    )
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--base-submission", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--disagreements", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.95)
    parser.add_argument("--min-control-support", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, disagreements = analyze(
        read_jsonl(args.judgments),
        read_submission(args.base_submission),
        confidence_threshold=args.confidence_threshold,
        min_control_support=args.min_control_support,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(args.disagreements, disagreements)
    print(json.dumps({
        "test_disagreements": report["test_disagreements"],
        "calibration_gated_for_review": report["calibration_gated_for_review"],
        "calibration_gated_ids": report["calibration_gated_ids"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
