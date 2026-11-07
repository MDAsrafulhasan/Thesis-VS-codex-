from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


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


def read_labels(path: Path) -> dict[int, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "label"]:
            raise ValueError(f"{path} must contain exactly id,label")
        labels = {int(row["id"]): int(row["label"]) for row in reader}
    if any(label not in {0, 1} for label in labels.values()):
        raise ValueError(f"{path} contains a non-binary label")
    return labels


def reconstruct_sample_cascade(
    router_labels: dict[int, int], cascade_report: dict
) -> dict[int, int]:
    labels = dict(router_labels)
    for decision in cascade_report.get("decisions", []):
        labels[int(decision["row_index"])] = int(decision["new"])
    return labels


def direction(base: int, proposed: int) -> str:
    return f"{base}_to_{proposed}"


def analyze(
    trace: list[dict],
    sample_truth: dict[int, int],
    sample_base: dict[int, int],
    test_base: dict[int, int],
) -> tuple[dict, list[dict]]:
    sample = [row for row in trace if row.get("split") == "sample"]
    test = [row for row in trace if row.get("split") == "test"]
    if not sample or not test:
        raise ValueError("trace must contain sample and test records")

    sample_base_errors = [
        row_id
        for row_id, truth in sample_truth.items()
        if sample_base.get(row_id) != truth
    ]
    sample_decided = [row for row in sample if row.get("consensus") in {0, 1}]
    sample_disagreements = [
        row
        for row in sample_decided
        if int(row["consensus"]) != sample_base[int(row["id"])]
    ]
    sample_fixes = [
        row
        for row in sample_disagreements
        if int(row["consensus"]) == sample_truth[int(row["id"])]
    ]

    direction_controls = {}
    for key in ("0_to_1", "1_to_0"):
        controls = [
            row
            for row in sample_disagreements
            if direction(
                sample_base[int(row["id"])], int(row["consensus"])
            )
            == key
        ]
        fixes = [
            row
            for row in controls
            if int(row["consensus"]) == sample_truth[int(row["id"])]
        ]
        direction_controls[key] = {
            "disagreements": len(controls),
            "fixes": len(fixes),
            "harms": len(controls) - len(fixes),
            "correction_precision": len(fixes) / len(controls)
            if controls
            else None,
        }

    test_decided = [row for row in test if row.get("consensus") in {0, 1}]
    test_disagreements = []
    for row in test_decided:
        row_id = int(row["id"])
        if row_id not in test_base:
            raise ValueError(f"test id {row_id} missing from v85")
        proposed = int(row["consensus"])
        base = test_base[row_id]
        if proposed == base:
            continue
        key = direction(base, proposed)
        control = direction_controls[key]
        test_disagreements.append(
            {
                "id": row_id,
                "base_label": base,
                "proposed_label": proposed,
                "direction": key,
                "has_context": bool(row.get("has_context")),
                "min_confidence": row.get("min_confidence"),
                "control_disagreements": control["disagreements"],
                "control_fixes": control["fixes"],
                "control_correction_precision": control["correction_precision"],
                "calibration_gate": bool(
                    control["disagreements"] >= 5
                    and control["correction_precision"] == 1.0
                ),
                "prompt_bn": row.get("prompt_bn", ""),
                "response_bn": row.get("response_bn", ""),
            }
        )

    test_disagreements.sort(key=lambda row: (row["direction"], row["id"]))
    gated = [row for row in test_disagreements if row["calibration_gate"]]
    report = {
        "policy": (
            "diagnostic only; a test direction requires at least five analogous "
            "released-cascade disagreements at 100% correction precision"
        ),
        "sample_rows": len(sample),
        "sample_base_errors": sample_base_errors,
        "sample_consensus_decisions": len(sample_decided),
        "sample_consensus_counts": dict(
            Counter(str(row["consensus"]) for row in sample)
        ),
        "sample_disagreements": len(sample_disagreements),
        "sample_fixes": len(sample_fixes),
        "sample_harms": len(sample_disagreements) - len(sample_fixes),
        "sample_correction_precision": (
            len(sample_fixes) / len(sample_disagreements)
            if sample_disagreements
            else None
        ),
        "direction_controls": direction_controls,
        "test_rows": len(test),
        "test_consensus_decisions": len(test_decided),
        "test_disagreements": len(test_disagreements),
        "test_direction_counts": dict(
            Counter(row["direction"] for row in test_disagreements)
        ),
        "calibration_gated_candidates": len(gated),
        "calibration_gated_ids": [row["id"] for row in gated],
    }
    return report, test_disagreements


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "id",
        "base_label",
        "proposed_label",
        "direction",
        "has_context",
        "min_confidence",
        "control_disagreements",
        "control_fixes",
        "control_correction_precision",
        "calibration_gate",
        "prompt_bn",
        "response_bn",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure whether Gemma disagreements fix the released cascade."
    )
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--sample-router", type=Path, required=True)
    parser.add_argument("--sample-cascade-report", type=Path, required=True)
    parser.add_argument("--test-base", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--disagreements", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_rows = read_json(args.sample)
    sample_truth = {
        index: int(row["label"]) for index, row in enumerate(sample_rows)
    }
    sample_base = reconstruct_sample_cascade(
        read_labels(args.sample_router),
        read_json(args.sample_cascade_report),
    )
    report, disagreements = analyze(
        read_jsonl(args.trace),
        sample_truth,
        sample_base,
        read_labels(args.test_base),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(args.disagreements, disagreements)
    print(
        json.dumps(
            {
                "sample_disagreements": report["sample_disagreements"],
                "sample_fixes": report["sample_fixes"],
                "test_disagreements": report["test_disagreements"],
                "calibration_gated_candidates": report[
                    "calibration_gated_candidates"
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
