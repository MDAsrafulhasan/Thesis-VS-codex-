from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from audit_source_origin_conflicts import (
    exact_origin_entries,
    group_entries,
    has_answer_conflict,
    load_frame,
)
from datathon_utils import metric_row, normalize_bn
from source_answer_bank import BankEntry, compare_response, read_bank


def repository_votes(response: str, entries: list[BankEntry]) -> dict[str, int]:
    """Return one high-confidence vote for each internally consistent repository."""
    labels: dict[str, list[int]] = defaultdict(list)
    for entry in entries:
        decision = compare_response(response, entry)
        if decision is not None and decision.confidence >= 0.90:
            labels[entry.source_repo].append(int(decision.label))

    return {
        repository: values[0]
        for repository, values in labels.items()
        if values and len(set(values)) == 1
    }


def strict_majority(votes: dict[str, int]) -> int | None:
    if not votes:
        return None
    counts = Counter(votes.values())
    ordered = counts.most_common()
    if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
        return None
    return int(ordered[0][0])


def audit_conflict_votes(
    frame: pd.DataFrame,
    entries: list[BankEntry],
    *,
    base_labels: dict[int, int] | None = None,
) -> tuple[list[dict], dict]:
    grouped = group_entries(entries)
    records: list[dict] = []
    conflict_rows = 0
    exact_origin_rows = 0

    for row in frame.itertuples(index=False):
        candidates = grouped.get(normalize_bn(row.prompt_bn), [])
        if not candidates or not has_answer_conflict(candidates):
            continue
        conflict_rows += 1

        origins = exact_origin_entries(str(row.prompt_bn), candidates)
        if not origins:
            continue
        exact_origin_rows += 1

        votes = repository_votes(str(row.response_bn), origins)
        prediction = strict_majority(votes)
        if prediction is None:
            continue

        row_id = int(row.id)
        counts = Counter(votes.values())
        record = {
            "id": row_id,
            "label": prediction,
            "repository_votes": dict(sorted(votes.items())),
            "vote_counts": {str(label): count for label, count in sorted(counts.items())},
            "source_repositories": sorted({entry.source_repo for entry in origins}),
            "prompt_bn": str(row.prompt_bn),
            "response_bn": str(row.response_bn),
        }
        if hasattr(row, "label"):
            record["truth"] = int(row.label)
            record["correct"] = prediction == int(row.label)
        if base_labels is not None:
            record["base_label"] = int(base_labels[row_id])
            record["disagrees_with_base"] = prediction != int(base_labels[row_id])
        records.append(record)

    summary: dict[str, object] = {
        "rows": len(frame),
        "conflict_rows": conflict_rows,
        "exact_origin_rows": exact_origin_rows,
        "majority_decisions": len(records),
    }
    if records and all("truth" in record for record in records):
        truth = [int(record["truth"]) for record in records]
        predictions = [int(record["label"]) for record in records]
        summary["metrics"] = metric_row(truth, predictions)
        summary["correct"] = sum(
            expected == actual for expected, actual in zip(truth, predictions)
        )
    if base_labels is not None:
        disagreements = [
            int(record["id"]) for record in records if record["disagrees_with_base"]
        ]
        summary["base_disagreements"] = len(disagreements)
        summary["base_disagreement_ids"] = disagreements
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic-only audit of strict repository-majority decisions for "
            "exact-origin questions whose public answer keys conflict."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--bank", type=Path, action="append", required=True)
    parser.add_argument("--base-submission", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = load_frame(args.input)
    entries = [entry for path in args.bank for entry in read_bank(path)]
    base_labels = None
    if args.base_submission:
        base = pd.read_csv(args.base_submission)
        base_labels = dict(zip(base["id"].astype(int), base["label"].astype(int)))
        missing = set(frame["id"].astype(int)) - set(base_labels)
        if missing:
            raise ValueError(f"base submission is missing IDs: {sorted(missing)[:10]}")

    records, summary = audit_conflict_votes(
        frame,
        entries,
        base_labels=base_labels,
    )
    payload = {"summary": summary, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
