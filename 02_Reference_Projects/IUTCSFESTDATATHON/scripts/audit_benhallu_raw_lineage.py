from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from audit_source_origin_conflicts import load_frame, raw_signature
from datathon_utils import clean_context, normalize_bn


LineageRecord = tuple[str, str, str, int, str, str]
LineageKey = tuple[str, str, str]


def load_lineage_records(
    gold_path: Path,
    hallucination_path: Path,
) -> list[LineageRecord]:
    gold = pd.read_csv(gold_path).fillna("")
    hallucinations = pd.read_csv(hallucination_path).fillna("")
    source_lookup = gold.set_index("id")
    records: list[LineageRecord] = []

    for row in gold.itertuples(index=False):
        records.append(
            (
                str(row.context),
                str(row.question),
                str(row.correct_answer),
                1,
                "gold",
                str(row.id),
            )
        )
    for row in hallucinations.itertuples(index=False):
        if row.source_id not in source_lookup.index:
            continue
        source = source_lookup.loc[row.source_id]
        records.append(
            (
                str(source["context"]),
                str(source["question"]),
                str(row.hallucinated_answer),
                0,
                str(row.pattern),
                str(row.id),
            )
        )
    return records


def normalized_key(context: object, question: object, answer: object) -> LineageKey:
    return (
        normalize_bn(context),
        normalize_bn(question),
        normalize_bn(answer),
    )


def raw_key(context: object, question: object, answer: object) -> LineageKey:
    return (
        raw_signature(context),
        raw_signature(question),
        raw_signature(answer),
    )


def audit_raw_lineage(
    frame: pd.DataFrame,
    source_records: list[LineageRecord],
    *,
    base_labels: dict[int, int] | None = None,
) -> tuple[list[dict], dict]:
    normalized: dict[LineageKey, list[LineageRecord]] = defaultdict(list)
    raw: dict[LineageKey, list[LineageRecord]] = defaultdict(list)
    for record in source_records:
        context, question, answer, *_ = record
        normalized[normalized_key(context, question, answer)].append(record)
        raw[raw_key(context, question, answer)].append(record)

    normalized_conflicts = {
        key for key, values in normalized.items() if len({item[3] for item in values}) > 1
    }
    raw_conflicts = {
        key for key, values in raw.items() if len({item[3] for item in values}) > 1
    }
    lineage = {
        key: values[0][3]
        for key, values in normalized.items()
        if key not in normalized_conflicts
    }

    records: list[dict] = []
    for row in frame.itertuples(index=False):
        context = clean_context(row.context)
        key = normalized_key(context, row.prompt_bn, row.response_bn)
        if key not in lineage:
            continue
        exact = raw.get(raw_key(context, row.prompt_bn, row.response_bn), [])
        row_id = int(row.id)
        item = {
            "id": row_id,
            "label": int(lineage[key]),
            "raw_exact": bool(exact),
            "normalized_source_records": len(normalized[key]),
            "source_patterns": sorted({record[4] for record in normalized[key]}),
            "prompt_bn": str(row.prompt_bn),
            "response_bn": str(row.response_bn),
        }
        if hasattr(row, "label"):
            item["truth"] = int(row.label)
            item["correct"] = int(lineage[key]) == int(row.label)
        if base_labels is not None:
            item["base_label"] = int(base_labels[row_id])
            item["disagrees_with_base"] = int(lineage[key]) != int(base_labels[row_id])
        records.append(item)

    normalized_only = [int(item["id"]) for item in records if not item["raw_exact"]]
    summary: dict[str, object] = {
        "source_records": len(source_records),
        "normalized_keys": len(normalized),
        "raw_keys": len(raw),
        "normalized_label_conflicts": len(normalized_conflicts),
        "raw_label_conflicts": len(raw_conflicts),
        "matched_rows": len(records),
        "raw_exact_rows": sum(bool(item["raw_exact"]) for item in records),
        "normalized_only_rows": len(normalized_only),
        "normalized_only_ids": normalized_only,
    }
    if records and all("truth" in item for item in records):
        summary["correct"] = sum(bool(item["correct"]) for item in records)
    if base_labels is not None:
        disagreements = [
            int(item["id"]) for item in records if item["disagrees_with_base"]
        ]
        summary["base_disagreements"] = len(disagreements)
        summary["base_disagreement_ids"] = disagreements
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether normalized BenHallu lineage matches are also exact raw "
            "context-question-response matches. Diagnostic only."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--hallucinations", type=Path, required=True)
    parser.add_argument("--base-submission", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = load_frame(args.input)
    source_records = load_lineage_records(args.gold, args.hallucinations)
    base_labels = None
    if args.base_submission:
        base = pd.read_csv(args.base_submission)
        base_labels = dict(zip(base["id"].astype(int), base["label"].astype(int)))
    records, summary = audit_raw_lineage(
        frame,
        source_records,
        base_labels=base_labels,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
