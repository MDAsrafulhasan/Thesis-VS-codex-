from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from datathon_utils import OUTPUT_DIR, load_sample, load_test, metric_row, normalize_bn
from source_answer_bank import compare_response, decide_row, index_bank, read_bank


# Weights are precision tiers, not corpus sizes. The official 299-label audit
# produced 100% confident-decision precision for the 4.0 tier, 97.4% for
# 80-20, and 78.3% for job_mcq. The low-quality mirror may break ties only in
# agreement; it cannot overrule one trusted source.
SOURCE_WEIGHTS = {
    "SumanMondal/bengali_qa": 4.0,
    "hishab/bangla-mmlu": 4.0,
    "mojnupust/api.farhan-mcq": 4.0,
    "Nh-em0n/QDoc": 4.0,
    "farihashifa/SOMAJGYAAN": 3.0,
    "minhaj1403/bluck": 3.0,
    "tanvirmahfuz100/80-20-exam": 1.5,
    "NewtonGain/job_mcq": 0.25,
}


def weighted_decision(response: str, entries, min_score: float, min_margin: float) -> dict | None:
    by_source = defaultdict(list)
    for entry in entries:
        decision = compare_response(response, entry)
        if decision is not None:
            by_source[entry.source_repo].append(decision)

    scores = {0: 0.0, 1: 0.0}
    votes = []
    for source, decisions in by_source.items():
        labels = {decision.label for decision in decisions}
        if len(labels) != 1:
            continue
        label = next(iter(labels))
        weight = SOURCE_WEIGHTS.get(source, 1.0)
        scores[label] += weight
        votes.append(
            {
                "source": source,
                "label": label,
                "weight": weight,
                "decisions": [asdict(decision) for decision in decisions],
            }
        )
    if not votes:
        return None
    label = int(scores[1] > scores[0])
    if scores[0] == scores[1]:
        return None
    winner, loser = scores[label], scores[1 - label]
    if winner < min_score or winner - loser < min_margin:
        return None
    return {"label": label, "scores": scores, "margin": winner - loser, "votes": votes}


def main() -> None:
    parser = argparse.ArgumentParser(description="Reliability-weighted exact-source decisions for conflict abstentions.")
    parser.add_argument("--split", choices=["sample", "test"], required=True)
    parser.add_argument("--bank", type=Path, default=OUTPUT_DIR / "source_answer_bank_complete_bcs.jsonl")
    parser.add_argument("--base-submission", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--min-score", type=float, default=1.5)
    parser.add_argument("--min-margin", type=float, default=1.0)
    args = parser.parse_args()

    df = load_sample() if args.split == "sample" else load_test()
    if "id" not in df.columns:
        df = df.copy()
        df["id"] = np.arange(len(df), dtype=int)
    bank = read_bank(args.bank)
    bank_index = index_bank(bank)
    records = []
    for row in df.itertuples():
        entries = bank_index.get(normalize_bn(row.prompt_bn), [])
        if not entries:
            continue
        deterministic, _ = decide_row(str(row.response_bn), entries)
        if deterministic is not None:
            continue
        decision = weighted_decision(str(row.response_bn), entries, args.min_score, args.min_margin)
        if decision is not None:
            records.append(
                {
                    "id": int(row.id),
                    "prompt": str(row.prompt_bn),
                    "response": str(row.response_bn),
                    **decision,
                }
            )

    summary: dict[str, object] = {
        "split": args.split,
        "accepted": len(records),
        "min_score": args.min_score,
        "min_margin": args.min_margin,
        "label_counts": pd.Series([record["label"] for record in records])
        .value_counts()
        .sort_index()
        .to_dict(),
    }
    if args.split == "sample":
        truth = df.set_index("id")["label"]
        y_true = np.array([int(truth.loc[record["id"]]) for record in records])
        y_pred = np.array([int(record["label"]) for record in records])
        summary["metrics"] = metric_row(y_true, y_pred) if len(records) else None
        summary["correct"] = int((y_true == y_pred).sum())
        for record, expected in zip(records, y_true):
            record["truth"] = int(expected)
            record["correct"] = int(record["label"]) == int(expected)
    else:
        if args.base_submission is None or args.output is None:
            raise ValueError("test split requires --base-submission and --output")
        output = pd.read_csv(args.base_submission)
        changed = 0
        for record in records:
            mask = output["id"] == record["id"]
            old = int(output.loc[mask, "label"].iloc[0])
            output.loc[mask, "label"] = int(record["label"])
            record["old"] = old
            record["changed"] = old != int(record["label"])
            changed += record["changed"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(args.output, index=False)
        summary.update(
            {
                "changed": changed,
                "output": str(args.output),
                "output_counts": output["label"].value_counts().sort_index().to_dict(),
            }
        )

    args.trace.parent.mkdir(parents=True, exist_ok=True)
    args.trace.write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote", args.trace)


if __name__ == "__main__":
    main()
