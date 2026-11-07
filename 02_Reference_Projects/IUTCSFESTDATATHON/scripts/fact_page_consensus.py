from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from datathon_utils import load_test


def load_trace(path: Path) -> dict[int, dict]:
    return {
        int(item["id"]): item
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for item in [json.loads(line)]
    }


def consensus_prediction(items: list[dict], min_confidence: float, min_agree: int) -> tuple[int | None, list[dict]]:
    voters = [
        item
        for item in items
        if item.get("pred") in {0, 1} and float(item.get("confidence", 0.0)) >= min_confidence
    ]
    votes = [int(item["pred"]) for item in voters]
    if len(votes) < min_agree or len(set(votes)) != 1:
        return None, voters
    winner, count = Counter(votes).most_common(1)[0]
    return (winner if count >= min_agree else None), voters


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply only unanimous, multi-view fact-page decisions.")
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--base-submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.85)
    parser.add_argument("--min-agree", type=int, default=2)
    args = parser.parse_args()

    test = load_test()
    base = pd.read_csv(args.base_submission)
    if len(base) != len(test) or not (base["id"].to_numpy() == test["id"].to_numpy()).all():
        raise ValueError("base submission IDs do not match test")
    traces = [(path.name, load_trace(path)) for path in args.trace]
    row_ids = sorted(set().union(*(set(trace) for _, trace in traces)))
    old_by_id = base.set_index("id")["label"].to_dict()
    output = base.copy()
    records = []
    for row_id in row_ids:
        named_items = [(name, trace[row_id]) for name, trace in traces if row_id in trace]
        pred, voters = consensus_prediction([item for _, item in named_items], args.min_confidence, args.min_agree)
        old = int(old_by_id[row_id])
        if pred is not None:
            output.loc[output["id"] == row_id, "label"] = int(pred)
        vote_records = []
        for name, item in named_items:
            vote_records.append(
                {
                    "trace": name,
                    "pred": item.get("pred"),
                    "verdict": item.get("verdict"),
                    "confidence": item.get("confidence"),
                    "expected_answer": item.get("expected_answer"),
                }
            )
        records.append(
            {
                "id": row_id,
                "old": old,
                "consensus": pred,
                "changed": pred is not None and int(pred) != old,
                "votes": vote_records,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    summary = {
        "trace_count": len(traces),
        "min_confidence": args.min_confidence,
        "min_agree": args.min_agree,
        "rows_considered": len(records),
        "accepted": sum(item["consensus"] is not None for item in records),
        "changed": sum(item["changed"] for item in records),
        "output": str(args.output),
        "output_counts": output["label"].value_counts().sort_index().to_dict(),
    }
    args.audit.write_text(json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for item in records:
        if item["changed"]:
            print("CHANGE", item["id"], item["old"], "->", item["consensus"])


if __name__ == "__main__":
    main()
