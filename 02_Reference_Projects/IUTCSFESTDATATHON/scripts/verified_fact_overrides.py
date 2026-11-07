from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from datathon_utils import load_sample, load_test
from source_answer_bank import canonical_answer


def load_verified_entries(path: Path) -> dict[str, dict]:
    return {
        canonical_answer(item["question"]): item
        for item in json.loads(path.read_text(encoding="utf-8"))
        if len(set(item.get("sources", []))) >= 2
    }


def predict_verified(response: str, evidence: dict) -> int:
    expected = {canonical_answer(answer) for answer in evidence["answers"]}
    return int(canonical_answer(response) in expected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a small two-source verified public fact bank.")
    parser.add_argument("--split", choices=["sample", "test"], required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--base-submission", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()

    df = load_sample() if args.split == "sample" else load_test()
    entries = load_verified_entries(args.evidence)
    base = pd.read_csv(args.base_submission) if args.base_submission else None
    records = []
    for row_index, row in df.iterrows():
        evidence = entries.get(canonical_answer(row.prompt_bn))
        if evidence is None:
            continue
        prediction = predict_verified(str(row.response_bn), evidence)
        item = {
            "row_index": int(row_index),
            "id": int(row.id) if "id" in df.columns else int(row_index),
            "prediction": prediction,
            "prompt": str(row.prompt_bn),
            "response": str(row.response_bn),
            "expected_answers": evidence["answers"],
            "sources": evidence["sources"],
            "rule": "two_source_verified_fact",
        }
        if "label" in df.columns:
            item.update({"truth": int(row.label), "correct": prediction == int(row.label)})
        if base is not None:
            mask = base.id == int(row.id)
            old = int(base.loc[mask, "label"].iloc[0])
            item.update({"old": old, "changed": old != prediction})
            base.loc[mask, "label"] = prediction
        records.append(item)

    summary = {
        "split": args.split,
        "resolved": len(records),
        "correct": sum(r.get("correct", False) for r in records) if "label" in df.columns else None,
        "changed": sum(r.get("changed", False) for r in records) if base is not None else None,
    }
    if base is not None:
        if args.output is None:
            raise ValueError("--base-submission requires --output")
        base.to_csv(args.output, index=False)
        summary.update({"output": str(args.output), "output_counts": base.label.value_counts().sort_index().to_dict()})
    args.trace.write_text(json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for item in records:
        if item.get("changed"):
            print("CHANGE", item["id"], item["old"], "->", item["prediction"], item["response"], item["expected_answers"])


if __name__ == "__main__":
    main()
