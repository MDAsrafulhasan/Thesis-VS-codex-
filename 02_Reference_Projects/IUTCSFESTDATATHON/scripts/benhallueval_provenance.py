from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import pandas as pd

from datathon_utils import load_sample, load_test, metric_row, normalize_bn


def load_provenance(gold_path: Path, hallucination_path: Path):
    gold = pd.read_csv(gold_path).fillna("")
    generated = pd.read_csv(hallucination_path).fillna("")
    correct: dict[str, set[str]] = collections.defaultdict(set)
    hallucinated: dict[str, set[str]] = collections.defaultdict(set)
    for row in gold.itertuples():
        correct[normalize_bn(row.question)].add(normalize_bn(row.correct_answer))
    for row in generated.itertuples():
        key = normalize_bn(row.question)
        correct[key].add(normalize_bn(row.right_answer))
        hallucinated[key].add(normalize_bn(row.hallucinated_answer))
    return correct, hallucinated


def decide(question: str, response: str, correct, hallucinated):
    key, answer = normalize_bn(question), normalize_bn(response)
    if not key or not answer:
        return None, ""
    positive, negative = correct.get(key, set()), hallucinated.get(key, set())
    if answer in positive and answer not in negative:
        return 1, "released_ground_truth_exact"
    if answer in negative and answer not in positive:
        return 0, "released_hallucination_exact"
    return None, ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact provenance labels from the released BenHalluEval GQA corpus.")
    parser.add_argument("--split", choices=["sample", "test"], required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--hallucinations", type=Path, required=True)
    parser.add_argument("--base-submission", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()
    df = load_sample() if args.split == "sample" else load_test()
    correct, hallucinated = load_provenance(args.gold, args.hallucinations)
    base = pd.read_csv(args.base_submission) if args.base_submission else None
    records, y, p = [], [], []
    for row_index, row in df.iterrows():
        label, rule = decide(row.prompt_bn, row.response_bn, correct, hallucinated)
        if label is None:
            continue
        item = {"row_index": int(row_index), "id": int(row.id) if "id" in df else int(row_index),
                "label": label, "rule": rule, "question": row.prompt_bn, "response": row.response_bn}
        if "label" in df:
            item["truth"] = int(row.label); y.append(int(row.label)); p.append(label)
        if base is not None:
            mask = base.id == int(row.id); old = int(base.loc[mask, "label"].iloc[0])
            base.loc[mask, "label"] = label; item.update({"old": old, "changed": old != label})
        records.append(item)
    summary = {"resolved": len(records), "changed": sum(r.get("changed", False) for r in records)}
    if y:
        summary["metrics"] = metric_row(pd.Series(y).to_numpy(), pd.Series(p).to_numpy())
    if base is not None:
        if args.output is None: raise ValueError("output required")
        base.to_csv(args.output, index=False); summary["output"] = str(args.output)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    args.trace.write_text(json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
