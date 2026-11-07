from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from source_answer_bank import canonical_answer, number_signature
from source_artifact_overrides import compatible


def cluster_votes(evidence: list[dict], min_score: float) -> list[list[dict]]:
    clusters: list[list[dict]] = []
    for item in evidence:
        if (
            float(item.get("score", 0)) < min_score
            or not item.get("answer")
            or item.get("relation_match") is False
        ):
            continue
        for cluster in clusters:
            if compatible(str(item["answer"]), str(cluster[0]["answer"])):
                cluster.append(item)
                break
        else:
            clusters.append([item])
    return clusters


def decide(response: str, evidence: list[dict], min_score: float, min_sources: int) -> tuple[int | None, dict]:
    clusters = cluster_votes(evidence, min_score)
    qualified = []
    for cluster in clusters:
        sources = {str(item.get("source_repo", "")) for item in cluster if item.get("source_repo")}
        if len(sources) >= min_sources:
            qualified.append((len(sources), max(float(x.get("score", 0)) for x in cluster), cluster, sources))
    if not qualified:
        return None, {}
    qualified.sort(key=lambda item: (item[0], item[1]), reverse=True)
    source_count, _, cluster, sources = qualified[0]
    answer = str(cluster[0]["answer"])
    if compatible(response, answer):
        prediction = 1
    else:
        response_numbers, answer_numbers = number_signature(response), number_signature(answer)
        short_typed_fact = len(canonical_answer(response)) <= 80 and len(canonical_answer(answer)) <= 80
        if response_numbers and answer_numbers and response_numbers != answer_numbers:
            prediction = 0
        elif short_typed_fact:
            prediction = 0
        else:
            return None, {}
    return prediction, {
        "rule": "independent_retrieval_consensus",
        "expected_answer": answer,
        "source_count": source_count,
        "sources": sorted(sources),
        "evidence": cluster,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply only independent-source hybrid retrieval consensus.")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--base-submission", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--min-score", type=float, default=0.65)
    parser.add_argument("--min-sources", type=int, default=2)
    args = parser.parse_args()
    records = []
    base = pd.read_csv(args.base_submission) if args.base_submission else None
    for line in args.evidence.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prediction, details = decide(str(row["response_bn"]), row.get("evidence", []), args.min_score, args.min_sources)
        if prediction is None:
            continue
        item = {
            "id": int(row["id"]),
            "prediction": prediction,
            "prompt": row["prompt_bn"],
            "response": row["response_bn"],
            **details,
        }
        if "label" in row:
            item.update({"truth": int(row["label"]), "correct": prediction == int(row["label"])})
        if base is not None:
            mask = base.id == int(row["id"])
            old = int(base.loc[mask, "label"].iloc[0])
            item.update({"old": old, "changed": old != prediction})
            base.loc[mask, "label"] = prediction
        records.append(item)
    summary = {
        "decisions": len(records),
        "correct": sum(item.get("correct", False) for item in records) if any("truth" in x for x in records) else None,
        "changed": sum(item.get("changed", False) for item in records) if base is not None else None,
        "min_score": args.min_score,
        "min_sources": args.min_sources,
    }
    if base is not None:
        if args.output is None:
            raise ValueError("--base-submission requires --output")
        base.to_csv(args.output, index=False)
        summary.update({"output": str(args.output), "counts": base.label.value_counts().sort_index().to_dict()})
    args.trace.write_text(json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
