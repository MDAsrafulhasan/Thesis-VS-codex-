#!/usr/bin/env python3
"""Audit candidate labels against BenHalluEval's gold-plus-four GQA groups.

    The join is deliberately conservative: contexts and questions must match after
    Bengali text normalization, and the competition response must exactly match
    either the gold answer or one of the four generated hallucinations. Requiring
    the question prevents cross-question collisions when a passage has several QA
    records whose answers are reused as distractors.
"""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from datathon_utils import normalize_bn


def audit(
    test: pd.DataFrame,
    submission: pd.DataFrame,
    gold: pd.DataFrame,
    hallucinations: pd.DataFrame,
) -> pd.DataFrame:
    candidate = test.merge(submission, on="id", validate="one_to_one")
    gold = gold.copy()
    gold["_context"] = gold["context"].fillna("").map(normalize_bn)
    by_context: dict[str, list[int]] = {}
    for index, context in gold["_context"].items():
        by_context.setdefault(context, []).append(index)

    rows: list[dict[str, object]] = []
    for row in candidate.itertuples(index=False):
        context = normalize_bn("" if pd.isna(row.context) else str(row.context))
        response = normalize_bn(str(row.response_bn))
        for index in by_context.get(context, []):
            source = gold.loc[index]
            normalized_question = normalize_bn(str(row.prompt_bn))
            normalized_source_question = normalize_bn(str(source["question"]))
            if normalized_question != normalized_source_question:
                continue
            source_id = source["id"]
            source_hallucinations = hallucinations[
                hallucinations["source_id"] == source_id
            ]["hallucinated_answer"].map(normalize_bn)
            gold_answer = normalize_bn(str(source["correct_answer"]))
            if response == gold_answer:
                evidence_label = 1
                side = "gold"
            elif response in set(source_hallucinations):
                evidence_label = 0
                side = "hallucination"
            else:
                continue
            rows.append(
                {
                    "id": int(row.id),
                    "candidate_label": int(row.label),
                    "evidence_label": evidence_label,
                    "agrees": int(row.label) == evidence_label,
                    "side": side,
                    "source_id": source_id,
                    "question_similarity": SequenceMatcher(
                        None,
                        normalized_question,
                        normalized_source_question,
                    ).ratio(),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--hallucinations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    result = audit(
        pd.read_csv(args.test),
        pd.read_csv(args.submission),
        pd.read_csv(args.gold),
        pd.read_csv(args.hallucinations),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    disagreements = int((~result["agrees"]).sum()) if len(result) else 0
    print(f"matched={len(result)} disagreements={disagreements} output={args.output}")


if __name__ == "__main__":
    main()
