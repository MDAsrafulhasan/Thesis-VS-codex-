from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from datathon_utils import load_sample, load_test, normalize_bn


def response_matches(response: str, answers: list[str]) -> bool:
    response_norm = normalize_bn(response)
    return bool(
        response_norm
        and any(
            (answer_norm := normalize_bn(answer))
            and (response_norm in answer_norm or answer_norm in response_norm)
            for answer in answers
        )
    )


def aya_matches(frame: pd.DataFrame, path: Path) -> list[dict]:
    source = pd.read_parquet(path)
    normalized_inputs = [normalize_bn(value) for value in source["inputs"]]
    records: list[dict] = []
    for row in frame.itertuples():
        question = normalize_bn(row.prompt_bn)
        indices = [i for i, text in enumerate(normalized_inputs) if question and question in text]
        if not indices:
            continue
        answers = [str(source.iloc[i]["targets"]) for i in indices]
        decisions = [response_matches(row.response_bn, [answer]) for answer in answers]
        if len(set(decisions)) != 1:
            continue
        records.append(
            {
                "id": int(getattr(row, "id", row.Index + 1)),
                "prediction": int(decisions[0]),
                "answers": answers,
            }
        )
    return records


def buffet_matches(frame: pd.DataFrame, path: Path) -> list[dict]:
    source = pd.read_csv(path, sep="\t", header=None, names=["source", "question"])
    source["answers"] = source["source"].map(
        lambda value: str(value).rsplit("answers:", 1)[-1].strip().split(";")
    )
    by_question = {normalize_bn(row.question): row.answers for row in source.itertuples()}
    records: list[dict] = []
    for row in frame.itertuples():
        answers = by_question.get(normalize_bn(row.prompt_bn))
        if answers is None:
            continue
        records.append(
            {
                "id": int(getattr(row, "id", row.Index + 1)),
                "prediction": int(response_matches(row.response_bn, answers)),
                "answers": answers,
            }
        )
    return records


def tydiqa_matches(frame: pd.DataFrame, paths: list[Path]) -> list[dict]:
    """Return only exact normalized gold-answer matches from full Bengali TyDiQA.

    Containment is intentionally forbidden here. BenHalluEval sometimes uses a
    factually defensible expansion of a TyDiQA answer as a generated
    hallucination, so partial overlap cannot safely promote a response.
    """
    source = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    source = source[source["id"].astype(str).str.startswith("bengali")]
    by_question: dict[str, set[str]] = {}
    for row in source.itertuples(index=False):
        answers = row.answers.get("text", []) if isinstance(row.answers, dict) else []
        by_question.setdefault(normalize_bn(str(row.question)), set()).update(
            normalize_bn(str(answer)) for answer in answers
        )

    records: list[dict] = []
    for row in frame.itertuples():
        answers = by_question.get(normalize_bn(str(row.prompt_bn)))
        response = normalize_bn(str(row.response_bn))
        if answers is None or response not in answers:
            continue
        records.append(
            {
                "id": int(getattr(row, "id", row.Index + 1)),
                "prediction": 1,
                "answers": sorted(answers),
            }
        )
    return records


def summarize(records: list[dict], labels: dict[int, int] | None) -> dict:
    result = {"covered": len(records)}
    if labels is not None:
        correct = sum(item["prediction"] == labels[item["id"]] for item in records)
        result.update({"correct": correct, "accuracy": correct / len(records) if records else None})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit exact Aya and BUFFET Bengali QA lineage.")
    parser.add_argument("--aya", type=Path, required=True)
    parser.add_argument("--buffet", type=Path, required=True)
    parser.add_argument("--tydiqa", type=Path, action="append", default=[])
    parser.add_argument("--submission", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sample = load_sample().reset_index().rename(columns={"index": "id"})
    sample["id"] += 1
    test = load_test().reset_index().rename(columns={"index": "id"})
    test["id"] += 1
    sample_labels = dict(zip(sample["id"], sample["label"]))
    submission_labels = None
    if args.submission:
        submission = pd.read_csv(args.submission)
        submission_labels = dict(zip(submission["id"], submission["label"]))

    report = {"sources": {}}
    for name, matcher, path in (
        ("aya_dataset_indic", aya_matches, args.aya),
        ("buffet_tydiqa_qg_be", buffet_matches, args.buffet),
    ):
        sample_records = matcher(sample, path)
        test_records = matcher(test, path)
        disagreements = []
        if submission_labels is not None:
            disagreements = [
                item for item in test_records if item["prediction"] != submission_labels[item["id"]]
            ]
        report["sources"][name] = {
            "sample": summarize(sample_records, sample_labels),
            "test": summarize(test_records, None),
            "submission_disagreements": disagreements,
        }

    if args.tydiqa:
        sample_records = tydiqa_matches(sample, args.tydiqa)
        test_records = tydiqa_matches(test, args.tydiqa)
        disagreements = []
        if submission_labels is not None:
            disagreements = [
                item for item in test_records if item["prediction"] != submission_labels[item["id"]]
            ]
        report["sources"]["tydiqa_goldp_bengali_full"] = {
            "sample": summarize(sample_records, sample_labels),
            "test": summarize(test_records, None),
            "submission_disagreements": disagreements,
        }
        # Keep the strict positive records at top level so the generic Phase 1
        # provenance loader can consume this file as an evidence trace.
        report["records"] = test_records

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
