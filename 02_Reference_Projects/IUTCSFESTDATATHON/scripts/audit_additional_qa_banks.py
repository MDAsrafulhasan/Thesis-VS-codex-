from __future__ import annotations

import argparse
import json
import tarfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

from datathon_utils import load_sample, load_test, normalize_bn


def add(bank: dict[str, set[str]], question: object, answers: list[object]) -> None:
    key = normalize_bn(str(question))
    values = {normalize_bn(str(answer)) for answer in answers if normalize_bn(str(answer))}
    if key and values:
        bank[key].update(values)


def read_squad_object(obj: dict, bank: dict[str, set[str]]) -> None:
    for article in obj.get("data", []):
        # BanglaRQA stores qas directly on an item. SQuAD/IndicQA nest them
        # under paragraphs.
        blocks = [article] if "qas" in article else article.get("paragraphs", [])
        for block in blocks:
            for qa in block.get("qas", []):
                question = qa.get("question") or qa.get("question_text")
                raw_answers = qa.get("answers", [])
                if isinstance(raw_answers, dict):
                    answers = raw_answers.get("answer_text") or raw_answers.get("text") or []
                else:
                    answers = [item.get("text", "") if isinstance(item, dict) else item for item in raw_answers]
                add(bank, question, answers)


def load_json_squad(paths: list[Path]) -> dict[str, set[str]]:
    bank: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        read_squad_object(json.loads(path.read_text(encoding="utf-8")), bank)
    return bank


def load_squad_archive(path: Path) -> dict[str, set[str]]:
    bank: dict[str, set[str]] = defaultdict(set)
    with tarfile.open(path, "r:bz2") as archive:
        for member in archive.getmembers():
            if not member.name.endswith(".json"):
                continue
            handle = archive.extractfile(member)
            if handle is not None:
                read_squad_object(json.load(handle), bank)
    return bank


def load_pair_json(path: Path) -> dict[str, set[str]]:
    bank: dict[str, set[str]] = defaultdict(set)
    for row in json.loads(path.read_text(encoding="utf-8")):
        add(bank, row.get("input", ""), [row.get("output", "")])
    return bank


def audit(frame: pd.DataFrame, bank: dict[str, set[str]], labels: dict[int, int] | None) -> dict:
    question_matches = []
    positive_records = []
    for row in frame.itertuples(index=False):
        answers = bank.get(normalize_bn(str(row.prompt_bn)))
        if answers is None:
            continue
        response = normalize_bn(str(row.response_bn))
        exact = response in answers
        item = {
            "id": int(row.id),
            "prediction": int(exact),
            "response": response,
            "answers": sorted(answers),
        }
        question_matches.append(item)
        if exact:
            positive_records.append(item)

    result = {
        "questions_in_bank": len(bank),
        "question_coverage": len(question_matches),
        "exact_positive_coverage": len(positive_records),
        "question_matches": question_matches,
        "records": positive_records,
    }
    if labels is not None:
        result["question_match_accuracy"] = (
            sum(item["prediction"] == labels[item["id"]] for item in question_matches)
            / len(question_matches)
            if question_matches
            else None
        )
        result["exact_positive_accuracy"] = (
            sum(labels[item["id"]] == 1 for item in positive_records) / len(positive_records)
            if positive_records
            else None
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit strict exact lineage in additional Bengali QA banks.")
    parser.add_argument("--banglarqa", type=Path, action="append", default=[])
    parser.add_argument("--indicqa", type=Path, action="append", default=[])
    parser.add_argument("--squad-bn-archive", type=Path)
    parser.add_argument("--pair-json", type=Path)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sources: dict[str, dict[str, set[str]]] = {}
    if args.banglarqa:
        sources["sartajekram/BanglaRQA"] = load_json_squad(args.banglarqa)
    if args.indicqa:
        sources["ai4bharat/IndicQA"] = load_json_squad(args.indicqa)
    if args.squad_bn_archive:
        sources["csebuetnlp/squad_bn"] = load_squad_archive(args.squad_bn_archive)
    if args.pair_json:
        sources["rasheduzzaman/Bangla_question_answer_pair_70K_dataset"] = load_pair_json(args.pair_json)

    sample = load_sample().reset_index().rename(columns={"index": "id"})
    sample["id"] += 1
    test = load_test()
    predictions = dict(pd.read_csv(args.submission)[["id", "label"]].itertuples(index=False, name=None))
    labels = dict(sample[["id", "label"]].itertuples(index=False, name=None))

    report = {"sources": {}}
    for name, bank in sources.items():
        sample_result = audit(sample, bank, labels)
        test_result = audit(test, bank, None)
        test_result["submission_disagreements"] = [
            item for item in test_result["question_matches"] if item["prediction"] != predictions[item["id"]]
        ]
        # Only exact positive records are safe to feed into provenance. A QA
        # source may omit valid aliases, so answer mismatch is never a trusted
        # hallucination label.
        report["sources"][name] = {"sample": sample_result, "test": test_result}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        name: {
            "sample_questions": values["sample"]["question_coverage"],
            "sample_exact": values["sample"]["exact_positive_coverage"],
            "sample_exact_accuracy": values["sample"].get("exact_positive_accuracy"),
            "test_questions": values["test"]["question_coverage"],
            "test_exact": values["test"]["exact_positive_coverage"],
            "v85_disagreements": len(values["test"]["submission_disagreements"]),
        }
        for name, values in report["sources"].items()
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
