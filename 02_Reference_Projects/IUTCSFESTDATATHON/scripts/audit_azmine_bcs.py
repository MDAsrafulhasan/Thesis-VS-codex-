from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from datathon_utils import metric_row, normalize_bn, prepare_frame
from fuzzy_source_overrides import FuzzyMcqMatcher
from source_answer_bank import BankEntry, decide_row, index_bank


def load_azmine_file(path: Path) -> list[BankEntry]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    entries: list[BankEntry] = []
    for row in rows:
        options = [str(item).strip() for item in row.get("options", [])]
        try:
            correct_index = int(str(row.get("answer", "")).strip()) - 1
        except ValueError:
            continue
        if not 0 <= correct_index < len(options):
            continue
        entries.append(
            BankEntry(
                question=str(row.get("question", "")).strip(),
                answers=[options[correct_index]],
                choices=options,
                correct_index=correct_index,
                source_repo=f"azminetoushikwasi/{path.stem}",
                source_file=str(row.get("file_name", "")),
                license="Apache-2.0",
                source_kind="mcq",
            )
        )
    return [entry for entry in entries if entry.question]


def exact_records(
    frame: pd.DataFrame,
    entries: list[BankEntry],
    base_labels: dict[int, int] | None = None,
) -> list[dict]:
    bank = index_bank(entries)
    records = []
    for row in frame.itertuples(index=False):
        matches = bank.get(normalize_bn(row.prompt_bn), [])
        if not matches:
            continue
        decision, all_decisions = decide_row(str(row.response_bn), matches)
        if decision is None:
            continue
        record = {
            "id": int(row.id),
            "prompt_bn": str(row.prompt_bn),
            "response_bn": str(row.response_bn),
            "label": int(decision.label),
            "rule": decision.rule,
            "answers": sorted({answer for entry in matches for answer in entry.answers}),
            "source_files": sorted({entry.source_file for entry in matches}),
            "all_decisions": [
                {
                    "label": int(item.label),
                    "confidence": float(item.confidence),
                    "rule": item.rule,
                    "source_repo": item.source_repo,
                    "answer": item.answer,
                }
                for item in all_decisions
            ],
        }
        if hasattr(row, "label"):
            record["truth"] = int(row.label)
            record["correct"] = int(decision.label) == int(row.label)
        if base_labels is not None:
            record["base_label"] = int(base_labels[int(row.id)])
            record["disagrees_with_base"] = (
                int(decision.label) != int(base_labels[int(row.id)])
            )
        records.append(record)
    return records


def audit(
    frame: pd.DataFrame,
    entries: list[BankEntry],
    base_labels: dict[int, int] | None = None,
) -> dict:
    frame = prepare_frame(frame)
    exact = exact_records(frame, entries, base_labels)
    exact_questions = set(index_bank(entries))
    candidates = frame[
        ~frame["prompt_bn"].map(normalize_bn).isin(exact_questions)
    ].reset_index(drop=True)
    fuzzy = []
    if not candidates.empty:
        fuzzy = FuzzyMcqMatcher(entries).match_rows(
            candidates,
            exclude_same_question=True,
            min_cosine=0.90,
            min_jaccard=0.65,
            min_sequence=0.82,
        )
    summary: dict[str, object] = {
        "input_rows": len(frame),
        "bank_entries": len(entries),
        "bank_questions": len(exact_questions),
        "exact_decisions": len(exact),
        "fuzzy_decisions": len(fuzzy),
    }
    if exact and all("truth" in item for item in exact):
        truth = [int(item["truth"]) for item in exact]
        pred = [int(item["label"]) for item in exact]
        summary["exact_metrics"] = metric_row(truth, pred)
        summary["exact_correct"] = sum(t == p for t, p in zip(truth, pred))
        truth_by_id = dict(zip(frame["id"].astype(int), frame["label"].astype(int)))
        summary["fuzzy_correct"] = sum(
            int(item["decision"]["label"]) == truth_by_id[int(item["id"])]
            for item in fuzzy
        )
    if base_labels is not None:
        summary["exact_base_disagreement_ids"] = [
            int(item["id"]) for item in exact if item.get("disagrees_with_base")
        ]
        summary["fuzzy_base_disagreement_ids"] = [
            int(item["id"])
            for item in fuzzy
            if int(item["decision"]["label"]) != int(base_labels[int(item["id"])])
        ]
    return {"summary": summary, "exact_records": exact, "fuzzy_records": fuzzy}


def load_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        frame = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    else:
        frame = pd.read_csv(path)
    if "id" not in frame:
        frame["id"] = range(len(frame))
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the public azminetoushikwasi BCS MCQ files."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mcq-json", type=Path, action="append", required=True)
    parser.add_argument("--base-submission", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    entries = [
        entry
        for path in args.mcq_json
        for entry in load_azmine_file(path)
    ]
    frame = load_frame(args.input)
    base_labels = None
    if args.base_submission:
        base = pd.read_csv(args.base_submission)
        base_labels = dict(zip(base["id"].astype(int), base["label"].astype(int)))
    payload = audit(frame, entries, base_labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
