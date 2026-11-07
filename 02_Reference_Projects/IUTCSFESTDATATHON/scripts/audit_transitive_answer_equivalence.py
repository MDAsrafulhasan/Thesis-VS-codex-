from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from datathon_utils import load_sample, load_test, metric_row, normalize_bn
from source_answer_bank import (
    BankEntry,
    compare_response,
    index_bank,
    read_bank,
    source_answers_conflict,
)


@dataclass(frozen=True)
class Calibration:
    threshold: float | None
    selected: int
    positive: int
    negative: int
    precision: float | None
    reason: str


def answer_bank_abstentions(
    frame: pd.DataFrame,
    entries: Iterable[BankEntry],
    *,
    exclude_conflicts: bool,
) -> list[dict]:
    """Return exact-question rows that lexical answer comparison cannot decide.

    Conflicted answer banks are excluded because a high semantic score against
    an incorrect repository answer is not evidence of faithfulness.
    """
    indexed = index_bank(list(entries))
    records: list[dict] = []
    for row_index, row in frame.iterrows():
        sources = indexed.get(normalize_bn(str(row.prompt_bn)), [])
        conflict = source_answers_conflict(sources) if sources else False
        if not sources or (exclude_conflicts and conflict):
            continue
        decisions = [
            decision
            for entry in sources
            if (decision := compare_response(str(row.response_bn), entry)) is not None
        ]
        labels = {decision.label for decision in decisions if decision.confidence >= 0.90}
        if not conflict and len(labels) == 1 and decisions:
            continue
        answers = sorted(
            {
                str(answer).strip()
                for entry in sources
                for answer in entry.answers
                if str(answer).strip()
            }
        )
        records.append(
            {
                "row_index": int(row_index),
                "id": int(row.id) if "id" in frame.columns else int(row_index + 1),
                "prompt_bn": str(row.prompt_bn),
                "response_bn": str(row.response_bn),
                "answers": answers,
                "category": str(row.category),
                "has_context": bool(row.has_context),
                "source_conflict": conflict,
                **({"truth": int(row.label)} if "label" in frame.columns else {}),
            }
        )
    return records


def no_conflict_abstentions(
    frame: pd.DataFrame,
    entries: Iterable[BankEntry],
) -> list[dict]:
    return answer_bank_abstentions(frame, entries, exclude_conflicts=True)


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def load_encoder(model_path: Path):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval()
    return tokenizer, model


def embed_texts(texts: list[str], tokenizer, model, batch_size: int = 32) -> np.ndarray:
    import torch
    from torch.nn.functional import normalize

    vectors = []
    for offset in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[offset : offset + batch_size],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        with torch.inference_mode():
            hidden = model(**batch).last_hidden_state
        pooled = mean_pool(hidden, batch["attention_mask"])
        vectors.append(normalize(pooled).cpu().numpy())
    return np.concatenate(vectors, axis=0) if vectors else np.empty((0, 0))


def add_similarity_scores(
    records: list[dict],
    tokenizer,
    model,
    batch_size: int = 32,
) -> None:
    pairs = [
        (record_index, answer)
        for record_index, record in enumerate(records)
        for answer in record["answers"]
    ]
    if not pairs:
        return
    query_vectors = embed_texts(
        ["query: " + records[index]["response_bn"] for index, _ in pairs],
        tokenizer,
        model,
        batch_size,
    )
    answer_vectors = embed_texts(
        ["passage: " + answer for _, answer in pairs],
        tokenizer,
        model,
        batch_size,
    )
    similarities = np.sum(query_vectors * answer_vectors, axis=1)
    best: dict[int, tuple[float, str]] = {}
    for (record_index, answer), score in zip(pairs, similarities):
        if record_index not in best or score > best[record_index][0]:
            best[record_index] = (float(score), answer)
    for index, record in enumerate(records):
        score, answer = best[index]
        record["similarity"] = score
        record["nearest_answer"] = answer


def calibrate_threshold(
    scores: Iterable[float],
    truths: Iterable[int],
    *,
    min_precision: float = 1.0,
    min_positive_controls: int = 3,
) -> Calibration:
    paired = sorted(
        ((float(score), int(truth)) for score, truth in zip(scores, truths)),
        reverse=True,
    )
    if not paired:
        return Calibration(None, 0, 0, 0, None, "no_controls")

    candidates: list[Calibration] = []
    thresholds = sorted({score for score, _ in paired}, reverse=True)
    for threshold in thresholds:
        selected = [truth for score, truth in paired if score >= threshold]
        positive = sum(selected)
        negative = len(selected) - positive
        precision = positive / len(selected)
        if precision >= min_precision and positive >= min_positive_controls:
            candidates.append(
                Calibration(
                    threshold,
                    len(selected),
                    positive,
                    negative,
                    precision,
                    "passed",
                )
            )
    if not candidates:
        return Calibration(
            None,
            0,
            0,
            0,
            None,
            "insufficient_zero_false_positive_controls",
        )
    # Use the least restrictive threshold that still clears the precision
    # requirement, maximizing covered controls without test-set tuning.
    return max(candidates, key=lambda item: (item.selected, -float(item.threshold)))


def summarize_records(records: list[dict]) -> dict:
    scores = [record["similarity"] for record in records]
    return {
        "rows": len(records),
        "similarity_min": min(scores) if scores else None,
        "similarity_max": max(scores) if scores else None,
        "similarity_mean": float(np.mean(scores)) if scores else None,
    }


def load_judge_trace(path: Path) -> dict[int, dict]:
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            records[int(row["id"])] = row
    return records


def summarize_judge(
    records: list[dict],
    trace: dict[int, dict],
    *,
    sample: bool,
    base_labels: dict[int, int] | None = None,
) -> dict:
    joined = []
    for record in records:
        trace_key = record["row_index"] if sample else record["id"]
        prediction = trace.get(trace_key)
        if prediction is None:
            continue
        item = {
            **record,
            "judge_label": int(prediction["pred"]),
            "judge_confidence": float(prediction.get("confidence", 0.0)),
        }
        if base_labels is not None:
            item["v85_label"] = int(base_labels[record["id"]])
            item["disagrees_v85"] = item["judge_label"] != item["v85_label"]
        joined.append(item)

    result = {"rows": len(joined)}
    if sample and joined:
        truth = np.asarray([record["truth"] for record in joined], dtype=int)
        pred = np.asarray([record["judge_label"] for record in joined], dtype=int)
        result["metrics"] = metric_row(truth, pred)
        result["wrong"] = [
            record for record in joined if record["truth"] != record["judge_label"]
        ]
    if base_labels is not None:
        disagreements = [record for record in joined if record["disagrees_v85"]]
        result["disagreements"] = len(disagreements)
        result["zero_to_one"] = sum(
            record["v85_label"] == 0 and record["judge_label"] == 1
            for record in disagreements
        )
        result["one_to_zero"] = sum(
            record["v85_label"] == 1 and record["judge_label"] == 0
            for record in disagreements
        )
        result["disagreement_records"] = disagreements
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit semantic answer equivalence only among exact-question, "
            "non-conflicted source-bank abstentions."
        )
    )
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base-submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-positive-controls", type=int, default=3)
    parser.add_argument("--judge-sample-trace", type=Path)
    parser.add_argument("--judge-test-trace", type=Path)
    args = parser.parse_args()

    entries = read_bank(args.bank)
    sample = load_sample()
    test = load_test()
    sample_records = no_conflict_abstentions(sample, entries)
    test_records = no_conflict_abstentions(test, entries)
    tokenizer, model = load_encoder(args.model)
    add_similarity_scores(sample_records, tokenizer, model, args.batch_size)
    add_similarity_scores(test_records, tokenizer, model, args.batch_size)

    calibration = calibrate_threshold(
        [record["similarity"] for record in sample_records],
        [record["truth"] for record in sample_records],
        min_precision=1.0,
        min_positive_controls=args.min_positive_controls,
    )

    base = pd.read_csv(args.base_submission)
    labels = dict(base[["id", "label"]].itertuples(index=False, name=None))
    for record in test_records:
        record["v85_label"] = int(labels[record["id"]])

    candidates = []
    if calibration.threshold is not None:
        candidates = [
            record
            for record in test_records
            if record["v85_label"] == 0
            and record["similarity"] >= calibration.threshold
        ]

    report = {
        "method": {
            "model": str(args.model),
            "scope": "exact-question non-conflicted answer-bank abstentions",
            "promotion": "v85 label 0 to label 1 only",
            "min_precision": 1.0,
            "min_positive_controls": args.min_positive_controls,
        },
        "sample": summarize_records(sample_records),
        "test": summarize_records(test_records),
        "calibration": asdict(calibration),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "sample_records": sample_records,
        "test_records": test_records,
    }
    if args.judge_sample_trace or args.judge_test_trace:
        all_sample_records = answer_bank_abstentions(
            sample,
            entries,
            exclude_conflicts=False,
        )
        all_test_records = answer_bank_abstentions(
            test,
            entries,
            exclude_conflicts=False,
        )
        report["judge"] = {}
        if args.judge_sample_trace:
            report["judge"]["sample"] = summarize_judge(
                all_sample_records,
                load_judge_trace(args.judge_sample_trace),
                sample=True,
            )
        if args.judge_test_trace:
            report["judge"]["test"] = summarize_judge(
                all_test_records,
                load_judge_trace(args.judge_test_trace),
                sample=False,
                base_labels=labels,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    console = {
        key: report[key]
        for key in ("sample", "test", "calibration", "candidate_count")
    }
    if "judge" in report:
        console["judge"] = {
            split: {
                key: value
                for key, value in summary.items()
                if key not in {"wrong", "disagreement_records"}
            }
            for split, summary in report["judge"].items()
        }
    print(json.dumps(console, indent=2))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
