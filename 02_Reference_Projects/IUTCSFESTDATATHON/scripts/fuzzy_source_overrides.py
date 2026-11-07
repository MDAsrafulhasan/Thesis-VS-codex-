from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from datathon_utils import OUTPUT_DIR, load_sample, load_test, metric_row, normalize_bn
from source_answer_bank import (
    BankEntry,
    Decision,
    decide_row,
    index_bank,
    number_signature,
    read_bank,
)


TOKEN_RE = re.compile(r"[0-9a-z\u0980-\u09ff]+")


@dataclass(frozen=True)
class MatchFeatures:
    cosine: float
    margin: float
    token_jaccard: float
    token_coverage: float
    sequence_ratio: float
    same_numbers: bool
    same_relation: bool


RELATION_PATTERNS = {
    "phase:start": re.compile(r"(?:শুরু|আরম্ভ|সূচনা|সূচিত)"),
    "phase:end": re.compile(r"(?:শেষ|সমাপ্ত|অবসান)"),
    "life:birth": re.compile(r"(?:জন্ম|জন্মগ্রহণ)"),
    "life:death": re.compile(r"(?:মৃত্যু|মারা যান|মৃত্যুবরণ)"),
    "work:written": re.compile(r"(?:রচনা|লিখে|লেখেন)"),
    "work:published": re.compile(r"(?:প্রকাশ|প্রকাশিত|মুদ্রিত)"),
    "place:source": re.compile(r"(?:উৎস|উৎপত্তি)"),
    "place:confluence": re.compile(r"(?:মিলিত|মিশে|সঙ্গম)"),
}


def relation_signature(text: str) -> frozenset[str]:
    normalized = normalize_bn(text)
    return frozenset(name for name, pattern in RELATION_PATTERNS.items() if pattern.search(normalized))


def relation_compatible(query: str, candidate: str) -> bool:
    """Reject near-duplicate questions that swap a material relation."""
    query_relations = relation_signature(query)
    candidate_relations = relation_signature(candidate)
    if not query_relations and not candidate_relations:
        return True
    relation_families = ("phase:", "life:", "work:", "place:")
    for family in relation_families:
        query_family = {value for value in query_relations if value.startswith(family)}
        candidate_family = {value for value in candidate_relations if value.startswith(family)}
        if query_family or candidate_family:
            if query_family != candidate_family:
                return False
    return True


def token_set(text: str) -> set[str]:
    return set(TOKEN_RE.findall(normalize_bn(text)))


def match_features(query: str, candidate: str, cosine: float, runner_up: float) -> MatchFeatures:
    query_tokens = token_set(query)
    candidate_tokens = token_set(candidate)
    intersection = len(query_tokens & candidate_tokens)
    return MatchFeatures(
        cosine=float(cosine),
        margin=float(cosine - runner_up),
        token_jaccard=intersection / max(1, len(query_tokens | candidate_tokens)),
        token_coverage=intersection / max(1, min(len(query_tokens), len(candidate_tokens))),
        sequence_ratio=SequenceMatcher(None, normalize_bn(query), normalize_bn(candidate)).ratio(),
        same_numbers=number_signature(query) == number_signature(candidate),
        same_relation=relation_compatible(query, candidate),
    )


def eligible(
    features: MatchFeatures,
    *,
    min_cosine: float = 0.90,
    min_jaccard: float = 0.65,
    min_sequence: float = 0.82,
) -> bool:
    # Numeric equality is deliberately mandatory. Character retrieval otherwise
    # confuses arithmetic/current-affairs templates whose only changed fact is a
    # year, amount, rate, or inequality bound.
    return bool(
        features.same_numbers
        and features.same_relation
        and features.cosine >= min_cosine
        and features.token_jaccard >= min_jaccard
        and features.sequence_ratio >= min_sequence
    )


class FuzzyMcqMatcher:
    def __init__(
        self,
        entries: list[BankEntry],
        max_features: int = 180_000,
        include_open_qa: bool = False,
    ):
        grouped: dict[str, list[BankEntry]] = {}
        for entry in entries:
            if entry.source_kind == "mcq" or include_open_qa:
                grouped.setdefault(normalize_bn(entry.question), []).append(entry)
        self.grouped = grouped
        self.questions = list(grouped)
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=max_features,
            min_df=1,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.matrix = self.vectorizer.fit_transform(self.questions)

    def match_rows(
        self,
        df: pd.DataFrame,
        *,
        exclude_same_question: bool,
        min_cosine: float,
        min_jaccard: float,
        min_sequence: float,
        batch_size: int = 128,
    ) -> list[dict]:
        query_matrix = self.vectorizer.transform(df["prompt_bn"].tolist())
        records: list[dict] = []
        top_k = min(20, len(self.questions))
        for start in range(0, len(df), batch_size):
            scores = linear_kernel(query_matrix[start : start + batch_size], self.matrix)
            for offset, row_scores in enumerate(scores):
                row = df.iloc[start + offset]
                normalized_query = normalize_bn(row.prompt_bn)
                order = np.argpartition(row_scores, -top_k)[-top_k:]
                order = order[np.argsort(row_scores[order])[::-1]]
                candidates = [
                    int(index)
                    for index in order
                    if not (exclude_same_question and self.questions[int(index)] == normalized_query)
                ]
                if len(candidates) < 2:
                    continue
                best_index, runner_index = candidates[:2]
                matched_key = self.questions[best_index]
                features = match_features(
                    str(row.prompt_bn),
                    matched_key,
                    float(row_scores[best_index]),
                    float(row_scores[runner_index]),
                )
                if not eligible(
                    features,
                    min_cosine=min_cosine,
                    min_jaccard=min_jaccard,
                    min_sequence=min_sequence,
                ):
                    continue
                decision, all_decisions = decide_row(str(row.response_bn), self.grouped[matched_key])
                if decision is None:
                    continue
                records.append(
                    {
                        "id": int(row.id),
                        "prompt": str(row.prompt_bn),
                        "response": str(row.response_bn),
                        "matched_question": self.grouped[matched_key][0].question,
                        "features": asdict(features),
                        "decision": asdict(decision),
                        "all_decisions": [asdict(item) for item in all_decisions],
                    }
                )
        return records


def apply_records(base: pd.DataFrame, records: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    output = base.copy()
    trace = []
    for record in records:
        mask = output["id"] == record["id"]
        if mask.sum() != 1:
            raise ValueError(f"submission does not contain exactly one row for id={record['id']}")
        old = int(output.loc[mask, "label"].iloc[0])
        new = int(record["decision"]["label"])
        output.loc[mask, "label"] = new
        trace.append({**record, "old": old, "new": new, "changed": old != new})
    return output, trace


def main() -> None:
    parser = argparse.ArgumentParser(description="High-precision near-duplicate MCQ source overrides.")
    parser.add_argument(
        "--bank",
        type=Path,
        action="append",
        help="Answer bank JSONL; repeat to search several banks jointly.",
    )
    parser.add_argument(
        "--exclude-source",
        action="append",
        default=[],
        help="Source repository to exclude; repeat as needed.",
    )
    parser.add_argument("--split", choices=["sample", "test"], required=True)
    parser.add_argument("--base-submission", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--min-cosine", type=float, default=0.90)
    parser.add_argument("--min-jaccard", type=float, default=0.65)
    parser.add_argument("--min-sequence", type=float, default=0.82)
    parser.add_argument("--max-features", type=int, default=180_000)
    parser.add_argument("--include-open-qa", action="store_true")
    args = parser.parse_args()

    bank_paths = args.bank or [OUTPUT_DIR / "source_answer_bank.jsonl"]
    entries = [entry for path in bank_paths for entry in read_bank(path)]
    excluded_sources = set(args.exclude_source)
    entries = [entry for entry in entries if entry.source_repo not in excluded_sources]
    exact_index = index_bank(entries)
    df = load_sample() if args.split == "sample" else load_test()
    if "id" not in df.columns:
        df = df.copy()
        df["id"] = np.arange(len(df), dtype=int)
    if args.split == "test":
        # The Phase-1 exam/GK block is the only target. Exact matches are already
        # resolved by source_answer_bank.py and must retain priority.
        df = df[(df["id"] >= 396) & (df["id"] <= 2104)].copy()
        df = df[~df["prompt_bn"].map(normalize_bn).isin(exact_index)].reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    matcher = FuzzyMcqMatcher(
        entries,
        max_features=args.max_features,
        include_open_qa=args.include_open_qa,
    )
    records = matcher.match_rows(
        df,
        exclude_same_question=True,
        min_cosine=args.min_cosine,
        min_jaccard=args.min_jaccard,
        min_sequence=args.min_sequence,
    )
    summary: dict[str, object] = {
        "split": args.split,
        "candidate_rows": len(df),
        "accepted": len(records),
        "thresholds": {
            "min_cosine": args.min_cosine,
            "min_jaccard": args.min_jaccard,
            "min_sequence": args.min_sequence,
            "same_numbers": True,
            "include_open_qa": args.include_open_qa,
        },
        "banks": [str(path) for path in bank_paths],
        "excluded_sources": sorted(excluded_sources),
        "label_counts": pd.Series([record["decision"]["label"] for record in records])
        .value_counts()
        .sort_index()
        .to_dict(),
    }
    if args.split == "sample":
        truth = df.set_index("id")["label"]
        y_true = np.array([int(truth.loc[record["id"]]) for record in records])
        y_pred = np.array([int(record["decision"]["label"]) for record in records])
        if len(records):
            summary["metrics"] = metric_row(y_true, y_pred)
            summary["correct"] = int((y_true == y_pred).sum())
            for record, expected in zip(records, y_true):
                record["truth"] = int(expected)
                record["correct"] = int(record["decision"]["label"]) == int(expected)
    elif args.base_submission and args.output:
        base = pd.read_csv(args.base_submission)
        output, records = apply_records(base, records)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(args.output, index=False)
        summary["changed"] = sum(record["changed"] for record in records)
        summary["output"] = str(args.output)
        summary["output_counts"] = output["label"].value_counts().sort_index().to_dict()
    else:
        raise ValueError("test split requires --base-submission and --output")

    args.trace.parent.mkdir(parents=True, exist_ok=True)
    args.trace.write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote", args.trace)


if __name__ == "__main__":
    main()
