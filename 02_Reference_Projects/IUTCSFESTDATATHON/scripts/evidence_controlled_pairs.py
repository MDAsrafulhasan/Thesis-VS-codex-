from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from datathon_utils import OUTPUT_DIR, clean_text, normalize_bn


QUOTE_RE = re.compile(r'"([^"]+)"|“([^”]+)”|‘([^’]+)’')


def extract_term(question: str) -> str:
    match = QUOTE_RE.search(question)
    if match:
        return next(x for x in match.groups() if x)
    return clean_text(question).split(" এর ", 1)[0]


def lexical_task(question: str) -> str:
    if "ভাবার্থ" in question:
        return "figurative"
    if "শাব্দিক" in question:
        return "literal"
    if "বিপরীত" in question:
        return "antonym"
    if "সমার্থক" in question:
        return "synonym"
    return "meaning"


def compact(text: str, max_len: int = 450) -> str:
    text = re.sub(r"\s+", " ", clean_text(text)).strip()
    return text[:max_len]


def add_row(rows: list[dict], item, response: str, label: int, source: str) -> None:
    rows.append(
        {
            "context": clean_text(item.context),
            "prompt_bn": clean_text(item.question),
            "response_bn": compact(response),
            "label": int(label),
            "source": source,
            "origin": "external_evidence",
            "source_repo": clean_text(item.source_repo),
        }
    )


def build_lexical_pairs(corpus: pd.DataFrame, max_rows: int | None = None) -> pd.DataFrame:
    df = corpus.copy().fillna("")
    df["term"] = df["question"].map(extract_term)
    df["task"] = df["question"].map(lexical_task)
    if max_rows:
        df = df.head(max_rows)

    rows: list[dict] = []
    by_entry = {(row.source_repo, row.source_file, row.term, row.task): row for row in df.itertuples(index=False)}
    by_task_answers = {
        task: group["answer"].drop_duplicates().tolist()
        for task, group in df.groupby("task")
    }
    for idx, item in enumerate(df.itertuples(index=False)):
        add_row(rows, item, item.answer, 1, "lexical_evidence_faithful")

        other_task = "literal" if item.task == "figurative" else "figurative" if item.task == "literal" else None
        if other_task:
            sibling = by_entry.get((item.source_repo, item.source_file, item.term, other_task))
            if sibling is not None and normalize_bn(sibling.answer) != normalize_bn(item.answer):
                add_row(rows, item, sibling.answer, 0, "lexical_evidence_task_flip")

        pool = by_task_answers.get(item.task, [])
        if len(pool) > 1:
            wrong = pool[(idx * 7919 + 23) % len(pool)]
            if normalize_bn(wrong) == normalize_bn(item.answer):
                wrong = pool[(idx * 7919 + 24) % len(pool)]
            if normalize_bn(wrong) != normalize_bn(item.answer):
                add_row(rows, item, wrong, 0, "lexical_evidence_answer_swap")

    out = pd.DataFrame(rows).drop_duplicates(["prompt_bn", "response_bn", "label", "source"])
    return out


def sample_balanced(df: pd.DataFrame, max_per_label: int | None, seed: int) -> pd.DataFrame:
    if not max_per_label:
        return df
    parts = []
    for _, group in df.groupby("label"):
        if len(group) > max_per_label:
            group = group.sample(max_per_label, random_state=seed)
        parts.append(group)
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def write_jsonl(df: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output, orient="records", lines=True, force_ascii=False)
    df.to_csv(output.with_suffix(".csv"), index=False)
    print("wrote", output, output.with_suffix(".csv"), "rows", len(df))
    print("labels", df["label"].value_counts().sort_index().to_dict())
    print("sources", df["source"].value_counts().to_dict())
    print("repos", df["source_repo"].value_counts().head(10).to_dict())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lexical-corpus", default=str(OUTPUT_DIR / "lexical_evidence_corpus.jsonl"))
    parser.add_argument("--output", default=str(OUTPUT_DIR / "lexical_controlled_pairs.jsonl"))
    parser.add_argument("--max-source-rows", type=int)
    parser.add_argument("--max-per-label", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    corpus = pd.read_json(args.lexical_corpus, lines=True)
    pairs = build_lexical_pairs(corpus, max_rows=args.max_source_rows)
    pairs = sample_balanced(pairs, args.max_per_label, args.seed)
    write_jsonl(pairs, Path(args.output))


if __name__ == "__main__":
    main()
