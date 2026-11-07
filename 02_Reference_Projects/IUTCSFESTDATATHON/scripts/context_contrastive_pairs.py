from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd

from datathon_utils import OUTPUT_DIR, bn_tokens, clean_text, normalize_bn, prompt_category
from train_verifier import extract_numbers


def compact(text: str, max_len: int = 360) -> str:
    text = re.sub(r"\s+", " ", clean_text(text))
    return text[:max_len]


def corrupt_number(text: str, rng: random.Random) -> str:
    numbers = extract_numbers(text)
    if not numbers:
        return ""
    target = rng.choice(numbers)
    try:
        value = int(float(target))
    except ValueError:
        return ""
    delta = rng.choice([-10, -5, -3, -2, -1, 1, 2, 3, 5, 10])
    replacement = str(max(0, value + delta))
    return re.sub(rf"(?<!\d){re.escape(target)}(?!\d)", replacement, normalize_bn(text), count=1)


def context_span(context: str, answer: str, rng: random.Random) -> str:
    context = clean_text(context)
    answer = clean_text(answer)
    if answer and answer in context:
        idx = context.find(answer)
        left = max(0, idx - 80)
        right = min(len(context), idx + len(answer) + 120)
        return compact(context[left:right])
    tokens = bn_tokens(context)
    if len(tokens) < 5:
        return compact(answer)
    width = min(len(tokens), rng.randint(6, 18))
    start = rng.randint(0, max(0, len(tokens) - width))
    return " ".join(tokens[start : start + width])


def load_corpus(path: Path, max_rows: int | None, seed: int) -> pd.DataFrame:
    df = pd.read_json(path, lines=True).fillna("")
    for col in ["question", "answer", "context", "source_repo"]:
        if col not in df.columns:
            df[col] = ""
    df = df[df["question"].astype(str).str.len().gt(0) & df["answer"].astype(str).str.len().gt(0)]
    df = df[df["context"].astype(str).str.len().gt(0)].copy()
    df["category"] = df["question"].map(prompt_category)
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed)
    return df.reset_index(drop=True)


def build_answer_pools(df: pd.DataFrame) -> dict[tuple[str, str], list[str]]:
    pools: dict[tuple[str, str], list[str]] = {}
    for key, group in df.groupby(["source_repo", "category"]):
        answers = group["answer"].map(compact).drop_duplicates().tolist()
        if len(answers) >= 2:
            pools[key] = answers
    return pools


def wrong_answer(item: dict, pools: dict[tuple[str, str], list[str]], rng: random.Random) -> str:
    key = (item["source_repo"], item["category"])
    candidates = pools.get(key, [])
    if not candidates:
        return ""
    answer_norm = normalize_bn(item["answer"])
    context_norm = normalize_bn(item["context"])
    for _ in range(30):
        candidate = rng.choice(candidates)
        norm = normalize_bn(candidate)
        if norm != answer_norm and norm not in context_norm:
            return compact(candidate)
    return ""


def make_pairs(df: pd.DataFrame, seed: int, max_per_label: int | None) -> pd.DataFrame:
    rng = random.Random(seed)
    pools = build_answer_pools(df)
    rows: list[dict] = []
    for item in df.to_dict("records"):
        context = compact(item["context"], max_len=1400)
        question = compact(item["question"])
        answer = compact(item["answer"])
        if not question or not answer or not context:
            continue
        rows.append(
            {
                "context": context,
                "prompt_bn": question,
                "response_bn": answer,
                "label": 1,
                "source": "external_context_faithful_answer",
                "source_repo": item["source_repo"],
            }
        )
        span = context_span(context, answer, rng)
        if span and normalize_bn(span) != normalize_bn(answer):
            rows.append(
                {
                    "context": context,
                    "prompt_bn": question,
                    "response_bn": span,
                    "label": 1,
                    "source": "external_context_supported_span",
                    "source_repo": item["source_repo"],
                }
            )
        wrong = wrong_answer(item, pools, rng)
        if wrong:
            rows.append(
                {
                    "context": context,
                    "prompt_bn": question,
                    "response_bn": wrong,
                    "label": 0,
                    "source": "external_context_unsupported_swap",
                    "source_repo": item["source_repo"],
                }
            )
        corrupted = corrupt_number(answer, rng) or corrupt_number(context, rng)
        if corrupted and normalize_bn(corrupted) != normalize_bn(answer):
            rows.append(
                {
                    "context": context,
                    "prompt_bn": question,
                    "response_bn": compact(corrupted),
                    "label": 0,
                    "source": "external_context_number_corrupt",
                    "source_repo": item["source_repo"],
                }
            )

    out = pd.DataFrame(rows).drop_duplicates(["context", "prompt_bn", "response_bn", "label"]).reset_index(drop=True)
    if max_per_label:
        parts = []
        for label, group in out.groupby("label"):
            take = min(len(group), max_per_label)
            parts.append(group.sample(take, random_state=seed + int(label)))
        out = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(OUTPUT_DIR / "external_qa_corpus.jsonl"))
    parser.add_argument("--output-prefix", default=str(OUTPUT_DIR / "external_context_contrastive_pairs"))
    parser.add_argument("--max-source-rows", type=int, default=120_000)
    parser.add_argument("--max-per-label", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    corpus = load_corpus(Path(args.corpus), args.max_source_rows, args.seed)
    pairs = make_pairs(corpus, args.seed, args.max_per_label)
    output = Path(args.output_prefix)
    output.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_json(output.with_suffix(".jsonl"), orient="records", lines=True, force_ascii=False)
    pairs.to_csv(output.with_suffix(".csv"), index=False)
    print("source rows", len(corpus))
    print("wrote", output.with_suffix(".jsonl"), output.with_suffix(".csv"), "rows", len(pairs))
    print("labels", pairs["label"].value_counts().sort_index().to_dict())
    print("sources", pairs["source"].value_counts().to_dict())


if __name__ == "__main__":
    main()
