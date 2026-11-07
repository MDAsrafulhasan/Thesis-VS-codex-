from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd

from datathon_utils import OUTPUT_DIR, bn_tokens, load_sample, load_test, normalize_bn


BN_DIGITS = "০১২৩৪৫৬৭৮৯"
AR_DIGITS = "0123456789"


def compact(text: str, max_len: int = 220) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text[:max_len]


def extract_numbers(text: str) -> list[str]:
    normalized = normalize_bn(text)
    return re.findall(r"\d+(?:\.\d+)?", normalized)


def corrupt_number_text(text: str, rng: random.Random) -> str | None:
    numbers = extract_numbers(text)
    if not numbers:
        return None
    target = rng.choice(numbers)
    try:
        value = int(float(target))
    except ValueError:
        return None
    delta = rng.choice([-3, -2, -1, 1, 2, 3, 5, 10])
    replacement = str(max(0, value + delta))
    normalized = normalize_bn(text)
    corrupted_norm = re.sub(rf"(?<!\d){re.escape(target)}(?!\d)", replacement, normalized, count=1)
    if corrupted_norm == normalized:
        return None
    return corrupted_norm


def sample_context_span(context: str, response: str, rng: random.Random) -> str:
    response = str(response).strip()
    context = str(context).strip()
    if response and response in context:
        return compact(response)
    tokens = bn_tokens(context)
    if not tokens:
        return compact(response)
    width = rng.randint(3, min(10, max(3, len(tokens))))
    start = rng.randint(0, max(0, len(tokens) - width))
    return " ".join(tokens[start : start + width])


def make_context_synthetic(sample: pd.DataFrame, test: pd.DataFrame, rng: random.Random) -> list[dict]:
    rows: list[dict] = []
    sample_ctx = sample[sample["has_context"]][["context", "prompt_bn", "response_bn", "category"]].copy()
    sample_ctx["origin"] = "sample"
    test_ctx = test[test["has_context"]][["context", "prompt_bn", "response_bn", "category"]].copy()
    test_ctx["origin"] = "test"
    ctx_df = pd.concat([sample_ctx, test_ctx], ignore_index=True)
    responses_by_cat = {
        cat: group["response_bn"].astype(str).tolist()
        for cat, group in ctx_df.groupby("category")
    }
    all_responses = ctx_df["response_bn"].astype(str).tolist()

    for item in ctx_df.itertuples(index=False):
        faithful = sample_context_span(item.context, item.response_bn, rng)
        if faithful:
            rows.append(
                {
                    "context": item.context,
                    "prompt_bn": item.prompt_bn,
                    "response_bn": faithful,
                    "label": 1,
                    "source": "synthetic_context_span",
                    "origin": item.origin,
                }
            )

        candidates = [x for x in responses_by_cat.get(item.category, all_responses) if normalize_bn(x) not in normalize_bn(item.context)]
        if candidates:
            rows.append(
                {
                    "context": item.context,
                    "prompt_bn": item.prompt_bn,
                    "response_bn": compact(rng.choice(candidates)),
                    "label": 0,
                    "source": "synthetic_context_swap",
                    "origin": item.origin,
                }
            )

        corrupted = corrupt_number_text(item.response_bn, rng) or corrupt_number_text(item.context, rng)
        if corrupted:
            rows.append(
                {
                    "context": item.context,
                    "prompt_bn": item.prompt_bn,
                    "response_bn": compact(corrupted),
                    "label": 0,
                    "source": "synthetic_number_corrupt",
                    "origin": item.origin,
                }
            )

    return rows


def make_no_context_synthetic(sample: pd.DataFrame, test: pd.DataFrame, rng: random.Random) -> list[dict]:
    rows: list[dict] = []
    # Use only released labeled no-context rows as reliable seeds; create nearby corruptions.
    noctx = sample[~sample["has_context"]].copy()
    by_cat = {cat: group["response_bn"].astype(str).tolist() for cat, group in noctx.groupby("category")}
    all_responses = noctx["response_bn"].astype(str).tolist()

    for item in noctx.itertuples(index=False):
        rows.append(
            {
                "context": "",
                "prompt_bn": item.prompt_bn,
                "response_bn": item.response_bn,
                "label": int(item.label),
                "source": "real_noctx_seed",
                "origin": "sample",
            }
        )
        candidates = [x for x in by_cat.get(item.category, all_responses) if normalize_bn(x) != normalize_bn(item.response_bn)]
        if candidates:
            rows.append(
                {
                    "context": "",
                    "prompt_bn": item.prompt_bn,
                    "response_bn": compact(rng.choice(candidates)),
                    "label": 0,
                    "source": "synthetic_noctx_swap",
                    "origin": "sample",
                }
            )
        corrupted = corrupt_number_text(item.response_bn, rng)
        if corrupted:
            rows.append(
                {
                    "context": "",
                    "prompt_bn": item.prompt_bn,
                    "response_bn": compact(corrupted),
                    "label": 0,
                    "source": "synthetic_noctx_number_corrupt",
                    "origin": "sample",
                }
            )

    # Pseudo-negative answer-to-wrong-question examples from test distribution.
    noctx_test = test[~test["has_context"]].copy()
    for cat, group in noctx_test.groupby("category"):
        values = group["response_bn"].astype(str).tolist()
        prompts = group["prompt_bn"].astype(str).tolist()
        if len(values) < 2:
            continue
        for prompt in prompts[: min(120, len(prompts))]:
            response = rng.choice(values)
            rows.append(
                {
                    "context": "",
                    "prompt_bn": prompt,
                    "response_bn": compact(response),
                    "label": 0,
                    "source": "synthetic_test_wrong_question",
                    "origin": "test",
                }
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(OUTPUT_DIR / "synthetic_train.jsonl"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    sample = load_sample()
    test = load_test()
    rows = make_context_synthetic(sample, test, rng)
    rows.extend(make_no_context_synthetic(sample, test, rng))
    rng.shuffle(rows)

    output = Path(args.output)
    output.parent.mkdir(exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("wrote", output, "rows", len(rows), "labels", pd.Series([r["label"] for r in rows]).value_counts().to_dict())


if __name__ == "__main__":
    main()
