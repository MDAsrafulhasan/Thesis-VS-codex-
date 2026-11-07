from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from datathon_utils import apply_context_rule, load_sample, load_test, normalize_bn, print_metrics


LEXICAL_CATEGORIES = {"meaning", "literal", "synonym", "antonym"}


def token_set(text: str) -> set[str]:
    return set(normalize_bn(text).split())


def overlap_score(response: str, answer: str) -> float:
    response_tokens = token_set(response)
    answer_tokens = token_set(answer)
    if not response_tokens or not answer_tokens:
        return 0.0
    return len(response_tokens & answer_tokens) / max(1, min(len(response_tokens), len(answer_tokens)))


def load_evidence(path: Path) -> dict[int, list[dict]]:
    rows: dict[int, list[dict]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[int(row["row_index"])] = row.get("evidence", [])
    return rows


def load_trace(path: Path, n_rows: int) -> np.ndarray:
    preds = np.full(n_rows, -1, dtype=int)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        preds[int(row["i"])] = int(row["pred"])
    return preds


def lexical_prediction(row, evidence: list[dict], threshold: float) -> tuple[int | None, float, int]:
    if row.category not in LEXICAL_CATEGORIES or row.has_context:
        return None, 0.0, 0
    exact = [item for item in evidence if item.get("exact_question")]
    if not exact:
        return None, 0.0, 0
    best = max(overlap_score(row.response_bn, item.get("answer", "")) for item in exact)
    return (1 if best >= threshold else 0), best, len(exact)


def command_apply(args) -> None:
    df = load_sample() if args.split == "sample" else load_test()
    evidence = load_evidence(Path(args.evidence_jsonl))

    if args.base_submission:
        sub = pd.read_csv(args.base_submission)
        labels = sub["label"].to_numpy(dtype=int).copy()
        if len(labels) != len(df):
            raise ValueError(f"base submission has {len(labels)} rows, expected {len(df)}")
    elif args.base_trace:
        base = load_trace(Path(args.base_trace), len(df))
        labels = np.where(base == -1, args.default_label, base)
        labels = apply_context_rule(df, labels)
    else:
        labels = np.full(len(df), args.default_label, dtype=int)
        labels = apply_context_rule(df, labels)

    overrides = []
    scores = np.zeros(len(df), dtype=float)
    for i, row in df.iterrows():
        pred, score, n_exact = lexical_prediction(row, evidence.get(int(i), []), args.threshold)
        scores[int(i)] = score
        if pred is not None:
            old = int(labels[int(i)])
            labels[int(i)] = int(pred)
            overrides.append({"row_index": int(i), "old": old, "new": int(pred), "score": score, "n_exact": n_exact})

    print("lexical overrides", len(overrides), "changed", sum(1 for item in overrides if item["old"] != item["new"]))
    print("label counts", pd.Series(labels).value_counts().sort_index().to_dict())
    if "label" in df.columns:
        print_metrics("lexical override", df["label"].to_numpy(), labels)
        lex_mask = (~df["has_context"].to_numpy()) & df["category"].isin(LEXICAL_CATEGORIES).to_numpy()
        print_metrics("lexical override / lexical no-context", df.loc[lex_mask, "label"].to_numpy(), labels[lex_mask])

    if args.output:
        if "id" not in df.columns:
            raise ValueError("CSV output requires split=test with id column")
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": df["id"].to_numpy(), "label": labels}).to_csv(output, index=False)
        print("wrote", output)
    if args.trace_output:
        output = Path(args.trace_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(overrides, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote", output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["sample", "test"], required=True)
    parser.add_argument("--evidence-jsonl", required=True)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--base-submission")
    parser.add_argument("--base-trace")
    parser.add_argument("--default-label", type=int, default=0)
    parser.add_argument("--output")
    parser.add_argument("--trace-output")
    args = parser.parse_args()
    command_apply(args)


if __name__ == "__main__":
    main()
