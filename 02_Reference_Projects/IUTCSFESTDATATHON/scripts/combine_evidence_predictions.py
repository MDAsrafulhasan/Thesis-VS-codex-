from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from datathon_utils import OUTPUT_DIR, apply_context_rule, load_sample, load_test, print_metrics


def load_trace(path: Path, n_rows: int) -> np.ndarray:
    preds = np.full(n_rows, -1, dtype=int)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        preds[int(row["i"])] = int(row["pred"])
    return preds


def load_top_scores(path: Path, n_rows: int) -> np.ndarray:
    scores = np.zeros(n_rows, dtype=float)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        evidence = row.get("evidence", [])
        scores[int(row["row_index"])] = float(evidence[0]["score"]) if evidence else 0.0
    return scores


def combine(args) -> None:
    df = load_sample() if args.split == "sample" else load_test()
    base = load_trace(Path(args.base_trace), len(df))
    evidence = load_trace(Path(args.evidence_trace), len(df))
    scores = load_top_scores(Path(args.evidence_jsonl), len(df))
    has_context = df["has_context"].to_numpy()
    eligible = (~has_context) & (base != -1) & (evidence != -1) & (scores >= args.score_threshold)

    labels = np.where(base == -1, args.default_label, base)
    labels[eligible] = evidence[eligible]
    labels = apply_context_rule(df, labels)

    print("eligible evidence overrides", int(eligible.sum()), "threshold", args.score_threshold)
    print("label counts", pd.Series(labels).value_counts().sort_index().to_dict())
    if "label" in df.columns:
        print_metrics("combined", df["label"].to_numpy(), labels)
        for threshold in args.sweep_threshold:
            sweep_eligible = (~has_context) & (base != -1) & (evidence != -1) & (scores >= threshold)
            sweep_labels = np.where(base == -1, args.default_label, base)
            sweep_labels[sweep_eligible] = evidence[sweep_eligible]
            sweep_labels = apply_context_rule(df, sweep_labels)
            print_metrics(f"combined threshold={threshold} overrides={sweep_eligible.sum()}", df["label"].to_numpy(), sweep_labels)

    if args.output:
        if "id" not in df.columns:
            raise ValueError("submission output requires split=test with id column")
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": df["id"].to_numpy(), "label": labels}).to_csv(output, index=False)
        print("wrote", output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["sample", "test"], required=True)
    parser.add_argument("--base-trace", required=True)
    parser.add_argument("--evidence-trace", required=True)
    parser.add_argument("--evidence-jsonl", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--default-label", type=int, default=0)
    parser.add_argument("--output")
    parser.add_argument("--sweep-threshold", type=float, action="append", default=[])
    args = parser.parse_args()
    combine(args)


if __name__ == "__main__":
    main()
