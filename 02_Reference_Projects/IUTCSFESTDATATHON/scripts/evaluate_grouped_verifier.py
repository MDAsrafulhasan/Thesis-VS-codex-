from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from train_portable_verifier import load_external_training, make_pipeline, prepare_frame


def metrics(y_true, pred) -> dict:
    return {
        "n": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "f1_0": float(f1_score(y_true, pred, labels=[0], average="macro", zero_division=0)),
        "f1_1": float(f1_score(y_true, pred, labels=[1], average="macro", zero_division=0)),
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(pred))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Leave-source/domain-out validation for the portable verifier.")
    parser.add_argument("--train-file", type=Path, action="append", required=True)
    parser.add_argument("--group-column", choices=["source", "domain", "source_file"], default="source")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-rows-per-file", type=int, default=30_000)
    parser.add_argument("--max-features", type=int, default=80_000)
    parser.add_argument("--c", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    df = load_external_training(args.train_file, args.max_rows_per_file, 42)
    if args.group_column not in df:
        raise ValueError(f"missing group column {args.group_column}")
    groups = df[args.group_column].astype(str).replace("", "unknown")
    unique = groups.nunique()
    folds = min(args.folds, unique)
    if folds < 2:
        raise ValueError(f"need at least two {args.group_column} groups, got {unique}")
    splitter = GroupKFold(n_splits=folds)
    oof = np.zeros(len(df), dtype=int)
    fold_records = []
    y = df.label.astype(int).to_numpy()
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(df, y, groups), 1):
        model = make_pipeline(args.c, args.max_features)
        model.fit(df.iloc[train_idx], y[train_idx])
        pred = model.predict(df.iloc[valid_idx]).astype(int)
        oof[valid_idx] = pred
        record = metrics(y[valid_idx], pred)
        record.update({
            "fold": fold,
            "held_out_groups": sorted(groups.iloc[valid_idx].unique().tolist()),
            "train_rows": int(len(train_idx)),
        })
        fold_records.append(record)
        print("fold", fold, record)
    report = {
        "group_column": args.group_column,
        "groups": int(unique),
        "folds": fold_records,
        "overall": metrics(y, oof),
        "by_domain": {},
    }
    if "domain" in df:
        for domain, part in df.groupby("domain"):
            report["by_domain"][str(domain)] = metrics(y[part.index], oof[part.index])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
