from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from datathon_utils import metric_row, normalize_bn


def row_key(row: pd.Series) -> str:
    fields = [normalize_bn(str(row.get(name, ""))) for name in ("context", "prompt_bn", "response_bn")]
    return hashlib.sha256("\x1f".join(fields).encode("utf-8")).hexdigest()


def build_released_lookup(released: pd.DataFrame) -> dict[str, int]:
    if "label" not in released:
        raise ValueError("released reference must contain label")
    lookup: dict[str, int] = {}
    for _, row in released.iterrows():
        key = row_key(row)
        label = int(row.label)
        if key in lookup and lookup[key] != label:
            raise ValueError("conflicting released labels for identical row")
        lookup[key] = label
    return lookup


def diagnostic_predictions(target: pd.DataFrame, lookup: dict[str, int]) -> list[int]:
    predictions = []
    for _, row in target.iterrows():
        key = row_key(row)
        if key not in lookup:
            raise ValueError("diagnostic target contains a row outside the released labeled set")
        predictions.append(lookup[key])
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-explicit released-set fit diagnostic; never for submissions.")
    parser.add_argument("--released", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    released = pd.read_json(args.released) if args.released.suffix == ".json" else pd.read_csv(args.released)
    target = pd.read_json(args.target) if args.target.suffix == ".json" else pd.read_csv(args.target)
    predictions = diagnostic_predictions(target, build_released_lookup(released))
    if "label" not in target:
        raise ValueError("diagnostic target must contain labels; refusing submission-like output")
    report = {
        "warning": "LEAKED DIAGNOSTIC: exact released-label lookup; not a generalization score and not submission-safe",
        "rows": len(target),
        "metrics": metric_row(target.label.astype(int).to_numpy(), predictions),
        "submission_safe": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
