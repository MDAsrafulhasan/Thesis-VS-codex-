from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from datathon_utils import OUTPUT_DIR, apply_context_rule, load_test
import train_portable_verifier as portable


def register_joblib_functions() -> None:
    main_module = sys.modules["__main__"]
    for name in ["text_all", "text_prompt_response", "numeric_features"]:
        setattr(main_module, name, getattr(portable, name))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(OUTPUT_DIR / "portable_verifier.joblib"))
    parser.add_argument("--base-submission", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--low", type=float, default=0.25)
    parser.add_argument("--high", type=float, default=0.75)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--no-context-only", action="store_true")
    args = parser.parse_args()

    register_joblib_functions()
    bundle = joblib.load(args.model)
    model = bundle["model"]
    threshold = float(args.threshold if args.threshold is not None else bundle["threshold"])

    test = portable.prepare_frame(load_test()).fillna("")
    base = pd.read_csv(args.base_submission)
    pred = base["label"].astype(int).to_numpy().copy()
    original = pred.copy()
    proba = model.predict_proba(test)[:, 1]
    mask = (proba <= args.low) | (proba >= args.high)
    if args.no_context_only:
        mask &= test["context"].astype(str).str.len().to_numpy() == 0
    pred[mask] = (proba[mask] >= threshold).astype(int)
    pred = apply_context_rule(test, pred)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": base["id"].to_numpy(), "label": pred}).to_csv(output, index=False)

    print("wrote", output)
    print(
        "threshold",
        threshold,
        "low",
        args.low,
        "high",
        args.high,
        "selected",
        int(mask.sum()),
        "changed",
        int((pred != original).sum()),
    )
    print("counts", pd.Series(pred).value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
