from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from datathon_utils import load_test


def recover_seed_labels(frame: pd.DataFrame, variants_per_seed: int = 15) -> pd.DataFrame:
    """Recover the source pseudo-label represented by exact-weight variants.

    The public synthetic package contains exactly fifteen weight-1 paraphrases
    for each test seed. Lower-weight rows belong to a separate labeled-control
    augmentation and must not participate in reconstruction.
    """
    exact = frame[frame["sample_weight"].eq(1.0)]
    grouped = exact.groupby("seed_sample_index")["label"]
    counts = grouped.size()
    unique = grouped.nunique()
    if not counts.eq(variants_per_seed).all():
        bad = counts[counts.ne(variants_per_seed)].to_dict()
        raise ValueError(f"expected {variants_per_seed} exact variants per seed: {bad}")
    if not unique.eq(1).all():
        bad = unique[unique.ne(1)].to_dict()
        raise ValueError(f"conflicting pseudo-labels within seeds: {bad}")
    labels = grouped.first().rename("pseudo_label").reset_index()
    labels["id"] = labels["seed_sample_index"] + 1
    return labels[["id", "pseudo_label"]]


def load_uncertain_ids(path: Path | None) -> set[int]:
    if path is None:
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(item["id"]) for item in payload.get("uncertainty", [])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a test-derived synthetic pseudo-label vector.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--router-trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    pseudo = recover_seed_labels(pd.read_csv(args.train))
    submission = pd.read_csv(args.submission).rename(columns={"label": "submission_label"})
    test = load_test()
    merged = test.merge(submission[["id", "submission_label"]], on="id").merge(pseudo, on="id")
    merged["disagrees"] = merged["submission_label"].ne(merged["pseudo_label"])
    uncertain = load_uncertain_ids(args.router_trace)
    merged["router_uncertain"] = merged["id"].isin(uncertain)
    disagreements = merged[merged["disagrees"]].copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    disagreements.to_csv(args.output, index=False)
    summary = {
        "rows": len(merged),
        "disagreements": len(disagreements),
        "directions": {
            f"{old}->{new}": int(count)
            for (old, new), count in disagreements.groupby(
                ["submission_label", "pseudo_label"]
            ).size().items()
        },
        "uncertain_disagreements": int(disagreements["router_uncertain"].sum()),
        "warning": (
            "These are participant pseudo-labels, not competition truth. "
            "Use only as a review queue and independently verify every change."
        ),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
