from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from datathon_utils import load_sample, metric_row


def load_index_votes(path: Path) -> dict[int, int]:
    votes = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        index = item.get("i", item.get("row_index"))
        label = item.get("pred", item.get("prediction", item.get("new")))
        if index is not None and label in (0, 1, "0", "1"):
            votes[int(index)] = int(label)
    return votes


def agreement_report(truth: np.ndarray, a: dict[int, int], b: dict[int, int]) -> dict:
    indices = sorted(set(a) & set(b))
    agreed = [index for index in indices if a[index] == b[index]]
    predictions = np.asarray([a[index] for index in agreed], dtype=int)
    targets = truth[agreed]
    return {
        "joint_rows": len(indices),
        "agreement_rows": len(agreed),
        "agreement_rate": len(agreed) / max(1, len(indices)),
        "metrics_on_agreement": metric_row(targets, predictions) if agreed else None,
        "agreement_indices": agreed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cross-family judge agreement on the released sample.")
    parser.add_argument("--judge-a", type=Path, required=True)
    parser.add_argument("--judge-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sample = load_sample()
    report = agreement_report(sample.label.astype(int).to_numpy(), load_index_votes(args.judge_a), load_index_votes(args.judge_b))
    by_category = {}
    for category, part in sample.loc[report["agreement_indices"]].groupby("category"):
        idx = part.index.tolist()
        a = load_index_votes(args.judge_a)
        pred = np.asarray([a[index] for index in idx], dtype=int)
        by_category[str(category)] = {"rows": len(idx), **metric_row(part.label.astype(int).to_numpy(), pred)}
    report["by_category"] = by_category
    report.pop("agreement_indices", None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
