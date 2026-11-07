from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from datathon_utils import clean_context, metric_row


def read_predictions(path: Path) -> dict[int, int]:
    predictions: dict[int, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        predictions[int(item["i"])] = int(item["pred"])
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a selective context/no-context judge behind the Phase 2 router.")
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--router-submission", type=Path, required=True)
    parser.add_argument("--router-trace", type=Path, required=True)
    parser.add_argument("--context-judge", type=Path, required=True)
    parser.add_argument("--no-context-judge", type=Path, action="append", required=True)
    parser.add_argument("--no-context-policy", choices=["single", "unanimous-positive"], default="single")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sample = pd.DataFrame(json.loads(args.sample.read_text(encoding="utf-8")))
    router = pd.read_csv(args.router_submission)
    trace = json.loads(args.router_trace.read_text(encoding="utf-8"))
    if len(sample) != len(router):
        raise ValueError("sample and router submission lengths differ")

    context_predictions = read_predictions(args.context_judge)
    no_context_predictions = [read_predictions(path) for path in args.no_context_judge]
    if args.no_context_policy == "single" and len(no_context_predictions) != 1:
        raise ValueError("single no-context policy requires exactly one judge trace")
    uncertain = {int(item["row_index"]) for item in trace["uncertainty"]}
    labels = router.label.astype(int).to_numpy(copy=True)
    selected: list[dict] = []
    missing: list[int] = []
    for i in sorted(uncertain):
        has_context = bool(clean_context(sample.iloc[i]["context"]))
        sources = [context_predictions] if has_context else no_context_predictions
        if any(i not in source for source in sources):
            missing.append(i)
            continue
        old = int(labels[i])
        votes = [source[i] for source in sources]
        labels[i] = votes[0] if has_context or args.no_context_policy == "single" else int(all(votes))
        selected.append({
            "row_index": i, "has_context": has_context, "old": old,
            "new": int(labels[i]), "votes": votes,
        })

    truth = sample.label.astype(int).to_numpy()
    covered = sorted(set(range(len(sample))) - uncertain)
    payload = {
        "rows": len(sample),
        "uncertain_rows": len(uncertain),
        "judge_covered_rows": len(selected),
        "missing_judge_rows": missing,
        "metrics_router": metric_row(truth, router.label.astype(int).to_numpy()),
        "metrics_cascade": metric_row(truth, labels),
        "metrics_router_covered": metric_row(truth[covered], router.label.astype(int).to_numpy()[covered]),
        "metrics_uncertain_after_judge": metric_row(truth[sorted(uncertain)], labels[sorted(uncertain)]),
        "context_judge": str(args.context_judge),
        "no_context_judges": [str(path) for path in args.no_context_judge],
        "no_context_policy": args.no_context_policy,
        "decisions": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "decisions"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
