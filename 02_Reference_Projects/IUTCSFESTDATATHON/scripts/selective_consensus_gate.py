from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


LABEL_FIELDS = ("prediction", "new", "pred", "label", "consensus")


def extract_records(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    if "id" in payload:
        return [payload]
    for key in ("records", "decisions"):
        if isinstance(payload.get(key), list):
            return [item for item in payload[key] if isinstance(item, dict)]
    return []


def record_label(item: dict) -> int | None:
    for field in LABEL_FIELDS:
        value = item.get(field)
        if value in (0, 1, "0", "1"):
            return int(value)
    return None


def load_votes(path: Path) -> dict[int, dict]:
    raw = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in raw.splitlines() if line.strip()]
    votes = {}
    for item in extract_records(payload):
        if "id" not in item:
            continue
        label = record_label(item)
        if label is not None:
            votes[int(item["id"])] = {"label": label, "record": item, "trace": str(path)}
    return votes


def consensus_decisions(
    judge_a: dict[int, dict], judge_b: dict[int, dict], evidence: dict[int, dict], require_evidence: bool = True
) -> list[dict]:
    decisions = []
    for row_id in sorted(set(judge_a) & set(judge_b)):
        a, b = judge_a[row_id]["label"], judge_b[row_id]["label"]
        if a != b:
            continue
        corroboration = evidence.get(row_id)
        if require_evidence and (corroboration is None or corroboration["label"] != a):
            continue
        decisions.append({
            "id": row_id,
            "prediction": a,
            "judge_a_trace": judge_a[row_id]["trace"],
            "judge_b_trace": judge_b[row_id]["trace"],
            "evidence_trace": corroboration["trace"] if corroboration else None,
            "rule": "cross_family_consensus_with_evidence" if corroboration else "cross_family_consensus",
        })
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply only cross-family judge decisions corroborated by evidence.")
    parser.add_argument("--base-submission", type=Path, required=True)
    parser.add_argument("--judge-a", type=Path, required=True)
    parser.add_argument("--judge-b", type=Path, required=True)
    parser.add_argument("--evidence-trace", type=Path, action="append", default=[])
    parser.add_argument("--allow-uncorroborated", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()

    evidence: dict[int, dict] = {}
    for path in args.evidence_trace:
        evidence.update(load_votes(path))
    decisions = consensus_decisions(
        load_votes(args.judge_a), load_votes(args.judge_b), evidence, require_evidence=not args.allow_uncorroborated
    )
    output = pd.read_csv(args.base_submission)
    records = []
    for item in decisions:
        mask = output.id == item["id"]
        if not mask.any():
            continue
        old = int(output.loc[mask, "label"].iloc[0])
        output.loc[mask, "label"] = item["prediction"]
        records.append({**item, "old": old, "changed": old != item["prediction"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    payload = {
        "summary": {"eligible": len(records), "changed": sum(item["changed"] for item in records), "require_evidence": not args.allow_uncorroborated},
        "records": records,
    }
    args.trace.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
