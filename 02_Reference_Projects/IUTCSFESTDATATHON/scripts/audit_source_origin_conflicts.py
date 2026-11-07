from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from datathon_utils import clean_text, metric_row, normalize_bn
from source_answer_bank import BankEntry, decide_row, read_bank


ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200d\ufeff]")
SPACE_RE = re.compile(r"\s+")
EMPTY_SVG_MARKDOWN_RE = re.compile(
    r"!\[[^\]]*\]\(data:image/svg\+xml,[^)]*fill-opacity(?:%3D|=)(?:%22|\")0(?:%22|\")[^)]*\)",
    flags=re.I,
)


def raw_signature(value: object, *, strip_empty_svg: bool = False) -> str:
    """Normalize encoding noise while preserving source-specific punctuation."""
    text = html.unescape(unicodedata.normalize("NFKC", clean_text(value)))
    text = ZERO_WIDTH_RE.sub("", text)
    if strip_empty_svg:
        text = EMPTY_SVG_MARKDOWN_RE.sub("", text)
    return SPACE_RE.sub(" ", text).strip()


def group_entries(entries: list[BankEntry]) -> dict[str, list[BankEntry]]:
    grouped: dict[str, list[BankEntry]] = defaultdict(list)
    for entry in entries:
        grouped[normalize_bn(entry.question)].append(entry)
    return grouped


def has_answer_conflict(entries: list[BankEntry]) -> bool:
    decisions = set()
    for entry in entries:
        decisions.update(normalize_bn(answer) for answer in entry.answers)
    return len(decisions) > 1


def exact_origin_entries(prompt: str, entries: list[BankEntry]) -> list[BankEntry]:
    signature = raw_signature(prompt)
    matches = [entry for entry in entries if raw_signature(entry.question) == signature]
    if matches:
        return matches

    # A few source records lost their diagrams during CSV serialization. Removing
    # only a demonstrably empty SVG placeholder recovers the textual question; it
    # does not attempt to infer what the missing diagram showed.
    stripped = raw_signature(prompt, strip_empty_svg=True)
    if stripped == signature:
        return []
    return [
        entry
        for entry in entries
        if raw_signature(entry.question, strip_empty_svg=True) == stripped
    ]


def origin_decision(prompt: str, response: str, entries: list[BankEntry]) -> dict | None:
    origins = exact_origin_entries(prompt, entries)
    if not origins:
        return None
    decision, all_decisions = decide_row(response, origins)
    if decision is None:
        return None
    repositories = sorted({entry.source_repo for entry in origins})
    return {
        "label": int(decision.label),
        "confidence": float(decision.confidence),
        "rule": decision.rule,
        "source_repositories": repositories,
        "source_files": sorted({entry.source_file for entry in origins}),
        "answers": sorted({answer for entry in origins for answer in entry.answers}),
        "all_decisions": [asdict(item) for item in all_decisions],
        "origin_count": len(origins),
    }


def audit_frame(
    frame: pd.DataFrame,
    entries: list[BankEntry],
    *,
    base_labels: dict[int, int] | None = None,
) -> tuple[list[dict], dict]:
    grouped = group_entries(entries)
    records: list[dict] = []
    for row in frame.itertuples(index=False):
        candidates = grouped.get(normalize_bn(row.prompt_bn), [])
        if not candidates or not has_answer_conflict(candidates):
            continue
        result = origin_decision(str(row.prompt_bn), str(row.response_bn), candidates)
        if result is None:
            continue
        row_id = int(row.id)
        record = {
            "id": row_id,
            "prompt_bn": str(row.prompt_bn),
            "response_bn": str(row.response_bn),
            **result,
        }
        if hasattr(row, "label"):
            record["truth"] = int(row.label)
            record["correct"] = int(result["label"]) == int(row.label)
        if base_labels is not None:
            record["base_label"] = int(base_labels[row_id])
            record["disagrees_with_base"] = int(result["label"]) != int(base_labels[row_id])
        records.append(record)

    summary: dict[str, object] = {
        "rows": len(frame),
        "conflicted_exact_origins": len(records),
        "distinct_ids": len({item["id"] for item in records}),
        "source_repositories": sorted({
            source for item in records for source in item["source_repositories"]
        }),
    }
    if records and all("truth" in item for item in records):
        truth = [int(item["truth"]) for item in records]
        pred = [int(item["label"]) for item in records]
        summary["metrics"] = metric_row(truth, pred)
        summary["correct"] = sum(t == p for t, p in zip(truth, pred))
    if base_labels is not None:
        summary["base_disagreements"] = sum(
            bool(item.get("disagrees_with_base")) for item in records
        )
        summary["base_disagreement_ids"] = [
            int(item["id"]) for item in records if item.get("disagrees_with_base")
        ]
    return records, summary


def load_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        frame = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    else:
        frame = pd.read_csv(path)
    if "id" not in frame:
        frame["id"] = range(len(frame))
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit conflicting public answer banks using exact raw source-origin "
            "fingerprints. This is diagnostic only and never writes a submission."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--bank", type=Path, action="append", required=True)
    parser.add_argument("--base-submission", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = load_frame(args.input)
    entries = [entry for path in args.bank for entry in read_bank(path)]
    base_labels = None
    if args.base_submission:
        base = pd.read_csv(args.base_submission)
        base_labels = dict(zip(base["id"].astype(int), base["label"].astype(int)))
        missing = set(frame["id"].astype(int)) - set(base_labels)
        if missing:
            raise ValueError(f"base submission is missing IDs: {sorted(missing)[:10]}")

    records, summary = audit_frame(frame, entries, base_labels=base_labels)
    payload = {"summary": summary, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
