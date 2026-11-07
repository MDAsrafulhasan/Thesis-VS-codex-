from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from birth_date_overrides import DateValue, MONTHS
from datathon_utils import BN_DIGITS


MONTH_PATTERN = "|".join(map(re.escape, MONTHS))
DAY_SUFFIX = r"(?:লা|ই|শে|রা|ঠা|তম)?"
DAY_MONTH_YEAR = re.compile(
    rf"(?<!\d)(\d{{1,2}}){DAY_SUFFIX}\s+({MONTH_PATTERN})\s*,?\s*(\d{{3,4}})(?!\d)"
)
MONTH_DAY_YEAR = re.compile(
    rf"({MONTH_PATTERN})\s+(\d{{1,2}}){DAY_SUFFIX}(?:,\s*|\s+)(\d{{3,4}})(?!\d)"
)
YEAR_DAY_MONTH = re.compile(
    rf"(?<!\d)(\d{{3,4}})(?!\d)\s*(?:সালের|খ্রিষ্টাব্দের|খ্রিস্টাব্দের)"
    rf"\s*(\d{{1,2}}){DAY_SUFFIX}\s+({MONTH_PATTERN})"
)
HIJRI_YEAR = re.compile(r"(?<!\d)(\d{2,4})(?!\d)\s*(?:হিজরি|হিজরী)")
STANDALONE_YEAR = re.compile(r"(?<!\d)(\d{3,4})(?!\d)")


def preserve_normalize(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).translate(BN_DIGITS).lower()
    return re.sub(r"\s+", " ", text)


def strict_parse_date(value: object) -> DateValue | None:
    """Parse dates without allowing a four-digit year to split into day + year."""
    text = preserve_normalize(value)
    candidates: list[tuple[int, int, DateValue]] = []
    for match in DAY_MONTH_YEAR.finditer(text):
        candidates.append(
            (
                match.start(),
                match.end(),
                DateValue(
                    year=int(match.group(3)),
                    month=MONTHS[match.group(2)],
                    day=int(match.group(1)),
                ),
            )
        )
    for match in MONTH_DAY_YEAR.finditer(text):
        candidates.append(
            (
                match.start(),
                match.end(),
                DateValue(
                    year=int(match.group(3)),
                    month=MONTHS[match.group(1)],
                    day=int(match.group(2)),
                ),
            )
        )
    for match in YEAR_DAY_MONTH.finditer(text):
        candidates.append(
            (
                match.start(),
                match.end(),
                DateValue(
                    year=int(match.group(1)),
                    month=MONTHS[match.group(3)],
                    day=int(match.group(2)),
                ),
            )
        )

    occupied = [(start, end) for start, end, _ in candidates]
    for pattern in (HIJRI_YEAR, STANDALONE_YEAR):
        for match in pattern.finditer(text):
            if any(start <= match.start() < end for start, end in occupied):
                continue
            candidates.append(
                (match.start(), match.end(), DateValue(year=int(match.group(1))))
            )
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    return candidates[0][2]


def compare_strict_date(
    prompt: str,
    response: str,
    expected: DateValue,
) -> int | None:
    candidate = strict_parse_date(response)
    if candidate is None:
        return None
    if candidate.year != expected.year:
        return 0
    if "কত সালে" in prompt:
        return 1
    if expected.month is not None and candidate.month != expected.month:
        return 0
    if expected.day is not None and candidate.day != expected.day:
        return 0
    if expected.month is None and candidate.month is not None:
        return 0
    return 1


def audit_context_trace(
    frame: pd.DataFrame,
    trace: dict,
    base_labels: dict[int, int],
) -> tuple[list[dict], dict]:
    by_id = frame.set_index("id")
    context_records = [
        record for record in trace["decisions"] if record.get("stage") == "context_rule"
    ]
    records: list[dict] = []
    strict_disagreements: list[int] = []
    parser_anomalies: list[int] = []
    base_disagreements: list[int] = []

    for record in context_records:
        row_id = int(record["id"])
        base_label = int(base_labels[row_id])
        trace_label = int(record["new"])
        if trace_label != base_label:
            base_disagreements.append(row_id)

        item = {
            "id": row_id,
            "rule": record.get("rule"),
            "trace_label": trace_label,
            "base_label": base_label,
        }
        if record.get("rule") == "birth_date":
            row = by_id.loc[row_id]
            trace_date = DateValue(**record["evidence"]["date"])
            strict_date = strict_parse_date(record["evidence"]["evidence"])
            strict_label = (
                compare_strict_date(
                    str(row.prompt_bn),
                    str(row.response_bn),
                    strict_date,
                )
                if strict_date is not None
                else None
            )
            item.update(
                {
                    "trace_date": asdict(trace_date),
                    "strict_date": asdict(strict_date) if strict_date else None,
                    "strict_label": strict_label,
                }
            )
            if strict_date != trace_date:
                parser_anomalies.append(row_id)
            if strict_label is not None and strict_label != trace_label:
                strict_disagreements.append(row_id)
        records.append(item)

    summary = {
        "context_rule_rows": len(context_records),
        "rule_counts": dict(Counter(record.get("rule") for record in context_records)),
        "base_disagreements": len(base_disagreements),
        "base_disagreement_ids": base_disagreements,
        "birth_date_rows": sum(
            record.get("rule") == "birth_date" for record in context_records
        ),
        "strict_birth_parser_disagreements": len(strict_disagreements),
        "strict_birth_parser_disagreement_ids": strict_disagreements,
        "birth_parser_anomalies": len(parser_anomalies),
        "birth_parser_anomaly_ids": parser_anomalies,
    }
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit context-rule trace decisions with an independent strict birth "
            "date parser. This never changes inference outputs."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--base-submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    base = pd.read_csv(args.base_submission)
    base_labels = dict(zip(base["id"].astype(int), base["label"].astype(int)))
    records, summary = audit_context_trace(frame, trace, base_labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
