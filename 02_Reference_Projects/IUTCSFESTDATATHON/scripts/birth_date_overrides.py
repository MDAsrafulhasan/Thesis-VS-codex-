from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from datathon_utils import BN_DIGITS, load_sample, load_test, normalize_bn


MONTHS = {
    "জানুয়ারি": 1,
    "জানুয়ারি": 1,
    "ফেব্রুয়ারি": 2,
    "ফেব্রুয়ারি": 2,
    "মার্চ": 3,
    "এপ্রিল": 4,
    "মে": 5,
    "জুন": 6,
    "জুলাই": 7,
    "আগস্ট": 8,
    "সেপ্টেম্বর": 9,
    "অক্টোবর": 10,
    "নভেম্বর": 11,
    "ডিসেম্বর": 12,
}
MONTH_PATTERN = "|".join(map(re.escape, MONTHS))
DAY_SUFFIX = r"(?:ই|শে|রা|ঠা|তম)?"
FULL_DATE_PATTERNS = [
    re.compile(rf"(?<!\d)(\d{{1,2}}){DAY_SUFFIX}\s+({MONTH_PATTERN})\s*,?\s*(\d{{3,4}})(?!\d)"),
    re.compile(rf"({MONTH_PATTERN})\s+(\d{{1,2}}){DAY_SUFFIX}\s*,?\s*(\d{{3,4}})(?!\d)"),
    re.compile(rf"(?<!\d)(\d{{3,4}})(?!\d)\s*(?:সালের|খ্রিষ্টাব্দের|খ্রিস্টাব্দের)\s*(\d{{1,2}}){DAY_SUFFIX}\s+({MONTH_PATTERN})"),
]


@dataclass(frozen=True)
class DateValue:
    year: int
    month: int | None = None
    day: int | None = None


@dataclass(frozen=True)
class BirthEvidence:
    date: DateValue
    rule: str
    evidence: str


def is_birth_date_question(prompt: str) -> bool:
    return "জন্ম" in prompt and ("কবে" in prompt or "কত সালে" in prompt or "জন্মদিন" in prompt)


def preserve_normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).translate(BN_DIGITS).lower()
    return re.sub(r"\s+", " ", value)


def find_dates(text: str) -> list[tuple[int, int, DateValue]]:
    value = preserve_normalize(text)
    candidates: list[tuple[int, int, DateValue]] = []
    for pattern_index, pattern in enumerate(FULL_DATE_PATTERNS):
        for match in pattern.finditer(value):
            if pattern_index == 0:
                date = DateValue(year=int(match.group(3)), month=MONTHS[match.group(2)], day=int(match.group(1)))
            elif pattern_index == 1:
                date = DateValue(year=int(match.group(3)), month=MONTHS[match.group(1)], day=int(match.group(2)))
            else:
                date = DateValue(year=int(match.group(1)), month=MONTHS[match.group(3)], day=int(match.group(2)))
            candidates.append((match.start(), match.end(), date))
    occupied = [(start, end) for start, end, _ in candidates]
    for match in re.finditer(r"(?<!\d)(\d{2,4})(?!\d)\s*(?:হিজরি|হিজরী)", value):
        if not any(start <= match.start() < end for start, end in occupied):
            candidates.append((match.start(), match.end(), DateValue(year=int(match.group(1)))))
    for match in re.finditer(r"(?<!\d)(\d{3,4})(?!\d)", value):
        if not any(start <= match.start() < end for start, end in occupied):
            candidates.append((match.start(), match.end(), DateValue(year=int(match.group(1)))))
    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    return candidates


def parse_date(text: str) -> DateValue | None:
    dates = find_dates(text)
    return dates[0][2] if dates else None


def date_text_pattern() -> str:
    day_month = rf"\d{{1,2}}{DAY_SUFFIX}\s+(?:{MONTH_PATTERN})\s*,?\s*\d{{3,4}}"
    month_day = rf"(?:{MONTH_PATTERN})\s+\d{{1,2}}{DAY_SUFFIX}\s*,?\s*\d{{3,4}}"
    year_day_month = rf"\d{{3,4}}\s*(?:সালের|খ্রিষ্টাব্দের|খ্রিস্টাব্দের)\s*\d{{1,2}}{DAY_SUFFIX}\s+(?:{MONTH_PATTERN})"
    year = r"\d{3,4}"
    return rf"(?:{day_month}|{month_day}|{year_day_month}|{year})"


def extract_birth_date(context: str) -> BirthEvidence | None:
    text = preserve_normalize(re.sub(r"\[[^]]*\]", " ", str(context)))

    # Most biography pages put the fact in a `(জন্ম: DATE)` lead. Limit the
    # window so a later election/appointment year cannot be captured.
    for match in re.finditer(r"জন্ম(?!সূত্রে|স্থান|দিন|গ্রহণ)\s*:?[ ]*", text):
        segment = text[match.end() : match.end() + 120]
        date = parse_date(segment)
        if date:
            return BirthEvidence(date=date, rule="birth_marker", evidence=segment)

    # Lead-style biographies often omit the word জন্ম: `Name (DATE – DATE)`.
    range_match = re.search(rf"({date_text_pattern()})\s*[-–—]\s*(?:মৃত্যু\s*:?)?", text[:800])
    if range_match:
        date = parse_date(range_match.group(1))
        if date:
            return BirthEvidence(date=date, rule="lead_date_range", evidence=range_match.group(0))

    # Some Bengali leads use `DATE জন্মগ্রহণ করেন`. Choose the first date in
    # the immediately preceding clause; it may be a Hijri/Gregorian pair.
    for match in re.finditer(r"জন্মগ্রহণ\s+কর(?:েন|ে|েছিলেন)", text):
        segment = text[max(0, match.start() - 120) : match.end()]
        dates = find_dates(segment)
        if dates:
            return BirthEvidence(date=dates[0][2], rule="date_before_birth_verb", evidence=segment)
    return None


def compare_date(prompt: str, response: str, expected: DateValue) -> int | None:
    candidate = parse_date(response)
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
    # Do not accept invented day/month precision when the context states only
    # a year and the prompt asks the open-ended `কবে`.
    if expected.month is None and candidate.month is not None:
        return 0
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve explicit context-grounded Bengali birth-date questions.")
    parser.add_argument("--split", choices=["sample", "test"], required=True)
    parser.add_argument("--base-submission", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()

    df = load_sample() if args.split == "sample" else load_test()
    base = pd.read_csv(args.base_submission) if args.base_submission else None
    records = []
    for row_index, row in df[df["has_context"]].iterrows():
        prompt = str(row.prompt_bn)
        if not is_birth_date_question(prompt):
            continue
        evidence = extract_birth_date(str(row.context))
        if evidence is None:
            continue
        prediction = compare_date(prompt, str(row.response_bn), evidence.date)
        if prediction is None:
            continue
        item = {
            "row_index": int(row_index),
            "id": int(row.id) if "id" in df.columns else int(row_index),
            "prompt": prompt,
            "response": str(row.response_bn),
            "prediction": prediction,
            "expected": asdict(evidence.date),
            "rule": evidence.rule,
            "evidence": evidence.evidence,
        }
        if "label" in df.columns:
            item.update({"truth": int(row.label), "correct": prediction == int(row.label)})
        if base is not None:
            mask = base["id"] == int(row.id)
            old = int(base.loc[mask, "label"].iloc[0])
            item.update({"old": old, "changed": old != prediction})
            base.loc[mask, "label"] = prediction
        records.append(item)

    summary = {
        "split": args.split,
        "resolved": len(records),
        "correct": sum(item.get("correct", False) for item in records) if "label" in df.columns else None,
        "changed": sum(item.get("changed", False) for item in records) if base is not None else None,
        "rules": pd.Series([item["rule"] for item in records]).value_counts().to_dict(),
    }
    if base is not None:
        if args.output is None:
            raise ValueError("--base-submission requires --output")
        base.to_csv(args.output, index=False)
        summary.update({"output": str(args.output), "output_counts": base.label.value_counts().sort_index().to_dict()})
    args.trace.write_text(json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for item in records:
        if item.get("changed"):
            print("CHANGE", item["id"], item["old"], "->", item["prediction"], item["response"], item["expected"], item["rule"])


if __name__ == "__main__":
    main()
