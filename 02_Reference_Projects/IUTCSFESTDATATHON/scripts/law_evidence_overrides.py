from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from datathon_utils import OUTPUT_DIR, load_sample, load_test, normalize_bn
from source_answer_bank import canonical_answer, number_signature


WORD_NUMBERS = {
    "শূন্য": 0,
    "এক": 1,
    "দুই": 2,
    "তিন": 3,
    "চার": 4,
    "পাঁচ": 5,
    "ছয়": 6,
    "ছয়": 6,
    "সাত": 7,
    "আট": 8,
    "নয়": 9,
    "নয়": 9,
    "দশ": 10,
    "এগারো": 11,
    "বারো": 12,
    "তেরো": 13,
    "চৌদ্দ": 14,
    "পনেরো": 15,
    "ষোল": 16,
    "সতেরো": 17,
    "আঠারো": 18,
    "আঠার": 18,
    "উনিশ": 19,
    "বিশ": 20,
    "একুশ": 21,
}


@dataclass(frozen=True)
class Expected:
    value: int | None
    scale: str | None
    answer_text: str
    rule: str
    source_file: str
    evidence: str


def read_law(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_law_corpus(path: Path | None, act_ids: set[str] | None = None) -> dict[str, str]:
    """Load normalized government-law chunks, grouped by bdlaws act ID."""
    if path is None or not path.exists():
        return {}
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        acts = payload.get("acts", payload)
        return {
            str(act_id): str(item.get("text", item) if isinstance(item, dict) else item)
            for act_id, item in acts.items()
            if act_ids is None or str(act_id) in act_ids
        }
    grouped: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            match = re.search(r"act-print-(\d+)\.html", str(item.get("url", "")))
            if match is None or (act_ids is not None and match.group(1) not in act_ids):
                continue
            grouped.setdefault(match.group(1), []).append(str(item.get("context", "")))
    return {act_id: "\n".join(parts) for act_id, parts in grouped.items()}


def read_act(
    act_id: str,
    laws_dir: Path | None,
    law_corpus: Mapping[str, str] | None,
) -> tuple[str, str] | None:
    if law_corpus and act_id in law_corpus:
        return law_corpus[act_id], f"bdlaws://act-{act_id}"
    if laws_dir is None:
        return None
    path = laws_dir / f"act-{act_id}.mdx"
    if not path.exists():
        return None
    return read_law(path), str(path)


def first_value(text: str) -> int | None:
    signature = number_signature(text)
    if signature:
        try:
            return int(float(signature[0]))
        except ValueError:
            pass
    normalized = normalize_bn(text)
    for token in normalized.split():
        token = re.sub(r"(?:টি|বার|বছর|বৎসর|মাস)$", "", token)
        if token in WORD_NUMBERS:
            return WORD_NUMBERS[token]
    return None


def amount_scale(text: str) -> str | None:
    normalized = normalize_bn(text)
    if "কোটি" in normalized:
        return "কোটি"
    if "লক্ষ" in normalized or "লাখ" in normalized:
        return "লক্ষ"
    if "হাজার" in normalized:
        return "হাজার"
    return None


def section_text(text: str, section: int) -> str | None:
    normalized = text.translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789"))
    pattern = re.compile(
        rf"(?m)^(?:\.?){section}[।.]\s*(.*?)(?=^###\s|^(?:\.?){section + 1}[।.]|\Z)",
        flags=re.S,
    )
    match = pattern.search(normalized)
    return match.group(1).strip() if match else None


def extract_maximum(text: str, kind: str) -> tuple[int, str | None] | None:
    normalized = text.translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789"))
    if kind == "imprisonment_years":
        match = re.search(r"অনধিক\s+(\d+)\s*\([^)]*\)\s*বৎসর কারাদণ্ড", normalized)
        return (int(match.group(1)), None) if match else None
    if kind == "fine":
        match = re.search(r"অনধিক\s+(\d+)\s*\([^)]*\)\s*(লক্ষ|হাজার) টাকা অর্থদণ্ড", normalized)
        return (int(match.group(1)), match.group(2)) if match else None
    return None


def _expected(
    value: int | None,
    scale: str | None,
    answer_text: str,
    rule: str,
    path: Path | str,
    evidence: str,
) -> Expected:
    return Expected(value, scale, answer_text, rule, str(path), re.sub(r"\s+", " ", evidence).strip()[:1200])


def dna_expected(
    prompt: str, laws_dir: Path | None, law_corpus: Mapping[str, str] | None = None,
) -> Expected | None:
    if "ডিএনএ আইন" not in prompt:
        return None
    source = read_act("1151", laws_dir, law_corpus)
    if source is None:
        return None
    text, path = source
    normalized_prompt = normalize_bn(prompt)

    if "জেনেটিক মার্কার" in prompt:
        match = re.search(r"অন্যূন\s+([০-৯0-9]+)\s*\([^)]*\)\s*টি জেনেটিক মার্কার", text)
        if match:
            value = int(match.group(1).translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")))
            return _expected(value, None, f"{value}টি", "dna_minimum_markers", path, match.group(0))

    if "উপদেষ্টা পরিষদের সভা" in prompt:
        evidence = section_text(text, 18)
        match = re.search(r"বৎসরে কমপক্ষে\s+([০-৯0-9]+)", evidence or "")
        if match:
            value = int(match.group(1).translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")))
            return _expected(value, None, f"{value} বার", "dna_minimum_meetings", path, evidence or "")

    section_match = re.search(r"(?:এর|ধারা)\s*([০-৯0-9]+)\s*ধারা", prompt)
    if not section_match:
        return None
    section = int(section_match.group(1).translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")))
    evidence = section_text(text, section)
    if not evidence:
        return None
    if "কারাদণ্ড" in normalized_prompt:
        maximum = extract_maximum(evidence, "imprisonment_years")
        if maximum:
            return _expected(maximum[0], None, f"{maximum[0]} বছর", "dna_max_imprisonment", path, evidence)
    if "অর্থদণ্ড" in normalized_prompt:
        maximum = extract_maximum(evidence, "fine")
        if maximum:
            value, scale = maximum
            return _expected(value, scale, f"{value} {scale} টাকা", "dna_max_fine", path, evidence)
    return None


def child_marriage_expected(
    prompt: str, laws_dir: Path | None, law_corpus: Mapping[str, str] | None = None,
) -> Expected | None:
    if "বাল্যবিবাহ নিরোধ আইন, ১৯২৯" not in prompt:
        return None
    source = read_act("149", laws_dir, law_corpus)
    if source is None:
        return None
    text, path = source
    if "পুরুষদের ক্ষেত্রে" in prompt:
        match = re.search(r"if a male, is under (twenty-one) years", text)
        if match:
            return _expected(21, None, "২১ বছরের কম", "child_age_male", path, match.group(0))
    if "মহিলাদের ক্ষেত্রে" in prompt:
        match = re.search(r"if a female, is under (eighteen) years", text)
        if match:
            return _expected(18, None, "১৮ বছরের কম", "child_age_female", path, match.group(0))
    if "নিষেধাজ্ঞা অমান্য" in prompt and "কত মাস" in prompt:
        evidence = section_text(text, 12)
        match = re.search(r"extend to (three) months", evidence or "")
        if match:
            return _expected(3, None, "৩ মাস", "child_injunction_max_months", path, evidence or "")
    return None


def rti_expected(
    prompt: str, laws_dir: Path | None, law_corpus: Mapping[str, str] | None = None,
) -> Expected | None:
    if "তথ্য অধিকার আইন, ২০০৯" not in prompt:
        return None
    source = read_act("1011", laws_dir, law_corpus)
    if source is None:
        return None
    text, path = source
    evidence = section_text(text, 1) or text[:2500]
    if "ব্যতীত" in prompt:
        return _expected(None, None, "২০ অক্টোবর, ২০০৮", "rti_other_sections_effective", path, evidence)
    if all(number in prompt for number in ["৮", "২৪", "২৫"]):
        return _expected(None, None, "১ জুলাই, ২০০৯", "rti_named_sections_effective", path, evidence)
    return None


def trademark_expected(
    prompt: str, laws_dir: Path | None, law_corpus: Mapping[str, str] | None = None,
) -> Expected | None:
    # Two competition prompts name the wrong act, but their operative wording
    # is a near-verbatim match to sections 74/75 of the Trademark Act, 2009.
    source = read_act("1010", laws_dir, law_corpus)
    if source is None:
        return None
    text, path = source
    if "পণ্য উৎপাদনকারী প্রতিষ্ঠান" in prompt and "দ্বিতীয়বার" in prompt:
        evidence = section_text(text, 74)
        match = re.search(r"দ্বিতীয়বার.*?অনধিক তিন বৎসর", evidence or "", flags=re.S)
        if match:
            return _expected(3, None, "৩ বছর", "trademark_repeat_offence_max_years", path, evidence or "")
    if "তুলার সুতা" in prompt and "অর্থদণ্ড" in prompt:
        evidence = section_text(text, 75)
        match = re.search(r"অনুধিক\s+([০-৯0-9]+)\s*\([^)]*\)\s*হাজার টাকা", evidence or "")
        if match:
            value = int(match.group(1).translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")))
            return _expected(value, "হাজার", f"{value} হাজার টাকা", "trademark_unmarked_yarn_max_fine", path, evidence or "")
    return None


def expected_for_prompt(
    prompt: str,
    laws_dir: Path | None,
    law_corpus: Mapping[str, str] | None = None,
) -> Expected | None:
    for handler in (dna_expected, child_marriage_expected, rti_expected, trademark_expected):
        result = handler(prompt, laws_dir, law_corpus)
        if result is not None:
            return result
    return None


def compare_expected(response: str, expected: Expected) -> int:
    if expected.value is None:
        return int(canonical_answer(response) == canonical_answer(expected.answer_text))
    response_value = first_value(response)
    if response_value != expected.value:
        return 0
    if expected.scale is not None and amount_scale(response) != expected.scale:
        return 0
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify law questions directly from public Bangladesh law text.")
    parser.add_argument("--split", choices=["sample", "test"], required=True)
    parser.add_argument("--laws-dir", type=Path, default=Path("/tmp/laws-bd/laws"))
    parser.add_argument("--law-corpus", type=Path, action="append", default=[])
    parser.add_argument("--base-submission", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()
    law_corpus = {}
    for path in args.law_corpus:
        law_corpus.update(load_law_corpus(path, {"149", "1010", "1011", "1151"}))

    df = load_sample() if args.split == "sample" else load_test()
    base = pd.read_csv(args.base_submission) if args.base_submission else None
    records = []
    for row_index, row in df.iterrows():
        expected = expected_for_prompt(str(row.prompt_bn), args.laws_dir, law_corpus)
        if expected is None:
            continue
        prediction = compare_expected(str(row.response_bn), expected)
        record = {
            "row_index": int(row_index),
            "id": int(row.id) if "id" in df.columns else int(row_index),
            "prompt": str(row.prompt_bn),
            "response": str(row.response_bn),
            "prediction": prediction,
            "expected": asdict(expected),
        }
        if "label" in df.columns:
            record["truth"] = int(row.label)
            record["correct"] = prediction == int(row.label)
        if base is not None:
            mask = base["id"] == int(row.id)
            old = int(base.loc[mask, "label"].iloc[0])
            record["old"] = old
            record["changed"] = old != prediction
            base.loc[mask, "label"] = prediction
        records.append(record)

    summary = {
        "split": args.split,
        "resolved": len(records),
        "correct": sum(record.get("correct", False) for record in records) if "label" in df.columns else None,
        "changed": sum(record.get("changed", False) for record in records) if base is not None else None,
        "rules": pd.Series([record["expected"]["rule"] for record in records]).value_counts().to_dict(),
    }
    if base is not None:
        if args.output is None:
            raise ValueError("--base-submission requires --output")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        base.to_csv(args.output, index=False)
        summary["output"] = str(args.output)
        summary["output_counts"] = base["label"].value_counts().sort_index().to_dict()
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    args.trace.write_text(json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote", args.trace)


if __name__ == "__main__":
    main()
