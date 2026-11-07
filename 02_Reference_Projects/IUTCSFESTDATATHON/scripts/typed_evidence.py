from __future__ import annotations

import json
import re
from pathlib import Path

from datathon_utils import normalize_bn
from source_answer_bank import canonical_answer


INTENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("literal_meaning", re.compile(r"(?:শাব্দিক|আক্ষরিক)\s+অর্থ")),
    ("figurative_meaning", re.compile(r"(?:ভাবার্থ|বাগধার|প্রবাদ).*(?:অর্থ|মানে)")),
    ("serialization_start", re.compile(r"ধারাবাহিক\s+প্রকাশ.*(?:শুরু|আরম্ভ)")),
    (
        "birth_date",
        re.compile(r"(?=.*(?:জন্ম|জন্মগ্রহণ))(?=.*(?:কবে|কত\s+সাল|তারিখ))"),
    ),
    (
        "death_date",
        re.compile(r"(?=.*(?:মৃত্যু|মৃত্যুবরণ|মারা\s+যান))(?=.*(?:কবে|কত\s+সাল|তারিখ))"),
    ),
    ("construction_start", re.compile(r"নির্মাণ.*(?:শুরু|আরম্ভ)")),
    ("construction_completion", re.compile(r"নির্মাণ.*(?:শেষ|সমাপ্ত)")),
    ("publication_date", re.compile(r"(?:প্রকাশ|প্রকাশিত|মুদ্রিত).*(?:কবে|কত\s+সাল)")),
    ("written_date", re.compile(r"(?:রচনা|লিখে|লেখেন).*(?:কবে|কত\s+সাল)")),
    ("river_source", re.compile(r"(?:নদীর?\s+উৎস|কোন\s+নদী\s+থেকে.*উৎপত্তি)")),
    ("river_confluence", re.compile(r"(?:নদী.*(?:মিলিত|মিশে|সঙ্গম))")),
)


def infer_intent(prompt: str) -> str | None:
    normalized = normalize_bn(prompt)
    for intent, pattern in INTENT_PATTERNS:
        if pattern.search(normalized):
            return intent
    return None


def _canonical_values(values: list[object]) -> set[str]:
    return {canonical_answer(value) for value in values if canonical_answer(value)}


def load_typed_evidence(paths: list[Path]) -> dict[str, list[dict]]:
    index: dict[str, list[dict]] = {}
    for path in paths:
        for number, item in enumerate(json.loads(path.read_text(encoding="utf-8")), start=1):
            question = str(item.get("question", "")).strip()
            intent = str(item.get("intent", "")).strip()
            accepted = _canonical_values(item.get("accepted_answers", []))
            rejected = _canonical_values(item.get("rejected_answers", []))
            sources = sorted(set(item.get("sources", [])))
            if not question or not intent:
                raise ValueError(f"{path}:{number}: question and intent are required")
            inferred = infer_intent(question)
            if inferred != intent:
                raise ValueError(
                    f"{path}:{number}: declared intent {intent!r} disagrees with inferred {inferred!r}"
                )
            if not accepted and not rejected:
                raise ValueError(f"{path}:{number}: at least one accepted or rejected answer is required")
            if accepted & rejected:
                raise ValueError(f"{path}:{number}: accepted and rejected answers overlap")
            if len(sources) < 2:
                raise ValueError(f"{path}:{number}: at least two source URLs are required")
            record = {
                **item,
                "question": question,
                "intent": intent,
                "_accepted": accepted,
                "_rejected": rejected,
                "_sources": sources,
                "_source_file": str(path),
            }
            index.setdefault(canonical_answer(question), []).append(record)
    return index


def _context_matches(context: str, item: dict) -> bool:
    normalized = normalize_bn(context)
    required_all = [normalize_bn(value) for value in item.get("context_contains_all", [])]
    required_any = [normalize_bn(value) for value in item.get("context_contains_any", [])]
    if required_all and not all(value in normalized for value in required_all):
        return False
    if required_any and not any(value in normalized for value in required_any):
        return False
    return True


def typed_evidence_decision(
    prompt: str,
    response: str,
    context: str,
    index: dict[str, list[dict]],
) -> tuple[int | None, dict]:
    prompt_intent = infer_intent(prompt)
    candidates = index.get(canonical_answer(prompt), [])
    answer = canonical_answer(response)
    decisions: list[tuple[int, dict]] = []
    for item in candidates:
        if item["intent"] != prompt_intent or not _context_matches(context, item):
            continue
        if answer in item["_accepted"]:
            label = 1
        elif answer in item["_rejected"]:
            label = 0
        else:
            continue
        decisions.append(
            (
                label,
                {
                    "rule": "typed_exact_evidence",
                    "intent": item["intent"],
                    "accepted_answers": item.get("accepted_answers", []),
                    "rejected_answers": item.get("rejected_answers", []),
                    "sources": item["_sources"],
                    "source_file": item["_source_file"],
                },
            )
        )
    labels = {label for label, _ in decisions}
    if len(labels) == 1:
        label, evidence = decisions[0]
        evidence["matching_records"] = len(decisions)
        return label, evidence
    if len(labels) > 1:
        return None, {
            "rule": "typed_evidence_conflict",
            "intent": prompt_intent,
            "matching_records": len(decisions),
        }
    return None, {}
