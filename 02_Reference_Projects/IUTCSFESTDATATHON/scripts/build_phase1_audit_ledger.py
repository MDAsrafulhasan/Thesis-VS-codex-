from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from datathon_utils import clean_context, normalize_bn


BN_DIGITS = "০১২৩৪৫৬৭৮৯"
NUMBER_RE = re.compile(rf"[0-9{BN_DIGITS}]+(?:[.,][0-9{BN_DIGITS}]+)?")
ANTONYM_MARKERS = ("বিপরীত শব্দ", "বিপরীতার্থক")
IDIOM_MARKERS = ("বাগধারা", "প্রবাদ")
REASONING_MARKERS = ("হিসাব", "নির্ণয়", "সমীকরণ", "অনুপাত", "শতকরা", "কত", "যোগফল", "মান কত")
DATE_MARKERS = ("সাল", "তারিখ", "কবে", "কত সালে", "জন্ম", "মৃত্যু")
PLACE_MARKERS = ("কোথায়", "কোথায়", "দেশ", "জেলা", "রাজ্য", "নদী", "সীমান্ত", "রাজধানী")
PERSON_MARKERS = ("কে", "কার", "লেখক", "রচয়িতা", "প্রতিষ্ঠাতা", "কবি")


def classify_route(prompt: str, response: str, context: str, category: str) -> str:
    text = normalize_bn(prompt)
    if any(marker in text for marker in ANTONYM_MARKERS):
        return "lexical_antonym"
    if any(marker in text for marker in IDIOM_MARKERS):
        return "lexical_idiom"
    if context:
        return "context_atomic_entailment"
    if category == "numeric_or_date" or NUMBER_RE.search(prompt + " " + response):
        if any(marker in text for marker in DATE_MARKERS):
            return "structured_date_fact"
        if any(marker in text for marker in REASONING_MARKERS):
            return "structured_reasoning"
        return "structured_numeric_fact"
    if category == "person_entity" or any(marker in text for marker in PERSON_MARKERS):
        return "entity_fact"
    if category == "place" or any(marker in text for marker in PLACE_MARKERS):
        return "place_fact"
    return "closed_book_qa"


def likely_error_family(route: str, prompt: str, response: str, context: str) -> str:
    if route in {"structured_reasoning"}:
        return "arithmetic_or_formula_error"
    if route in {"structured_date_fact", "structured_numeric_fact", "entity_fact", "place_fact"}:
        return "factual_substitution"
    if route == "lexical_antonym":
        return "lexical_relation_or_valid_alternative"
    if route == "lexical_idiom":
        return "figurative_semantic_drift"
    if route == "context_atomic_entailment":
        response_tokens = set(normalize_bn(response).split())
        context_tokens = set(normalize_bn(context).split())
        overlap = len(response_tokens & context_tokens) / max(1, len(response_tokens))
        if overlap < 0.25:
            return "fabricated_or_nonfactual_addition"
        if any(token in normalize_bn(response) for token in ("না", "নয়", "নেই", "ভুল")):
            return "possible_direct_contradiction"
        return "unsupported_inference_or_specificity"
    return "comprehension_specificity_or_inference"


def evidence_requirement(route: str) -> str:
    requirements = {
        "structured_reasoning": "deterministic derivation with unit test",
        "structured_date_fact": "two independent authoritative sources or exact released provenance",
        "structured_numeric_fact": "deterministic derivation or two independent authoritative sources",
        "lexical_antonym": "two independent dictionaries explicitly accepting the candidate relation",
        "lexical_idiom": "expert idiom corpus plus independent dictionary or exam key",
        "context_atomic_entailment": "atomic claim supported or contradicted by the supplied context",
        "entity_fact": "two independent authoritative biographical or institutional sources",
        "place_fact": "two independent government, institutional, or encyclopedic sources",
        "closed_book_qa": "canonical answer plus two-source or cross-family consensus",
    }
    return requirements[route]


def build_ledger(unresolved: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    by_id = test.set_index("id", drop=False)
    records = []
    for raw in unresolved.to_dict("records"):
        row_id = int(raw["id"])
        row = by_id.loc[row_id]
        prompt = str(row.prompt_bn)
        response = str(row.response_bn)
        context = clean_context(row.context)
        category = str(raw.get("category", "other"))
        route = classify_route(prompt, response, context, category)
        records.append({
            "id": row_id,
            "row_index": int(raw.get("row_index", row_id)),
            "block": int(raw.get("block", -1)),
            "category": category,
            "has_context": bool(context),
            "prompt_bn": prompt,
            "response_bn": response,
            "context": context,
            "current_label": int(raw["base_label"]),
            "route": route,
            "likely_error_family": likely_error_family(route, prompt, response, context),
            "evidence_requirement": evidence_requirement(route),
            "canonical_answer": "",
            "proposed_label": None,
            "confidence_tier": "U",
            "evidence_sources": [],
            "rationale": "",
            "submission_eligible": False,
            "review_status": "pending",
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the row-level Phase 1 unresolved audit ledger.")
    parser.add_argument("--unresolved", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()

    unresolved = pd.read_csv(args.unresolved)
    test = pd.read_csv(args.test)
    records = build_ledger(unresolved, test)
    route_counts = pd.Series([item["route"] for item in records]).value_counts().to_dict()
    family_counts = pd.Series([item["likely_error_family"] for item in records]).value_counts().to_dict()
    payload = {"summary": {"rows": len(records), "route_counts": route_counts, "error_family_counts": family_counts}, "records": records}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_records = [{**item, "evidence_sources": " | ".join(item["evidence_sources"])} for item in records]
    pd.DataFrame(csv_records).to_csv(args.csv, index=False)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
