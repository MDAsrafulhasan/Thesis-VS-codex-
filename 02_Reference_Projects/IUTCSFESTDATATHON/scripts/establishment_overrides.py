from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path

import pandas as pd

from birth_date_overrides import parse_date, preserve_normalize
from datathon_utils import bn_tokens, load_sample, load_test, normalize_bn


STOP = {
    "কত", "সালে", "কবে", "প্রতিষ্ঠিত", "প্রতিষ্ঠা", "হয়", "হয়", "হয়েছিল", "হয়েছিল",
    "এর", "কাজ", "কাজের", "শুরু", "থেকে", "নির্মাণ", "মর্যাদা", "দেওয়া", "দেওয়া",
}
PRONOUN_PREFIXES = ("এটি ", "দলটি ", "সমিতিটি ", "এই সমিতিটি ", "এর সূচনা", "প্রতিষ্ঠানটি ")
ESTABLISH_RELATIONS = (
    "প্রতিষ্ঠিত", "প্রতিষ্ঠা লাভ", "প্রতিষ্ঠা করা", "প্রতিষ্ঠা করেন", "প্রতিষ্ঠা করে",
    "প্রতিষ্ঠা হয়", "প্রতিষ্ঠা হয়", "গঠিত", "সূচনা হয়", "সূচনা হয়", "যাত্রা আরম্ভ",
)
PERSON_MARKERS = ("ছিলেন একজন", "হলেন একজন", "একজন বাংলাদেশী", "একজন বাংলাদেশি", "একজন ভারতীয়", "একজন ব্রিটিশ")
PERSON_OCCUPATIONS = ("রাজনীতিবিদ", "সমাজকর্মী", "লেখক", "অভিনেতা", "কবি", "বিজ্ঞানী", "কর্মকর্তা", "খেলোয়াড়")


def split_sentences(context: str) -> list[str]:
    return [item.strip() for item in re.split(r"(?<=[।!?])\s+|\n+", preserve_normalize(context)) if item.strip()]


def subject_from_prompt(prompt: str) -> str:
    value = preserve_normalize(prompt)
    value = re.split(r"\s+(?:কত সালে|কবে|কত সাল)", value, maxsplit=1)[0]
    return value.strip(" ?'\"")


def subject_overlap(subject: str, sentence: str) -> float:
    left = set(bn_tokens(subject)) - STOP
    right = set(bn_tokens(sentence)) - STOP
    overlap = len(left & right) / max(1, len(left))
    if len(left) == 1 and right:
        token = next(iter(left))
        # A few translated pages differ by one vowel mark in the entity name.
        # Restrict fuzzy linking to single-token subjects to avoid joining a
        # related sub-entity merely because it shares one generic name token.
        if max(difflib.SequenceMatcher(None, token, item).ratio() for item in right) >= 0.88:
            return 1.0
    return overlap


def years_with_positions(text: str) -> list[tuple[int, int]]:
    years = []
    for match in re.finditer(r"(?<!\d)(\d{3,4})(?!\d)", preserve_normalize(text)):
        year = int(match.group(1))
        # Three-digit quantities (for example, "১১০ acres") otherwise look
        # like ancient establishment years when they occur near the relation.
        if 1000 <= year <= 2026:
            years.append((match.start(), year))
    return years


def remove_parenthetical_dates(text: str) -> str:
    return re.sub(r"\([^)]*\d{3,4}[^)]*\)", " ", text)


def nearest_relation_year(sentence: str, relations: tuple[str, ...]) -> int | None:
    value = remove_parenthetical_dates(preserve_normalize(sentence))
    years = years_with_positions(value)
    choices = []
    for relation in relations:
        for match in re.finditer(re.escape(relation), value):
            for position, year in years:
                distance = abs(position - match.start())
                if distance <= 120:
                    choices.append((distance, position > match.start(), year))
    if not choices:
        return None
    choices.sort(key=lambda item: (item[0], item[1]))
    return choices[0][2]


def extract_establishment_year(prompt: str, context: str) -> tuple[int | None, dict]:
    subject = subject_from_prompt(prompt)
    sentences = split_sentences(context)
    found = []
    entity_active = False
    aliases: set[str] = set()
    subject_token_count = len(set(bn_tokens(subject)) - STOP)
    for index, sentence in enumerate(sentences):
        overlap = subject_overlap(subject, sentence)
        direct_threshold = 0.40 if subject_token_count >= 4 else 0.55
        direct = overlap >= direct_threshold
        alias_link = entity_active and any(sentence.startswith(alias + " ") for alias in aliases)
        pronoun_link = any(0 <= sentence.find(prefix) <= 12 for prefix in PRONOUN_PREFIXES)
        pronoun = entity_active and (pronoun_link or alias_link)
        if direct or pronoun:
            year = nearest_relation_year(sentence, ESTABLISH_RELATIONS)
            if year is not None:
                found.append((year, sentence, overlap, "direct" if direct else "pronoun"))
        if direct:
            # Source leads commonly introduce a short Bengali acronym which is
            # then used for the historical sentence (e.g. এলআরডি).
            for alias in re.findall(r"\(([^()]*)\)", sentence):
                alias = alias.strip()
                if 2 <= len(alias) <= 20 and not re.search(r"\d", alias):
                    aliases.add(alias)
            entity_active = True

    unique = sorted({item[0] for item in found})
    if len(unique) == 1:
        year = unique[0]
        best = max((item for item in found if item[0] == year), key=lambda item: item[2])
        return year, {"rule": "entity_relation_year", "evidence": best[1], "subject_overlap": best[2], "link": best[3]}
    if len(unique) > 1:
        return None, {"rule": "ambiguous_entity_relation_year", "candidate_years": unique}

    return None, {}


def extract_construction_start_year(prompt: str, context: str) -> tuple[int | None, dict]:
    subject = subject_from_prompt(prompt)
    found = []
    for sentence in split_sentences(context):
        overlap = subject_overlap(subject, sentence)
        if overlap < 0.45 or "শুরু" not in sentence or "নির্মাণ" not in sentence:
            continue
        year = nearest_relation_year(sentence, ("শুরু",))
        if year is not None:
            found.append((year, sentence, overlap))
    unique = sorted({item[0] for item in found})
    if len(unique) != 1:
        return None, {"rule": "ambiguous_construction_start", "candidate_years": unique} if unique else {}
    best = max(found, key=lambda item: item[2])
    return unique[0], {"rule": "construction_start_year", "evidence": best[1], "subject_overlap": best[2]}


def extract_status_year(prompt: str, context: str) -> tuple[int | None, dict]:
    subject = subject_from_prompt(prompt)
    found = []
    for sentence in split_sentences(context):
        overlap = subject_overlap(subject, sentence)
        if overlap < 0.45 or "মর্যাদা" not in sentence:
            continue
        year = nearest_relation_year(sentence, ("মর্যাদা",))
        if year is not None:
            found.append((year, sentence, overlap))
    unique = sorted({item[0] for item in found})
    if len(unique) != 1:
        return None, {}
    best = max(found, key=lambda item: item[2])
    return unique[0], {"rule": "status_year", "evidence": best[1], "subject_overlap": best[2]}


def negative_establishment_evidence(prompt: str, response: str, context: str) -> dict:
    """Detect relation swaps that cannot answer an establishment question."""
    subject = subject_from_prompt(prompt)
    sentences = split_sentences(context)
    if not sentences or subject_overlap(subject, sentences[0]) < 0.55:
        return {}
    lead = sentences[0]
    if any(marker in lead for marker in PERSON_MARKERS) and any(job in lead for job in PERSON_OCCUPATIONS):
        return {"rule": "person_cannot_be_established", "evidence": lead}
    if "প্রস্তাবিত" in lead and any(marker in context for marker in ("প্রস্তাব গৃহীত", "গঠনের প্রস্তাব", "প্রস্তাব দেওয়া", "প্রস্তাব দেওয়া")):
        return {"rule": "proposal_is_not_establishment", "evidence": lead}

    candidate = parse_date(response)
    if candidate is None:
        return {}
    subject_tokens = len(set(bn_tokens(subject)) - STOP)
    threshold = 0.40 if subject_tokens >= 4 else 0.55
    for sentence in sentences[1:]:
        year = nearest_relation_year(sentence, ESTABLISH_RELATIONS)
        if year != candidate.year or subject_overlap(subject, sentence) >= threshold:
            continue
        # A separately named entity's establishment year is not evidence for
        # the target introduced in the lead. Pronoun/acronym continuations are
        # handled positively by extract_establishment_year and are excluded.
        relation_position = min((sentence.find(rel) for rel in ESTABLISH_RELATIONS if rel in sentence), default=len(sentence))
        has_target_pronoun = any(prefix.strip() in sentence[:relation_position] for prefix in PRONOUN_PREFIXES)
        if not has_target_pronoun:
            return {"rule": "other_entity_establishment_year", "evidence": sentence, "copied_year": year}
    return {}


def decide(prompt: str, response: str, context: str) -> tuple[int | None, dict]:
    normalized = normalize_bn(prompt)
    if "নির্মাণ" in normalized and "শুরু" in normalized:
        year, evidence = extract_construction_start_year(prompt, context)
    elif "মর্যাদা" in normalized and ("দেওয়া" in normalized or "দেওয়া" in prompt):
        year, evidence = extract_status_year(prompt, context)
    elif "প্রতিষ্ঠিত" in normalized or "প্রতিষ্ঠা কাল" in normalized:
        year, evidence = extract_establishment_year(prompt, context)
        if year is None:
            negative = negative_establishment_evidence(prompt, response, context)
            if negative:
                return 0, negative
    else:
        return None, {}
    candidate = parse_date(response)
    if year is None or candidate is None:
        return None, evidence
    return int(candidate.year == year), {**evidence, "expected_year": year}


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve explicit establishment/construction relation years.")
    parser.add_argument("--split", choices=["sample", "test"], required=True)
    parser.add_argument("--base-submission", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()

    df = load_sample() if args.split == "sample" else load_test()
    base = pd.read_csv(args.base_submission) if args.base_submission else None
    records = []
    for row_index, row in df[df.has_context].iterrows():
        pred, evidence = decide(str(row.prompt_bn), str(row.response_bn), str(row.context))
        if pred is None:
            continue
        item = {
            "row_index": int(row_index),
            "id": int(row.id) if "id" in df.columns else int(row_index),
            "prompt": str(row.prompt_bn),
            "response": str(row.response_bn),
            "prediction": pred,
            **evidence,
        }
        if "label" in df.columns:
            item.update({"truth": int(row.label), "correct": pred == int(row.label)})
        if base is not None:
            mask = base.id == int(row.id)
            old = int(base.loc[mask, "label"].iloc[0])
            item.update({"old": old, "changed": old != pred})
            base.loc[mask, "label"] = pred
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
            expected = item.get("expected_year", item.get("copied_year", "relation contradiction"))
            print("CHANGE", item["id"], item["old"], "->", item["prediction"], item["response"], expected, item["rule"])


if __name__ == "__main__":
    main()
