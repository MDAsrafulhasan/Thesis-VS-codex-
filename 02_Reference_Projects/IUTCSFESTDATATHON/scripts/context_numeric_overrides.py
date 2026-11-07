from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pandas as pd

from bn_num2words import num2words

from datathon_utils import bn_tokens, load_sample, load_test, normalize_bn


QUESTION_STOP = {
    "কত", "কবে", "কি", "কী", "কে", "কোন", "কয়", "কয়", "সালে", "হয়", "হয়",
    "ছিল", "ছিলেন", "এর", "টি", "ভাবে", "থেকে", "বর্তমান", "প্রায়", "প্রায়", "জন",
    "করেন", "করা", "র",
}
ANSWER_DROP = {
    "প্রায়", "প্রায়", "সালে", "সাল", "খ্রিষ্টাব্দে", "খ্রিস্টাব্দে", "জন", "টি", "টাকা",
    "বছর", "বৎসর", "দীর্ঘ",
}
SCALES = {
    "হাজার": 1_000,
    "লক্ষ": 100_000,
    "লাখ": 100_000,
    "কোটি": 10_000_000,
    "মিলিয়ন": 1_000_000,
    "মিলিয়ন": 1_000_000,
    "ট্রিলিয়ন": 1_000_000_000_000,
    "ট্রিলিয়ন": 1_000_000_000_000,
}
RELATION_GROUPS = [
    (("চাহিদা",), ("চাহিদা",)),
    (("জনসংখ্যা",), ("জনসংখ্যা",)),
    (("প্রতিষ্ঠিত", "প্রতিষ্ঠা"), ("প্রতিষ্ঠিত", "প্রতিষ্ঠা", "স্থাপিত")),
    (("জন্ম",), ("জন্ম",)),
    (("মারা", "মৃত্যু"), ("মারা", "মৃত্যু", "নিহত", "প্রাণ হার")),
    (("দৈর্ঘ্য", "দীর্ঘ"), ("দৈর্ঘ্য", "দীর্ঘ")),
    (("উচ্চতা",), ("উচ্চতা", "উঁচু")),
    (("ভাগে", "ভাগ"), ("ভাগে", "ভাগ")),
    (("নির্মিত", "নির্মাণ"), ("নির্মিত", "নির্মাণ")),
    (("প্রকাশিত", "প্রকাশ"), ("প্রকাশিত", "প্রকাশ")),
    (("রচিত", "রচনা"), ("রচিত", "রচনা")),
    (("উদ্বোধন", "চালু"), ("উদ্বোধন", "চালু")),
]


def answer_canonical(text: str) -> str:
    value = normalize_bn(text).replace("লাখ", "লক্ষ").replace("শো", "শত").replace("শ’", "শত")
    for scale in ("হাজার", "লক্ষ", "কোটি", "মিলিয়ন", "মিলিয়ন"):
        value = value.replace(f"{scale}েরও", scale).replace(f"{scale}ের", scale)
    value = value.replace("'", "")
    tokens = [token for token in value.split() if token not in ANSWER_DROP]
    value = " ".join(tokens)
    return re.sub(r"([\w\u0980-\u09ff]+)\s+শত", r"\1শত", value)


def numeric_components(text: str) -> list[float]:
    raw = re.sub(r"\[[^]]*\]", " ", str(text)).replace(",", "").replace("٬", "")
    value = normalize_bn(raw)
    scale_pattern = "|".join(map(re.escape, SCALES))
    spans = []
    output = []
    for match in re.finditer(rf"(\d+)\s+দশমিক\s+(\d+)\s*({scale_pattern})?", value):
        number = float(f"{match.group(1)}.{match.group(2)}") * SCALES.get(match.group(3), 1)
        output.append(number)
        spans.append(match.span())
    for match in re.finditer(rf"(?<!\d)(\d+(?:\.\d+)?)\s*({scale_pattern})?", value):
        if any(start <= match.start() < end for start, end in spans):
            continue
        output.append(float(match.group(1)) * SCALES.get(match.group(2), 1))
    return output


def number_word_variants(number: int) -> set[str]:
    variants = {answer_canonical(num2words(number, lang="bn"))}
    if 100 <= number <= 9999:
        hundreds, remainder = divmod(number, 100)
        if remainder:
            phrase = answer_canonical(f"{num2words(hundreds, lang='bn')}শ {num2words(remainder, lang='bn')}")
            variants.add(phrase)
            variants.add(phrase.replace("োশ", "শ"))
    return variants


@lru_cache(maxsize=1)
def number_word_index() -> dict[str, int]:
    entries: dict[str, int] = {}
    for number in range(2, 10_000):
        for variant in number_word_variants(number):
            if len(variant) >= 3:
                entries[variant] = number
    entries["অষ্টাশি"] = 88
    return entries


def word_numeric_components(text: str) -> list[float]:
    value = answer_canonical(text)
    tokens = value.split()
    lookup = number_word_index()
    matches: list[tuple[int, int, str, int]] = []
    for start in range(len(tokens)):
        for end in range(start + 1, min(len(tokens), start + 8) + 1):
            phrase = " ".join(tokens[start:end])
            number = lookup.get(phrase)
            if number is None and phrase.endswith("টি"):
                number = lookup.get(phrase[:-2])
            if number is not None:
                matches.append((start, end, phrase, number))
    matches.sort(key=lambda item: (-(item[1] - item[0]), -len(item[2]), item[0]))
    occupied: set[int] = set()
    output = []
    for start, end, _, number in matches:
        span = set(range(start, end))
        if occupied & span:
            continue
        occupied.update(span)
        output.append(float(number))
    return output


def multiset_subset(left: list[float], right: list[float]) -> bool:
    remaining = Counter(right)
    for value in left:
        match = next((item for item in remaining if remaining[item] and abs(value - item) < 1e-9), None)
        if match is None:
            return False
        remaining[match] -= 1
    return bool(left)


def numeric_equivalent(response: str, sentence: str) -> bool:
    response_values = numeric_components(response)
    sentence_values = numeric_components(sentence)
    if response_values:
        if sentence_values and multiset_subset(response_values, sentence_values):
            return True
        if len(response_values) == 1 and response_values[0].is_integer() and response_values[0] < 10_000_000:
            sentence_norm = answer_canonical(sentence)
            for variant in number_word_variants(int(response_values[0])):
                if re.search(rf"(^|\s){re.escape(variant)}($|\s)", sentence_norm):
                    return True
        return False
    response_norm = answer_canonical(response)
    for value in sentence_values:
        if value.is_integer() and value < 10_000_000 and response_norm in number_word_variants(int(value)):
            return True
    return False


def cross_format_only(response: str, sentence: str) -> bool:
    """Require a genuine surface conversion, not a shared digit substring."""
    response_norm = normalize_bn(response)
    sentence_norm = normalize_bn(sentence)
    response_digits = re.findall(r"\d+(?:\.\d+)?", response_norm)
    sentence_digits = re.findall(r"\d+(?:\.\d+)?", sentence_norm)
    if bool(response_digits) != bool(sentence_digits):
        return True
    if response_digits and sentence_digits:
        # Allow Indian/western scale equivalence such as 1 crore == 10
        # million, but reject same-surface dates and years.
        response_scales = {scale for scale in SCALES if scale in response_norm}
        sentence_scales = {scale for scale in SCALES if scale in sentence_norm}
        return response_digits != sentence_digits and response_scales != sentence_scales
    return False


def relation_aligned(prompt: str, sentence: str) -> bool:
    if "কবে" in prompt and not any(
        marker in sentence for marker in ("সাল", "বছর", "খ্রিস্টাব্দ", "খ্রিষ্টাব্দ", "তারিখ")
    ):
        return False
    for triggers, requirements in RELATION_GROUPS:
        if any(trigger in prompt for trigger in triggers):
            if not any(requirement in sentence for requirement in requirements):
                return False
    return True


def split_sentences(context: str) -> list[str]:
    # Semicolons frequently separate independent factual clauses in the test
    # contexts. Keeping both clauses together can introduce a second year and
    # force an otherwise exact single-number claim to abstain.
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[।!?])\s+|[;；]+|\n+", context)
        if sentence.strip()
    ]
    clauses = []
    for sentence in sentences:
        parts = re.split(r"\s+এবং\s+", sentence)
        numeric_parts = sum(
            bool(numeric_components(part) or word_numeric_components(part))
            for part in parts
        )
        if len(parts) > 1 and numeric_parts >= 2:
            clauses.extend(part.strip() for part in parts if part.strip())
        else:
            clauses.append(sentence)
    return clauses


def overlap_score(prompt: str, sentence: str) -> float:
    question = set(bn_tokens(prompt)) - QUESTION_STOP
    evidence = set(bn_tokens(sentence)) - QUESTION_STOP
    return len(question & evidence) / max(1, len(question))


def shared_compound_topic(prompt: str, sentence: str) -> bool:
    question = normalize_bn(prompt).replace("কী বোর্ড", "কীবোর্ড")
    evidence = normalize_bn(sentence).replace("কী বোর্ড", "কীবোর্ড")
    tokens = [token for token in bn_tokens(question) if len(token) >= 5 and token not in QUESTION_STOP]
    return any(token in evidence for token in tokens)


def decide(prompt: str, response: str, context: str, min_overlap: float) -> tuple[int | None, dict]:
    if not any(marker in prompt for marker in ("কত", "কবে", "কয়", "কয়")):
        return None, {}
    if any(
        marker in prompt
        for marker in (
            "সর্বোচ্চ", "সর্বনিম্ন", "ন্যূনতম", "অনধিক", "কমপক্ষে",
            "দীর্ঘতম", "বৃহত্তম", "সর্বাধিক",
        )
    ):
        return None, {}
    candidates = []
    contradictions = []
    numeric_claims = []
    for sentence in split_sentences(context):
        score = overlap_score(prompt, sentence)
        compound_topic = shared_compound_topic(prompt, sentence)
        if (score < min_overlap and not compound_topic) or not relation_aligned(prompt, sentence):
            continue
        sentence_numbers = numeric_components(sentence)
        if not sentence_numbers:
            sentence_numbers = word_numeric_components(sentence)
        response_numbers = numeric_components(response)
        if not response_numbers:
            response_numbers = word_numeric_components(response)
        if score >= min_overlap and sentence_numbers and response_numbers:
            numeric_claims.append((score, sentence, sentence_numbers, response_numbers))
        if (
            len(sentence_numbers) >= 2 and len(response_numbers) >= 2
            and not multiset_subset(response_numbers, sentence_numbers)
        ):
            contradictions.append((score, sentence, sentence_numbers, response_numbers))
            continue
        if score < min_overlap:
            continue
        single_number_claim = len(response_numbers) == 1 and len(sentence_numbers) == 1
        if (
            len(sentence_numbers) <= 1
            and (cross_format_only(response, sentence) or single_number_claim)
            and numeric_equivalent(response, sentence)
        ):
            candidates.append((score, sentence))
    if numeric_claims:
        best_score = max(item[0] for item in numeric_claims)
        top_claims = [item for item in numeric_claims if abs(item[0] - best_score) < 1e-9]
        signatures = {tuple(sorted(item[2])) for item in top_claims}
        if len(signatures) == 1:
            score, sentence, sentence_numbers, response_numbers = top_claims[0]
            if (
                len(response_numbers) == len(sentence_numbers)
                and multiset_subset(response_numbers, sentence_numbers)
            ):
                return 1, {
                    "rule": "relation_bound_exact_numeric_signature",
                    "overlap": score,
                    "evidence_sentence": sentence,
                    "context_numbers": sentence_numbers,
                    "response_numbers": response_numbers,
                }
            if (
                len(response_numbers) == 1
                and len(sentence_numbers) == 1
                and answer_canonical(response) not in answer_canonical(context)
            ):
                return 0, {
                    "rule": "relation_bound_single_numeric_contradiction",
                    "overlap": score,
                    "evidence_sentence": sentence,
                    "context_numbers": sentence_numbers,
                    "response_numbers": response_numbers,
                }
    if candidates:
        score, sentence = max(candidates, key=lambda item: item[0])
        return 1, {"rule": "question_conditioned_numeric_equivalence", "overlap": score, "evidence_sentence": sentence}
    if contradictions:
        response_norm = answer_canonical(response)
        context_norm = answer_canonical(context)
        if response_norm and response_norm in context_norm:
            return None, {}
        score, sentence, sentence_numbers, response_numbers = max(contradictions, key=lambda item: item[0])
        return 0, {
            "rule": "question_conditioned_numeric_range_contradiction", "overlap": score,
            "evidence_sentence": sentence, "context_numbers": sentence_numbers,
            "response_numbers": response_numbers,
        }
    return None, {}


def main() -> None:
    parser = argparse.ArgumentParser(description="High-precision context numeric-equivalence overrides.")
    parser.add_argument("--split", choices=["sample", "test"], required=True)
    parser.add_argument("--base-submission", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--min-overlap", type=float, default=0.4)
    args = parser.parse_args()

    df = load_sample() if args.split == "sample" else load_test()
    base = pd.read_csv(args.base_submission) if args.base_submission else None
    records = []
    for row_index, row in df[df["has_context"]].iterrows():
        pred, evidence = decide(str(row.prompt_bn), str(row.response_bn), str(row.context), args.min_overlap)
        if pred is None:
            continue
        item = {
            "row_index": int(row_index),
            "id": int(row.id) if "id" in df.columns else int(row_index),
            "prediction": pred,
            "prompt": str(row.prompt_bn),
            "response": str(row.response_bn),
            **evidence,
        }
        if "label" in df.columns:
            item.update({"truth": int(row.label), "correct": pred == int(row.label)})
        if base is not None:
            mask = base["id"] == int(row.id)
            old = int(base.loc[mask, "label"].iloc[0])
            item.update({"old": old, "changed": old != pred})
            base.loc[mask, "label"] = pred
        records.append(item)
    summary = {
        "split": args.split,
        "min_overlap": args.min_overlap,
        "decisions": len(records),
        "correct": sum(item.get("correct", False) for item in records) if "label" in df.columns else None,
        "changed": sum(item.get("changed", False) for item in records) if base is not None else None,
    }
    if base is not None:
        if args.output is None:
            raise ValueError("--base-submission requires --output")
        base.to_csv(args.output, index=False)
        summary.update({"output": str(args.output), "output_counts": base["label"].value_counts().sort_index().to_dict()})
    args.trace.write_text(json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for item in records:
        if item.get("changed"):
            print("CHANGE", item["id"], item["old"], "->", item["prediction"], item["response"], item["evidence_sentence"][:180])


if __name__ == "__main__":
    main()
