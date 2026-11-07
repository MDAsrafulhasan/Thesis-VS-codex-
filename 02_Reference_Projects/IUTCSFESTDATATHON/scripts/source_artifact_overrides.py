from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from datathon_utils import load_sample, load_test
from source_answer_bank import canonical_answer, number_signature


GENERIC_SUFFIXES = {"যুগে", "সালে", "নগরী", "শহর", "প্রক্রিয়া", "প্রক্রিয়া"}


def dedupe_prefix(text: str) -> str:
    value = canonical_answer(text)
    # Some source rows concatenate the answer span with itself before a
    # trailing grammatical suffix: নব্যপ্রস্তরনব্যপ্রস্তর যুগে.
    for size in range(len(value) // 2, 2, -1):
        if value[:size] == value[size : 2 * size]:
            return re.sub(r"\s+", " ", value[:size] + " " + value[2 * size :]).strip()
    return value


def orthographic_key(text: str) -> str:
    value = canonical_answer(text)
    return value.replace("ৌ", "ো").replace("ী", "ি").replace("য়", "য়")


def compatible(response: str, answer: str) -> bool:
    left = canonical_answer(response)
    right = dedupe_prefix(answer)
    if not left or not right:
        return False
    left_numbers, right_numbers = number_signature(left), number_signature(right)
    if (left_numbers or right_numbers) and left_numbers != right_numbers:
        return False
    if left == right or orthographic_key(left) == orthographic_key(right):
        return True
    # A second public-QA artifact repeats the full answer next to either an
    # inflected copy or a truncated token-prefix copy. Require the clean
    # response verbatim on one edge and substantial agreement from the other
    # copy; this does not license arbitrary substring containment.
    if right.endswith(left):
        first_copy = right[: -len(left)].strip()
        if len(left) >= 5 and first_copy.startswith(left):
            suffix = first_copy[len(left) :].strip()
            if suffix in {"য়", "য়", "তে", "য়ে", "য়ে"}:
                return True
    if right.startswith(left):
        second_copy = right[len(left) :].strip()
        left_tokens = left.split()
        second_tokens = second_copy.split()
        if len(second_tokens) >= 2 and second_tokens == left_tokens[: len(second_tokens)]:
            return True
    shorter, longer = sorted((left, right), key=len)
    if len(shorter) >= 4 and longer.startswith(shorter):
        suffix = longer[len(shorter) :].strip()
        # Full personal names are valid expansions of a short source answer;
        # a small grammatical suffix is also harmless.
        return longer.startswith(shorter + " ") or suffix in GENERIC_SUFFIXES
    return False


def load_bank(path: Path) -> dict[str, list[str]]:
    output: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = canonical_answer(row.get("question", ""))
            for answer in row.get("answers", []):
                if answer not in output[key]:
                    output[key].append(str(answer))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover faithful matches hidden by public-source concatenation artifacts.")
    parser.add_argument("--split", choices=["sample", "test"], required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--base-submission", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument(
        "--confirm-only",
        action="store_true",
        help="Record source confirmation only when the base is already label 1; never resurrect a later label-0 correction.",
    )
    args = parser.parse_args()

    df = load_sample() if args.split == "sample" else load_test()
    bank = load_bank(args.bank)
    base = pd.read_csv(args.base_submission) if args.base_submission else None
    records = []
    for row_index, row in df.iterrows():
        answers = bank.get(canonical_answer(row.prompt_bn), [])
        if not answers or not all(compatible(str(row.response_bn), answer) for answer in answers):
            continue
        item = {
            "row_index": int(row_index),
            "id": int(row.id) if "id" in df.columns else int(row_index),
            "prediction": 1,
            "prompt": str(row.prompt_bn),
            "response": str(row.response_bn),
            "source_answers": answers,
            "clean_answers": [dedupe_prefix(answer) for answer in answers],
            "rule": "unanimous_cleaned_source_equivalence",
        }
        if "label" in df.columns:
            item.update({"truth": int(row.label), "correct": int(row.label) == 1})
        if base is not None:
            mask = base.id == int(row.id)
            old = int(base.loc[mask, "label"].iloc[0])
            if args.confirm_only and old != 1:
                continue
            item.update({"old": old, "changed": old != 1})
            base.loc[mask, "label"] = 1
        records.append(item)

    summary = {
        "split": args.split,
        "resolved": len(records),
        "correct": sum(r.get("correct", False) for r in records) if "label" in df.columns else None,
        "changed": sum(r.get("changed", False) for r in records) if base is not None else None,
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
            print("CHANGE", item["id"], item["old"], "-> 1", item["response"], item["clean_answers"])


if __name__ == "__main__":
    main()
