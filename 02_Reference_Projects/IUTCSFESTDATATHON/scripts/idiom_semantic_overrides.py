from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from datathon_utils import clean_context
from source_answer_bank import canonical_answer


IDIOM_PROMPT_PATTERNS = (
    re.compile(r"[\"'“‘]([^\"'”’]+)[\"'”’]\s*(?:বাগধারা|প্রবাদ)\s*/\s*(?:প্রবাদ|বাগধারা)"),
    re.compile(r"[\"'“‘]([^\"'”’]+)[\"'”’]\s*(?:প্রবাদ|বাগধারা)\s*/\s*(?:বাগধারা|প্রবাদ)"),
)


def extract_idiom(prompt: str) -> str | None:
    """Extract the quoted idiom only from prompts that explicitly ask for an idiom/proverb."""
    for pattern in IDIOM_PROMPT_PATTERNS:
        match = pattern.search(str(prompt))
        if match:
            return canonical_answer(match.group(1))
    return None


def load_idiom_dataset(path: Path) -> dict[str, dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict] = {}
    for item in records:
        names = [item.get("idiom", ""), *item.get("alternative_idioms", [])]
        for name in names:
            key = canonical_answer(str(name))
            if key:
                result[key] = item
    return result


def load_corrections(path: Path, idioms: dict[str, dict]) -> dict[tuple[str, str], dict]:
    result: dict[tuple[str, str], dict] = {}
    for item in json.loads(path.read_text(encoding="utf-8")):
        idiom = canonical_answer(item["idiom"])
        source = idioms.get(idiom)
        if source is None:
            raise ValueError(f"correction idiom is absent from the cited dataset: {item['idiom']}")
        expected = canonical_answer(item["figurative_meaning_bn"])
        actual = canonical_answer(source.get("figurative_meaning_bn", ""))
        if expected != actual:
            raise ValueError(f"figurative meaning disagrees with cited dataset: {item['idiom']}")
        if len(set(item.get("sources", []))) < 2:
            raise ValueError(f"correction needs two independent source URLs: {item['idiom']}")
        label = int(item["correct_label"])
        if label not in (0, 1):
            raise ValueError(f"invalid label for {item['idiom']}: {label}")
        for response in item["rejected_responses"]:
            key = (idiom, canonical_answer(response))
            if key in result:
                raise ValueError(f"duplicate idiom/response correction: {key}")
            result[key] = {**item, "source_dataset_record": source}
    return result


def correction_for(prompt: str, response: str, corrections: dict[tuple[str, str], dict]) -> dict | None:
    idiom = extract_idiom(prompt)
    if idiom is None:
        return None
    return corrections.get((idiom, canonical_answer(response)))


def apply_corrections(
    frame: pd.DataFrame,
    base: pd.DataFrame,
    corrections: dict[tuple[str, str], dict],
) -> tuple[pd.DataFrame, list[dict]]:
    if list(base.columns) != ["id", "label"]:
        raise ValueError("base submission must have exactly the columns id,label")
    if len(frame) != len(base) or not frame["id"].equals(base["id"]):
        raise ValueError("base submission IDs do not match the input frame")

    output = base.copy()
    trace: list[dict] = []
    for row in frame.itertuples(index=False):
        correction = correction_for(str(row.prompt_bn), str(row.response_bn), corrections)
        if correction is None:
            continue
        mask = output["id"] == row.id
        old = int(output.loc[mask, "label"].iloc[0])
        new = int(correction["correct_label"])
        output.loc[mask, "label"] = new
        source = correction["source_dataset_record"]
        trace.append(
            {
                "id": int(row.id),
                "old": old,
                "new": new,
                "changed": old != new,
                "idiom": correction["idiom"],
                "prompt": str(row.prompt_bn),
                "response": str(row.response_bn),
                "context": clean_context(row.context),
                "source_literal_meaning": source.get("literal_meaning"),
                "source_figurative_meaning": source.get("figurative_meaning_bn"),
                "reason": correction["reason"],
                "sources": correction["sources"],
                "rule": "exact_idiom_literal_figurative_correction",
            }
        )
    return output, trace


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply exact content-keyed literal-vs-figurative idiom corrections to a base submission."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--base-submission", type=Path, required=True)
    parser.add_argument("--idiom-dataset", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    required = {"id", "context", "prompt_bn", "response_bn"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"input is missing columns: {sorted(missing)}")
    base = pd.read_csv(args.base_submission)
    idioms = load_idiom_dataset(args.idiom_dataset)
    corrections = load_corrections(args.corrections, idioms)
    output, records = apply_corrections(frame, base, corrections)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    summary = {
        "input_rows": len(frame),
        "source_idioms": len(idioms),
        "matched_corrections": len(records),
        "changed": sum(item["changed"] for item in records),
        "output": str(args.output),
        "output_counts": output["label"].value_counts().sort_index().to_dict(),
    }
    args.trace.write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for item in records:
        print("CHANGE", item["id"], item["old"], "->", item["new"], item["idiom"], item["response"])


if __name__ == "__main__":
    main()
