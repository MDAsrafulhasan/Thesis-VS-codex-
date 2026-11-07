from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from datathon_utils import OUTPUT_DIR, clean_context, clean_text, prompt_category


DEFAULT_INPUTS = [
    OUTPUT_DIR / "lexical_controlled_pairs_80k.jsonl",
    OUTPUT_DIR / "external_qa_pairs.jsonl",
    OUTPUT_DIR / "wikidata_bangladesh_pairs.jsonl",
    OUTPUT_DIR / "external_context_contrastive_pairs.jsonl",
    OUTPUT_DIR / "synthetic_plus_curated.jsonl",
]


def source_name(path: Path) -> str:
    name = path.name
    if name.endswith(".jsonl"):
        name = name[:-6]
    return name


def normalize_row(row: dict[str, Any], path: Path, row_number: int) -> dict[str, Any] | None:
    if "label" not in row:
        return None
    try:
        label = int(row["label"])
    except (TypeError, ValueError):
        return None
    if label not in {0, 1}:
        return None

    context = clean_context(row.get("context", ""))
    prompt = clean_text(row.get("prompt_bn", ""))
    response = clean_text(row.get("response_bn", ""))
    if not prompt or not response:
        return None

    source = clean_text(row.get("source", "")) or source_name(path)
    return {
        "teacher_item_id": f"{source_name(path)}:{row_number}",
        "context": context,
        "prompt_bn": prompt,
        "response_bn": response,
        "label": label,
        "gold_label": label,
        "source": source,
        "source_repo": clean_text(row.get("source_repo", "")),
        "source_file": path.as_posix(),
        "category": clean_text(row.get("category", "")) or prompt_category(prompt),
        "has_context": bool(context),
    }


def reservoir_add(bucket: list[dict[str, Any]], row: dict[str, Any], seen: int, limit: int, rng: random.Random) -> None:
    if len(bucket) < limit:
        bucket.append(row)
        return
    idx = rng.randrange(seen)
    if idx < limit:
        bucket[idx] = row


def sample_file(path: Path, rows_per_file: int, seed: int) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    per_label = max(1, rows_per_file // 2)
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen: dict[int, int] = defaultdict(int)
    rng = random.Random(seed + sum(path.as_posix().encode("utf-8")))

    with path.open(encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = normalize_row(json.loads(line), path, row_number)
            if row is None:
                continue
            label = int(row["label"])
            seen[label] += 1
            reservoir_add(buckets[label], row, seen[label], per_label, rng)

    rows = buckets[0] + buckets[1]
    rng.shuffle(rows)
    print(
        "sampled",
        path,
        "seen",
        dict(seen),
        "kept",
        {label: len(buckets[label]) for label in sorted(buckets)},
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a balanced pool for open-weight LLM teacher labeling.")
    parser.add_argument("--input", action="append", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "teacher_pool.jsonl")
    parser.add_argument("--rows-per-file", type=int, default=1000)
    parser.add_argument("--max-total", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = args.input or DEFAULT_INPUTS
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(sample_file(path, rows_per_file=args.rows_per_file, seed=args.seed))

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.max_total and len(rows) > args.max_total:
        rows = rows[: args.max_total]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows):
            row["teacher_pool_index"] = idx
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = {
        "rows": len(rows),
        "labels": {str(label): sum(int(row["label"]) == label for row in rows) for label in [0, 1]},
        "has_context": {
            "false": sum(not row["has_context"] for row in rows),
            "true": sum(bool(row["has_context"]) for row in rows),
        },
    }
    print("wrote", args.output, json.dumps(counts, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
