from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from datathon_utils import clean_text, normalize_bn


NUMBER_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)(?!\d)")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write(handle, row: dict, seen: set[tuple]) -> bool:
    key = (
        normalize_bn(row.get("context", "")),
        normalize_bn(row.get("prompt_bn", "")),
        normalize_bn(row.get("response_bn", "")),
        int(row["label"]),
    )
    if not key[1] or not key[2] or key in seen:
        return False
    seen.add(key)
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return True


def corrupt_number(text: str) -> str | None:
    value = normalize_bn(text)
    match = NUMBER_RE.search(value)
    if not match:
        return None
    old = match.group(1)
    try:
        number = float(old)
    except ValueError:
        return None
    if number.is_integer():
        replacement = str(int(number) + (1 if number < 100 else 7))
    else:
        replacement = format(number + 0.1, ".10g")
    return value[: match.start()] + replacement + value[match.end() :]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build provenance-preserving Bengali verifier pairs.")
    parser.add_argument("--nli", type=Path, action="append", default=[])
    parser.add_argument("--evidence", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-evidence-rows", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    evidence = []
    for path in args.evidence:
        evidence.extend(read_jsonl(path))
    if len(evidence) > args.max_evidence_rows:
        evidence = rng.sample(evidence, args.max_evidence_rows)
    by_domain = defaultdict(list)
    for row in evidence:
        by_domain[row.get("domain", "unknown")].append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen, counts = set(), Counter()
    with args.output.open("w", encoding="utf-8") as out:
        for path in args.nli:
            for row in read_jsonl(path):
                if write(out, row, seen):
                    counts[row.get("hallucination_type", "nli")] += 1

        for row in evidence:
            base = {
                "prompt_bn": clean_text(row["question"]),
                "response_bn": clean_text(row["answer"]),
                "context": clean_text(row.get("context", "")),
                "source": row.get("source_repo", "unknown"),
                "source_file": row.get("source_file", ""),
                "domain": row.get("domain", "unknown"),
                "license": row.get("license", "unknown"),
                "split": "external",
            }
            positive = {**base, "label": 1, "hallucination_type": "faithful_evidence_answer"}
            if write(out, positive, seen):
                counts[positive["hallucination_type"]] += 1

            pool = by_domain[base["domain"]]
            if len(pool) > 1:
                decoy = rng.choice(pool)
                for _ in range(8):
                    if normalize_bn(decoy["answer"]) != normalize_bn(base["response_bn"]):
                        break
                    decoy = rng.choice(pool)
                negative = {
                    **base,
                    "response_bn": clean_text(decoy["answer"]),
                    "label": 0,
                    "hallucination_type": "same_domain_unsupported_answer",
                    "decoy_source_file": decoy.get("source_file", ""),
                }
                if write(out, negative, seen):
                    counts[negative["hallucination_type"]] += 1

            corrupted = corrupt_number(base["response_bn"])
            if corrupted and normalize_bn(corrupted) not in normalize_bn(base["context"]):
                negative = {
                    **base,
                    "response_bn": corrupted,
                    "label": 0,
                    "hallucination_type": "numeric_corruption",
                }
                if write(out, negative, seen):
                    counts[negative["hallucination_type"]] += 1

    summary = {"output": str(args.output), "rows": sum(counts.values()), "types": dict(counts)}
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
