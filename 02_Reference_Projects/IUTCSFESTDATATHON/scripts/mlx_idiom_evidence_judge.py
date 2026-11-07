from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
from mlx_lm import generate, load

from datathon_utils import BN_DIGITS, OUTPUT_DIR, load_test, normalize_bn
from mlx_qwen_judge import MODELS, apply_chat


SYSTEM = """তুমি বাংলা বাগধারার উত্তরের কঠোর যাচাইকারী।
প্রশ্নের বাগধারাটির জন্য একটি অভিধান-এন্ট্রি দেওয়া হবে। অভিধানের ভাবার্থ এবং প্রার্থিত উত্তর একই অর্থ প্রকাশ করে কি না শুধু সেটি নির্ণয় করো।

- সমার্থক শব্দ, সংক্ষিপ্ত ব্যাখ্যা, এবং ছোট বানানভেদ গ্রহণযোগ্য।
- আংশিকভাবে সম্পর্কিত হলেও মূল অর্থ আলাদা হলে গ্রহণযোগ্য নয়।
- অভিধান-এন্ট্রির বাইরে নিজের স্মৃতি দিয়ে সিদ্ধান্ত বদলাবে না।

শুধু compact JSON দাও:
{"label":0 অথবা 1,"confidence":0.0 থেকে 1.0,"reason":"খুব সংক্ষিপ্ত"}

label=1 মানে অভিধানের ভাবার্থের সঙ্গে উত্তরটি অর্থগতভাবে মেলে।
label=0 মানে মেলে না।"""


def idiom_from_prompt(prompt: str) -> str | None:
    if "বাগধারা" not in prompt and "প্রবাদ" not in prompt:
        return None
    match = re.search(r"'([^']+)'", prompt)
    return match.group(1).strip() if match else None


def compact_evidence(context: str) -> str:
    text = str(context).strip()
    match = re.search(r"বাগধারা:\s*(.*?)\s+শাব্দিক অর্থ:.*?\s+ভাবার্থ:\s*(.*?)(?:\s+বিকল্প রূপ:|\s+উদাহরণ:|$)", text)
    if match:
        return f"বাগধারা: {match.group(1).strip()}\nভাবার্থ: {match.group(2).strip()}"
    return text[:1100]


def build_evidence_index(corpus_path: Path) -> list[dict]:
    records = []
    with corpus_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            context = str(item.get("context", ""))
            if "বাগধারা:" not in context or "ভাবার্থ:" not in context:
                continue
            headword_match = re.match(r"\s*বাগধারা:\s*(.*?)\s+শাব্দিক অর্থ:", context)
            if not headword_match:
                continue
            records.append(
                {
                    "normalized": normalize_bn(context),
                    "headword": normalize_bn(headword_match.group(1)),
                    "evidence": compact_evidence(context),
                    "source_repo": item.get("source_repo", ""),
                    "source_file": item.get("source_file", ""),
                }
            )
    return records


def find_exact_entry(idiom: str, records: list[dict]) -> dict | None:
    needle = normalize_bn(idiom)
    # Do not treat a mention in an example or an "alternative form" field as
    # exact evidence: a few crowd-sourced alternative mappings are noisy.
    candidates = [record for record in records if needle and record["headword"] == needle]
    if not candidates:
        return None
    candidates.sort(key=lambda record: len(record["evidence"]))
    return candidates[0]


def make_prompt(idiom: str, response: str, evidence: str) -> str:
    return "\n".join(
        [
            SYSTEM,
            "",
            "অভিধান-এন্ট্রি:",
            evidence,
            "",
            f"যাচাইযোগ্য বাগধারা: {idiom}",
            f"প্রার্থিত উত্তর: {response}",
            "",
            "JSON:",
        ]
    )


def parse_output(raw: str) -> tuple[int, float, str, str | None]:
    text = (raw or "").strip().translate(BN_DIGITS)
    match = re.search(r"\{.*?\}", text, flags=re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
            label = int(obj["label"])
            confidence = max(0.0, min(1.0, float(obj.get("confidence", 0.5))))
            reason = str(obj.get("reason", "")).strip()
            if label in {0, 1}:
                return label, confidence, reason, None
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            parse_error = f"json:{exc}"
        else:
            parse_error = "json:invalid_label"
    else:
        parse_error = "json:not_found"
    labels = re.findall(r'"?label"?\s*[:=]\s*([01])', text)
    if labels:
        return int(labels[-1]), 0.5, "", parse_error
    return 0, 0.0, "", parse_error + ";label:not_found"


def load_trace(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                rows[int(item["id"])] = item
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify exact-retrieved Bengali idioms with a local MLX judge.")
    parser.add_argument("--model", choices=MODELS, default="qwen36_27b")
    parser.add_argument("--base-submission", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=OUTPUT_DIR / "lexical_evidence_corpus.jsonl")
    parser.add_argument("--trace", type=Path, default=OUTPUT_DIR / "mlx_qwen36_27b_idiom_evidence.jsonl")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "submission_idiom_evidence.csv")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    test = load_test()
    records = build_evidence_index(args.corpus)
    selected = []
    for row in test.itertuples():
        idiom = idiom_from_prompt(str(row.prompt_bn))
        evidence = find_exact_entry(idiom, records) if idiom else None
        if evidence:
            selected.append((row, idiom, evidence))

    existing = load_trace(args.trace)
    remaining = [item for item in selected if int(item[0].id) not in existing]
    print("selected", len(selected), "existing", len(existing), "remaining", len(remaining))

    if remaining:
        print("loading", MODELS[args.model])
        model, tokenizer = load(str(MODELS[args.model]))
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        with args.trace.open("a", encoding="utf-8") as handle:
            for offset, (row, idiom, evidence) in enumerate(remaining, start=1):
                prompt = apply_chat(tokenizer, make_prompt(idiom, str(row.response_bn), evidence["evidence"]))
                raw = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False)
                label, confidence, reason, parse_error = parse_output(raw)
                record = {
                    "id": int(row.id),
                    "pred": label,
                    "confidence": confidence,
                    "idiom": idiom,
                    "response": str(row.response_bn),
                    "evidence": evidence["evidence"],
                    "source_repo": evidence["source_repo"],
                    "source_file": evidence["source_file"],
                    "reason": reason,
                    "parse_error": parse_error,
                    "raw": raw,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                existing[int(row.id)] = record
                if offset % 10 == 0:
                    elapsed = (time.time() - started) / 60
                    print(f"done {offset}/{len(remaining)} elapsed_min={elapsed:.1f}", flush=True)

    base = pd.read_csv(args.base_submission)
    if len(base) != len(test) or not (base["id"].to_numpy() == test["id"].to_numpy()).all():
        raise ValueError("base submission IDs do not match test")
    output = base.copy()
    changes = []
    for row, _, _ in selected:
        item = existing[int(row.id)]
        old = int(output.loc[output["id"] == row.id, "label"].iloc[0])
        new = int(item["pred"])
        output.loc[output["id"] == row.id, "label"] = new
        if old != new:
            changes.append((int(row.id), old, new, float(item["confidence"])))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print("wrote", args.output, output["label"].value_counts().sort_index().to_dict())
    print("overridden", len(selected), "changed", len(changes))
    print("changes", changes)


if __name__ == "__main__":
    main()
