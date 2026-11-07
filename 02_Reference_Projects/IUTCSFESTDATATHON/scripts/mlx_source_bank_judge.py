from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
from mlx_lm import generate, load

from datathon_utils import BN_DIGITS, OUTPUT_DIR, load_sample, load_test, metric_row, normalize_bn
from mlx_qwen_judge import MODELS, apply_chat
from source_answer_bank import BankEntry, compare_response, decide_row, index_bank, read_bank


SYSTEM = """তুমি প্রমাণ-নির্ভর বাংলা উত্তর যাচাইকারী।
তোমাকে প্রশ্ন, প্রার্থিত উত্তর, এবং এক বা একাধিক প্রকাশিত answer-key দেওয়া হবে।

কাজ:
- প্রসঙ্গ দেওয়া থাকলে সেটিই সর্বোচ্চ অগ্রাধিকার: উত্তরটি প্রশ্নের দাবিসহ প্রসঙ্গ দ্বারা সরাসরি সমর্থিত হতে হবে।
- answer-key-তে একই entity থাকলেও প্রসঙ্গ যদি শুধু দুর্বল/ভিন্ন দাবি সমর্থন করে, তবে 0।
- answer-key-র সঠিক উত্তর/সঠিক MCQ option-এর সঙ্গে প্রার্থিত উত্তরের অর্থ তুলনা করো।
- বাংলা/ইংরেজি সংখ্যা, কথায় লেখা সংখ্যা, একক, উপাধি, সংক্ষিপ্ত রূপ, বানানভেদ ও সমার্থক রূপ গ্রহণযোগ্য।
- ভুল option, ভুল ব্যক্তি/স্থান/তারিখ/সংখ্যা হলে 0।
- answer-key-তে স্পষ্ট OCR/টাইপো থাকলে প্রশ্ন ও options দেখে যুক্তিসঙ্গত সংশোধন করো।
- নিজের অপ্রাসঙ্গিক সাধারণ জ্ঞান দিয়ে প্রকাশিত evidence উপেক্ষা করো না।

শুধু compact JSON দাও:
{"label":0 অথবা 1,"confidence":0.0 থেকে 1.0}"""


def evidence_text(entries: list[BankEntry]) -> str:
    parts = []
    for rank, entry in enumerate(entries[:4], start=1):
        block = [f"[{rank}] source={entry.source_repo}", f"correct_answer={entry.answers[0]}"]
        if entry.choices:
            for i, choice in enumerate(entry.choices):
                marker = "CORRECT" if i == entry.correct_index else "wrong"
                block.append(f"{chr(65+i)} ({marker}): {choice}")
        parts.append("\n".join(block))
    return "\n\n".join(parts)


def make_prompt(row, entries: list[BankEntry]) -> str:
    parts = [SYSTEM]
    context = str(getattr(row, "context", "") or "").strip()
    if context:
        parts.extend(["", "প্রসঙ্গ (সর্বোচ্চ অগ্রাধিকার):", context[:2400]])
    parts.extend(
        [
            "",
            "প্রশ্ন:",
            str(row.prompt_bn),
            "",
            "প্রার্থিত উত্তর:",
            str(row.response_bn),
            "",
            "প্রকাশিত answer-key:",
            evidence_text(entries),
            "",
            "JSON:",
        ]
    )
    return "\n".join(parts)


def parse_output(raw: str) -> tuple[int, float, str | None]:
    text = (raw or "").strip().translate(BN_DIGITS)
    match = re.search(r"\{.*?\}", text, flags=re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
            label = int(obj["label"])
            confidence = max(0.0, min(1.0, float(obj.get("confidence", 0.5))))
            if label in {0, 1}:
                return label, confidence, None
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            error = f"json:{exc}"
        else:
            error = "json:invalid_label"
    else:
        error = "json:not_found"
    labels = re.findall(r'"?label"?\s*[:=]\s*([01])', text)
    if labels:
        return int(labels[-1]), 0.5, error
    return 0, 0.0, error + ";label:not_found"


def load_trace(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    output = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                output[int(row["id"])] = row
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Use local Qwen as a semantic comparator for answer-bank abstentions.")
    parser.add_argument("--split", choices=["sample", "test"], required=True)
    parser.add_argument("--model", choices=MODELS, default="qwen35_9b")
    parser.add_argument("--bank", type=Path, default=OUTPUT_DIR / "source_answer_bank.jsonl")
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--base-submission", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--all-matched", action="store_true")
    args = parser.parse_args()

    df = load_sample() if args.split == "sample" else load_test()
    entries = read_bank(args.bank)
    bank_index = index_bank(entries)
    selected = []
    for row in df.itertuples():
        sources = bank_index.get(normalize_bn(row.prompt_bn), [])
        if not sources:
            continue
        deterministic, _ = decide_row(str(row.response_bn), sources)
        if args.all_matched or deterministic is None:
            selected.append((row, sources))

    trace_path = args.trace or OUTPUT_DIR / f"mlx_{args.model}_source_bank_{args.split}.jsonl"
    existing = load_trace(trace_path)
    remaining = [item for item in selected if int(getattr(item[0], "id", item[0].Index)) not in existing]
    print("selected", len(selected), "existing", len(existing), "remaining", len(remaining))

    if remaining:
        print("loading", MODELS[args.model])
        model, tokenizer = load(str(MODELS[args.model]))
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        with trace_path.open("a", encoding="utf-8") as handle:
            for offset, (row, sources) in enumerate(remaining, start=1):
                raw = generate(
                    model,
                    tokenizer,
                    prompt=apply_chat(tokenizer, make_prompt(row, sources)),
                    max_tokens=args.max_tokens,
                    verbose=False,
                )
                label, confidence, parse_error = parse_output(raw)
                row_id = int(getattr(row, "id", row.Index))
                record = {
                    "id": row_id,
                    "pred": label,
                    "confidence": confidence,
                    "parse_error": parse_error,
                    "sources": sorted({entry.source_repo for entry in sources}),
                    "raw": raw,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                existing[row_id] = record
                if offset % 10 == 0:
                    print(f"done {offset}/{len(remaining)} elapsed_min={(time.time()-started)/60:.1f}", flush=True)

    chosen = [existing[int(getattr(row, "id", row.Index))] for row, _ in selected]
    if args.split == "sample":
        y = pd.Series([int(row.label) for row, _ in selected]).to_numpy()
        pred = pd.Series([int(item["pred"]) for item in chosen]).to_numpy()
        print(json.dumps(metric_row(y, pred), indent=2))
        wrong = [
            (int(row.Index), int(row.label), int(item["pred"]), float(item["confidence"]), item["raw"])
            for (row, _), item in zip(selected, chosen)
            if int(row.label) != int(item["pred"])
        ]
        print("wrong", wrong)

    if args.output:
        if args.split != "test" or args.base_submission is None:
            raise ValueError("--output requires --split test and --base-submission")
        base = pd.read_csv(args.base_submission)
        output = base.copy()
        changed = 0
        for (row, _), item in zip(selected, chosen):
            mask = output["id"] == row.id
            old = int(output.loc[mask, "label"].iloc[0])
            new = int(item["pred"])
            output.loc[mask, "label"] = new
            changed += old != new
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(args.output, index=False)
        print("wrote", args.output, "changed", changed, output.label.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
