from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
from mlx_lm import generate, load

from datathon_utils import BN_DIGITS, OUTPUT_DIR, load_test
from mlx_qwen_judge import MODELS, apply_chat


SYSTEM = """তুমি বাংলা ব্যাকরণ, শব্দার্থ ও পরীক্ষার প্রশ্নের বিশেষজ্ঞ যাচাইকারী।
প্রথমে মনে মনে প্রশ্নটির সঠিক উত্তর নির্ণয় করবে, তারপর প্রার্থিত উত্তরের সঙ্গে তুলনা করবে।

গুরুত্বপূর্ণ:
- দেওয়া প্রসঙ্গটি অনেক ক্ষেত্রে শুধু প্রশ্নের টেমপ্লেট, ব্যাসবাক্য, বা উদাহরণ; সেটি সঠিক উত্তর নাও বলতে পারে।
- প্রসঙ্গে সরাসরি প্রাসঙ্গিক উত্তর থাকলে ব্যবহার করবে। উত্তর না থাকলে বাংলা ব্যাকরণের নিয়ম ও নিজের জ্ঞান দিয়ে সমাধান করবে।
- সমাসে ব্যাসবাক্য দেখে বিভক্তি/সম্পর্ক নির্ণয় করবে।
- সন্ধিতে যুক্ত শব্দটি সত্যিই নিয়মমাফিক কি না যাচাই করবে।
- বাগধারা, বিপরীত শব্দ ও উপসর্গে প্রচলিত মান্য অর্থ/শ্রেণি ব্যবহার করবে।
- ছোট বানানভেদ গ্রহণযোগ্য; তথ্যগতভাবে আলাদা উত্তর গ্রহণযোগ্য নয়।

শুধু compact JSON দাও:
{"correct_answer":"সংক্ষিপ্ত সঠিক উত্তর","label":0 অথবা 1,"confidence":0.0 থেকে 1.0}

label=1 যদি দেওয়া উত্তরটি সঠিক বা গ্রহণযোগ্য সমার্থক হয়।
label=0 যদি দেওয়া উত্তরটি ভুল/বানোয়াট হয়।"""


def is_structured_prompt(prompt: str) -> bool:
    return any(
        marker in prompt
        for marker in [
            "সমাসের উদাহরণ",
            "সন্ধিতে কোন শব্দ গঠিত",
            "প্রবাদ/বাগধারাটির অর্থ",
            "বাগধারা/প্রবাদটির অর্থ",
            "শুদ্ধ বিপরীত শব্দ",
            "উপসর্গটি কোন শ্রেণির",
        ]
    )


def make_prompt(row) -> str:
    parts = [SYSTEM, ""]
    if row.context:
        parts.extend(["প্রসঙ্গ/সহায়ক তথ্য:", str(row.context)[:900], ""])
    parts.extend(["প্রশ্ন:", row.prompt_bn, "", "দেওয়া উত্তর:", row.response_bn, "", "JSON:"])
    return "\n".join(parts)


def parse_output(raw: str) -> tuple[int, float, str, str | None]:
    text = (raw or "").strip().translate(BN_DIGITS)
    match = re.search(r"\{.*?\}", text, flags=re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
            label = int(obj["label"])
            confidence = max(0.0, min(1.0, float(obj.get("confidence", 0.5))))
            answer = str(obj.get("correct_answer", "")).strip()
            if label in {0, 1}:
                return label, confidence, answer, None
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
                row = json.loads(line)
                rows[int(row["id"])] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge non-extractive structured Bengali rows with local MLX Qwen.")
    parser.add_argument("--model", choices=MODELS, default="qwen36_27b")
    parser.add_argument("--id-min", type=int, default=2101)
    parser.add_argument("--id-max", type=int, default=2350)
    parser.add_argument("--base-submission", type=Path, required=True)
    parser.add_argument("--trace", type=Path, default=OUTPUT_DIR / "mlx_qwen36_27b_structured_test.jsonl")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "submission_qwen36_structured.csv")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    test = load_test()
    selected = test[
        test["id"].between(args.id_min, args.id_max)
        & test["prompt_bn"].map(is_structured_prompt)
    ]
    existing = load_trace(args.trace)
    remaining = selected[~selected["id"].isin(existing)]
    print("selected", len(selected), "existing", len(existing), "remaining", len(remaining))

    if len(remaining):
        print("loading", MODELS[args.model])
        model, tokenizer = load(str(MODELS[args.model]))
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        with args.trace.open("a", encoding="utf-8") as handle:
            for offset, row in enumerate(remaining.itertuples(), start=1):
                prompt = apply_chat(tokenizer, make_prompt(row))
                raw = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False)
                label, confidence, correct_answer, parse_error = parse_output(raw)
                record = {
                    "id": int(row.id),
                    "row_index": int(row.Index),
                    "pred": label,
                    "confidence": confidence,
                    "correct_answer": correct_answer,
                    "parse_error": parse_error,
                    "raw": raw,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                existing[int(row.id)] = record
                if offset % args.progress_every == 0:
                    elapsed = (time.time() - started) / 60
                    counts = pd.Series([item["pred"] for item in existing.values()]).value_counts().to_dict()
                    print(f"done {offset}/{len(remaining)} elapsed_min={elapsed:.1f} counts={counts}", flush=True)

    base = pd.read_csv(args.base_submission)
    if len(base) != len(test) or not (base["id"].to_numpy() == test["id"].to_numpy()).all():
        raise ValueError("base submission IDs do not match test")
    output = base.copy()
    by_id = output.set_index("id")["label"].to_dict()
    changes = []
    for row_id in selected["id"]:
        item = existing.get(int(row_id))
        if item is None:
            raise ValueError(f"missing trace for id {row_id}")
        old = int(by_id[int(row_id)])
        new = int(item["pred"])
        output.loc[output["id"] == row_id, "label"] = new
        if old != new:
            changes.append((int(row_id), old, new, float(item["confidence"])))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print("wrote", args.output, output["label"].value_counts().sort_index().to_dict())
    print("overridden", len(selected), "changed", len(changes))
    print("change directions", pd.Series([(old, new) for _, old, new, _ in changes]).value_counts().to_dict())


if __name__ == "__main__":
    main()
