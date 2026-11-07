from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
from mlx_lm import generate, load

from datathon_utils import BN_DIGITS, OUTPUT_DIR, load_sample, load_test, metric_row
from mlx_qwen_judge import MODELS, apply_chat


SYSTEM = """তুমি প্রশ্ন-নির্ভর বাংলা context entailment verifier।

অবশ্যই এই ক্রমে বিচার করবে:
1. প্রশ্নটি context থেকে ঠিক কোন তথ্য চায় তা নির্ণয় করো।
2. context থেকে সেই প্রশ্নের সংক্ষিপ্ত সঠিক উত্তর বের করো।
3. প্রার্থিত উত্তরটির সঙ্গে সেই সঠিক উত্তর তুলনা করো।

কঠোর নিয়ম:
- প্রার্থিত শব্দ context-এর অন্য কোথাও আছে—এটুকু যথেষ্ট নয়; সেটি প্রশ্নের চাওয়া সম্পর্ক/তারিখ/ব্যক্তিই হতে হবে।
- “প্রথম/শেষ”, “জন্ম/মৃত্যু”, “শুরু/শেষ”, “প্রতিষ্ঠা/মর্যাদা”, ব্যক্তি ও তার আত্মীয়—এই সম্পর্কগুলো গুলিয়ে ফেলবে না।
- বাংলা/ইংরেজি সংখ্যা, কথায় লেখা সংখ্যা, বানানভেদ, একক, সংক্ষিপ্ত রূপ ও সমার্থক নাম গ্রহণযোগ্য।
- উত্তর প্রশ্নের ধরন ভুল হলে 0: যেমন প্রশ্ন সাল চায় কিন্তু উত্তর চলচ্চিত্রের নাম দেয়।
- context-এ উত্তর সরাসরি বা স্পষ্টভাবে নিহিত থাকলে গ্রহণ করো; বাইরের জ্ঞান দিয়ে context বদলাবে না।

উদাহরণ:
context: প্রথম চলচ্চিত্র অরণ্য (১৯৮৫), সর্বশেষ চলচ্চিত্র নদী (২০১০)।
প্রশ্ন: সর্বশেষ চলচ্চিত্রটি কত সালে মুক্তি পায়?
প্রার্থিত উত্তর: নদী
JSON: {"expected":"২০১০","label":0,"confidence":1.0}

context: প্রতিষ্ঠানটি ১৯১৩ সালে প্রতিষ্ঠিত হয় এবং ১৯৮৩ সালে জাতীয় মর্যাদা পায়।
প্রশ্ন: প্রতিষ্ঠানটি কত সালে প্রতিষ্ঠিত হয়?
প্রার্থিত উত্তর: ১৯৮৩
JSON: {"expected":"১৯১৩","label":0,"confidence":1.0}

context: করিম বাংলাদেশ জাতীয়তাবাদী দলের স্থায়ী কমিটির সদস্য ছিলেন।
প্রশ্ন: করিম কোন দলের সদস্য ছিলেন?
প্রার্থিত উত্তর: বিএনপি
JSON: {"expected":"বাংলাদেশ জাতীয়তাবাদী দল (বিএনপি)","label":1,"confidence":1.0}

শুধু compact JSON দাও:
{"expected":"context থেকে সংক্ষিপ্ত উত্তর","label":0 অথবা 1,"confidence":0.0 থেকে 1.0}"""


def make_prompt(row) -> str:
    return "\n".join(
        [
            SYSTEM,
            "",
            "context:",
            str(row.context)[:2600],
            "",
            "প্রশ্ন:",
            str(row.prompt_bn),
            "",
            "প্রার্থিত উত্তর:",
            str(row.response_bn),
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
            expected = str(obj.get("expected", "")).strip()
            if label in {0, 1}:
                return label, confidence, expected, None
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            error = f"json:{exc}"
        else:
            error = "json:invalid_label"
    else:
        error = "json:not_found"
    labels = re.findall(r'"?label"?\s*[:=]\s*([01])', text)
    if labels:
        return int(labels[-1]), 0.5, "", error
    return 0, 0.0, "", error + ";label:not_found"


def load_trace(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    output = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                output[int(row["row_index"])] = row
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Question-conditioned context verifier with local MLX Qwen.")
    parser.add_argument("--split", choices=["sample", "test"], required=True)
    parser.add_argument("--model", choices=MODELS, default="qwen35_9b")
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--base-submission", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--id-min", type=int)
    parser.add_argument("--id-max", type=int)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    df = load_sample() if args.split == "sample" else load_test()
    selected = df[df["has_context"]].copy()
    if args.id_min is not None:
        selected = selected[selected["id"] >= args.id_min]
    if args.id_max is not None:
        selected = selected[selected["id"] <= args.id_max]
    trace_path = args.trace or OUTPUT_DIR / f"mlx_{args.model}_context_verifier_{args.split}.jsonl"
    existing = load_trace(trace_path)
    remaining = selected[~selected.index.isin(existing)]
    print("selected", len(selected), "existing", len(existing), "remaining", len(remaining))

    if len(remaining):
        print("loading", MODELS[args.model])
        model, tokenizer = load(str(MODELS[args.model]))
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        with trace_path.open("a", encoding="utf-8") as handle:
            for offset, (row_index, row) in enumerate(remaining.iterrows(), start=1):
                raw = generate(
                    model,
                    tokenizer,
                    prompt=apply_chat(tokenizer, make_prompt(row)),
                    max_tokens=args.max_tokens,
                    verbose=False,
                )
                label, confidence, expected, parse_error = parse_output(raw)
                record = {
                    "row_index": int(row_index),
                    "id": int(row["id"]) if "id" in row else None,
                    "pred": label,
                    "confidence": confidence,
                    "expected": expected,
                    "parse_error": parse_error,
                    "raw": raw,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                existing[int(row_index)] = record
                if offset % 25 == 0:
                    print(f"done {offset}/{len(remaining)} elapsed_min={(time.time()-started)/60:.1f}", flush=True)

    predictions = pd.Series({i: int(existing[int(i)]["pred"]) for i in selected.index})
    if args.split == "sample":
        y = selected["label"].astype(int).to_numpy()
        pred = predictions.loc[selected.index].to_numpy()
        print(json.dumps(metric_row(y, pred), indent=2))
        wrong = selected.index[y != pred].tolist()
        print("wrong_indices", wrong)

    if args.output:
        if args.split != "test" or args.base_submission is None:
            raise ValueError("--output requires --split test and --base-submission")
        base = pd.read_csv(args.base_submission)
        output = base.copy()
        changed = 0
        for row_index, row in selected.iterrows():
            mask = output["id"] == row["id"]
            old = int(output.loc[mask, "label"].iloc[0])
            new = int(predictions.loc[row_index])
            output.loc[mask, "label"] = new
            changed += old != new
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(args.output, index=False)
        print("wrote", args.output, "changed", changed, output.label.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
