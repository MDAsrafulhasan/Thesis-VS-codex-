from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from mlx_lm import generate, load

from datathon_utils import BN_DIGITS, OUTPUT_DIR, clean_context, clean_text
from mlx_qwen_judge import MODELS, apply_chat


SYSTEM_INSTRUCTIONS = """তুমি বাংলা hallucination detection teacher model।
তোমাকে প্রসঙ্গ (থাকলে), প্রশ্ন, এবং উত্তর দেওয়া হবে।

লেবেল:
0 = উত্তরটি ভুল, বানানো, প্রশ্নের উত্তর নয়, প্রসঙ্গ দ্বারা অসমর্থিত, বা তথ্যগতভাবে অসঙ্গত।
1 = উত্তরটি সঠিক, প্রশ্নের উত্তর দেয়, এবং প্রসঙ্গ থাকলে প্রসঙ্গ দ্বারা সমর্থিত।

নিয়ম:
- প্রসঙ্গ থাকলে প্রসঙ্গকেই প্রধান প্রমাণ ধরবে।
- প্রসঙ্গ না থাকলে নিজের সাধারণ জ্ঞান দিয়ে যাচাই করবে।
- বানান/রূপভেদের ছোট পার্থক্য উপেক্ষা করবে, কিন্তু ভুল ব্যক্তি/স্থান/তারিখ/সংখ্যা/অর্থ ধরবে।
- উৎস বা metadata দিয়ে সিদ্ধান্ত নেবে না।
- শুধু compact JSON লিখবে: {"label":0 বা 1,"confidence":0.0 থেকে 1.0}"""


def make_teacher_prompt(row: dict[str, Any], max_context_chars: int) -> str:
    context = clean_context(row.get("context", ""))
    prompt = clean_text(row.get("prompt_bn", ""))
    response = clean_text(row.get("response_bn", ""))
    parts = [SYSTEM_INSTRUCTIONS, ""]
    if context:
        parts.append("প্রসঙ্গ:\n" + context[:max_context_chars])
        parts.append("")
    parts.extend(
        [
            "প্রশ্ন:\n" + prompt,
            "",
            "উত্তর:\n" + response,
            "",
            'JSON:',
        ]
    )
    return "\n".join(parts)


def parse_teacher_output(text: str) -> tuple[int, float, str | None]:
    raw = (text or "").strip().translate(BN_DIGITS)
    json_match = re.search(r"\{.*?\}", raw, flags=re.S)
    if json_match:
        try:
            obj = json.loads(json_match.group(0))
            label = int(obj["label"])
            confidence = float(obj.get("confidence", 0.5))
            if label in {0, 1}:
                return label, max(0.0, min(1.0, confidence)), None
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return 0, 0.0, f"json_parse_error:{exc}"

    label_match = re.search(r'"?label"?\s*[:=]\s*([01])', raw)
    if not label_match:
        label_match = re.search(r"(?<!\d)([01])(?!\d)", raw)
    label = int(label_match.group(1)) if label_match else 0

    conf_match = re.search(r'"?confidence"?\s*[:=]\s*([01](?:\.\d+)?)', raw)
    confidence = float(conf_match.group(1)) if conf_match else 0.5
    confidence = max(0.0, min(1.0, confidence))
    return label, confidence, None if label_match else "fallback_no_label_match"


def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get("teacher_item_id", row.get("teacher_pool_index", "")))
            if key:
                done.add(key)
    return done


def read_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Label JSONL rows with a local open-weight MLX teacher model.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "teacher_labels_qwen36_27b.jsonl")
    parser.add_argument("--model", choices=MODELS, default="qwen36_27b")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-context-chars", type=int, default=1800)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    rows = read_rows(args.input, args.limit)
    done = set() if args.no_resume else load_done(args.output)
    remaining = [
        row for row in rows if str(row.get("teacher_item_id", row.get("teacher_pool_index", ""))) not in done
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("loading", MODELS[args.model])
    model, tokenizer = load(str(MODELS[args.model]))
    print("rows", len(rows), "remaining", len(remaining), "output", args.output)

    start = time.time()
    counts = {0: 0, 1: 0}
    with args.output.open("a", encoding="utf-8") as handle:
        for offset, row in enumerate(remaining, start=1):
            prompt = apply_chat(tokenizer, make_teacher_prompt(row, args.max_context_chars))
            raw = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False)
            label, confidence, parse_error = parse_teacher_output(raw)
            counts[label] += 1
            out = dict(row)
            out.update(
                {
                    "teacher_model": args.model,
                    "teacher_label": label,
                    "teacher_confidence": confidence,
                    "teacher_raw": raw,
                    "teacher_parse_error": parse_error,
                }
            )
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            handle.flush()
            if offset % args.progress_every == 0:
                elapsed = (time.time() - start) / 60
                print(f"done {offset}/{len(remaining)} elapsed_min={elapsed:.1f} counts={counts}", flush=True)

    elapsed = (time.time() - start) / 60
    print(f"finished rows={len(remaining)} elapsed_min={elapsed:.2f} counts={counts}")


if __name__ == "__main__":
    main()
