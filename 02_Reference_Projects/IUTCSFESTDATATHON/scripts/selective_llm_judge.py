from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from datathon_utils import clean_context


BASE_EXAMPLES = """প্রশ্ন: বাংলাদেশের রাজধানী কী?
উত্তর: ঢাকা
লেবেল: 1

প্রশ্ন: বাংলাদেশের রাজধানী কী?
উত্তর: চট্টগ্রাম
লেবেল: 0

প্রশ্ন: "ধান্ধা" এর ভাবার্থ কী?
উত্তর: কোন অসৎ উদ্দেশ্য
লেবেল: 1

প্রশ্ন: শেখ মুজিবুর রহমান কোথায় জন্মগ্রহণ করেন?
উত্তর: চুরুলিয়া, বর্ধমান
লেবেল: 0

প্রসঙ্গ: অভ্র কীবোর্ড সফটওয়্যারটি মেহদী হাসান খান তৈরি করেন।
প্রশ্ন: অভ্র কিবোর্ড কে উদ্ভাবন করেন?
উত্তর: মেহদী হাসান খান
লেবেল: 1

প্রসঙ্গ: প্রথম চলচ্চিত্র সোনার বেড়ি এবং সর্বশেষ পূর্ণদৈর্ঘ্য চলচ্চিত্র রানওয়ে।
প্রশ্ন: সর্বশেষ চলচ্চিত্রটি কত সালে মুক্তি পায়?
উত্তর: রানওয়ে
লেবেল: 0"""

VIEW_HEADERS = {
    "context_nli": """তুমি বাংলা context-grounding verifier। প্রসঙ্গ থাকলে উত্তরটি সরাসরি প্রসঙ্গ দ্বারা সমর্থিত কি না দেখবে।
শুধু 0 অথবা 1 লিখবে।
0 = প্রসঙ্গে নেই, প্রসঙ্গের সঙ্গে মেলে না, বা তথ্যগতভাবে ভুল।
1 = প্রসঙ্গ/প্রশ্ন অনুযায়ী সঠিক।""",
    "noctx_fact": """তুমি বাংলা সাধারণ জ্ঞান ও পরীক্ষার প্রশ্ন যাচাইকারী। প্রসঙ্গ না থাকলে নিজের জ্ঞান দিয়ে fact-check করবে।
শুধু 0 অথবা 1 লিখবে।
0 = ভুল ব্যক্তি/স্থান/তারিখ/সংখ্যা/অর্থ, বানানো তথ্য, বা প্রশ্নের ভুল উত্তর।
1 = সঠিক উত্তর।""",
    "exam_strict": """তুমি BCS/ভর্তি/সাধারণ জ্ঞান ধাঁচের বাংলা MCQ/QA verifier। প্রশ্নের exact চাহিদা মিলাবে।
শুধু 0 অথবা 1 লিখবে।
0 = ভুল অপশন, ভুল তথ্য, ভুল সাল/সংখ্যা, ভুল লেখক/স্থান, অথবা বিভ্রান্তিকর উত্তর।
1 = পরীক্ষার answer key অনুযায়ী সঠিক উত্তর।""",
    "zero_bias": """তুমি অত্যন্ত সতর্ক hallucination detector। উত্তর নিশ্চিতভাবে সঠিক না হলে 0 দেবে।
শুধু 0 অথবা 1 লিখবে।
0 = সন্দেহজনক, অসমর্থিত, ভুল, আংশিক ভুল, বা প্রশ্নের সঙ্গে অমিল।
1 = নিশ্চিতভাবে সঠিক।""",
}


def make_prompt(row, evidence: list[dict], view: str = "context_nli") -> str:
    if view not in VIEW_HEADERS:
        raise ValueError(f"unknown judge view: {view}")
    parts = [
        VIEW_HEADERS[view], "", BASE_EXAMPLES, "",
        "এখন নতুন উদাহরণের লেবেল দাও। ব্যাখ্যা নয়, শুধু 0 অথবা 1।",
    ]
    context = clean_context(row.context)
    if context:
        parts.append("প্রসঙ্গ: " + context[:1800])
    snippets = "\n".join(f"- {item.get('context', '')[:700]}" for item in evidence[:3])
    if snippets:
        parts.append("বাহ্যিক প্রমাণ:\n" + snippets)
    parts.extend(("প্রশ্ন: " + row.prompt_bn, "উত্তর: " + row.response_bn, "লেবেল:"))
    return "\n".join(parts)


def apply_chat(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def consensus_label(view_labels: dict[str, int], has_context: bool) -> int:
    if has_context:
        return int(view_labels["context_nli"])
    required = ("noctx_fact", "exam_strict", "zero_bias")
    return int(all(int(view_labels[name]) == 1 for name in required))


def read_evidence(path: Path | None) -> dict[int, list[dict]]:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as handle:
        return {int(item["id"]): item.get("evidence", []) for line in handle if (item := json.loads(line))}


def binary_token_ids(tokenizer) -> dict[int, int]:
    """Find distinct single-token encodings for labels 0 and 1."""
    result: dict[int, int] = {}
    for label in (0, 1):
        for text in (str(label), f" {label}"):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 1:
                result[label] = int(ids[0])
                break
        if label not in result:
            raise ValueError(f"tokenizer has no single-token encoding for label {label}")
    if result[0] == result[1]:
        raise ValueError("tokenizer maps binary labels to the same token")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="4-bit selective open-weight LLM judge for unresolved router rows.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--base-submission", type=Path, required=True)
    parser.add_argument("--router-trace", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--model", required=True, help="Local Kaggle dataset path or open Hugging Face model ID.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    frame, base = pd.read_csv(args.input), pd.read_csv(args.base_submission)
    router = json.loads(args.router_trace.read_text(encoding="utf-8"))
    selected = {int(item["id"]) for item in router.get("uncertainty", [])}
    output, records = base.copy(), []
    rows = [row for row in frame.itertuples() if int(row.id) in selected]
    if not rows:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(args.output, index=False)
        args.trace.write_text(json.dumps({"model": args.model, "selected": 0, "records": []}, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote", args.output, "judged", 0, "changed", 0)
        return
    evidence = read_evidence(args.evidence)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    label_tokens = binary_token_ids(tokenizer)
    token_labels = {token: label for label, token in label_tokens.items()}
    allowed_tokens = sorted(token_labels)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", quantization_config=quant, torch_dtype=torch.float16)
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        row_views = [
            ["context_nli"] if clean_context(row.context) else ["noctx_fact", "exam_strict", "zero_bias"]
            for row in batch
        ]
        collected: list[dict[str, int]] = [{} for _ in batch]
        raw_views: list[dict[str, str]] = [{} for _ in batch]
        for view in VIEW_HEADERS:
            offsets = [offset for offset, views in enumerate(row_views) if view in views]
            if not offsets:
                continue
            prompts = [
                apply_chat(tokenizer, make_prompt(batch[offset], evidence.get(int(batch[offset].id), []), view))
                for offset in offsets
            ]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=6144).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded, max_new_tokens=1, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    prefix_allowed_tokens_fn=lambda _batch_id, _input_ids: allowed_tokens,
                )
            token_ids = generated[:, encoded.input_ids.shape[1]:][:, 0].tolist()
            for offset, token_id in zip(offsets, token_ids):
                collected[offset][view] = token_labels[int(token_id)]
                raw_views[offset][view] = tokenizer.decode([int(token_id)], skip_special_tokens=True)
        for offset, row in enumerate(batch):
            label = consensus_label(collected[offset], bool(clean_context(row.context)))
            mask = output.id == int(row.id)
            old = int(output.loc[mask, "label"].iloc[0])
            output.loc[mask, "label"] = label
            records.append({
                "id": int(row.id), "old": old, "new": label,
                "views": collected[offset], "raw_views": raw_views[offset],
                "changed": old != label,
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    args.trace.write_text(json.dumps({"model": args.model, "selected": len(rows), "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", args.output, "judged", len(rows), "changed", sum(item["changed"] for item in records))


if __name__ == "__main__":
    main()
