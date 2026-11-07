from __future__ import annotations

import argparse
import json
import re
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from datathon_utils import BN_DIGITS, load_sample, load_test, metric_row


SYSTEM = """You are a meticulous Bengali hallucination detector. Label 1 means the candidate answer is faithful and correct. Label 0 means it is fabricated, unsupported, answers the wrong relation, or is factually wrong.

Procedure:
1. Identify precisely what the Bengali question asks.
2. Derive the shortest expected answer. If a context is supplied, use it as the primary evidence and do not confuse nearby people, dates, quantities, first/last, birth/death, foundation/status, or parent/sub-entities. If context is NULL, use reliable knowledge and Bengali linguistic/cultural meaning.
3. Compare the candidate semantically. Accept harmless spelling variants, equivalent number scripts/words, abbreviations, and concise paraphrases. Reject a plausible answer to a related but different fact.
Return only compact JSON with expected, label, and confidence. Confidence 1.0 means directly explicit or certain; use below 0.8 for ambiguity."""


def prompt(row) -> str:
    context = str(row.context)
    return f"CONTEXT:\n{context[:6000]}\n\nQUESTION:\n{row.prompt_bn}\n\nCANDIDATE ANSWER:\n{row.response_bn}"


def parse(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw or "", flags=re.S)
    if not match:
        return {"pred": 0, "confidence": 0.0, "expected": "", "reason": "parse failure", "raw": raw}
    try:
        item = json.loads(match.group(0))
        label = int(str(item.get("label", 0)).translate(BN_DIGITS))
        confidence = max(0.0, min(1.0, float(item.get("confidence", 0.5))))
        if label not in (0, 1):
            raise ValueError(label)
        return {"pred": label, "confidence": confidence, "expected": str(item.get("expected", "")),
                "reason": str(item.get("reason", "")), "raw": raw}
    except Exception as exc:
        return {"pred": 0, "confidence": 0.0, "expected": "", "reason": f"parse failure: {exc}", "raw": raw}


def request_one(url: str, model: str, row_index: int, row, max_tokens: int) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "system", "content": "/no_think\n" + SYSTEM}, {"role": "user", "content": prompt(row)}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "reasoning_effort": "none",
        "response_format": {"type": "json_schema", "json_schema": {"name": "verdict", "strict": True, "schema": {
            "type": "object", "properties": {
                "expected": {"type": "string"}, "label": {"type": "integer", "enum": [0, 1]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1}},
            "required": ["expected", "label", "confidence"], "additionalProperties": False}}},
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as response:
        payload = json.load(response)
    result = parse(payload["choices"][0]["message"]["content"])
    return {"row_index": int(row_index), "id": int(row.id) if hasattr(row, "id") else int(row_index), **result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["sample", "test"], required=True)
    parser.add_argument("--model", default="gemma4-12b-qat")
    parser.add_argument("--url", default="http://127.0.0.1:1234/v1/chat/completions")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--ids", type=Path, help="Optional JSON list of actual IDs to judge.")
    args = parser.parse_args()
    df = load_sample() if args.split == "sample" else load_test()
    if args.ids:
        ids = set(json.loads(args.ids.read_text()))
        df = df[df.id.isin(ids)]
    existing = {}
    if args.trace.exists():
        with args.trace.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line); existing[int(item["row_index"])] = item
    todo = [(int(i), row) for i, row in df.iterrows() if int(i) not in existing]
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    with args.trace.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(request_one, args.url, args.model, i, row, args.max_tokens): i for i, row in todo}
        for count, future in enumerate(as_completed(futures), 1):
            item = future.result()
            with lock:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n"); handle.flush()
            existing[item["row_index"]] = item
            if count % 25 == 0:
                print("done", count, "/", len(todo), flush=True)
    selected = sorted(set(df.index) & set(existing))
    if args.split == "sample":
        y = df.loc[selected, "label"].astype(int).to_numpy()
        p = pd.Series({i: existing[i]["pred"] for i in selected}).loc[selected].to_numpy()
        print(json.dumps(metric_row(y, p), indent=2), "covered", len(selected))
        print("high_conf_wrong", [i for i in selected if existing[i]["confidence"] >= .9 and existing[i]["pred"] != int(df.loc[i, "label"])])


if __name__ == "__main__":
    main()
