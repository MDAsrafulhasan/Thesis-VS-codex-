from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

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
""".strip()


VIEW_HEADERS = {
    "context_nli": """তুমি বাংলা context-grounding verifier। প্রসঙ্গ থাকলে উত্তরটি প্রসঙ্গ দ্বারা সমর্থিত কি না দেখবে।""",
    "noctx_fact": """তুমি বাংলা সাধারণ জ্ঞান ও পরীক্ষার প্রশ্ন যাচাইকারী। নিজে সঠিক উত্তর বের করে দেওয়া উত্তরের সঙ্গে তুলনা করবে।""",
    "exam_strict": """তুমি BCS/ভর্তি/সাধারণ জ্ঞান ধাঁচের বাংলা QA verifier। প্রশ্নের exact চাহিদা, entity, relation, সাল ও সংখ্যা মেলাবে।""",
    "zero_bias": """তুমি অত্যন্ত সতর্ক hallucination detector। আংশিক ভুল, ভুল relation, অসমর্থিত বা প্রশ্নের সঙ্গে অমিল হলে 0 দেবে।""",
}


def make_prompt(row, view: str) -> str:
    parts = [
        VIEW_HEADERS[view],
        "শুধু 0 অথবা 1 লিখবে। 0 = ভুল/হ্যালুসিনেটেড; 1 = সঠিক/বিশ্বস্ত।",
        "",
        BASE_EXAMPLES,
        "",
        "এখন নতুন উদাহরণের লেবেল দাও। ব্যাখ্যা নয়, শুধু 0 অথবা 1।",
    ]
    context = clean_context(row.context)
    if context:
        parts.append("প্রসঙ্গ: " + context[:1800])
    parts.extend(("প্রশ্ন: " + str(row.prompt_bn), "উত্তর: " + str(row.response_bn), "লেবেল:"))
    return "\n".join(parts)


def parse_binary(text: str) -> int:
    match = re.search(r"(?<!\d)([01])(?:\.0)?(?!\d)", text.strip())
    if not match:
        raise ValueError(f"Judge returned no binary label: {text!r}")
    return int(match.group(1))


def stable_seed(row_id: int, view: str) -> int:
    digest = hashlib.sha256(f"{row_id}:{view}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def call_server(url: str, prompt: str, timeout: float, retries: int, seed: int = 42) -> tuple[int, str]:
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 2,
        "seed": seed,
        "cache_prompt": True,
        "grammar": "root ::= [01]",
        "chat_template_kwargs": {"enable_thinking": False},
        "thinking_budget_tokens": 0,
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            message = body["choices"][0]["message"]
            raw = str(message.get("content") or message.get("reasoning_content") or "")
            return parse_binary(raw), raw
        except (OSError, KeyError, ValueError, urllib.error.HTTPError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Judge request failed after {retries + 1} attempts") from last_error


def judge_row(row, url: str, timeout: float, retries: int) -> dict:
    labels, raw = {}, {}
    if clean_context(row.context):
        views = ["context_nli"]
    else:
        # The policy is unanimous-positive, so a single zero is decisive. Put
        # the empirically most skeptical views first and avoid needless calls.
        views = ["exam_strict", "zero_bias", "noctx_fact"]
    for view in views:
        labels[view], raw[view] = call_server(
            url,
            make_prompt(row, view),
            timeout,
            retries,
            seed=stable_seed(int(row.id), view),
        )
        if view != "context_nli" and labels[view] == 0:
            break
    label = (
        labels["context_nli"]
        if clean_context(row.context)
        else int(len(labels) == len(views) and all(label == 1 for label in labels.values()))
    )
    return {"id": int(row.id), "new": int(label), "views": labels, "raw_views": raw}


def main() -> None:
    parser = argparse.ArgumentParser(description="Selective deterministic GGUF judge through llama-server.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--base-submission", type=Path, required=True)
    parser.add_argument("--router-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8080")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--row-batch-size", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    started = time.perf_counter()
    frame, output = pd.read_csv(args.input), pd.read_csv(args.base_submission)
    router = json.loads(args.router_trace.read_text(encoding="utf-8"))
    selected = {int(item["id"]) for item in router.get("uncertainty", [])}
    rows = sorted(
        (row for row in frame.itertuples(index=False) if int(row.id) in selected),
        key=lambda row: int(row.id),
    )
    records = []
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    partial_trace = args.trace.with_suffix(args.trace.suffix + ".partial.jsonl")
    with partial_trace.open("w", encoding="utf-8") as partial:
        done = 0
        for start in range(0, len(rows), args.row_batch_size):
            batch = rows[start : start + args.row_batch_size]
            # Intentionally process each fixed, sorted batch serially. The
            # model still spans both T4 GPUs, while request order and slot
            # assignment remain stable across organizer reruns.
            for row in batch:
                record = judge_row(row, args.server_url, args.timeout, args.retries)
                mask = output.id.astype(int) == record["id"]
                record["old"] = int(output.loc[mask, "label"].iloc[0])
                record["changed"] = record["old"] != record["new"]
                output.loc[mask, "label"] = record["new"]
                records.append(record)
                partial.write(json.dumps(record, ensure_ascii=False) + "\n")
                partial.flush()
                done += 1
                if done % 25 == 0 or done == len(rows):
                    print(f"judged {done}/{len(rows)}", flush=True)

    output["label"] = output["label"].astype(int)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    elapsed = time.perf_counter() - started
    report = {
        "selected": len(rows), "seconds": elapsed,
        "rows_per_second": len(rows) / max(elapsed, 1e-9),
        "changed": sum(item["changed"] for item in records),
        "records": sorted(records, key=lambda item: item["id"]),
    }
    args.trace.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
