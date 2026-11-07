from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score


WORK = Path("/kaggle/working")
TEST_PATH = Path("/kaggle/input/competitions/bengali-hallucination/test set.csv")
SAMPLE_PATH = Path("/kaggle/input/competitions/bengali-hallucination/dataset samples.json")
MODEL_PATH = Path(
    "/kaggle/input/models/snowrab/qwen3-6-27b-q4-k-p/pytorch/default/1/"
    "Qwen3.6-27B-Q4_K_P.gguf"
)
SERVER_INPUT = Path(
    "/kaggle/input/datasets/seyamalam/bengali-hallucination-llama-cpp-cuda-b9637/"
    "llama-server"
)
SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
USER_AGENT = "Huntrix-Bengali-source-recovery-audit/1.0"

# Derived from the frozen v85 compliance router trace. These IDs select the
# unresolved research tail only; they are never used by the inference notebook.
UNCERTAIN_IDS = [
    10, 19, 23, 37, 55, 58, 63, 65, 68, 69, 92, 93, 94, 112, 126, 179,
    319, 320, 321, 322, 323, 324, 326, 327, 334, 336, 337, 339, 341, 343,
    344, 345, 346, 347, 349, 350, 352, 353, 354, 355, 356, 358, 359, 362,
    363, 366, 369, 371, 372, 373, 376, 378, 380, 381, 422, 471, 589, 698,
    714, 723, 904, 953, 1029, 1031, 1074, 1102, 1254, 1372, 1540, 1549,
    1630, 1679, 1680, 1688, 1800, 1823, 1911, 1924, 2069, 2091, 2158,
    2159, 2160, 2168, 2169, 2174, 2175, 2176, 2182, 2184, 2187, 2188,
    2189, 2190, 2191, 2192, 2193, 2194, 2195, 2196, 2198, 2199, 2200,
    2201, 2203, 2204, 2205, 2206, 2207, 2209, 2210, 2211, 2213, 2214,
    2216, 2218, 2219, 2220, 2221, 2222, 2226, 2227, 2266, 2268, 2269,
    2270, 2272, 2273, 2275, 2276, 2277, 2279, 2280, 2281, 2283, 2284,
    2285, 2286, 2287, 2288, 2289, 2290, 2291, 2292, 2293, 2294, 2297,
    2298, 2299, 2300, 2301, 2302, 2303, 2304, 2305, 2306, 2307, 2308,
    2309, 2310, 2311, 2314, 2316, 2317, 2318, 2319, 2339, 2340, 2341,
    2350,
]


def canonical(text: object) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).lower()
    value = re.sub(r"!\[\]\(data:image/[^)]+\)", " ", value)
    value = re.sub(r"[\u200b-\u200d\ufeff]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def is_null_context(value: object) -> bool:
    return canonical(value) in {"", "[null]", "null", "nan"}


def query_fragment(prompt: str) -> str:
    value = canonical(prompt)
    value = re.sub(r"[\"'‘’“”]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value[:180]


def request_text(url: str, timeout: float = 15.0) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def clean_html(value: str) -> str:
    value = re.sub(r"<script.*?</script>|<style.*?</style>", " ", value, flags=re.S | re.I)
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def grep_app_search(query: str, limit: int = 4) -> list[dict]:
    if len(query) < 5:
        return []
    url = "https://grep.app/api/search?" + urllib.parse.urlencode(
        {"q": query[:120], "regexp": "false", "case": "false"}
    )
    try:
        payload = json.loads(request_text(url))
    except (OSError, ValueError, json.JSONDecodeError):
        return []
    output = []
    for hit in payload.get("hits", {}).get("hits", [])[:limit]:
        repo = str(hit.get("repo", ""))
        path = str(hit.get("path", ""))
        snippet = clean_html(str(hit.get("content", {}).get("snippet", "")))
        output.append(
            {
                "engine": "grep.app",
                "url": f"https://github.com/{repo}/blob/HEAD/{path}",
                "title": f"{repo}/{path}",
                "snippet": snippet[:800],
            }
        )
    return output


def duckduckgo_search(query: str, limit: int = 5) -> list[dict]:
    if len(query) < 5:
        return []
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode(
        {"q": f'"{query[:180]}"'}
    )
    try:
        body = request_text(url)
    except OSError:
        return []
    results = []
    blocks = re.findall(
        r'<div[^>]+class="[^"]*result[^"]*"[^>]*>(.*?)</div>\s*</div>',
        body,
        flags=re.S | re.I,
    )
    if not blocks:
        blocks = re.split(r'class="result results_links', body)[1:]
    for block in blocks[: limit * 2]:
        link = re.search(
            r'class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            block,
            flags=re.S | re.I,
        )
        if not link:
            continue
        snippet_match = re.search(
            r'class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</',
            block,
            flags=re.S | re.I,
        )
        raw_url = html.unescape(link.group(1))
        parsed = urllib.parse.urlparse(raw_url)
        target = urllib.parse.parse_qs(parsed.query).get("uddg", [raw_url])[0]
        results.append(
            {
                "engine": "duckduckgo",
                "url": target,
                "title": clean_html(link.group(2))[:300],
                "snippet": clean_html(snippet_match.group(1) if snippet_match else "")[:800],
            }
        )
        if len(results) >= limit:
            break
    return results


def domain(url: str) -> str:
    return urllib.parse.urlparse(url).netloc.lower().removeprefix("www.")


def retrieve(row: dict) -> list[dict]:
    if not is_null_context(row["context"]):
        return []
    query = query_fragment(row["prompt_bn"])
    hits = grep_app_search(query)
    time.sleep(0.25)
    hits.extend(duckduckgo_search(query))
    unique = {}
    prompt_norm = canonical(row["prompt_bn"])
    response_norm = canonical(row["response_bn"])
    for hit in hits:
        key = hit["url"]
        haystack = canonical(hit["title"] + " " + hit["snippet"])
        hit["domain"] = domain(hit["url"])
        hit["exact_prompt_fragment"] = bool(
            len(query) >= 12 and canonical(query) in haystack
        )
        hit["mentions_response"] = bool(
            len(response_norm) >= 2 and response_norm in haystack
        )
        hit["prompt_token_overlap"] = len(
            set(prompt_norm.split()) & set(haystack.split())
        ) / max(1, len(set(prompt_norm.split())))
        unique[key] = hit
    return list(unique.values())[:8]


def stable_seed(split: str, row_id: int) -> int:
    digest = hashlib.sha256(f"{split}:{row_id}:source-recovery-v1".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


SYSTEM = """তুমি বাংলা benchmark source-recovery auditor।
প্রসঙ্গ, প্রশ্ন, প্রার্থী উত্তর এবং noisy public-search hits দেওয়া হবে।

দুটি আলাদা বিচার করো:
1. source_relation: exact_origin যদি search hit-এ একই প্রশ্ন/answer-key স্পষ্ট থাকে;
   independent_evidence যদি hit বা প্রদত্ত context প্রশ্নের exact relation সমর্থন/খণ্ডন করে;
   none যদি প্রমাণ অপর্যাপ্ত বা শুধু সম্পর্কিত বিষয় থাকে।
2. verdict: 1 শুধু উত্তরটি exact relation-এ সরাসরি সমর্থিত হলে; 0 সরাসরি বিরোধ হলে;
   -1 যদি প্রমাণ অপর্যাপ্ত/দ্বন্দ্বপূর্ণ হয়।

একই পেজে response শব্দ থাকা যথেষ্ট নয়। ব্যক্তি, সাল, সংখ্যা, শুরু/শেষ, নির্মাতা/পুনর্নির্মাতা,
প্রকাশ/রচনা, direct antonym/synonym relation ঠিকমতো মিলাতে হবে।

শুধু এক লাইনের JSON:
{"source_relation":"exact_origin|independent_evidence|none","verdict":-1|0|1,"confidence":0.0,"correct_answer":"সংক্ষিপ্ত","reason":"সংক্ষিপ্ত"}"""


def make_prompt(row: dict, hits: list[dict]) -> str:
    context = "" if is_null_context(row["context"]) else str(row["context"])[:2200]
    evidence = []
    for rank, hit in enumerate(hits[:6], start=1):
        evidence.append(
            f"[{rank}] {hit['title']} | {hit['url']}\n{hit['snippet'][:650]}"
        )
    return "\n".join(
        [
            SYSTEM,
            "",
            f"প্রসঙ্গ: {context or '[NULL]'}",
            f"প্রশ্ন: {row['prompt_bn']}",
            f"প্রার্থী উত্তর: {row['response_bn']}",
            "Public-search hits:",
            "\n\n".join(evidence) if evidence else "[NONE]",
            "JSON:",
        ]
    )


def post_json(payload: dict, timeout: float = 120.0) -> dict:
    request = urllib.request.Request(
        SERVER_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def parse_judgment(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw.strip(), flags=re.S)
    if not match:
        return {"parse_error": "json_not_found", "raw": raw}
    try:
        obj = json.loads(match.group(0))
        relation = str(obj.get("source_relation", "none"))
        verdict = int(obj.get("verdict", -1))
        confidence = float(obj.get("confidence", 0.0))
        if relation not in {"exact_origin", "independent_evidence", "none"}:
            raise ValueError("invalid source_relation")
        if verdict not in {-1, 0, 1}:
            raise ValueError("invalid verdict")
        obj["confidence"] = max(0.0, min(1.0, confidence))
        return obj
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return {"parse_error": str(exc), "raw": raw}


def judge(row: dict, hits: list[dict], split: str) -> dict:
    payload = {
        "model": "qwen",
        "messages": [{"role": "user", "content": make_prompt(row, hits)}],
        "temperature": 0,
        "seed": stable_seed(split, int(row["id"])),
        "max_tokens": 160,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = post_json(payload)
    raw = response["choices"][0]["message"]["content"]
    return parse_judgment(raw)


def wait_for_server(process: subprocess.Popen, timeout: float = 360.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited with {process.returncode}")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(2)
    raise TimeoutError("llama-server did not become healthy")


def start_server() -> tuple[subprocess.Popen, object]:
    server = WORK / "llama-server"
    shutil.copy2(SERVER_INPUT, server)
    server.chmod(0o755)
    log_handle = (WORK / "llama_server.log").open("w", encoding="utf-8")
    command = [
        str(server),
        "--model", str(MODEL_PATH),
        "--host", "127.0.0.1",
        "--port", "8080",
        "--n-gpu-layers", "99",
        "--split-mode", "layer",
        "--tensor-split", "1,1",
        "--ctx-size", "8192",
        "--parallel", "1",
        "--batch-size", "1024",
        "--ubatch-size", "256",
        "--flash-attn", "on",
        "--reasoning-budget", "0",
        "--chat-template-kwargs", '{"enable_thinking":false}',
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--no-webui",
    ]
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    wait_for_server(process)
    return process, log_handle


def load_rows() -> tuple[list[dict], list[dict]]:
    test = pd.read_csv(TEST_PATH)
    required = {"id", "context", "prompt_bn", "response_bn"}
    if not required.issubset(test.columns):
        raise ValueError(f"test columns missing {sorted(required - set(test.columns))}")
    test = test[test["id"].astype(int).isin(UNCERTAIN_IDS)].copy()
    if test["id"].astype(int).tolist() != UNCERTAIN_IDS:
        raise ValueError("frozen uncertain IDs do not match competition test ordering")

    sample = pd.DataFrame(json.loads(SAMPLE_PATH.read_text(encoding="utf-8")))
    if not {"context", "prompt_bn", "response_bn", "label"}.issubset(sample.columns):
        raise ValueError("released sample schema changed")
    sample.insert(0, "id", range(1, len(sample) + 1))
    return sample.to_dict("records"), test.to_dict("records")


def audit_split(
    rows: list[dict],
    split: str,
    retrieval_handle,
    judgment_handle,
) -> list[dict]:
    records = []
    for offset, row in enumerate(rows, start=1):
        hits = retrieve(row)
        retrieval_record = {
            "split": split,
            "id": int(row["id"]),
            "hits": hits,
            "hit_domains": sorted({hit["domain"] for hit in hits if hit["domain"]}),
        }
        retrieval_handle.write(json.dumps(retrieval_record, ensure_ascii=False) + "\n")
        retrieval_handle.flush()

        judgment = judge(row, hits, split)
        record = {
            "split": split,
            "id": int(row["id"]),
            "label": int(row["label"]) if "label" in row else None,
            "has_context": not is_null_context(row["context"]),
            "prompt_bn": str(row["prompt_bn"]),
            "response_bn": str(row["response_bn"]),
            "hit_count": len(hits),
            "hit_domains": retrieval_record["hit_domains"],
            **judgment,
        }
        judgment_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        judgment_handle.flush()
        records.append(record)
        if offset % 20 == 0:
            print(
                split,
                offset,
                "/",
                len(rows),
                "relations",
                Counter(item.get("source_relation", "parse_error") for item in records),
                flush=True,
            )
    return records


def summarize(sample_records: list[dict], test_records: list[dict], elapsed: float) -> dict:
    decided = [
        row for row in sample_records
        if row.get("verdict") in {0, 1} and row.get("confidence", 0.0) >= 0.95
    ]
    truth = [int(row["label"]) for row in decided]
    pred = [int(row["verdict"]) for row in decided]
    sample_metrics = {
        "covered": len(decided),
        "accuracy": accuracy_score(truth, pred) if truth else None,
        "f1_0": f1_score(truth, pred, pos_label=0, zero_division=0) if truth else None,
        "f1_1": f1_score(truth, pred, pos_label=1, zero_division=0) if truth else None,
        "macro_f1": f1_score(truth, pred, average="macro", zero_division=0) if truth else None,
    }
    high_confidence_test = [
        row for row in test_records
        if row.get("verdict") in {0, 1}
        and row.get("confidence", 0.0) >= 0.95
        and (
            row.get("source_relation") == "exact_origin"
            or len(row.get("hit_domains", [])) >= 2
            or row["has_context"]
        )
    ]
    return {
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "sample_rows": len(sample_records),
        "test_rows": len(test_records),
        "sample_high_confidence": sample_metrics,
        "sample_relation_counts": dict(Counter(
            row.get("source_relation", "parse_error") for row in sample_records
        )),
        "test_relation_counts": dict(Counter(
            row.get("source_relation", "parse_error") for row in test_records
        )),
        "test_high_confidence_candidates": len(high_confidence_test),
        "candidate_ids": [row["id"] for row in high_confidence_test],
        "elapsed_seconds": elapsed,
        "policy": (
            "diagnostic only; candidates require local source-origin review and "
            "released-control precision before any inference change"
        ),
    }


def main() -> None:
    started = time.perf_counter()
    if torch.cuda.device_count() != 2 or not all(
        "T4" in torch.cuda.get_device_name(index) for index in range(2)
    ):
        raise RuntimeError(
            "Expected Kaggle T4 x2, observed "
            f"{[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}"
        )
    for required in (TEST_PATH, SAMPLE_PATH, MODEL_PATH, SERVER_INPUT):
        if not required.is_file():
            raise FileNotFoundError(required)

    sample_rows, test_rows = load_rows()
    process, log_handle = start_server()
    retrieval_path = WORK / "source_retrieval_trace.jsonl"
    judgment_path = WORK / "source_judgment_trace.jsonl"
    try:
        with retrieval_path.open("w", encoding="utf-8") as retrieval_handle, judgment_path.open(
            "w", encoding="utf-8"
        ) as judgment_handle:
            sample_records = audit_split(
                sample_rows, "sample", retrieval_handle, judgment_handle
            )
            test_records = audit_split(
                test_rows, "test", retrieval_handle, judgment_handle
            )
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
        log_handle.close()

    report = summarize(
        sample_records,
        test_records,
        time.perf_counter() - started,
    )
    (WORK / "source_recovery_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # A diagnostic candidate table is intentionally not a Kaggle submission.
    pd.DataFrame(test_records).to_csv(
        WORK / "source_recovery_candidates.csv",
        index=False,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
