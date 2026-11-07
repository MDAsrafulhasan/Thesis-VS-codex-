from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score


WORK = Path("/kaggle/working")
TEST_PATH = Path("/kaggle/input/competitions/bengali-hallucination/test set.csv")
SAMPLE_PATH = Path(
    "/kaggle/input/competitions/bengali-hallucination/dataset samples.json"
)
MODEL_DIR = Path(
    "/kaggle/input/models/adityanathupadhyay/"
    "gemma-4-12b-it-qat-q4-0-gguf/gguf/image-text-to-text/1"
)
SERVER_INPUT = Path(
    "/kaggle/input/datasets/seyamalam/"
    "bengali-hallucination-llama-cpp-cuda-b9637/llama-server"
)
SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"

# These are the source-audit abstentions, not inferred labels. The list limits
# a diagnostic research run and is not imported into the frozen submission.
TARGET_IDS = [
    10, 19, 23, 37, 55, 58, 63, 65, 68, 69, 92, 93, 94, 112, 126, 179,
    319, 320, 321, 322, 323, 324, 326, 327, 334, 336, 337, 339, 341, 343,
    344, 345, 346, 347, 349, 350, 352, 353, 354, 355, 356, 358, 359, 362,
    363, 366, 369, 371, 372, 373, 376, 378, 380, 381, 422, 723, 904, 953,
    1029, 1074, 1372, 1540, 1549, 1679, 1680, 1688, 1823, 2158, 2159,
    2160, 2168, 2169, 2174, 2175, 2176, 2182, 2184, 2187, 2188, 2189,
    2190, 2191, 2192, 2193, 2194, 2195, 2196, 2198, 2199, 2200, 2201,
    2203, 2204, 2205, 2206, 2207, 2209, 2210, 2211, 2213, 2214, 2220,
    2268, 2269, 2270, 2272, 2273, 2275, 2276, 2277, 2279, 2280, 2281,
    2283, 2284, 2285, 2286, 2287, 2288, 2290, 2291, 2292, 2293, 2294,
    2297, 2298, 2299, 2300, 2301, 2302, 2303, 2304, 2305, 2306, 2307,
    2308, 2309, 2310, 2311, 2314, 2316, 2317, 2318, 2319, 2341,
]

EXAMPLES = """প্রশ্ন: বাংলাদেশের রাজধানী কী?
উত্তর: ঢাকা
লেবেল: 1

প্রশ্ন: বাংলাদেশের রাজধানী কী?
উত্তর: চট্টগ্রাম
লেবেল: 0

প্রসঙ্গ: বইটি ১৯২৮ সালে প্রকাশিত, কিন্তু ১৯২৭ সালে ধারাবাহিক প্রকাশ শুরু হয়।
প্রশ্ন: বইটির ধারাবাহিক প্রকাশ কখন শুরু হয়?
উত্তর: ১৯২৮ সালে
লেবেল: 0

প্রসঙ্গ: উপন্যাসটি তিনটি ভাগ ও পঁয়ত্রিশটি অধ্যায়ে বিভক্ত।
প্রশ্ন: উপন্যাসটি কয়টি ভাগে বিভক্ত?
উত্তর: পঁয়ত্রিশটি ভাগে
লেবেল: 0

প্রশ্ন: ২ + ৩ কত?
উত্তর: ৫
লেবেল: 1"""

VIEW_HEADERS = {
    "solve_compare": (
        "প্রথমে প্রশ্নটির exact সঠিক উত্তর বা সম্পর্ক মনে মনে নির্ণয় করো। তারপর "
        "প্রার্থী উত্তরের entity, relation, সাল, সংখ্যা এবং answer type সেই উত্তরের "
        "সঙ্গে মেলে কি না বিচার করো।"
    ),
    "adversarial_relation": (
        "প্রার্থী উত্তরে সূক্ষ্ম field swap খোঁজো: শুরু বনাম শেষ, নির্মাণ বনাম "
        "পুনর্নির্মাণ, রচনা বনাম প্রকাশ, ব্যক্তি বনাম প্রতিষ্ঠান, সাল/সংখ্যা/স্থান "
        "বদল, বা প্রশ্নের বদলে সম্পর্কিত অন্য তথ্য। সত্যিকারের বানান/সমার্থক "
        "বিকল্পকে ভুল বলবে না।"
    ),
}


def canonical(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = re.sub(r"[\u200b-\u200d\ufeff]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_context(value: object) -> str:
    text = canonical(value)
    return "" if text.lower() in {"", "[null]", "null", "nan"} else text


def prompt(row: dict, view: str) -> str:
    context = clean_context(row["context"])
    grounding = (
        "প্রসঙ্গ থাকলে বাইরের জ্ঞানের বদলে শুধু প্রসঙ্গের exact দাবি ব্যবহার করো।"
        if context
        else "প্রসঙ্গ নেই; প্রশ্নটি নিজে সমাধান করে প্রার্থী উত্তরের সঙ্গে তুলনা করো।"
    )
    fields = []
    if context:
        fields.append("প্রসঙ্গ: " + context[:2200])
    fields.extend(
        [
            "প্রশ্ন: " + str(row["prompt_bn"]),
            "উত্তর: " + str(row["response_bn"]),
            "লেবেল:",
        ]
    )
    return "\n\n".join(
        [
            "তুমি বাংলা hallucination detector।",
            VIEW_HEADERS[view],
            grounding,
            "0 = ভুল/অসমর্থিত/হ্যালুসিনেটেড; 1 = exact প্রশ্ন অনুযায়ী সঠিক ও বিশ্বস্ত।",
            "শুধু একটি অক্ষর 0 অথবা 1 লিখবে। কোনো ব্যাখ্যা নয়।",
            EXAMPLES,
            "\n".join(fields),
        ]
    )


def stable_seed(split: str, row_id: int, view: str) -> int:
    digest = hashlib.sha256(
        f"gemma-cross-family-v1:{split}:{row_id}:{view}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def parse_label(value: object) -> int:
    match = re.search(r"(?<!\d)([01])(?!\d)", str(value or "").strip())
    if not match:
        raise ValueError(f"no binary label in {value!r}")
    return int(match.group(1))


def token_confidence(choice: dict, label: int) -> float | None:
    content = (choice.get("logprobs") or {}).get("content") or []
    if not content:
        return None
    candidates = content[0].get("top_logprobs") or []
    scores = {}
    for item in candidates:
        token = str(item.get("token", "")).strip()
        if token in {"0", "1"}:
            scores[int(token)] = float(item["logprob"])
    if label not in scores:
        return None
    denominator = sum(math.exp(score) for score in scores.values())
    return math.exp(scores[label]) / denominator if denominator else None


def call_model(row: dict, split: str, view: str) -> dict:
    payload = {
        "messages": [{"role": "user", "content": prompt(row, view)}],
        "temperature": 0,
        "max_tokens": 2,
        "seed": stable_seed(split, int(row["id"]), view),
        "cache_prompt": True,
        "grammar": "root ::= [01]",
        "logprobs": True,
        "top_logprobs": 2,
    }
    request = urllib.request.Request(
        SERVER_URL,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    last_error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = json.loads(response.read().decode())
            choice = body["choices"][0]
            message = choice["message"]
            raw = message.get("content") or message.get("reasoning_content") or ""
            label = parse_label(raw)
            return {
                "label": label,
                "confidence": token_confidence(choice, label),
                "raw": str(raw),
            }
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2**attempt)
    raise RuntimeError(f"Gemma request failed for {split}:{row['id']}:{view}") from last_error


def wait_for_server(process: subprocess.Popen, timeout: float = 360.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited with {process.returncode}")
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:8080/health", timeout=2
            ) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(2)
    raise TimeoutError("llama-server did not become healthy")


def model_path() -> Path:
    files = sorted(MODEL_DIR.glob("*.gguf"))
    if len(files) != 1:
        raise FileNotFoundError(
            f"expected one pinned GGUF under {MODEL_DIR}, observed {files}"
        )
    return files[0]


def start_server(model: Path) -> tuple[subprocess.Popen, object]:
    server = WORK / "llama-server"
    shutil.copy2(SERVER_INPUT, server)
    server.chmod(0o755)
    log_handle = (WORK / "llama_server.log").open("w", encoding="utf-8")
    command = [
        str(server),
        "--model", str(model),
        "--host", "127.0.0.1",
        "--port", "8080",
        "--n-gpu-layers", "99",
        "--split-mode", "layer",
        "--tensor-split", "1,1",
        "--ctx-size", "4096",
        "--parallel", "1",
        "--batch-size", "1024",
        "--ubatch-size", "256",
        "--flash-attn", "on",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--no-webui",
    ]
    print("+", " ".join(command), flush=True)
    process = subprocess.Popen(
        command, stdout=log_handle, stderr=subprocess.STDOUT, text=True
    )
    wait_for_server(process)
    return process, log_handle


def load_rows() -> tuple[list[dict], list[dict]]:
    sample = pd.DataFrame(json.loads(SAMPLE_PATH.read_text(encoding="utf-8")))
    sample.insert(0, "id", range(len(sample)))
    test = pd.read_csv(TEST_PATH)
    required = {"id", "context", "prompt_bn", "response_bn"}
    if not required.issubset(sample.columns) or not required.issubset(test.columns):
        raise ValueError("competition schema changed")
    test = test[test.id.astype(int).isin(TARGET_IDS)].copy()
    if test.id.astype(int).tolist() != TARGET_IDS:
        raise ValueError("target IDs do not match sorted competition test rows")
    return sample.to_dict("records"), test.to_dict("records")


def audit(rows: list[dict], split: str, handle) -> list[dict]:
    records = []
    for offset, row in enumerate(rows, start=1):
        views = {
            view: call_model(row, split, view)
            for view in ("solve_compare", "adversarial_relation")
        }
        labels = [item["label"] for item in views.values()]
        consensus = labels[0] if len(set(labels)) == 1 else None
        confidences = [
            item["confidence"]
            for item in views.values()
            if item["confidence"] is not None
        ]
        record = {
            "split": split,
            "id": int(row["id"]),
            "label": int(row["label"]) if "label" in row else None,
            "has_context": bool(clean_context(row["context"])),
            "prompt_bn": str(row["prompt_bn"]),
            "response_bn": str(row["response_bn"]),
            "views": views,
            "consensus": consensus,
            "min_confidence": min(confidences) if len(confidences) == 2 else None,
        }
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        records.append(record)
        if offset % 25 == 0 or offset == len(rows):
            print(
                split,
                offset,
                "/",
                len(rows),
                "consensus",
                Counter(item["consensus"] for item in records),
                flush=True,
            )
    return records


def metrics(rows: list[dict], *, context: bool | None = None) -> dict:
    selected = [
        row
        for row in rows
        if row["consensus"] in {0, 1}
        and (context is None or row["has_context"] is context)
    ]
    truth = [int(row["label"]) for row in selected]
    pred = [int(row["consensus"]) for row in selected]
    return {
        "covered": len(selected),
        "accuracy": accuracy_score(truth, pred) if truth else None,
        "f1_0": f1_score(truth, pred, pos_label=0, zero_division=0)
        if truth
        else None,
        "f1_1": f1_score(truth, pred, pos_label=1, zero_division=0)
        if truth
        else None,
        "macro_f1": f1_score(truth, pred, average="macro", zero_division=0)
        if truth
        else None,
    }


def main() -> None:
    started = time.perf_counter()
    gpu_names = [
        torch.cuda.get_device_name(index)
        for index in range(torch.cuda.device_count())
    ]
    if len(gpu_names) != 2 or not all("T4" in name for name in gpu_names):
        raise RuntimeError(f"expected Tesla T4 x2, observed {gpu_names}")
    for path in (TEST_PATH, SAMPLE_PATH, SERVER_INPUT):
        if not path.is_file():
            raise FileNotFoundError(path)
    model = model_path()
    sample_rows, test_rows = load_rows()
    process, log_handle = start_server(model)
    trace_path = WORK / "gemma_cross_family_trace.jsonl"
    try:
        with trace_path.open("w", encoding="utf-8") as handle:
            sample_records = audit(sample_rows, "sample", handle)
            test_records = audit(test_rows, "test", handle)
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
        log_handle.close()

    report = {
        "gpu_names": gpu_names,
        "model": str(model),
        "model_gib": model.stat().st_size / 1024**3,
        "sample_rows": len(sample_records),
        "test_rows": len(test_records),
        "sample_consensus": metrics(sample_records),
        "sample_context_consensus": metrics(sample_records, context=True),
        "sample_no_context_consensus": metrics(sample_records, context=False),
        "sample_abstentions": sum(row["consensus"] is None for row in sample_records),
        "test_consensus_counts": dict(
            Counter(str(row["consensus"]) for row in test_records)
        ),
        "test_abstentions": sum(row["consensus"] is None for row in test_records),
        "elapsed_seconds": time.perf_counter() - started,
        "policy": (
            "diagnostic only; compare disagreements with the near-perfect released "
            "cascade and v85 before any manual evidence review"
        ),
    }
    (WORK / "gemma_cross_family_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
