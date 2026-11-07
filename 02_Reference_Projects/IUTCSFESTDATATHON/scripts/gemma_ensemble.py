from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from datathon_utils import (
    OUTPUT_DIR,
    ROOT,
    apply_context_rule,
    build_context_retriever,
    evaluate_by_slice,
    load_sample,
    load_test,
    load_trace_predictions,
    parse_label,
    print_metrics,
    retrieve_contexts,
)


LLAMA_SERVER = Path(
    "/Users/seyam/.lmstudio/extensions/backends/"
    "llama.cpp-mac-arm64-apple-metal-advsimd-2.23.1/llama-server"
)
HOST = "127.0.0.1"
PORT = 8099
SERVER_URL = f"http://{HOST}:{PORT}/v1/chat/completions"

MODELS = {
    "qat": ROOT / "models/lmstudio/gemma-4-12B-it-QAT-GGUF/gemma-4-12B-it-QAT-Q4_0.gguf",
    "lmstudio_qat": Path(
        "/Users/seyam/.lmstudio/models/lmstudio-community/"
        "gemma-4-12B-it-QAT-GGUF/gemma-4-12B-it-QAT-Q4_0.gguf"
    ),
    "lmstudio_nonqat": Path(
        "/Users/seyam/.lmstudio/models/lmstudio-community/"
        "gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf"
    ),
}


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


VARIANT_HEADERS = {
    "base": """তুমি বাংলা hallucination detector। শুধু 0 অথবা 1 লিখবে।
0 = উত্তরটি ভুল, বানানো, প্রশ্নের সঙ্গে মেলে না, অথবা প্রসঙ্গ দ্বারা সমর্থিত নয়।
1 = উত্তরটি সঠিক এবং প্রসঙ্গ থাকলে প্রসঙ্গ দ্বারা সমর্থিত।""",
    "strict": """তুমি কঠোর বাংলা fact-checker। আংশিক মিল থাকলেই 1 দেবে না।
শুধু 0 অথবা 1 লিখবে।
0 = ভুল সংখ্যা/তারিখ/ব্যক্তি/স্থান, বানানো তথ্য, প্রশ্নের উত্তর নয়, বা প্রসঙ্গে নেই।
1 = সম্পূর্ণ সঠিক এবং প্রাসঙ্গিক উত্তর।""",
    "skeptical": """তুমি সন্দেহপ্রবণ বাংলা উত্তর যাচাইকারী। নিশ্চিত না হলে 0 দেবে।
শুধু 0 অথবা 1 লিখবে।
0 = hallucinated/incorrect/unsupported.
1 = faithful/correct/supported.""",
    "lenient": """তুমি বাংলা উত্তর যাচাইকারী। বানান বা ভাষার ছোট পার্থক্য উপেক্ষা করবে।
শুধু 0 অথবা 1 লিখবে।
0 = তথ্যগতভাবে ভুল বা অসমর্থিত।
1 = তথ্যগতভাবে সঠিক, যদিও শব্দচয়ন আলাদা হতে পারে।""",
    "numeric": """তুমি বাংলা QA verifier। সংখ্যা, সাল, তারিখ, ব্যক্তি, স্থান ও প্রশ্নের ধরন খুব সতর্কভাবে মিলাবে।
শুধু 0 অথবা 1 লিখবে।
0 = কোনো গুরুত্বপূর্ণ entity/number/date/place/person ভুল অথবা উত্তর প্রশ্নের সঙ্গে মেলে না।
1 = উত্তর সঠিক।""",
    "evidence": """তুমি বাংলা retrieval-grounded fact-checker। দেওয়া প্রমাণ থাকলে সেটি ব্যবহার করবে।
শুধু 0 অথবা 1 লিখবে।
0 = উত্তর প্রমাণের সঙ্গে মেলে না, ভুল, বানানো, অথবা অপ্রমাণিত।
1 = উত্তর প্রমাণ/সাধারণ জ্ঞান অনুযায়ী সঠিক।""",
}


def make_prompt(row, variant: str, evidence: str = "", max_context_chars: int = 2200) -> str:
    header = VARIANT_HEADERS[variant]
    parts = [header, "", BASE_EXAMPLES, "", "এখন নতুন উদাহরণের লেবেল দাও। ব্যাখ্যা নয়, শুধু 0 অথবা 1।"]
    if row.context:
        parts.append("প্রসঙ্গ: " + row.context[:max_context_chars])
    elif evidence and variant == "evidence":
        parts.append("উদ্ধারকৃত সম্ভাব্য প্রমাণ:\n" + evidence)
    parts.append("প্রশ্ন: " + row.prompt_bn)
    parts.append("উত্তর: " + row.response_bn)
    parts.append("লেবেল:")
    return "\n".join(parts)


def wait_for_server(timeout: int = 180) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            payload = {
                "model": "local",
                "messages": [{"role": "user", "content": "শুধু 1 লিখো।"}],
                "temperature": 0,
                "max_tokens": 4,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            request = urllib.request.Request(
                SERVER_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    return False


def start_server(model_path: Path, ctx_size: int = 2048):
    if wait_for_server(timeout=4):
        print("Using already-running llama-server")
        return None
    cmd = [
        str(LLAMA_SERVER),
        "-m",
        str(model_path),
        "--host",
        HOST,
        "--port",
        str(PORT),
        "-c",
        str(ctx_size),
        "-ngl",
        "auto",
        "--log-disable",
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if not wait_for_server(timeout=240):
        process.terminate()
        raise RuntimeError("llama-server did not become ready")
    print("Started llama-server:", model_path)
    return process


def call_gemma(prompt: str, retries: int = 3) -> tuple[str, dict]:
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 8,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(payload).encode("utf-8")
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                SERVER_URL,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=240) as response:
                output = json.loads(response.read())
            message = output["choices"][0]["message"]
            text = message.get("content") or message.get("reasoning_content") or ""
            return text, output.get("usage", {})
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(3)
    raise RuntimeError("unreachable")


def row_indices(df: pd.DataFrame, rows: str, ids_file: str | None = None) -> np.ndarray:
    if ids_file:
        path = Path(ids_file)
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            selected = {int(item["id"]) for item in payload.get("records", payload)}
        elif path.suffix == ".csv":
            selected = set(pd.read_csv(path)["id"].astype(int))
        else:
            selected = {int(value) for value in path.read_text(encoding="utf-8").split()}
        return np.flatnonzero(df["id"].astype(int).isin(selected).to_numpy())
    if rows == "all":
        return np.arange(len(df))
    if rows == "no_context":
        return np.flatnonzero(~df["has_context"].to_numpy())
    if rows == "context":
        return np.flatnonzero(df["has_context"].to_numpy())
    raise ValueError(rows)


def run_variant(
    df: pd.DataFrame,
    variant: str,
    trace_path: Path,
    rows: str = "all",
    evidence: list[str] | None = None,
    ids_file: str | None = None,
) -> np.ndarray:
    indices = row_indices(df, rows, ids_file=ids_file)
    existing = []
    if trace_path.exists():
        existing = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    done = {int(row["i"]): row for row in existing}
    preds = np.full(len(df), -1, dtype=int)
    for row in existing:
        preds[int(row["i"])] = int(row["pred"])

    print(f"variant={variant} rows={rows} remaining={sum(i not in done for i in indices)} trace={trace_path}")
    start = time.time()
    with trace_path.open("a", encoding="utf-8") as handle:
        for count, i in enumerate(indices, start=1):
            if int(i) in done:
                continue
            row = df.iloc[int(i)]
            prompt = make_prompt(row, variant, evidence=evidence[int(i)] if evidence else "")
            generated, usage = call_gemma(prompt)
            pred = parse_label(generated)
            record = {"i": int(i), "pred": int(pred), "gen": generated, "usage": usage}
            if "id" in df.columns:
                record["id"] = int(df.iloc[int(i)]["id"])
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            preds[int(i)] = pred
            completed = len([idx for idx in indices if preds[int(idx)] != -1])
            if completed % 50 == 0:
                elapsed = (time.time() - start) / 60
                known = preds[indices][preds[indices] != -1]
                print(f"done {completed}/{len(indices)} elapsed_min={elapsed:.1f} counts={pd.Series(known).value_counts().to_dict()}")
    return preds


def build_evidence(sample_df: pd.DataFrame, test_df: pd.DataFrame | None, target_df: pd.DataFrame) -> list[str]:
    corpus_df, vectorizer, matrix = build_context_retriever(sample_df, test_df)
    return retrieve_contexts(target_df, corpus_df, vectorizer, matrix, top_k=3)


def evaluate_trace_set(sample_df: pd.DataFrame, trace_paths: list[Path], names: list[str]) -> dict[str, np.ndarray]:
    y = sample_df["label"].to_numpy()
    predictions = {}
    for name, path in zip(names, trace_paths):
        raw = load_partial_trace(path, len(sample_df))
        mask = raw != -1
        if mask.all():
            post = apply_context_rule(sample_df, raw)
            print_metrics(f"{name} raw", y, raw)
            print_metrics(f"{name} post", y, post)
            predictions[name] = post
        else:
            subset_y = y[mask]
            subset_pred = raw[mask]
            print_metrics(f"{name} partial", subset_y, subset_pred)
            predictions[name] = raw
    return predictions


def load_partial_trace(trace_path: Path, n_rows: int) -> np.ndarray:
    preds = np.full(n_rows, -1, dtype=int)
    if not trace_path.exists():
        return preds
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        preds[int(row["i"])] = int(row["pred"])
    return preds


def vote_predictions(predictions: list[np.ndarray], default: np.ndarray | None = None, threshold: float = 0.5) -> np.ndarray:
    stack = np.vstack(predictions)
    valid = stack != -1
    votes = np.where(valid, stack, 0).sum(axis=0)
    counts = valid.sum(axis=0)
    if default is None:
        out = (votes / np.maximum(counts, 1) >= threshold).astype(int)
    else:
        out = default.copy()
        mask = counts > 0
        out[mask] = (votes[mask] / counts[mask] >= threshold).astype(int)
    return out


def command_eval(args) -> None:
    sample_df = load_sample()
    test_df = load_test() if args.include_test_contexts else None
    evidence = build_evidence(sample_df, test_df, sample_df) if args.variant == "evidence" else None
    model_path = MODELS[args.model]
    process = start_server(model_path)
    try:
        trace = OUTPUT_DIR / f"gemma_{args.model}_{args.variant}_sample_{args.rows}.jsonl"
        preds = run_variant(sample_df, args.variant, trace, rows=args.rows, evidence=evidence, ids_file=args.ids_file)
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=10)

    y = sample_df["label"].to_numpy()
    mask = preds != -1
    if args.rows == "no_context":
        print_metrics(f"{args.model}/{args.variant}/no_context", y[mask], preds[mask])
        full = load_partial_trace(OUTPUT_DIR / "gemma_qat_base_sample_all.jsonl", len(sample_df))
        if (full == -1).all() and (OUTPUT_DIR / "gemma12_qat_sample_trace.jsonl").exists():
            full = load_trace_predictions(OUTPUT_DIR / "gemma12_qat_sample_trace.jsonl", len(sample_df))
        combined = apply_context_rule(sample_df, full if (full != -1).any() else np.zeros(len(sample_df), dtype=int))
        combined[mask] = preds[mask]
        print_metrics(f"{args.model}/{args.variant}/combined_context_rule", y, combined)
    else:
        evaluate_by_slice(sample_df, apply_context_rule(sample_df, preds), f"{args.model}/{args.variant}")


def command_predict(args) -> None:
    sample_df = load_sample()
    test_df = load_test()
    evidence = build_evidence(sample_df, test_df, test_df) if args.variant == "evidence" else None
    model_path = MODELS[args.model]
    process = start_server(model_path)
    try:
        trace = OUTPUT_DIR / f"gemma_{args.model}_{args.variant}_test_{args.rows}.jsonl"
        preds = run_variant(test_df, args.variant, trace, rows=args.rows, evidence=evidence, ids_file=args.ids_file)
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=10)

    base = np.zeros(len(test_df), dtype=int)
    if args.base_trace and Path(args.base_trace).exists():
        base = load_trace_predictions(Path(args.base_trace), len(test_df))
    final = apply_context_rule(test_df, base)
    mask = preds != -1
    final[mask] = preds[mask]
    output = Path(args.output)
    pd.DataFrame({"id": test_df["id"].to_numpy(), "label": final}).to_csv(output, index=False)
    print("wrote", output, pd.Series(final).value_counts().sort_index().to_dict())


def command_ensemble(args) -> None:
    sample_df = load_sample()
    test_df = load_test()
    y = sample_df["label"].to_numpy()

    sample_preds = []
    test_preds = []
    for spec in args.member:
        sample_trace, test_trace = [Path(part) for part in spec.split(":", 1)]
        sample_pred = load_partial_trace(sample_trace, len(sample_df))
        test_pred = load_partial_trace(test_trace, len(test_df))
        sample_preds.append(sample_pred)
        test_preds.append(test_pred)

    base_sample = load_trace_predictions(OUTPUT_DIR / "gemma12_qat_sample_trace.jsonl", len(sample_df))
    base_test = load_trace_predictions(OUTPUT_DIR / "gemma12_qat_test_trace.jsonl", len(test_df))
    sample_default = apply_context_rule(sample_df, base_sample)
    test_default = apply_context_rule(test_df, base_test)

    best = None
    for threshold in np.linspace(0.34, 0.67, 34):
        pred = vote_predictions(sample_preds, default=sample_default, threshold=float(threshold))
        pred = apply_context_rule(sample_df, pred)
        metrics = print_metrics(f"ensemble threshold={threshold:.2f}", y, pred)
        key = (metrics["macro_f1"], metrics["f1_0"])
        if best is None or key > best[0]:
            best = (key, threshold, pred, metrics)

    assert best is not None
    threshold = best[1]
    final = vote_predictions(test_preds, default=test_default, threshold=float(threshold))
    final = apply_context_rule(test_df, final)
    output = Path(args.output)
    pd.DataFrame({"id": test_df["id"].to_numpy(), "label": final}).to_csv(output, index=False)
    print("best threshold", threshold, best[3])
    print("wrote", output, pd.Series(final).value_counts().sort_index().to_dict())


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    eval_parser = sub.add_parser("eval")
    eval_parser.add_argument("--model", choices=MODELS, default="qat")
    eval_parser.add_argument("--variant", choices=VARIANT_HEADERS, required=True)
    eval_parser.add_argument("--rows", choices=["all", "no_context", "context"], default="no_context")
    eval_parser.add_argument("--ids-file")
    eval_parser.add_argument("--include-test-contexts", action="store_true")
    eval_parser.set_defaults(func=command_eval)

    predict_parser = sub.add_parser("predict")
    predict_parser.add_argument("--model", choices=MODELS, default="qat")
    predict_parser.add_argument("--variant", choices=VARIANT_HEADERS, required=True)
    predict_parser.add_argument("--rows", choices=["all", "no_context", "context"], default="no_context")
    predict_parser.add_argument("--ids-file")
    predict_parser.add_argument("--base-trace", default=str(OUTPUT_DIR / "gemma12_qat_test_trace.jsonl"))
    predict_parser.add_argument("--output", default=str(OUTPUT_DIR / "submission_ensemble_candidate.csv"))
    predict_parser.set_defaults(func=command_predict)

    ensemble_parser = sub.add_parser("ensemble")
    ensemble_parser.add_argument(
        "--member",
        action="append",
        required=True,
        help="sample_trace:test_trace pair. May be repeated.",
    )
    ensemble_parser.add_argument("--output", default=str(OUTPUT_DIR / "submission_gemma_vote_ensemble.csv"))
    ensemble_parser.set_defaults(func=command_ensemble)

    args = parser.parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
