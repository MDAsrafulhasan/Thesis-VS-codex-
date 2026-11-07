from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from mlx_lm import generate, load

from datathon_utils import (
    OUTPUT_DIR,
    apply_context_rule,
    evaluate_by_slice,
    load_sample,
    load_test,
    parse_label,
    print_metrics,
)


MODELS = {
    "qwen35_9b": Path("/Users/seyam/.lmstudio/models/lmstudio-community/Qwen3.5-9B-MLX-4bit"),
    "qwen36_27b": Path("/Users/seyam/.lmstudio/models/lmstudio-community/Qwen3.6-27B-MLX-4bit"),
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
    "strict": """তুমি কঠোর বাংলা hallucination detector। শুধু 0 অথবা 1 লিখবে।
0 = উত্তরটি ভুল, বানানো, প্রশ্নের সঙ্গে মেলে না, অথবা প্রসঙ্গ দ্বারা সমর্থিত নয়।
1 = উত্তরটি সম্পূর্ণ সঠিক এবং প্রসঙ্গ থাকলে প্রসঙ্গ দ্বারা সমর্থিত।""",
    "lenient": """তুমি বাংলা উত্তর যাচাইকারী। বানান বা শব্দচয়নের ছোট পার্থক্য উপেক্ষা করবে।
শুধু 0 অথবা 1 লিখবে।
0 = তথ্যগতভাবে ভুল, বানানো, প্রশ্নের উত্তর নয়, অথবা অসমর্থিত।
1 = তথ্যগতভাবে সঠিক বা সমার্থকভাবে সঠিক।""",
    "noctx_fact": """তুমি বাংলা সাধারণ জ্ঞান ও পরীক্ষার প্রশ্ন যাচাইকারী। প্রসঙ্গ না থাকলে নিজের জ্ঞান দিয়ে fact-check করবে।
শুধু 0 অথবা 1 লিখবে।
0 = ভুল ব্যক্তি/স্থান/তারিখ/সংখ্যা/অর্থ, বানানো তথ্য, বা প্রশ্নের ভুল উত্তর।
1 = সঠিক উত্তর।""",
    "context_nli": """তুমি বাংলা context-grounding verifier। প্রসঙ্গ থাকলে উত্তরটি সরাসরি প্রসঙ্গ দ্বারা সমর্থিত কি না দেখবে।
শুধু 0 অথবা 1 লিখবে।
0 = প্রসঙ্গে নেই, প্রসঙ্গের সঙ্গে মেলে না, বা তথ্যগতভাবে ভুল।
1 = প্রসঙ্গ/প্রশ্ন অনুযায়ী সঠিক।""",
    "meaning": """তুমি বাংলা শব্দার্থ, বাগধারা, ভাবার্থ ও শাব্দিক অর্থ যাচাইকারী।
শুধু 0 অথবা 1 লিখবে।
0 = অর্থ/ভাবার্থ/শাব্দিক অর্থ ভুল।
1 = অর্থ/ভাবার্থ/শাব্দিক অর্থ সঠিক বা গ্রহণযোগ্য সমার্থক।""",
    "solve_compare": """তুমি বাংলা পরীক্ষার প্রশ্ন সমাধানকারী ও fact-checker। মনে মনে সঠিক উত্তর বের করবে, তারপর দেওয়া উত্তরটির সঙ্গে তুলনা করবে।
শুধু 0 অথবা 1 লিখবে।
0 = দেওয়া উত্তরটি সঠিক উত্তরের সঙ্গে মেলে না, ভুল entity/number/date/meaning দেয়, বা প্রশ্নের উত্তর নয়।
1 = দেওয়া উত্তরটি সঠিক উত্তরের সঙ্গে মেলে বা গ্রহণযোগ্য সমার্থক।""",
    "exam_strict": """তুমি BCS/ভর্তি/সাধারণ জ্ঞান ধাঁচের বাংলা MCQ/QA verifier। প্রশ্নের exact চাহিদা মিলাবে।
শুধু 0 অথবা 1 লিখবে।
0 = ভুল অপশন, ভুল তথ্য, ভুল সাল/সংখ্যা, ভুল লেখক/স্থান, অথবা বিভ্রান্তিকর উত্তর।
1 = পরীক্ষার answer key অনুযায়ী সঠিক উত্তর।""",
    "zero_bias": """তুমি অত্যন্ত সতর্ক hallucination detector। উত্তর নিশ্চিতভাবে সঠিক না হলে 0 দেবে।
শুধু 0 অথবা 1 লিখবে।
0 = সন্দেহজনক, অসমর্থিত, ভুল, আংশিক ভুল, বা প্রশ্নের সঙ্গে অমিল।
1 = নিশ্চিতভাবে সঠিক।""",
    "one_bias": """তুমি বাংলা উত্তর যাচাইকারী। একই অর্থ হলে 1 দেবে, ছোট বানান/রূপভেদে 0 দেবে না।
শুধু 0 অথবা 1 লিখবে।
0 = স্পষ্টভাবে ভুল বা hallucinated।
1 = উত্তরটি মূলত সঠিক।""",
    "evidence_fact": """তুমি বাংলা hallucination detector। দেওয়া বাহ্যিক প্রমাণগুলো noisy হতে পারে, তাই আগে প্রমাণটি প্রশ্নের সঙ্গে সম্পর্কিত কি না যাচাই করবে।
শুধু 0 অথবা 1 লিখবে।
0 = উত্তরটি প্রাসঙ্গিক প্রমাণ বা নিজের নির্ভরযোগ্য জ্ঞানের সঙ্গে মেলে না, ভুল entity/date/number/meaning দেয়, অথবা প্রশ্নের উত্তর নয়।
1 = উত্তরটি প্রাসঙ্গিক প্রমাণ দ্বারা সমর্থিত বা নির্ভরযোগ্যভাবে সঠিক।""",
}


def format_external_evidence(row, max_evidence_chars: int = 2400) -> str:
    evidence = getattr(row, "external_evidence", "")
    if not evidence:
        return ""
    return str(evidence)[:max_evidence_chars]


def make_prompt(row, variant: str, max_context_chars: int = 1800, max_evidence_chars: int = 2400) -> str:
    parts = [
        VARIANT_HEADERS[variant],
        "",
        BASE_EXAMPLES,
        "",
        "এখন নতুন উদাহরণের লেবেল দাও। ব্যাখ্যা নয়, শুধু 0 অথবা 1।",
    ]
    if row.context:
        parts.append("প্রসঙ্গ: " + row.context[:max_context_chars])
    evidence_text = format_external_evidence(row, max_evidence_chars=max_evidence_chars)
    if evidence_text:
        parts.append("বাহ্যিক প্রমাণ:\n" + evidence_text)
    parts.append("প্রশ্ন: " + row.prompt_bn)
    parts.append("উত্তর: " + row.response_bn)
    parts.append("লেবেল:")
    return "\n".join(parts)


def apply_chat(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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


def row_indices(df: pd.DataFrame, rows: str, ids_file: str | None = None) -> np.ndarray:
    selected_ids: set[int] | None = None
    if ids_file:
        path = Path(ids_file)
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            items = payload.get("records", payload.get("uncertainty", payload))
            selected_ids = {int(item["id"]) for item in items}
        elif path.suffix == ".csv":
            selected_ids = set(pd.read_csv(path)["id"].astype(int))
        else:
            selected_ids = {int(value) for value in path.read_text(encoding="utf-8").split()}
    if rows == "all":
        mask = np.ones(len(df), dtype=bool)
    elif rows == "no_context":
        mask = (~df["has_context"].to_numpy()).copy()
    elif rows == "context":
        mask = df["has_context"].to_numpy(copy=True)
    elif rows.startswith("category:"):
        category = rows.split(":", 1)[1]
        mask = (df["category"].to_numpy() == category).copy()
    else:
        raise ValueError(rows)
    if selected_ids is not None:
        mask &= df["id"].astype(int).isin(selected_ids).to_numpy()
    return np.flatnonzero(mask)


def load_external_evidence(path: str | None, top_k: int = 3) -> dict[int, str]:
    if not path:
        return {}
    evidence_map = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        parts = []
        for item in row.get("evidence", [])[:top_k]:
            segment = (
                f"[{item.get('rank', len(parts) + 1)}] "
                f"প্রশ্ন: {item.get('question', '')} "
                f"উত্তর: {item.get('answer', '')}"
            )
            context = item.get("context", "")
            if context:
                segment += f" প্রসঙ্গ: {context[:450]}"
            segment += f" উৎস: {item.get('source_repo', '')}"
            parts.append(segment)
        evidence_map[int(row["row_index"])] = "\n".join(parts)
    return evidence_map


def attach_external_evidence(df: pd.DataFrame, evidence_path: str | None, top_k: int) -> pd.DataFrame:
    df = df.copy()
    evidence_map = load_external_evidence(evidence_path, top_k=top_k)
    df["external_evidence"] = [evidence_map.get(int(i), "") for i in df.index]
    if evidence_path:
        print("loaded external evidence", evidence_path, "rows", len(evidence_map))
    return df


def run_variant(model, tokenizer, df: pd.DataFrame, model_name: str, variant: str, split: str, rows: str, ids_file: str | None = None) -> np.ndarray:
    evidence_tag = ""
    if "external_evidence" in df.columns and df["external_evidence"].astype(bool).any():
        evidence_tag = "_evidence"
    trace_path = OUTPUT_DIR / f"mlx_{model_name}_{variant}{evidence_tag}_{split}_{rows.replace(':', '_')}.jsonl"
    indices = row_indices(df, rows, ids_file=ids_file)
    preds = load_partial_trace(trace_path, len(df))
    remaining = [int(i) for i in indices if preds[int(i)] == -1]
    print(f"model={model_name} variant={variant} split={split} rows={rows} remaining={len(remaining)} trace={trace_path}")

    start = time.time()
    with trace_path.open("a", encoding="utf-8") as handle:
        for i in remaining:
            row = df.iloc[i]
            prompt = apply_chat(tokenizer, make_prompt(row, variant))
            generated = generate(model, tokenizer, prompt=prompt, max_tokens=8, verbose=False)
            pred = parse_label(generated)
            record = {"i": i, "pred": int(pred), "gen": generated}
            if "id" in df.columns:
                record["id"] = int(df.iloc[i]["id"])
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            preds[i] = pred
            done = len([idx for idx in indices if preds[int(idx)] != -1])
            if done % 25 == 0:
                elapsed = (time.time() - start) / 60
                known = preds[indices][preds[indices] != -1]
                print(f"done {done}/{len(indices)} elapsed_min={elapsed:.1f} counts={pd.Series(known).value_counts().to_dict()}", flush=True)
    return preds


def command_eval(args) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = attach_external_evidence(load_sample(), args.evidence, top_k=args.evidence_top_k)
    print("loading", MODELS[args.model])
    model, tokenizer = load(str(MODELS[args.model]))
    y = df["label"].to_numpy()
    for variant in args.variant:
        preds = run_variant(model, tokenizer, df, args.model, variant, "sample", args.rows, args.ids_file)
        mask = preds != -1
        if args.rows == "all":
            post = apply_context_rule(df, preds)
            evaluate_by_slice(df, post, f"{args.model}/{variant}/post")
        else:
            print_metrics(f"{args.model}/{variant}/{args.rows}", y[mask], preds[mask])
            # Use deterministic context rule outside the evaluated row subset.
            combined = apply_context_rule(df, np.zeros(len(df), dtype=int))
            combined[mask] = preds[mask]
            combined = apply_context_rule(df, combined)
            print_metrics(f"{args.model}/{variant}/combined_context_rule", y, combined)


def command_predict(args) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = attach_external_evidence(load_test(), args.evidence, top_k=args.evidence_top_k)
    print("loading", MODELS[args.model])
    model, tokenizer = load(str(MODELS[args.model]))
    final = apply_context_rule(df, np.zeros(len(df), dtype=int))
    for variant in args.variant:
        preds = run_variant(model, tokenizer, df, args.model, variant, "test", args.rows, args.ids_file)
        mask = preds != -1
        out = final.copy()
        out[mask] = preds[mask]
        out = apply_context_rule(df, out)
        output = Path(args.output or OUTPUT_DIR / f"submission_mlx_{args.model}_{variant}_{args.rows}.csv")
        if len(args.variant) > 1:
            output = OUTPUT_DIR / f"submission_mlx_{args.model}_{variant}_{args.rows}.csv"
        pd.DataFrame({"id": df["id"].to_numpy(), "label": out}).to_csv(output, index=False)
        print("wrote", output, pd.Series(out).value_counts().sort_index().to_dict())


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    eval_parser = sub.add_parser("eval")
    eval_parser.add_argument("--model", choices=MODELS, required=True)
    eval_parser.add_argument("--variant", choices=VARIANT_HEADERS, action="append", required=True)
    eval_parser.add_argument("--rows", default="all")
    eval_parser.add_argument("--evidence")
    eval_parser.add_argument("--evidence-top-k", type=int, default=3)
    eval_parser.add_argument("--ids-file")
    eval_parser.set_defaults(func=command_eval)

    predict_parser = sub.add_parser("predict")
    predict_parser.add_argument("--model", choices=MODELS, required=True)
    predict_parser.add_argument("--variant", choices=VARIANT_HEADERS, action="append", required=True)
    predict_parser.add_argument("--rows", default="no_context")
    predict_parser.add_argument("--output")
    predict_parser.add_argument("--evidence")
    predict_parser.add_argument("--evidence-top-k", type=int, default=3)
    predict_parser.add_argument("--ids-file")
    predict_parser.set_defaults(func=command_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
