from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from mlx_lm import generate, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from datathon_utils import BN_DIGITS, OUTPUT_DIR, load_test
from mlx_qwen_judge import MODELS, apply_chat


SYSTEM = """তুমি উৎস-নির্ভর বাংলা fact verifier। নিচে বাংলা বিশ্বকোষের কয়েকটি উদ্ধৃতি দেওয়া হবে।
প্রশ্নের যে সম্পর্কটি জানতে চাওয়া হয়েছে (যেমন জন্ম, মৃত্যু, প্রকাশ, রচনা, উদ্বোধন, সংখ্যা), ঠিক সেই সম্পর্কের তথ্য মিলাবে।
একই সংখ্যা পৃষ্ঠার অন্য কোনো ঘটনায় থাকলে সেটিকে প্রমাণ হিসেবে গ্রহণ করবে না।
উদ্ধৃতি প্রশ্নের উত্তর না দিলে অনুমান করবে না এবং verdict=insufficient দেবে।

শুধু compact JSON দাও:
{"verdict":"supported|contradicted|insufficient","expected_answer":"সংক্ষিপ্ত উত্তর অথবা খালি","confidence":0.0 থেকে 1.0,"evidence_rank":1}

supported = দেওয়া উত্তরটি উদ্ধৃতির সংশ্লিষ্ট তথ্যের সঙ্গে মেলে।
contradicted = উদ্ধৃতি একই প্রশ্নের স্পষ্ট ভিন্ন উত্তর দেয়।
insufficient = উদ্ধৃতি দিয়ে নিশ্চিত সিদ্ধান্ত সম্ভব নয়।"""


def split_chunks(page: dict) -> list[dict]:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[।!?])\s+|\n+", str(page.get("extract", "")))
        if len(sentence.strip()) >= 18
    ]
    chunks = []
    for index in range(len(sentences)):
        # Two adjacent sentences preserve short lead facts without making every
        # number on the page look related to every question.
        text = " ".join(sentences[index : index + 2])
        chunks.append(
            {
                "query": page.get("query", ""),
                "title": page.get("title", ""),
                "text": text,
            }
        )
    return chunks


def build_retriever(corpus: Path):
    pages = json.loads(corpus.read_text(encoding="utf-8"))
    chunks = [chunk for page in pages if page.get("extract") for chunk in split_chunks(page)]
    documents = [f"{item['query']} {item['title']} {item['text']}" for item in chunks]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(documents)
    return chunks, vectorizer, matrix


def build_page_retriever(corpus: Path, max_page_chars: int):
    pages = [
        {
            "query": item.get("query", ""),
            "title": item.get("title", ""),
            "text": str(item.get("extract", ""))[:max_page_chars],
        }
        for item in json.loads(corpus.read_text(encoding="utf-8"))
        if item.get("extract")
    ]
    # Page selection should depend on entity/work names, not incidental words
    # buried in the article body.
    documents = [f"{item['query']} {item['title']}" for item in pages]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=1,
        sublinear_tf=True,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(documents)
    return pages, vectorizer, matrix


def retrieve(prompt: str, chunks: list[dict], vectorizer, matrix, top_k: int) -> list[dict]:
    scores = linear_kernel(vectorizer.transform([prompt]), matrix)[0]
    order = np.argsort(scores)[::-1]
    selected = []
    seen = set()
    for index in order:
        item = chunks[int(index)]
        key = (item["title"], item["text"])
        if key in seen:
            continue
        seen.add(key)
        selected.append({**item, "score": float(scores[int(index)])})
        if len(selected) == top_k:
            break
    return selected


def make_prompt(row, evidence: list[dict]) -> str:
    parts = [SYSTEM, "", "বিশ্বকোষের উদ্ধৃতি:"]
    for rank, item in enumerate(evidence, start=1):
        parts.append(f"[{rank}] পৃষ্ঠা: {item['title']}\n{item['text']}")
    parts.extend(["", f"প্রশ্ন: {row.prompt_bn}", f"দেওয়া উত্তর: {row.response_bn}", "JSON:"])
    return "\n".join(parts)


def parse_output(raw: str) -> tuple[str, float, str, int | None, str | None]:
    text = (raw or "").strip().translate(BN_DIGITS)
    match = re.search(r"\{.*?\}", text, flags=re.S)
    if not match:
        return "insufficient", 0.0, "", None, "json:not_found"
    try:
        obj = json.loads(match.group(0))
        verdict = str(obj.get("verdict", "insufficient")).lower().strip()
        if verdict not in {"supported", "contradicted", "insufficient"}:
            raise ValueError("invalid verdict")
        confidence = max(0.0, min(1.0, float(obj.get("confidence", 0.0))))
        expected = str(obj.get("expected_answer", "")).strip()
        rank = obj.get("evidence_rank")
        rank = int(rank) if rank is not None and str(rank).isdigit() else None
        return verdict, confidence, expected, rank, None
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return "insufficient", 0.0, "", None, f"json:{exc}"


def load_trace(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    return {
        int(item["id"]): item
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for item in [json.loads(line)]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a factual test block against exact-title Bengali Wikipedia pages.")
    parser.add_argument("--model", choices=MODELS, default="qwen36_27b")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--base-submission", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id-min", type=int, default=319)
    parser.add_argument("--id-max", type=int, default=381)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--retrieval-mode", choices=["chunks", "page"], default="chunks")
    parser.add_argument("--max-page-chars", type=int, default=16_000)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--min-confidence", type=float, default=0.85)
    args = parser.parse_args()

    test = load_test()
    selected = test[test["id"].between(args.id_min, args.id_max)]
    if args.retrieval_mode == "page":
        chunks, vectorizer, matrix = build_page_retriever(args.corpus, args.max_page_chars)
    else:
        chunks, vectorizer, matrix = build_retriever(args.corpus)
    existing = load_trace(args.trace)
    remaining = selected[~selected["id"].isin(existing)]
    print("selected", len(selected), "chunks", len(chunks), "existing", len(existing), "remaining", len(remaining))

    if len(remaining):
        print("loading", MODELS[args.model])
        model, tokenizer = load(str(MODELS[args.model]))
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        with args.trace.open("a", encoding="utf-8") as handle:
            for offset, row in enumerate(remaining.itertuples(), start=1):
                evidence = retrieve(row.prompt_bn, chunks, vectorizer, matrix, args.top_k)
                prompt = apply_chat(tokenizer, make_prompt(row, evidence))
                raw = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False)
                verdict, confidence, expected, rank, parse_error = parse_output(raw)
                record = {
                    "id": int(row.id),
                    "row_index": int(row.Index),
                    "verdict": verdict,
                    "pred": 1 if verdict == "supported" else 0 if verdict == "contradicted" else None,
                    "confidence": confidence,
                    "expected_answer": expected,
                    "evidence_rank": rank,
                    "parse_error": parse_error,
                    "evidence": evidence,
                    "raw": raw,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                existing[int(row.id)] = record
                if offset % 5 == 0 or offset == len(remaining):
                    elapsed = (time.time() - started) / 60
                    counts = pd.Series([item["verdict"] for item in existing.values()]).value_counts().to_dict()
                    print(f"done {offset}/{len(remaining)} elapsed_min={elapsed:.1f} verdicts={counts}", flush=True)

    base = pd.read_csv(args.base_submission)
    if len(base) != len(test) or not np.array_equal(base["id"].to_numpy(), test["id"].to_numpy()):
        raise ValueError("base submission IDs do not match test")
    output = base.copy()
    old_by_id = base.set_index("id")["label"].to_dict()
    changes = []
    accepted = []
    for row_id in selected["id"]:
        item = existing[int(row_id)]
        pred = item.get("pred")
        if pred is None or float(item.get("confidence", 0.0)) < args.min_confidence:
            continue
        new = int(pred)
        old = int(old_by_id[int(row_id)])
        output.loc[output["id"] == row_id, "label"] = new
        accepted.append(int(row_id))
        if old != new:
            changes.append((int(row_id), old, new, item["verdict"], item["confidence"], item["expected_answer"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print("wrote", args.output, output["label"].value_counts().sort_index().to_dict())
    print("accepted", len(accepted), "changed", len(changes))
    for change in changes:
        print("CHANGE", *change)


if __name__ == "__main__":
    main()
