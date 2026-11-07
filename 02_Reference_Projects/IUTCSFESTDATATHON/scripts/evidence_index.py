from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from datathon_utils import OUTPUT_DIR, clean_text, load_sample, load_test, normalize_bn


DEFAULT_INDEX_DIR = OUTPUT_DIR / "evidence_index" / "external_qa_char_tfidf"


def evidence_doc(row) -> str:
    context = clean_text(row.get("context", ""))
    return (
        clean_text(row.get("question", ""))
        + " [ANS] "
        + clean_text(row.get("answer", ""))
        + " [CTX] "
        + context
    )


def query_text(row, mode: str = "prompt_response") -> str:
    prompt = clean_text(row["prompt_bn"])
    response = clean_text(row["response_bn"])
    if mode == "prompt":
        return prompt
    if mode == "response":
        return response
    if mode == "prompt_response":
        return prompt + " [ANS] " + response
    raise ValueError(f"unknown query mode: {mode}")


def load_corpus(path: Path, max_docs: int | None = None) -> pd.DataFrame:
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_json(path, lines=True)
    if max_docs:
        df = df.head(max_docs)
    keep = ["question", "answer", "context", "source_repo", "source_file", "license"]
    for col in keep:
        if col not in df.columns:
            df[col] = ""
    df = df[keep].fillna("").drop_duplicates(["question", "answer", "context"]).reset_index(drop=True)
    df["doc_text"] = [evidence_doc(row) for _, row in df.iterrows()]
    return df


def write_jsonl_gz(df: pd.DataFrame, path: Path) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in df.drop(columns=["doc_text"], errors="ignore").to_dict("records"):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl_gz(path: Path) -> pd.DataFrame:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows).fillna("")


def build_index(args) -> None:
    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus(Path(args.corpus), max_docs=args.max_docs)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(args.min_ngram, args.max_ngram),
        max_features=args.max_features,
        min_df=args.min_df,
        sublinear_tf=True,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(corpus["doc_text"].tolist())
    sparse.save_npz(index_dir / "matrix.npz", matrix, compressed=True)
    joblib.dump(vectorizer, index_dir / "vectorizer.joblib", compress=3)
    write_jsonl_gz(corpus, index_dir / "corpus.jsonl.gz")
    metadata = {
        "corpus": str(Path(args.corpus)),
        "rows": int(len(corpus)),
        "matrix_shape": list(matrix.shape),
        "max_features": int(args.max_features),
        "min_df": int(args.min_df),
        "ngram_range": [int(args.min_ngram), int(args.max_ngram)],
        "source_counts": corpus["source_repo"].value_counts().to_dict(),
        "license_counts": corpus["license"].value_counts().to_dict(),
    }
    (index_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote index", index_dir)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def load_index(index_dir: Path):
    vectorizer = joblib.load(index_dir / "vectorizer.joblib")
    matrix = sparse.load_npz(index_dir / "matrix.npz")
    corpus = read_jsonl_gz(index_dir / "corpus.jsonl.gz")
    return corpus, vectorizer, matrix


def topk_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    if scores.shape[0] <= top_k:
        return np.argsort(scores)[::-1]
    idx = np.argpartition(scores, -top_k)[-top_k:]
    return idx[np.argsort(scores[idx])[::-1]]


def retrieve(args) -> None:
    index_dir = Path(args.index_dir)
    corpus, vectorizer, matrix = load_index(index_dir)
    df = load_sample() if args.split == "sample" else load_test()
    if args.only_no_context:
        df = df[~df["has_context"]].copy()
    if args.limit:
        df = df.head(args.limit).copy()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    queries = [query_text(row, mode=args.query_mode) for _, row in df.iterrows()]
    q_matrix = vectorizer.transform(queries)
    exact_lookup: dict[str, list[int]] = {}
    if args.inject_exact_question:
        for corpus_i, question in enumerate(corpus["question"].tolist()):
            key = normalize_bn(question)
            if key:
                exact_lookup.setdefault(key, []).append(corpus_i)

    records = []
    with output.open("w", encoding="utf-8") as handle:
        for start in range(0, q_matrix.shape[0], args.batch_size):
            stop = min(start + args.batch_size, q_matrix.shape[0])
            sims = linear_kernel(q_matrix[start:stop], matrix)
            for local_i, scores in enumerate(sims):
                df_index = df.index[start + local_i]
                row = df.iloc[start + local_i]
                exact = exact_lookup.get(normalize_bn(row["prompt_bn"]), [])
                best = list(exact[: args.top_k])
                for idx in topk_indices(scores, args.top_k + len(best)):
                    idx = int(idx)
                    if idx not in best:
                        best.append(idx)
                    if len(best) >= args.top_k:
                        break
                evidence = []
                for rank, corpus_i in enumerate(best[: args.top_k], start=1):
                    exact_question = corpus_i in exact
                    score = float(scores[corpus_i])
                    if exact_question:
                        score = max(score, 1.0)
                    if score <= args.min_score:
                        continue
                    item = corpus.iloc[int(corpus_i)]
                    evidence.append(
                        {
                            "rank": rank,
                            "score": round(score, 6),
                            "exact_question": bool(exact_question),
                            "question": item["question"],
                            "answer": item["answer"],
                            "context": item["context"],
                            "source_repo": item["source_repo"],
                            "source_file": item["source_file"],
                            "license": item["license"],
                        }
                    )
                record = {
                    "row_index": int(df_index),
                    "id": int(row["id"]) if "id" in df.columns else None,
                    "split": args.split,
                    "has_context": bool(row["has_context"]),
                    "category": row["category"],
                    "prompt_bn": row["prompt_bn"],
                    "response_bn": row["response_bn"],
                    "query": queries[start + local_i],
                    "evidence": evidence,
                }
                if "label" in df.columns:
                    record["label"] = int(row["label"])
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)

    summary = summarize_records(records)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", output, "rows", len(records))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def summarize_records(records: list[dict]) -> dict:
    top_scores = [row["evidence"][0]["score"] if row["evidence"] else 0.0 for row in records]
    evidence_counts = [len(row["evidence"]) for row in records]
    by_category = {}
    for row, score in zip(records, top_scores, strict=True):
        bucket = by_category.setdefault(row["category"], {"n": 0, "top_score_sum": 0.0, "with_evidence": 0})
        bucket["n"] += 1
        bucket["top_score_sum"] += score
        bucket["with_evidence"] += int(score > 0)
    for bucket in by_category.values():
        bucket["mean_top_score"] = round(bucket.pop("top_score_sum") / max(bucket["n"], 1), 6)
    return {
        "rows": len(records),
        "with_evidence": int(sum(count > 0 for count in evidence_counts)),
        "mean_evidence_count": round(float(np.mean(evidence_counts)) if evidence_counts else 0.0, 4),
        "mean_top_score": round(float(np.mean(top_scores)) if top_scores else 0.0, 6),
        "median_top_score": round(float(np.median(top_scores)) if top_scores else 0.0, 6),
        "p25_top_score": round(float(np.quantile(top_scores, 0.25)) if top_scores else 0.0, 6),
        "p75_top_score": round(float(np.quantile(top_scores, 0.75)) if top_scores else 0.0, 6),
        "by_category": by_category,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--corpus", default=str(OUTPUT_DIR / "external_qa_corpus.jsonl"))
    build.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    build.add_argument("--max-docs", type=int)
    build.add_argument("--max-features", type=int, default=200_000)
    build.add_argument("--min-df", type=int, default=2)
    build.add_argument("--min-ngram", type=int, default=3)
    build.add_argument("--max-ngram", type=int, default=5)
    build.set_defaults(func=build_index)

    get = sub.add_parser("retrieve")
    get.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    get.add_argument("--split", choices=["sample", "test"], required=True)
    get.add_argument("--output", required=True)
    get.add_argument("--top-k", type=int, default=5)
    get.add_argument("--batch-size", type=int, default=64)
    get.add_argument("--min-score", type=float, default=0.0)
    get.add_argument("--query-mode", choices=["prompt_response", "prompt", "response"], default="prompt_response")
    get.add_argument("--inject-exact-question", action="store_true")
    get.add_argument("--limit", type=int)
    get.add_argument("--all-rows", dest="only_no_context", action="store_false")
    get.set_defaults(func=retrieve, only_no_context=True)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
