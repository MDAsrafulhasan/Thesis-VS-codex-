from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from datathon_utils import OUTPUT_DIR, load_test, normalize_bn


def load_contexts(path: Path, max_chars: int) -> list[dict]:
    grouped: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            context = str(row.get("context", "")).strip()
            if not context:
                continue
            key = normalize_bn(context[:max_chars])
            item = grouped.setdefault(
                key,
                {
                    "context": context[:max_chars],
                    "questions": [],
                    "sources": set(),
                },
            )
            if len(item["questions"]) < 4:
                item["questions"].append(str(row.get("question", "")))
            item["sources"].add(str(row.get("source_repo", "")))
    output = list(grouped.values())
    for item in output:
        item["sources"] = sorted(item["sources"])
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve topic/page contexts rather than individual QA rows.")
    parser.add_argument("--corpus", type=Path, default=OUTPUT_DIR / "external_qa_corpus.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id-start", type=int, required=True)
    parser.add_argument("--id-end", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=5000)
    parser.add_argument("--max-features", type=int, default=180_000)
    args = parser.parse_args()

    contexts = load_contexts(args.corpus, args.max_context_chars)
    documents = [
        " ".join(item["questions"]) + " " + item["context"]
        for item in contexts
    ]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=args.max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(documents)
    test = load_test()
    selected = test[(test["id"] >= args.id_start) & (test["id"] <= args.id_end)]
    queries = vectorizer.transform(selected["prompt_bn"].tolist())
    records = []
    for start in range(0, len(selected), 64):
        scores = linear_kernel(queries[start : start + 64], matrix)
        for offset, row_scores in enumerate(scores):
            row = selected.iloc[start + offset]
            order = np.argpartition(row_scores, -args.top_k)[-args.top_k:]
            order = order[np.argsort(row_scores[order])[::-1]]
            evidence = []
            for rank, index in enumerate(order, start=1):
                item = contexts[int(index)]
                evidence.append(
                    {
                        "rank": rank,
                        "score": float(row_scores[index]),
                        "questions": item["questions"],
                        "sources": item["sources"],
                        "context": item["context"],
                    }
                )
            records.append(
                {
                    "id": int(row.id),
                    "prompt": str(row.prompt_bn),
                    "response": str(row.response_bn),
                    "evidence": evidence,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print("contexts", len(contexts), "matrix", matrix.shape)
    print("wrote", args.output, "rows", len(records))


if __name__ == "__main__":
    main()
