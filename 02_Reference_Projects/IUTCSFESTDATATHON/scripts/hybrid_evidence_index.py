from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from datathon_utils import load_sample, load_test, normalize_bn


RELATIONS = {
    "birth": ("জন্ম", "জন্মগ্রহণ"),
    "death": ("মৃত্যু", "মারা", "নিহত"),
    "law_penalty": ("কারাদণ্ড", "অর্থদণ্ড", "জরিমানা", "শাস্তি"),
    "writing": ("রচনা করেন", "রচিত", "লেখেন", "লিখিত"),
    "publication": ("প্রকাশিত", "প্রকাশকাল", "প্রকাশ"),
    "recognition": ("স্বীকৃতি", "মর্যাদা"),
    "inauguration": ("উদ্বোধন", "চালু হয়", "চালু হয়"),
    "location": ("কোথায়", "কোথায়", "অবস্থিত", "স্থান"),
    "establishment": ("প্রতিষ্ঠিত", "প্রতিষ্ঠা", "গঠিত"),
    "construction": ("নির্মাণ", "নির্মিত"),
    "length": ("দৈর্ঘ্য", "কিলোমিটার দীর্ঘ"),
    "count": ("কতটি", "কয়টি", "কয়টি", "মোট কত"),
    "area": ("আয়তন", "আয়তন", "বর্গকিলোমিটার", "একর"),
    "population": ("জনসংখ্যা", "লোকসংখ্যা"),
    "language": ("ভাষা", "সরকারি ভাষা"),
}

DOMAIN_AUTHORITY = {
    "law": 1.00,
    "government": 1.08,
    "textbook": 1.05,
    "wikipedia": 1.00,
    "news": 0.98,
    "qa": 0.95,
    "unknown": 0.92,
}

SOURCE_AUTHORITY = {
    "sakhadib/Bangladesh-Legal-Acts-Dataset": 1.10,
    "bipinsaha/bangla-law-qna": 0.94,
    "md-nishat-008/Bangla-TextBook": 1.05,
    "rejauldu/bengali-wikipedia": 1.00,
    "zabir-nabil/bangla_newspaper_dataset": 0.98,
}

QUESTION_ANSWER_SOURCES = {
    "bipinsaha/bangla-law-qna",
}


def relation_types(text: str) -> tuple[str, ...]:
    value = normalize_bn(text)
    found: list[str] = []
    # These relations are expressed with separated tokens in natural questions
    # (for example, "নির্মাণ কাজ কত সালে শুরু হয়"), so exact phrases are brittle.
    if "নির্মাণ" in value and "শুরু" in value:
        found.append("construction_start")
    if "নির্মাণ" in value and any(marker in value for marker in ("সম্পন্ন", "সমাপ্ত", "শেষ")):
        found.append("construction_completion")
    for relation, markers in RELATIONS.items():
        if relation == "construction" and any(
            specific in found for specific in ("construction_start", "construction_completion")
        ):
            continue
        if any(marker in value for marker in markers):
            found.append(relation)
    return tuple(dict.fromkeys(found))


def relation_type(text: str) -> str:
    relations = relation_types(text)
    return relations[0] if relations else "other"


def source_relation(question: str, context: str, title: str = "") -> str:
    """Prefer the source question's requested relation over incidental context."""
    question_relation = relation_type(question)
    if question_relation != "other":
        return question_relation
    return relation_type(f"{title} {context}")


def source_relations(
    question: str,
    context: str,
    title: str = "",
    source_repo: str = "",
) -> tuple[str, ...]:
    """Return every relation represented by a source document.

    Only a known QA corpus receives question-first treatment. In article, law,
    news, and textbook corpora the historical ``question`` field is actually a
    title/lead, so reducing the whole document to its first marker loses valid
    relations and can assign a completely incidental one.
    """
    question_relations = relation_types(question)
    if source_repo in QUESTION_ANSWER_SOURCES and question_relations:
        return question_relations
    return relation_types(f"{title} {question} {context}")


def read_corpus(paths: list[Path], max_docs: int | None = None, seed: int = 42) -> pd.DataFrame:
    rows, protected = [], []
    seen = 0
    rng = random.Random(seed)
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                # Small QA banks are exact-question assets and should survive
                # reservoir sampling of large document corpora.
                if max_docs and str(item.get("domain", "qa")) == "qa":
                    protected.append(item)
                    continue
                if not max_docs or len(rows) < max_docs:
                    rows.append(item)
                else:
                    replacement = rng.randint(0, seen)
                    if replacement < max_docs:
                        rows[replacement] = item
                seen += 1
    if max_docs:
        keep = max(0, max_docs - len(protected))
        rows = rows[:keep] + protected[:max_docs]
    df = pd.DataFrame(rows).fillna("")
    for col in ["question", "answer", "context", "title", "source_repo", "source_file", "domain", "license", "document_date"]:
        if col not in df:
            df[col] = ""
        df[col] = df[col].astype(str)
    df.loc[df.domain.str.len() == 0, "domain"] = "qa"
    relation_sets = [
        source_relations(question, context, title, source_repo)
        for question, context, title, source_repo in zip(
            df.question, df.context, df.title, df.source_repo
        )
    ]
    df["relations"] = ["|".join(relations) for relations in relation_sets]
    df["relation"] = [relations[0] if relations else "other" for relations in relation_sets]
    df["document"] = (
        df.title + " [Q] " + df.question + " [A] " + df.answer + " [CTX] " + df.context
    ).map(normalize_bn)
    return df.drop_duplicates(["question", "answer", "source_repo", "source_file"]).reset_index(drop=True)


def save_corpus(df: pd.DataFrame, path: Path) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in df.drop(columns=["document"]).to_dict("records"):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_corpus(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return pd.DataFrame(json.loads(line) for line in handle if line.strip()).fillna("")


def retrieval_queries(df: pd.DataFrame, mode: str) -> list[str]:
    if mode == "prompt":
        values = df.prompt_bn.astype(str)
    elif mode == "prompt_response":
        values = df.prompt_bn.astype(str) + " [ANS] " + df.response_bn.astype(str)
    else:
        raise ValueError(f"unsupported query mode: {mode}")
    return values.map(normalize_bn).tolist()


def build(args) -> None:
    df = read_corpus(args.corpus, args.max_docs)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=args.min_df, max_features=args.max_features, sublinear_tf=True)
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=args.min_df, max_features=args.max_features // 2, sublinear_tf=True)
    char_matrix = char.fit_transform(df.document)
    word_matrix = word.fit_transform(df.document)
    args.index_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(char, args.index_dir / "char_vectorizer.joblib")
    joblib.dump(word, args.index_dir / "word_vectorizer.joblib")
    # Uncompressed CSR is materially faster to build/load in Kaggle kernels;
    # the final dataset archive supplies outer compression when needed.
    sparse.save_npz(args.index_dir / "char_matrix.npz", char_matrix, compressed=False)
    sparse.save_npz(args.index_dir / "word_matrix.npz", word_matrix, compressed=False)
    save_corpus(df, args.index_dir / "corpus.jsonl.gz")
    relation_counts: dict[str, int] = {}
    for tags in df.relations:
        for relation in str(tags).split("|"):
            if relation:
                relation_counts[relation] = relation_counts.get(relation, 0) + 1
    metadata = {
        "documents": len(df),
        "domains": df.domain.value_counts().to_dict(),
        "primary_relations": df.relation.value_counts().to_dict(),
        "relations": dict(sorted(relation_counts.items(), key=lambda item: item[1], reverse=True)),
        "char_shape": list(char_matrix.shape),
        "word_shape": list(word_matrix.shape),
    }
    (args.index_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def retrieve(args) -> None:
    corpus = load_corpus(args.index_dir / "corpus.jsonl.gz")
    char = joblib.load(args.index_dir / "char_vectorizer.joblib")
    word = joblib.load(args.index_dir / "word_vectorizer.joblib")
    char_matrix = sparse.load_npz(args.index_dir / "char_matrix.npz")
    word_matrix = sparse.load_npz(args.index_dir / "word_matrix.npz")
    # Convert transposes once. linear_kernel(query, matrix) otherwise repeats
    # an expensive CSC->CSR conversion for every row.
    char_transpose = char_matrix.T.tocsr()
    word_transpose = word_matrix.T.tocsr()
    df = load_sample() if args.split == "sample" else load_test()
    queries = retrieval_queries(df, args.query_mode)
    q_char, q_word = char.transform(queries), word.transform(queries)
    exact = {}
    for index, question in enumerate(corpus.question.astype(str)):
        exact.setdefault(normalize_bn(question), []).append(index)
    records = []
    if "relations" not in corpus:
        corpus["relations"] = corpus.relation.where(corpus.relation != "other", "")
    for i, row in enumerate(df.itertuples()):
        char_scores = (q_char[i] @ char_transpose).toarray().ravel()
        word_scores = (q_word[i] @ word_transpose).toarray().ravel()
        scores = args.char_weight * char_scores + (1 - args.char_weight) * word_scores
        relation = relation_type(str(row.prompt_bn))
        relation_matches = corpus.relations.astype(str).map(
            lambda tags: relation in tags.split("|") if tags else False
        ).to_numpy()
        if relation != "other":
            scores += relation_matches * args.relation_bonus
        authority = np.array([
            SOURCE_AUTHORITY.get(str(source), DOMAIN_AUTHORITY.get(str(domain), DOMAIN_AUTHORITY["unknown"]))
            for source, domain in zip(corpus.source_repo, corpus.domain)
        ])
        scores *= authority
        for index in exact.get(normalize_bn(str(row.prompt_bn)), []):
            scores[index] = max(scores[index], 1.25)
        top = np.argpartition(scores, -min(args.top_k, len(scores)))[-args.top_k:]
        top = top[np.argsort(scores[top])[::-1]]
        evidence = []
        for rank, index in enumerate(top, 1):
            item = corpus.iloc[int(index)]
            source_has_relation = bool(str(item.relations))
            relation_match = (
                None
                if relation == "other" or not source_has_relation
                else bool(relation_matches[int(index)])
            )
            evidence.append({
                "rank": rank,
                "score": round(float(scores[index]), 6),
                "relation_match": relation_match,
                "relations": str(item.relations),
                "question": item.question,
                "answer": item.answer,
                "context": item.context,
                "source_repo": item.source_repo,
                "source_file": item.source_file,
                "domain": item.domain,
                "license": item.license,
                "document_date": item.document_date,
            })
        record = {
            "id": int(getattr(row, "id", i)),
            "prompt_bn": str(row.prompt_bn),
            "response_bn": str(row.response_bn),
            "relation": relation,
            "evidence": evidence,
        }
        if hasattr(row, "label"):
            record["label"] = int(row.label)
        records.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("wrote", args.output, "rows", len(records))


def main() -> None:
    parser = argparse.ArgumentParser(description="Portable hybrid sparse Bengali evidence retrieval.")
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--corpus", type=Path, action="append", required=True)
    build_parser.add_argument("--index-dir", type=Path, required=True)
    build_parser.add_argument("--max-docs", type=int)
    build_parser.add_argument("--max-features", type=int, default=180_000)
    build_parser.add_argument("--min-df", type=int, default=1)
    build_parser.set_defaults(func=build)
    retrieve_parser = sub.add_parser("retrieve")
    retrieve_parser.add_argument("--index-dir", type=Path, required=True)
    retrieve_parser.add_argument("--split", choices=["sample", "test"], required=True)
    retrieve_parser.add_argument("--output", type=Path, required=True)
    retrieve_parser.add_argument("--top-k", type=int, default=5)
    retrieve_parser.add_argument("--char-weight", type=float, default=0.6)
    retrieve_parser.add_argument("--relation-bonus", type=float, default=0.08)
    retrieve_parser.add_argument(
        "--query-mode",
        choices=["prompt", "prompt_response"],
        default="prompt_response",
    )
    retrieve_parser.set_defaults(func=retrieve)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
