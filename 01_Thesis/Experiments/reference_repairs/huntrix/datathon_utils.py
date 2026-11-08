from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics.pairwise import linear_kernel


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "bengali-hallucination"
OUTPUT_DIR = ROOT / "outputs"

NO_CONTEXT = {"", "[NULL]", "NULL", "null", "nan", "NaN", "None"}
BN_DIGITS = str.maketrans("০১২৩৪۵۶۷۸۹۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


# Override the legacy mixed-script table above with complete, unambiguous
# mappings for Bengali, Arabic-Indic, and Eastern Arabic-Indic digits.
BN_DIGITS = str.maketrans(
    {
        **{ord(char): str(index) for index, char in enumerate("০১২৩৪৫৬৭৮৯")},
        **{ord(char): str(index) for index, char in enumerate("٠١٢٣٤٥٦٧٨٩")},
        **{ord(char): str(index) for index, char in enumerate("۰۱۲۳۴۵۶۷۸۹")},
    }
)


def clean_context(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text in NO_CONTEXT else text


def clean_text(value) -> str:
    return "" if pd.isna(value) else str(value).strip()


def normalize_bn(text: str) -> str:
    text = unicodedata.normalize("NFKC", clean_text(text)).translate(BN_DIGITS).lower()
    text = re.sub(r"[\u200c\u200d]", "", text)
    text = re.sub(r"[।,;:!?()\[\]{}\"“”‘’`*_\-–—/\\]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def bn_tokens(text: str) -> list[str]:
    return re.findall(r"[\w\u0980-\u09ff]+", normalize_bn(text))


def response_in_context(context: str, response: str) -> bool:
    response_norm = normalize_bn(response)
    return bool(response_norm and response_norm in normalize_bn(context))


def parse_label(text: str) -> int:
    text = (text or "").strip().translate(BN_DIGITS)
    match = re.search(r"(?<!\d)([01])(?!\d)", text)
    return int(match.group(1)) if match else 0


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["prompt_bn", "response_bn"]:
        df[col] = df[col].map(clean_text)
    df["context"] = df["context"].map(clean_context)
    df["has_context"] = df["context"].str.len() > 0
    df["category"] = df["prompt_bn"].map(prompt_category)
    return df


def load_sample() -> pd.DataFrame:
    with (DATA_DIR / "dataset samples.json").open(encoding="utf-8") as handle:
        return prepare_frame(pd.DataFrame(json.load(handle)))


def load_test() -> pd.DataFrame:
    return prepare_frame(pd.read_csv(DATA_DIR / "test set.csv"))


def prompt_category(prompt: str) -> str:
    prompt = clean_text(prompt)
    if "ভাবার্থ" in prompt:
        return "meaning"
    if "শাব্দিক অর্থ" in prompt:
        return "literal"
    if "সমার্থক" in prompt:
        return "synonym"
    if "বিপরীত" in prompt or "বিপরীতার্থক" in prompt:
        return "antonym"
    if "ইংরেজি" in prompt or "English" in prompt:
        return "english"
    if re.search(r"\d|[০-৯]", prompt):
        return "numeric_or_date"
    if any(key in prompt for key in ["কত", "কবে", "তারিখ", "সাল"]):
        return "numeric_or_date"
    if any(key in prompt for key in ["কে", "কার", "নাম", "লেখক", "রচয়িতা", "প্রতিষ্ঠাতা"]):
        return "person_entity"
    if any(key in prompt for key in ["কোথায়", "কোথায়", "কোন জেল", "স্থান"]):
        return "place"
    if any(key in prompt for key in ["কোন", "কী", "কি"]):
        return "qa"
    return "other"


def apply_context_rule(df: pd.DataFrame, labels: Iterable[int]) -> np.ndarray:
    labels = np.asarray(list(labels), dtype=int).copy()
    has_context = df["has_context"].to_numpy()
    exact_support = np.array(
        [response_in_context(row.context, row.response_bn) for row in df.itertuples()],
        dtype=bool,
    )
    labels[has_context & exact_support] = 1
    labels[has_context & ~exact_support] = 0
    return labels


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_0": f1_score(y_true, y_pred, pos_label=0),
        "f1_1": f1_score(y_true, y_pred, pos_label=1),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def print_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metrics = metric_row(y_true, y_pred)
    counts = pd.Series(y_pred).value_counts().sort_index().to_dict()
    print(
        f"{name}: n={len(y_true)} counts={counts} "
        f"acc={metrics['accuracy']:.4f} f1_0={metrics['f1_0']:.4f} "
        f"f1_1={metrics['f1_1']:.4f} macro={metrics['macro_f1']:.4f}"
    )
    print(confusion_matrix(y_true, y_pred))
    return metrics


def evaluate_by_slice(df: pd.DataFrame, labels: np.ndarray, name: str) -> None:
    y = df["label"].to_numpy()
    print_metrics(name, y, labels)
    for has_context in [False, True]:
        mask = df["has_context"].to_numpy() == has_context
        if mask.any():
            print_metrics(f"{name} / has_context={has_context}", y[mask], labels[mask])
    for category in sorted(df["category"].unique()):
        mask = df["category"].to_numpy() == category
        if mask.sum() >= 5:
            print_metrics(f"{name} / category={category}", y[mask], labels[mask])


def load_trace_predictions(trace_path: Path, n_rows: int) -> np.ndarray:
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != n_rows:
        raise ValueError(f"{trace_path} has {len(rows)} rows, expected {n_rows}")
    return np.array([int(row["pred"]) for row in rows], dtype=int)


def build_context_retriever(sample_df: pd.DataFrame, test_df: pd.DataFrame | None = None):
    frames = [sample_df[sample_df["has_context"]][["context", "prompt_bn", "response_bn"]]]
    if test_df is not None:
        frames.append(test_df[test_df["has_context"]][["context", "prompt_bn", "response_bn"]])
    corpus_df = pd.concat(frames, ignore_index=True).drop_duplicates("context")
    docs = (
        corpus_df["prompt_bn"].map(clean_text)
        + " [ANS] "
        + corpus_df["response_bn"].map(clean_text)
        + " [CTX] "
        + corpus_df["context"].map(clean_text)
    ).tolist()
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=120_000,
        min_df=1,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(docs)
    return corpus_df.reset_index(drop=True), vectorizer, matrix


def retrieve_contexts(
    df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    matrix,
    top_k: int = 3,
) -> list[str]:
    queries = (df["prompt_bn"].map(clean_text) + " [ANS] " + df["response_bn"].map(clean_text)).tolist()
    q_matrix = vectorizer.transform(queries)
    sims = linear_kernel(q_matrix, matrix)
    evidence = []
    for row_scores in sims:
        best = np.argsort(row_scores)[::-1][:top_k]
        parts = []
        for rank, idx in enumerate(best, start=1):
            score = row_scores[idx]
            if score <= 0:
                continue
            context = corpus_df.iloc[idx]["context"]
            parts.append(f"[{rank}] {context[:900]}")
        evidence.append("\n".join(parts))
    return evidence
