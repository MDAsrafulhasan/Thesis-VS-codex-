from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from datathon_utils import (
    OUTPUT_DIR,
    apply_context_rule,
    bn_tokens,
    load_sample,
    load_test,
    metric_row,
    normalize_bn,
    print_metrics,
    response_in_context,
)
from gemma_ensemble import load_partial_trace, vote_predictions


def text_all(df: pd.DataFrame) -> np.ndarray:
    return (
        df["prompt_bn"].astype(str)
        + " [ANS] "
        + df["response_bn"].astype(str)
        + " [CTX] "
        + df["context"].astype(str)
        + " [CAT] "
        + df["category"].astype(str)
    ).to_numpy()


def text_pr(df: pd.DataFrame) -> np.ndarray:
    return (df["prompt_bn"].astype(str) + " [ANS] " + df["response_bn"].astype(str)).to_numpy()


def extract_numbers(text: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?", normalize_bn(text))


def containment(short: str, long: str) -> float:
    a = set(bn_tokens(short))
    b = set(bn_tokens(long))
    return len(a & b) / len(a) if a else 0.0


def jaccard(a_text: str, b_text: str) -> float:
    a = set(bn_tokens(a_text))
    b = set(bn_tokens(b_text))
    return len(a & b) / len(a | b) if a and b else 0.0


def numeric_features(df: pd.DataFrame) -> np.ndarray:
    rows = []
    categories = ["literal", "meaning", "numeric_or_date", "other", "person_entity", "place", "qa", "antonym", "english", "synonym"]
    for row in df.itertuples():
        context = str(row.context)
        prompt = str(row.prompt_bn)
        response = str(row.response_bn)
        has_context = bool(getattr(row, "has_context", False))
        r_nums = extract_numbers(response)
        c_nums = extract_numbers(context)
        p_nums = extract_numbers(prompt)
        exact = response_in_context(context, response) if has_context else False
        nums_in_context = sum(n in c_nums for n in r_nums) / len(r_nums) if r_nums and has_context else 0.0
        rows.append(
            [
                int(has_context),
                len(prompt),
                len(response),
                len(context),
                len(bn_tokens(prompt)),
                len(bn_tokens(response)),
                len(bn_tokens(context)),
                int(exact),
                containment(response, context) if has_context else 0.0,
                containment(response, prompt),
                jaccard(response, context) if has_context else 0.0,
                jaccard(response, prompt),
                len(r_nums),
                len(c_nums),
                len(p_nums),
                nums_in_context,
                int(bool(r_nums)),
                int("**" in response),
                int("(" in response or ")" in response),
                int(any(ch in response for ch in [",", ";", "।"])),
            ]
            + [int(row.category == cat) for cat in categories]
        )
    return np.asarray(rows, dtype=float)


def make_pipeline(c: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "char_all",
                            Pipeline(
                                [
                                    ("sel", FunctionTransformer(text_all, validate=False)),
                                    (
                                        "tfidf",
                                        TfidfVectorizer(
                                            analyzer="char_wb",
                                            ngram_range=(2, 5),
                                            max_features=120_000,
                                            min_df=1,
                                            sublinear_tf=True,
                                        ),
                                    ),
                                ]
                            ),
                        ),
                        (
                            "char_pr",
                            Pipeline(
                                [
                                    ("sel", FunctionTransformer(text_pr, validate=False)),
                                    (
                                        "tfidf",
                                        TfidfVectorizer(
                                            analyzer="char",
                                            ngram_range=(3, 6),
                                            max_features=100_000,
                                            min_df=1,
                                            sublinear_tf=True,
                                        ),
                                    ),
                                ]
                            ),
                        ),
                        (
                            "word",
                            Pipeline(
                                [
                                    ("sel", FunctionTransformer(text_all, validate=False)),
                                    (
                                        "tfidf",
                                        TfidfVectorizer(
                                            analyzer="word",
                                            token_pattern=r"(?u)\b\w+\b",
                                            ngram_range=(1, 2),
                                            max_features=80_000,
                                            min_df=1,
                                            sublinear_tf=True,
                                        ),
                                    ),
                                ]
                            ),
                        ),
                        (
                            "num",
                            Pipeline(
                                [
                                    ("num", FunctionTransformer(numeric_features, validate=False)),
                                    ("scale", StandardScaler()),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    C=c,
                    class_weight="balanced",
                    random_state=42,
                    solver="liblinear",
                ),
            ),
        ]
    )


def load_synthetic(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    df["has_context"] = df["context"].astype(str).str.len() > 0
    from datathon_utils import prompt_category

    df["category"] = df["prompt_bn"].map(prompt_category)
    return df


def add_teacher_columns(sample: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Lightweight feature injection by appending teacher votes to text. This lets a
    # portable classifier learn from local teacher behavior without special model code.
    teacher_specs = [
        ("q27_noctx", "outputs/mlx_qwen36_27b_noctx_fact_sample_no_context.jsonl", "outputs/mlx_qwen36_27b_noctx_fact_test_no_context.jsonl"),
        ("q27_exam", "outputs/mlx_qwen36_27b_exam_strict_sample_no_context.jsonl", "outputs/mlx_qwen36_27b_exam_strict_test_no_context.jsonl"),
        ("q27_zero", "outputs/mlx_qwen36_27b_zero_bias_sample_no_context.jsonl", "outputs/mlx_qwen36_27b_zero_bias_test_no_context.jsonl"),
        ("gemma_base", "outputs/gemma12_qat_sample_trace.jsonl", "outputs/gemma12_qat_test_trace.jsonl"),
    ]
    sample = sample.copy()
    test = test.copy()
    for name, s_path, t_path in teacher_specs:
        s = load_partial_trace(Path(s_path), len(sample))
        t = load_partial_trace(Path(t_path), len(test))
        sample[name] = np.where(s == -1, 0.5, s)
        test[name] = np.where(t == -1, 0.5, t)
    for df in [sample, test]:
        teacher_text = (
            " [TQ27N] "
            + df["q27_noctx"].astype(str)
            + " [TQ27E] "
            + df["q27_exam"].astype(str)
            + " [TQ27Z] "
            + df["q27_zero"].astype(str)
            + " [TGEM] "
            + df["gemma_base"].astype(str)
        )
        df["response_bn"] = df["response_bn"].astype(str) + teacher_text
    return sample, test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", default=str(OUTPUT_DIR / "synthetic_train.jsonl"))
    parser.add_argument("--output", default=str(OUTPUT_DIR / "submission_verifier.csv"))
    parser.add_argument("--with-teachers", action="store_true")
    args = parser.parse_args()

    sample = load_sample()
    test = load_test()
    if args.with_teachers:
        sample, test = add_teacher_columns(sample, test)
    synthetic = load_synthetic(Path(args.synthetic))
    train_df = pd.concat([synthetic, sample], ignore_index=True)
    cv_synthetic = synthetic[synthetic.get("origin", "unknown").eq("test")].copy()

    y_real = sample["label"].to_numpy()
    print(
        "train rows",
        len(train_df),
        "synthetic",
        len(synthetic),
        "cv_synthetic_no_leak",
        len(cv_synthetic),
        "real",
        len(sample),
    )
    for c in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
        # Cross-validate only on real rows, training folds include synthetic plus real train folds.
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        proba = np.zeros(len(sample), dtype=float)
        for tr, va in cv.split(sample, y_real):
            fold_train = pd.concat([cv_synthetic, sample.iloc[tr]], ignore_index=True)
            fold_y = fold_train["label"].astype(int).to_numpy()
            model = make_pipeline(c)
            model.fit(fold_train, fold_y)
            proba[va] = model.predict_proba(sample.iloc[va])[:, 1]
        best = None
        for threshold in np.linspace(0.2, 0.8, 61):
            pred = (proba >= threshold).astype(int)
            pred = apply_context_rule(sample, pred)
            metrics = metric_row(y_real, pred)
            key = (metrics["macro_f1"], metrics["f1_0"])
            if best is None or key > best[0]:
                best = (key, threshold, metrics)
        print("C", c, "best_threshold", best[1], best[2])

    # Fit final with conservative default from best quick search observed manually.
    c = 0.5
    threshold = 0.5
    model = make_pipeline(c)
    model.fit(train_df, train_df["label"].astype(int).to_numpy())
    test_proba = model.predict_proba(test)[:, 1]
    pred = (test_proba >= threshold).astype(int)
    pred = apply_context_rule(test, pred)
    output = Path(args.output)
    pd.DataFrame({"id": test["id"].to_numpy(), "label": pred}).to_csv(output, index=False)
    print("wrote", output, pd.Series(pred).value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
