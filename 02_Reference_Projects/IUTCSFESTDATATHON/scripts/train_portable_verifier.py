from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
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
    prompt_category,
    response_in_context,
)
from train_verifier import extract_numbers


DEFAULT_TRAIN_FILES = [
    OUTPUT_DIR / "lexical_controlled_pairs_80k.jsonl",
    OUTPUT_DIR / "synthetic_plus_curated.jsonl",
]


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().fillna("")
    for col in ["context", "prompt_bn", "response_bn"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str)
    if "has_context" not in df.columns:
        df["has_context"] = df["context"].str.len() > 0
    if "category" not in df.columns:
        df["category"] = df["prompt_bn"].map(prompt_category)
    if "source" not in df.columns:
        df["source"] = "unknown"
    return df


def load_jsonl(path: Path) -> pd.DataFrame:
    return prepare_frame(pd.read_json(path, lines=True))


def balanced_sample(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if not max_rows or len(df) <= max_rows:
        return df
    per_label = max_rows // 2
    parts = []
    for label, group in df.groupby("label"):
        take = min(len(group), per_label)
        parts.append(group.sample(take, random_state=seed + int(label)))
    out = pd.concat(parts)
    if len(out) < max_rows:
        rest = df.drop(out.index, errors="ignore")
        if len(rest):
            out = pd.concat([out, rest.sample(min(len(rest), max_rows - len(out)), random_state=seed)])
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def load_external_training(paths: list[Path], max_rows_per_file: int | None, seed: int) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = load_jsonl(path)
        if "label" not in df.columns:
            raise ValueError(f"{path} has no label column")
        df = balanced_sample(df, max_rows_per_file, seed)
        df["train_file"] = path.as_posix()
        frames.append(df)
        print("loaded", path, len(df), df["label"].value_counts().sort_index().to_dict())
    return prepare_frame(pd.concat(frames, ignore_index=True))


def text_all(df: pd.DataFrame) -> np.ndarray:
    return (
        df["prompt_bn"].astype(str)
        + " [ANS] "
        + df["response_bn"].astype(str)
        + " [CTX] "
        + df["context"].astype(str)
        + " [CAT] "
        + df["category"].astype(str)
        + " [SRC] "
        + df["source"].astype(str)
    ).to_numpy()


def text_prompt_response(df: pd.DataFrame) -> np.ndarray:
    return (df["prompt_bn"].astype(str) + " [ANS] " + df["response_bn"].astype(str)).to_numpy()


def containment(short: str, long: str) -> float:
    a = set(bn_tokens(short))
    b = set(bn_tokens(long))
    return len(a & b) / len(a) if a else 0.0


def jaccard(a_text: str, b_text: str) -> float:
    a = set(bn_tokens(a_text))
    b = set(bn_tokens(b_text))
    return len(a & b) / len(a | b) if a and b else 0.0


def numeric_features(df: pd.DataFrame) -> np.ndarray:
    categories = ["literal", "meaning", "numeric_or_date", "other", "person_entity", "place", "qa", "antonym", "english", "synonym"]
    rows = []
    for row in df.itertuples():
        context = str(row.context)
        prompt = str(row.prompt_bn)
        response = str(row.response_bn)
        has_context = bool(getattr(row, "has_context", False))
        r_nums = extract_numbers(response)
        c_nums = extract_numbers(context)
        exact = response_in_context(context, response) if has_context else False
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
                sum(n in c_nums for n in r_nums) / len(r_nums) if r_nums and c_nums else 0.0,
                int("ভাবার্থ" in prompt),
                int("শাব্দিক" in prompt),
                int("কত" in prompt or "কবে" in prompt or "সাল" in prompt),
            ]
            + [int(row.category == cat) for cat in categories]
        )
    return np.asarray(rows, dtype=float)


def make_pipeline(c: float, max_features: int) -> Pipeline:
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
                                            ngram_range=(3, 5),
                                            max_features=max_features,
                                            min_df=2,
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
                                    ("sel", FunctionTransformer(text_prompt_response, validate=False)),
                                    (
                                        "tfidf",
                                        TfidfVectorizer(
                                            analyzer="char",
                                            ngram_range=(3, 6),
                                            max_features=max_features // 2,
                                            min_df=2,
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
                    C=c,
                    class_weight="balanced",
                    max_iter=2500,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def tune_threshold(y_true: np.ndarray, proba: np.ndarray, sample_df: pd.DataFrame) -> tuple[float, dict]:
    best_threshold = 0.5
    best_metrics = None
    best_key = None
    for threshold in np.linspace(0.2, 0.8, 61):
        pred = (proba >= threshold).astype(int)
        pred = apply_context_rule(sample_df, pred)
        metrics = metric_row(y_true, pred)
        key = (metrics["macro_f1"], metrics["f1_0"])
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics or {}


def cross_validate(external: pd.DataFrame, sample: pd.DataFrame, c: float, max_features: int) -> tuple[np.ndarray, float, dict]:
    y = sample["label"].astype(int).to_numpy()
    proba = np.zeros(len(sample), dtype=float)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr, va) in enumerate(cv.split(sample, y), start=1):
        train_df = pd.concat([external, sample.iloc[tr]], ignore_index=True)
        model = make_pipeline(c, max_features)
        model.fit(train_df, train_df["label"].astype(int).to_numpy())
        proba[va] = model.predict_proba(sample.iloc[va])[:, 1]
        print("fold", fold, "done")
    threshold, metrics = tune_threshold(y, proba, sample)
    return proba, threshold, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", action="append", type=Path, default=[])
    parser.add_argument("--max-rows-per-file", type=int, default=80_000)
    parser.add_argument("--max-features", type=int, default=90_000)
    parser.add_argument("--c", type=float, default=0.5)
    parser.add_argument("--model-output", default=str(OUTPUT_DIR / "portable_verifier.joblib"))
    parser.add_argument("--submission-output", default=str(OUTPUT_DIR / "submission_portable_verifier.csv"))
    parser.add_argument("--metadata-output", default=str(OUTPUT_DIR / "portable_verifier_metadata.json"))
    parser.add_argument("--oof-output", default=str(OUTPUT_DIR / "portable_verifier_oof.csv"))
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--fixed-threshold", type=float)
    parser.add_argument("--skip-final-fit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_files = args.train_file or DEFAULT_TRAIN_FILES
    external = load_external_training(train_files, args.max_rows_per_file, args.seed)
    sample = prepare_frame(load_sample())
    test = prepare_frame(load_test())
    print("external", len(external), external["label"].value_counts().sort_index().to_dict())

    if args.skip_cv:
        threshold = float(args.fixed_threshold if args.fixed_threshold is not None else 0.5)
        cv_metrics = {}
        print("skipped cv; threshold", threshold)
    else:
        proba, threshold, cv_metrics = cross_validate(external, sample, args.c, args.max_features)
        if args.fixed_threshold is not None:
            threshold = float(args.fixed_threshold)
            cv_metrics = metric_row(
                sample["label"].astype(int).to_numpy(),
                apply_context_rule(sample, (proba >= threshold).astype(int)),
            )
        print("cv threshold", threshold, cv_metrics)
        oof_pred = apply_context_rule(sample, (proba >= threshold).astype(int))
        pd.DataFrame(
            {
                "id": sample["id"].to_numpy() if "id" in sample.columns else np.arange(len(sample)),
                "label": sample["label"].astype(int).to_numpy(),
                "proba_1": proba,
                "pred": oof_pred,
            }
        ).to_csv(args.oof_output, index=False)
        print("wrote", args.oof_output)

    if args.skip_final_fit:
        return

    train_df = pd.concat([external, sample], ignore_index=True)
    model = make_pipeline(args.c, args.max_features)
    model.fit(train_df, train_df["label"].astype(int).to_numpy())
    test_proba = model.predict_proba(test)[:, 1]
    pred = (test_proba >= threshold).astype(int)
    pred = apply_context_rule(test, pred)

    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold, "train_files": [p.as_posix() for p in train_files]}, model_path, compress=3)
    pd.DataFrame({"id": test["id"].to_numpy(), "label": pred}).to_csv(args.submission_output, index=False)
    metadata = {
        "train_files": [p.as_posix() for p in train_files],
        "external_rows": int(len(external)),
        "sample_rows": int(len(sample)),
        "c": args.c,
        "max_features": args.max_features,
        "threshold": threshold,
        "cv_metrics": cv_metrics,
        "submission_counts": pd.Series(pred).value_counts().sort_index().to_dict(),
    }
    Path(args.metadata_output).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", model_path, args.submission_output, args.metadata_output)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
