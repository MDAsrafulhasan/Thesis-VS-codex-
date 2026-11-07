#!/usr/bin/env python3
"""Normalize the thesis source CSVs and create deterministic benchmark subsets."""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split


BENHALLU_COLUMNS = [
    "id",
    "task",
    "track",
    "hallucination_type",
    "question",
    "context",
    "candidate_answer",
    "ground_truth_label",
]
DIALECT_COLUMNS = [
    "id",
    "dialect",
    "question_type",
    "question",
    "response",
    "human_or_primary_score",
]
TASKS = ("bn_qa", "codemixed_qa", "summarization", "math")
TRACK_A = "Track_A"
TRACK_B = "Track_B"
HALLUCINATION_TYPES = (
    "Factualness",
    "Comprehension",
    "Specificity",
    "Inference",
    "Fabricated_Content",
    "Non_Factual_Addition",
    "Direct_Contradiction",
    "Arithmetic_Slip",
    "Formula_Misapplication",
    "Invalid_Deduction",
    "Hallucinated_Intermediate_Fact",
    "Semantic_Drift",
)

TYPE_ALIASES = {
    "factualness": "Factualness",
    "comprehension": "Comprehension",
    "specificity": "Specificity",
    "inference": "Inference",
    "fabricated_content": "Fabricated_Content",
    "fabricated content": "Fabricated_Content",
    "intrinsic": "Fabricated_Content",
    "non_factual_addition": "Non_Factual_Addition",
    "non-factual addition": "Non_Factual_Addition",
    "non-factual": "Non_Factual_Addition",
    "non factual": "Non_Factual_Addition",
    "direct_contradiction": "Direct_Contradiction",
    "direct contradiction": "Direct_Contradiction",
    "factual contradiction": "Direct_Contradiction",
    "arithmetic_slip": "Arithmetic_Slip",
    "arithmetic slip": "Arithmetic_Slip",
    "formula_misapplication": "Formula_Misapplication",
    "formula misapplication": "Formula_Misapplication",
    "invalid_deduction": "Invalid_Deduction",
    "invalid deduction": "Invalid_Deduction",
    "hallucinated_intermediate_fact": "Hallucinated_Intermediate_Fact",
    "hallucinated intermediate fact": "Hallucinated_Intermediate_Fact",
    "semantic_drift": "Semantic_Drift",
    "semantic drift": "Semantic_Drift",
}


def _clean(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _canonical_type(value: object) -> str:
    normalized = re.sub(r"\s+", " ", _clean(value).lower()).strip()
    if normalized in TYPE_ALIASES:
        return TYPE_ALIASES[normalized]
    underscored = normalized.replace(" ", "_")
    if underscored in TYPE_ALIASES:
        return TYPE_ALIASES[underscored]
    raise ValueError(f"Unrecognized Track_B hallucination type: {value!r}")


def _record(
    raw_id: object,
    task: str,
    track: str,
    hallucination_type: str | None,
    question: object,
    context: object,
    candidate_answer: object,
) -> dict[str, object]:
    raw_id_text = _clean(raw_id)
    if not raw_id_text:
        raise ValueError(f"Empty id in {track}/{task} source row")
    return {
        # Raw source ids recur between QA and code-mixed QA. The qualified id is
        # stable and unique, which is required for safe JSONL resumption.
        "id": f"{track}:{task}:{raw_id_text}",
        "task": task,
        "track": track,
        "hallucination_type": hallucination_type if hallucination_type else "None",
        "question": _clean(question),
        "context": _clean(context),
        "candidate_answer": _clean(candidate_answer),
        "ground_truth_label": 0 if track == TRACK_A else 1,
    }


def _qualify_source_rows(rows: list[dict[str, object]], path: Path) -> list[dict[str, object]]:
    """Add a stable file and row ordinal so even repeated raw ids stay unique."""
    source_name = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem)
    for row_index, row in enumerate(rows):
        row["id"] = f"{row['id']}:{source_name}:row{row_index:05d}"
    return rows


def _read_csv(path: Path) -> pd.DataFrame:
    # utf-8-sig also handles ordinary UTF-8 files and strips a possible BOM.
    return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)


def _normalize_raw_file(path: Path, track: str) -> list[dict[str, object]]:
    name = path.name.lower()
    df = _read_csv(path)
    rows: list[dict[str, object]] = []

    if track == TRACK_A:
        if "summarization" in name:
            for row in df.to_dict("records"):
                rows.append(_record(row.get("id"), "summarization", track, None,
                                    "Summarize the following text.", row.get("question"), row.get("summary")))
        elif "codemix" in name or "code_mix" in name:
            for row in df.to_dict("records"):
                rows.append(_record(row.get("id"), "codemixed_qa", track, None,
                                    row.get("codemix_question", row.get("question")),
                                    row.get("codemix_context", row.get("context")),
                                    row.get("codemix_answer", row.get("correct_answer"))))
        elif "somadhan" in name or "reasoning" in name or "math" in name:
            for row in df.to_dict("records"):
                rows.append(_record(row.get("question_id", row.get("id")), "math", track, None,
                                    row.get("question"), "", row.get("answer")))
        elif "qa" in name:
            for row in df.to_dict("records"):
                rows.append(_record(row.get("id"), "bn_qa", track, None,
                                    row.get("question"), row.get("context"), row.get("correct_answer")))
        else:
            raise ValueError(f"Cannot infer Track_A task from filename: {path.name}")
        return _qualify_source_rows(rows, path)

    # Track_B source filenames use their task-specific answer fields.
    if "summarization" in name:
        for row in df.to_dict("records"):
            rows.append(_record(row.get("id"), "summarization", track,
                                _canonical_type(row.get("pattern")),
                                "Summarize the following text.", row.get("document"),
                                row.get("hallucinated_summary")))
    elif "reasoning" in name or "math" in name:
        for row in df.to_dict("records"):
            rows.append(_record(row.get("id"), "math", track,
                                _canonical_type(row.get("hallucination_type", row.get("error_type"))),
                                row.get("question"), "", row.get("hallucinated_answer")))
    elif "codemix" in name or "code_mix" in name:
        for row in df.to_dict("records"):
            rows.append(_record(row.get("id"), "codemixed_qa", track,
                                _canonical_type(row.get("pattern")),
                                row.get("codemix_question"), row.get("codemix_context"),
                                row.get("hallucinated_answer")))
    elif "qa" in name:
        for row in df.to_dict("records"):
            rows.append(_record(row.get("id"), "bn_qa", track,
                                _canonical_type(row.get("pattern")),
                                row.get("question"), row.get("context"),
                                row.get("hallucinated_answer")))
    else:
        raise ValueError(f"Cannot infer Track_B task from filename: {path.name}")
    return _qualify_source_rows(rows, path)


def normalize_benhallu_input(input_path: Path) -> pd.DataFrame:
    if input_path.is_file():
        frame = _read_csv(input_path)
        if set(BENHALLU_COLUMNS).issubset(frame.columns):
            frame = frame[BENHALLU_COLUMNS].copy()
        else:
            raise ValueError(
                "A single CSV must already use the BenHalluEval schema. "
                "Pass the parent directory containing Track-A and Track-B source files to normalize raw files."
            )
    elif input_path.is_dir():
        candidates = sorted(input_path.rglob("*.csv"))
        canonical = []
        for path in candidates:
            try:
                header = list(pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns)
            except Exception:
                continue
            if set(BENHALLU_COLUMNS).issubset(header):
                canonical.append(path)
        if canonical:
            if len(canonical) != 1:
                raise ValueError(f"Expected one canonical BenHallu CSV, found {len(canonical)}: {canonical}")
            frame = _read_csv(canonical[0])[BENHALLU_COLUMNS].copy()
        else:
            file_rows: list[dict[str, object]] = []
            for path in candidates:
                parts = {part.lower() for part in path.parts}
                if any("track-a" in part or "track_a" in part for part in parts):
                    file_rows.extend(_normalize_raw_file(path, TRACK_A))
                elif any("track-b" in part or "track_b" in part for part in parts):
                    file_rows.extend(_normalize_raw_file(path, TRACK_B))
            if not file_rows:
                raise ValueError(f"No recognized Track-A/Track-B or canonical CSVs found under {input_path}")
            frame = pd.DataFrame(file_rows, columns=BENHALLU_COLUMNS)
    else:
        raise FileNotFoundError(input_path)

    frame = frame[BENHALLU_COLUMNS].copy()
    frame["id"] = frame["id"].astype(str)
    frame["task"] = frame["task"].astype(str)
    frame["track"] = frame["track"].astype(str)
    frame["hallucination_type"] = frame["hallucination_type"].fillna("None").astype(str)
    frame["ground_truth_label"] = pd.to_numeric(frame["ground_truth_label"], errors="raise").astype(int)
    for col in ("question", "context", "candidate_answer"):
        frame[col] = frame[col].fillna("").astype(str)

    if frame.empty:
        raise ValueError("BenHalluEval input is empty")
    if frame["id"].duplicated().any():
        examples = frame.loc[frame["id"].duplicated(keep=False), "id"].head(5).tolist()
        raise ValueError(f"BenHalluEval ids must be unique; duplicate examples: {examples}")
    if not set(frame["task"]).issubset(TASKS):
        raise ValueError(f"Unknown task values: {sorted(set(frame['task']) - set(TASKS))}")
    if not set(frame["track"]).issubset({TRACK_A, TRACK_B}):
        raise ValueError(f"Unknown track values: {sorted(set(frame['track']) - {TRACK_A, TRACK_B})}")
    if not set(frame["hallucination_type"]).issubset({"None", *HALLUCINATION_TYPES}):
        raise ValueError("Input contains hallucination_type values outside the required 12 labels")
    expected_label = frame["track"].map({TRACK_A: 0, TRACK_B: 1})
    if not frame["ground_truth_label"].equals(expected_label.astype(int)):
        raise ValueError("ground_truth_label must be 0 for Track_A and 1 for Track_B")
    if not (frame.loc[frame.track == TRACK_A, "hallucination_type"] == "None").all():
        raise ValueError("Track_A hallucination_type must be 'None'")
    if not frame.loc[frame.track == TRACK_B, "hallucination_type"].isin(HALLUCINATION_TYPES).all():
        raise ValueError("Every Track_B row must use one of the 12 hallucination types")
    return frame


def validate_full_benhallu(frame: pd.DataFrame) -> None:
    if len(frame) != 16_000:
        raise ValueError(f"Expected exactly 16,000 BenHalluEval rows, received {len(frame):,}")
    counts = frame["track"].value_counts().to_dict()
    if counts.get(TRACK_A, 0) != 4_000 or counts.get(TRACK_B, 0) != 12_000:
        raise ValueError(f"Expected 4,000 Track_A and 12,000 Track_B rows; observed {counts}")
    task_counts = frame.groupby(["track", "task"]).size().to_dict()
    expected = {
        (TRACK_A, "bn_qa"): 1_000,
        (TRACK_A, "codemixed_qa"): 1_000,
        (TRACK_A, "summarization"): 1_000,
        (TRACK_A, "math"): 1_000,
        (TRACK_B, "bn_qa"): 4_000,
        (TRACK_B, "codemixed_qa"): 4_000,
        (TRACK_B, "summarization"): 3_000,
        (TRACK_B, "math"): 1_000,
    }
    if task_counts != expected:
        raise ValueError(f"Unexpected per-track task counts. Expected {expected}; observed {task_counts}")
    observed_types = set(frame.loc[frame.track == TRACK_B, "hallucination_type"])
    if observed_types != set(HALLUCINATION_TYPES):
        raise ValueError(f"Track_B must include all 12 hallucination types; missing {set(HALLUCINATION_TYPES) - observed_types}")


def _stratified_sample(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if count > len(frame):
        raise ValueError(f"Cannot sample {count} rows from only {len(frame)}")
    strata = frame["_stratum"]
    sizes = strata.value_counts()
    if (sizes < 2).any():
        rare = sizes[sizes < 2].to_dict()
        raise ValueError(f"Every stratum needs at least 2 rows for stratified splitting; rare strata: {rare}")
    if count < strata.nunique():
        raise ValueError(f"Requested sample size {count} is smaller than {strata.nunique()} strata")
    sample, _ = train_test_split(
        frame,
        train_size=count,
        random_state=seed,
        stratify=strata,
    )
    return sample.drop(columns=["_stratum"]).sort_values("id", kind="stable").reset_index(drop=True)


def create_benhallu_splits(frame: pd.DataFrame, output_dir: Path, seed: int = 42) -> None:
    validate_full_benhallu(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = frame.copy()
    frame["_stratum"] = frame.apply(
        lambda row: f"{row['task']}_{row['track']}_{row['hallucination_type']}", axis=1
    )

    # Split Track_A and Track_B separately to guarantee the required 30/90 and
    # 400/1,200 class proportions while preserving the composite strata.
    plans = (
        ("benhallu_micro_debug_120.csv", 30, 90),
        ("benhallu_stratified_10pct.csv", 400, 1_200),
    )
    for filename, count_a, count_b in plans:
        split_a = _stratified_sample(frame[frame.track == TRACK_A], count_a, seed)
        split_b = _stratified_sample(frame[frame.track == TRACK_B], count_b, seed)
        combined = pd.concat([split_a, split_b], ignore_index=True)
        combined = combined[BENHALLU_COLUMNS].sort_values("id", kind="stable").reset_index(drop=True)
        if len(combined) != count_a + count_b:
            raise AssertionError(f"Internal error: {filename} has {len(combined)} rows")
        combined.to_csv(output_dir / filename, index=False, encoding="utf-8")

    full = frame[BENHALLU_COLUMNS].sort_values("id", kind="stable").reset_index(drop=True)
    full.to_csv(output_dir / "benhallu_full_16k.csv", index=False, encoding="utf-8")


def normalize_dialect_input(input_path: Path) -> pd.DataFrame:
    if input_path.is_file():
        paths = [input_path]
    elif input_path.is_dir():
        paths = sorted(input_path.rglob("*.csv"))
    else:
        raise FileNotFoundError(input_path)
    matches: list[pd.DataFrame] = []
    for path in paths:
        try:
            header = list(pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns)
        except Exception:
            continue
        if set(DIALECT_COLUMNS).issubset(header):
            matches.append(_read_csv(path)[DIALECT_COLUMNS].copy())
    if not matches:
        raise ValueError(f"No CSV using the Dialectal_Bias schema found at {input_path}")
    frame = pd.concat(matches, ignore_index=True)
    frame["id"] = frame["id"].astype(str)
    frame["dialect"] = frame["dialect"].astype(str)
    frame["question_type"] = frame["question_type"].astype(str)
    frame["question"] = frame["question"].fillna("").astype(str)
    frame["response"] = frame["response"].fillna("").astype(str)
    frame["human_or_primary_score"] = pd.to_numeric(frame["human_or_primary_score"], errors="raise")
    if frame["id"].duplicated().any():
        raise ValueError("Dialectal_Bias ids must be unique")
    if frame["id"].str.strip().eq("").any():
        raise ValueError("Dialectal_Bias ids must not be empty")
    if frame["dialect"].str.strip().eq("").any():
        raise ValueError("Dialectal_Bias dialect values must not be empty")
    if not frame["human_or_primary_score"].between(0, 10).all():
        raise ValueError("human_or_primary_score must be between 0 and 10")
    if not set(frame["question_type"]).issubset({"definitional", "factual"}):
        raise ValueError("question_type must be definitional or factual")
    return frame


ANCHOR_DIALECTS = ("Standard_Bengali", "Tangail", "Chittagong")


def create_dialect_split(frame: pd.DataFrame, output_dir: Path, seed: int = 42) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    subset = frame[frame["dialect"].isin(ANCHOR_DIALECTS)].copy()
    missing = set(ANCHOR_DIALECTS) - set(subset["dialect"])
    if missing:
        raise ValueError(f"Dialect input is missing anchor variants: {sorted(missing)}")
    sampled: list[pd.DataFrame] = []
    for offset, dialect in enumerate(ANCHOR_DIALECTS):
        group = subset[subset.dialect == dialect]
        if len(group) < 100:
            raise ValueError(f"Expected at least 100 rows for {dialect}; found {len(group)}")
        type_counts = group["question_type"].value_counts()
        if len(type_counts) > 1 and (type_counts < 2).any():
            raise ValueError(f"Every question_type needs at least 2 rows within {dialect} for stratification")
        if len(group) == 100:
            selected = group.copy()
        else:
            selected, _ = train_test_split(
                group,
                train_size=100,
                random_state=seed + offset,
                stratify=group["question_type"],
            )
        sampled.append(selected)
    result = pd.concat(sampled, ignore_index=True)
    result = result[DIALECT_COLUMNS].sort_values(["dialect", "id"], kind="stable").reset_index(drop=True)
    if len(result) != 300 or result["dialect"].value_counts().to_dict() != {d: 100 for d in ANCHOR_DIALECTS}:
        raise AssertionError("Internal error: dialect CCC subset is not 100 rows per anchor")
    result.to_csv(output_dir / "dialect_ccc_subset_300.csv", index=False, encoding="utf-8")


def _mock_primary_rows() -> list[dict[str, object]]:
    rng = random.Random(42)
    rows: list[dict[str, object]] = []
    bengali_topics = ["বাংলাদেশের নদী", "ঢাকার ইতিহাস", "গণিতের একটি সমস্যা", "বাংলা সাহিত্যের পরিচয়"]
    for task in TASKS:
        for i in range(1_000):
            topic = bengali_topics[i % len(bengali_topics)]
            if task == "bn_qa":
                question, context, answer = (
                    f"{topic} সম্পর্কে প্রশ্ন {i + 1}: প্রধান তথ্যটি কী?",
                    f"নথিতে {topic} সম্পর্কে নির্ভরযোগ্য তথ্য {i + 1} দেওয়া হয়েছে।",
                    f"নথির তথ্য অনুযায়ী উত্তর {i + 1}।",
                )
            elif task == "codemixed_qa":
                question, context, answer = (
                    f"{topic} niye question {i + 1}: main fact ta ki?",
                    f"Document-e {topic} niye verified information {i + 1} deya ache.",
                    f"Document onujayi answer {i + 1}.",
                )
            elif task == "summarization":
                question, context, answer = (
                    "Summarize the following text.",
                    f"এই অনুচ্ছেদে {topic} নিয়ে তিনটি যাচাইযোগ্য তথ্য আছে। নমুনা নম্বর {i + 1}।",
                    f"অনুচ্ছেদটি {topic} সম্পর্কে তথ্য দেয়; নমুনা নম্বর {i + 1}।",
                )
            else:
                question, context, answer = (
                    f"সমস্যা {i + 1}: ১৪০-এর ৪০ শতাংশ কত?",
                    "",
                    f"১৪০ × ০.৪ = {56 + (i % 3)}।",
                )
            rows.append({
                "id": f"mock:Track_A:{task}:{i:04d}", "task": task, "track": TRACK_A,
                "hallucination_type": "None", "question": question, "context": context,
                "candidate_answer": answer, "ground_truth_label": 0,
            })

    track_b_specs: list[tuple[str, str, int]] = []
    for task in ("bn_qa", "codemixed_qa"):
        track_b_specs.extend((task, kind, 1_000) for kind in HALLUCINATION_TYPES[:4])
    track_b_specs.extend(("summarization", kind, 1_000) for kind in HALLUCINATION_TYPES[4:7])
    track_b_specs.extend(("math", kind, 200) for kind in HALLUCINATION_TYPES[7:])
    for task, kind, amount in track_b_specs:
        for i in range(amount):
            if task == "bn_qa":
                question, context, candidate = (
                    f"বাংলাদেশের নদী সম্পর্কে প্রশ্ন {i + 1}: প্রধান তথ্যটি কী?",
                    f"নথিতে নদী সম্পর্কে নির্ভরযোগ্য তথ্য {i + 1} দেওয়া হয়েছে।",
                    f"নথির বিপরীত একটি বানানো উত্তর {i + 1}।",
                )
            elif task == "codemixed_qa":
                question, context, candidate = (
                    f"Bangladesh-er nodi niye question {i + 1}: main fact ta ki?",
                    f"Document-e verified information {i + 1} deya ache.",
                    f"Document-er sathe mil nei emon invented answer {i + 1}.",
                )
            elif task == "summarization":
                question, context, candidate = (
                    "Summarize the following text.",
                    f"এই অনুচ্ছেদে নদী নিয়ে যাচাইযোগ্য তথ্য আছে। নমুনা নম্বর {i + 1}।",
                    f"সারাংশে নথিতে না থাকা একটি অতিরিক্ত দাবি আছে; নমুনা {i + 1}।",
                )
            else:
                question, context, candidate = (
                    f"সমস্যা {i + 1}: ১৪০-এর ৪০ শতাংশ কত?",
                    "",
                    f"ভুল গণনা বা যুক্তির ফল {i + 1}।",
                )
            # The category has intentionally varied answer wording while the
            # compact source schema records the exact label for stratification.
            if rng.random() < 0.15:
                candidate += " এটি যাচাই করা দরকার।"
            rows.append({
                "id": f"mock:Track_B:{task}:{kind}:{i:04d}", "task": task, "track": TRACK_B,
                "hallucination_type": kind, "question": question, "context": context,
                "candidate_answer": candidate, "ground_truth_label": 1,
            })
    return rows


def generate_mock_data(output_dir: Path) -> tuple[Path, Path]:
    mock_dir = output_dir / "mock_data"
    mock_dir.mkdir(parents=True, exist_ok=True)
    primary_path = mock_dir / "benhallu_mock_16k.csv"
    dialect_path = mock_dir / "dialectal_bias_mock_4k.csv"
    primary = pd.DataFrame(_mock_primary_rows(), columns=BENHALLU_COLUMNS)
    if len(primary) != 16_000:
        raise AssertionError(f"Mock BenHallu data must have 16,000 rows; generated {len(primary)}")
    primary.to_csv(primary_path, index=False, encoding="utf-8")

    rng = random.Random(2026)
    dialect_specs = (
        ("Standard_Bengali", "বাংলাদেশের জাতীয় ফুল কোনটি?", "বাংলাদেশের জাতীয় ফুল শাপলা।"),
        ("Tangail", "তোমার এলাকার প্রধান নদীটার নাম কী?", "আমাগো এলাকার প্রধান নদীর নাম ধলেশ্বরী।"),
        ("Chittagong", "অঁনার বাড়ি কোন গন্ডায়?", "আঁর বাড়ি চট্টগ্রামের পাহাড়তলীত।"),
        ("Sylheti", "তুমার বাড়ি কই?", "আমার বাড়ি সিলেট শহরত।"),
    )
    dialect_rows = []
    for dialect, question, response in dialect_specs:
        for i in range(1_000):
            question_type = "definitional" if i % 2 == 0 else "factual"
            score = max(0.0, min(10.0, round(7.2 + rng.gauss(0, 1.05) - (0.25 if dialect == "Chittagong" else 0), 2)))
            dialect_rows.append({
                "id": f"dialect:{dialect}:{i:04d}",
                "dialect": dialect,
                "question_type": question_type,
                "question": f"{question} (নমুনা {i + 1})",
                "response": f"{response} নমুনা {i + 1}-এর উত্তর।",
                "human_or_primary_score": score,
            })
    dialect = pd.DataFrame(dialect_rows, columns=DIALECT_COLUMNS)
    if len(dialect) != 4_000:
        raise AssertionError(f"Mock Dialectal_Bias data must have 4,000 rows; generated {len(dialect)}")
    dialect.to_csv(dialect_path, index=False, encoding="utf-8")
    return primary_path, dialect_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", type=Path, help="Canonical CSV or the raw All data directory")
    parser.add_argument("--output_dir", type=Path, default=Path("."), help="Directory for generated splits and mock data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("benhallu", "dialect"), default="benhallu")
    parser.add_argument("--generate_mock_data", action="store_true", help="Generate exact-schema 16k and 4k mock datasets before splitting")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        mock_paths = generate_mock_data(args.output_dir) if args.generate_mock_data else None
        input_path = args.input_csv
        if mock_paths:
            input_path = mock_paths[0] if args.mode == "benhallu" else mock_paths[1]
        if input_path is None:
            raise ValueError("--input_csv is required unless --generate_mock_data is supplied")
        if args.mode == "benhallu":
            frame = normalize_benhallu_input(input_path)
            validate_full_benhallu(frame)
            create_benhallu_splits(frame, args.output_dir, args.seed)
            print(f"Validated {len(frame):,} BenHalluEval rows; wrote three splits to {args.output_dir.resolve()}")
        else:
            frame = normalize_dialect_input(input_path)
            create_dialect_split(frame, args.output_dir, args.seed)
            print(f"Validated {len(frame):,} Dialectal_Bias rows; wrote 300-row anchor subset to {args.output_dir.resolve()}")
        return 0
    except (ValueError, FileNotFoundError, pd.errors.ParserError) as exc:
        print(f"make_splits.py: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
