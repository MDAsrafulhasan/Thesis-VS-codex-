from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

from datathon_utils import OUTPUT_DIR, clean_text


HF_DATASETS = [
    {
        "repo": "shayekh/bengali-exams-public",
        "license": "cc-by-nc-4.0",
        "files": ["bcs_10th_to_45th.csv", "bcs_10th_to_45th_raw.json", "public_5_v0.csv"],
        "kind": "exam",
        "enabled": False,
    },
    {
        "repo": "shayekh/bengali-exams",
        "license": "cc-by-nc-4.0",
        "files": [
            "du_bengali_grammar_400_clean.json",
            "du_bengali_literature_400_clean.json",
            "ru_general_knowledge_bangladesh_all_clean.json",
            "ru_general_knowledge_world_all_clean.json",
            "ru_bengali_grammar_all_clean.json",
            "ru_bengali_literature_all_clean.json",
            "cu_general_knowledge_bangladesh_all_clean.json",
            "cu_general_knowledge_world_all_clean.json",
            "cu_bengali_grammar_all_clean.json",
            "cu_bengali_literature_all_clean.json",
            "cu_ict_all_clean.json",
        ],
        "kind": "exam",
        "enabled": False,
    },
    {
        "repo": "SumanMondal/bengali_qa",
        "license": "apache-2.0",
        "files": ["data/train-00000-of-00001.parquet"],
        "kind": "qa_context",
    },
    {
        "repo": "r-niloy/bengali-qa-from-alldata",
        "license": "unknown",
        "files": ["data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"],
        "kind": "qa_context",
    },
    {
        "repo": "Kowshik24/BengaliQADataset",
        "license": "unknown",
        "files": ["data/full_dataset.csv", "data/train.csv", "data/valid.csv"],
        "kind": "qa",
    },
    {
        "repo": "Lavlu118557/bonolota_qa_bengali",
        "license": "unknown",
        "files": ["data/qa_dataset.csv"],
        "kind": "qa",
    },
    {
        "repo": "Badhon/bangla_literature_qa_dataset_v_1",
        "license": "unknown",
        "files": [
            "qa_bangla_literature_oporichita_merged_all.json",
            "qa_bangla_literature_oporichita_output_no_duplicates.json",
        ],
        "kind": "qa",
    },
    {
        "repo": "sabbirRamim/bengali-question-answering-preindependence-1947-1971",
        "license": "mit",
        "files": ["thesis_qna.json"],
        "kind": "qa",
    },
]


QUESTION_KEYS = ["question", "প্রশ্ন", "Question", "query", "instruction", "prompt", "prompt_bn"]
ANSWER_KEYS = ["answer", "answers", "উত্তর", "Answer", "output", "completion", "response", "response_bn", "correct_answer"]
CONTEXT_KEYS = ["context", "passage", "title", "article", "source", "explanation"]
OPTION_KEYS = ["option", "options", "choices", "mcq", "a", "b", "c", "d", "A", "B", "C", "D"]


def flatten_records(obj: Any) -> list[dict]:
    if isinstance(obj, list):
        rows = []
        for item in obj:
            rows.extend(flatten_records(item))
        return rows
    if isinstance(obj, dict):
        # Common wrappers.
        for key in ["data", "train", "questions", "items", "records"]:
            if key in obj and isinstance(obj[key], list):
                return flatten_records(obj[key])
        return [obj]
    return []


def first_value(row: dict, keys: list[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in [None, "", []]:
            return row[key]
    return None


def answer_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            texts = [x.get("text") or x.get("answer") or x.get("value") for x in value]
            return "; ".join(str(x) for x in texts if x)
        return "; ".join(str(x) for x in value if x)
    if isinstance(value, dict):
        for key in ["text", "answer", "value", "label"]:
            if key in value:
                return answer_to_text(value[key])
    return str(value)


def options_to_context(row: dict) -> str:
    parts = []
    options = first_value(row, OPTION_KEYS)
    if isinstance(options, dict):
        for key, value in options.items():
            parts.append(f"{key}: {value}")
    elif isinstance(options, list):
        for i, value in enumerate(options):
            parts.append(f"{chr(65+i)}: {value}")
    else:
        for key in ["a", "b", "c", "d", "A", "B", "C", "D", "option_a", "option_b", "option_c", "option_d"]:
            if key in row and row[key]:
                parts.append(f"{key}: {row[key]}")
    return " | ".join(parts)


def normalize_row(row: dict, repo: str, file_name: str, license_name: str) -> dict | None:
    question = first_value(row, QUESTION_KEYS)
    answer = first_value(row, ANSWER_KEYS)
    context = first_value(row, CONTEXT_KEYS)
    if answer is None and "correct" in row:
        answer = row["correct"]
    if answer is None and "label" in row:
        answer = row["label"]
    question_text = clean_text(question)
    answer_text = answer_to_text(answer)
    context_text = clean_text(context)
    options_text = options_to_context(row)
    if options_text and options_text not in context_text:
        context_text = (context_text + " " + options_text).strip()
    if not question_text or not answer_text:
        return None
    return {
        "question": question_text,
        "answer": clean_text(answer_text),
        "context": context_text,
        "source_repo": repo,
        "source_file": file_name,
        "license": license_name,
    }


def read_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.DataFrame([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])
    if suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        return pd.DataFrame(flatten_records(obj))
    raise ValueError(f"unsupported file type: {path}")


def download_and_convert(
    output: Path,
    cache_dir: Path,
    max_rows_per_file: int | None = None,
    include_gated: bool = False,
) -> pd.DataFrame:
    rows = []
    for spec in HF_DATASETS:
        if not spec.get("enabled", True) and not include_gated:
            print("repo", spec["repo"], "SKIP disabled/gated; pass --include-gated if access is available")
            continue
        repo = spec["repo"]
        license_name = spec.get("license", "unknown")
        print("repo", repo)
        for file_name in spec["files"]:
            try:
                local = hf_hub_download(repo_id=repo, repo_type="dataset", filename=file_name, cache_dir=cache_dir)
                df = read_file(Path(local))
                if max_rows_per_file:
                    df = df.head(max_rows_per_file)
                converted = 0
                for row in df.to_dict("records"):
                    normalized = normalize_row(row, repo, file_name, license_name)
                    if normalized:
                        rows.append(normalized)
                        converted += 1
                print(" ", file_name, "raw", len(df), "converted", converted)
            except Exception as exc:
                print(" ", file_name, "ERR", repr(exc))
    out_df = pd.DataFrame(rows).drop_duplicates(["question", "answer", "context"])
    output.parent.mkdir(exist_ok=True)
    out_df.to_json(output.with_suffix(".jsonl"), orient="records", lines=True, force_ascii=False)
    out_df.to_csv(output.with_suffix(".csv"), index=False)
    print("wrote", output.with_suffix(".jsonl"), output.with_suffix(".csv"), "rows", len(out_df))
    return out_df


def make_hallucination_pairs(corpus: pd.DataFrame, output: Path) -> None:
    rows = []
    answer_pools = {
        src: group["answer"].drop_duplicates().tolist()
        for src, group in corpus.groupby("source_repo")
    }
    all_answers = corpus["answer"].drop_duplicates().tolist()
    for idx, item in corpus.iterrows():
        rows.append(
            {
                "context": item.get("context", ""),
                "prompt_bn": item["question"],
                "response_bn": item["answer"],
                "label": 1,
                "source": "external_qa_faithful",
                "source_repo": item["source_repo"],
            }
        )
        pool = answer_pools.get(item["source_repo"]) or all_answers
        if len(pool) > 1:
            wrong_answer = pool[(int(idx) * 7919 + 17) % len(pool)]
            if wrong_answer == item["answer"]:
                wrong_answer = pool[(int(idx) * 7919 + 18) % len(pool)]
            rows.append(
                {
                    "context": item.get("context", ""),
                    "prompt_bn": item["question"],
                    "response_bn": wrong_answer,
                    "label": 0,
                    "source": "external_qa_answer_swap",
                    "source_repo": item["source_repo"],
                }
            )
    df = pd.DataFrame(rows)
    output.parent.mkdir(exist_ok=True)
    df.to_json(output.with_suffix(".jsonl"), orient="records", lines=True, force_ascii=False)
    df.to_csv(output.with_suffix(".csv"), index=False)
    print("wrote", output.with_suffix(".jsonl"), output.with_suffix(".csv"), "rows", len(df), "labels", df.label.value_counts().to_dict())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", default=str(OUTPUT_DIR / "external_qa_corpus"))
    parser.add_argument("--cache-dir", default=str(OUTPUT_DIR / "hf_cache"))
    parser.add_argument("--max-rows-per-file", type=int)
    parser.add_argument("--include-gated", action="store_true")
    parser.add_argument("--pairs-output-prefix")
    args = parser.parse_args()

    output = Path(args.output_prefix)
    corpus = download_and_convert(
        output,
        Path(args.cache_dir),
        max_rows_per_file=args.max_rows_per_file,
        include_gated=args.include_gated,
    )
    if args.pairs_output_prefix:
        pairs_output = Path(args.pairs_output_prefix)
    elif output.name == "external_qa_corpus":
        pairs_output = output.parent / "external_qa_pairs"
    else:
        pairs_output = output.parent / f"{output.name}_pairs"
    make_hallucination_pairs(corpus, pairs_output)


if __name__ == "__main__":
    main()
