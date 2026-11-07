from __future__ import annotations

import argparse
import bz2
import csv
import json
import random
import tarfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from datathon_utils import clean_text


REGISTRY = {
    "indicxnli_bn": {
        "repo": "Divyanshu/indicxnli",
        "license": "cc0-1.0",
        "domain": "nli",
        "kind": "nli_json",
        "files": [
            "forward/train/xnli_bn.json",
            "forward/dev/xnli_bn.json",
            "forward/test/xnli_bn.json",
        ],
        "file_bytes": {
            "forward/train/xnli_bn.json": 217731050,
            "forward/dev/xnli_bn.json": 1321253,
            "forward/test/xnli_bn.json": 2669757,
        },
    },
    "xnli_bn": {
        "repo": "csebuetnlp/xnli_bn",
        "license": "cc-by-nc-sa-4.0",
        "domain": "nli",
        "kind": "xnli_tar",
        "files": ["data/xnli_bn.tar.bz2"],
        "file_bytes": {"data/xnli_bn.tar.bz2": 21437836},
    },
    "bangla_law_qna": {
        "repo": "bipinsaha/bangla-law-qna",
        "license": "cc-by-nc-4.0",
        "domain": "law",
        "kind": "table",
        "files": ["Law-QnA-Prothom-Alo-63.csv"],
        "file_bytes": {"Law-QnA-Prothom-Alo-63.csv": 460149},
    },
    "bangla_textbook": {
        "repo": "md-nishat-008/Bangla-TextBook",
        "license": "mit",
        "domain": "textbook",
        "kind": "table",
        "files": ["bangla_textbook_128w_cleaned.csv"],
        "file_bytes": {"bangla_textbook_128w_cleaned.csv": 188650371},
    },
    "bangla_newspaper": {
        "repo": "zabir-nabil/bangla_newspaper_dataset",
        "license": "mit",
        "domain": "news",
        "kind": "table",
        "files": [
            "data/train-00000-of-00003.parquet",
            "data/valid-00000-of-00001.parquet",
        ],
        "file_bytes": {
            "data/train-00000-of-00003.parquet": 169985871,
            "data/valid-00000-of-00001.parquet": 118164780,
        },
    },
    "bangla_wikipedia": {
        "repo": "rejauldu/bengali-wikipedia",
        "license": "cc-by-3.0",
        "domain": "wikipedia",
        "kind": "table",
        "files": ["data/test-00000-of-00001.parquet"],
        "file_bytes": {"data/test-00000-of-00001.parquet": 28135132},
    },
    "bangladesh_legal_acts": {
        "repo": "sakhadib/Bangladesh-Legal-Acts-Dataset",
        "license": "cc-by-4.0",
        "domain": "law",
        "kind": "legal_acts",
        "files": ["Contextualized_Bangladesh_Legal_Acts.json"],
        "file_bytes": {"Contextualized_Bangladesh_Legal_Acts.json": 61900000},
    },
}

QUESTION_KEYS = ("question", "প্রশ্ন", "query", "prompt", "instruction", "Problem", "problem")
ANSWER_KEYS = ("answer", "উত্তর", "response", "output", "target", "Solution", "solution")
CONTEXT_KEYS = ("context", "passage", "article", "text", "content", "body")
TITLE_KEYS = ("title", "শিরোনাম", "name")
DATE_KEYS = ("date", "published", "publication_date", "timestamp")


def first(row: dict, keys: Iterable[str]) -> object:
    for key in keys:
        if key in row and row[key] not in (None, "", []):
            return row[key]
    return ""


def flatten_text(value: object) -> str:
    if isinstance(value, list):
        return clean_text("; ".join(map(str, value)))
    if isinstance(value, dict):
        return clean_text(value.get("text") or value.get("answer") or json.dumps(value, ensure_ascii=False))
    return clean_text(value)


def nli_record(row: dict, source: str, source_file: str, license_name: str, split: str) -> dict | None:
    premise = flatten_text(row.get("premise", ""))
    hypothesis = flatten_text(row.get("hypothesis", ""))
    if not premise or not hypothesis:
        return None
    label = row.get("label")
    if isinstance(label, str):
        label_name = label.lower()
    else:
        # XNLI convention: entailment=0, neutral=1, contradiction=2.
        label_name = {0: "entailment", 1: "neutral", 2: "contradiction"}.get(int(label), "")
    if label_name not in {"entailment", "neutral", "contradiction"}:
        return None
    return {
        "prompt_bn": "প্রদত্ত বক্তব্য থেকে দাবিটি সমর্থিত কি?",
        "response_bn": hypothesis,
        "context": premise,
        "label": 1 if label_name == "entailment" else 0,
        "nli_label": label_name,
        "source": source,
        "source_file": source_file,
        "domain": "nli",
        "license": license_name,
        "split": split,
        "hallucination_type": "faithful_entailment" if label_name == "entailment" else f"nli_{label_name}",
    }


def evidence_record(row: dict, source: str, source_file: str, domain: str, license_name: str) -> dict | None:
    question = flatten_text(first(row, QUESTION_KEYS))
    answer = flatten_text(first(row, ANSWER_KEYS))
    context = flatten_text(first(row, CONTEXT_KEYS))
    title = flatten_text(first(row, TITLE_KEYS))
    if not context:
        context = flatten_text(row.get("text", ""))
    if not question and title:
        question = title
    if not answer and context:
        answer = context[:500]
    if not context and answer:
        context = answer
    if not question and context:
        question = context[:180]
    if not question or not answer:
        return None
    return {
        "question": question,
        "answer": answer,
        "context": context,
        "title": title,
        "source_repo": source,
        "source_file": source_file,
        "domain": domain,
        "license": license_name,
        "document_date": flatten_text(first(row, DATE_KEYS)),
    }


def iter_json(path: Path, file_name: str) -> Iterator[tuple[dict, str]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        for split, rows in obj.items():
            if isinstance(rows, list):
                for row in rows:
                    yield row, str(split)
    elif isinstance(obj, list):
        for row in obj:
            yield row, Path(file_name).stem


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def iter_table(path: Path, original_name: str, batch_size: int = 10_000) -> Iterator[dict]:
    if original_name.endswith(".parquet"):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size):
            yield from batch.to_pandas().fillna("").to_dict("records")
    else:
        for chunk in pd.read_csv(path, chunksize=batch_size, low_memory=False):
            yield from chunk.fillna("").to_dict("records")


def iter_xnli_tar(path: Path) -> Iterator[tuple[dict, str]]:
    with tarfile.open(path, mode="r:bz2") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            split = "train" if "train" in member.name else "validation" if "dev" in member.name else "test"
            handle = archive.extractfile(member)
            if handle is None:
                continue
            for raw in handle:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield row, split


def iter_legal_acts(path: Path) -> Iterator[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    acts = payload.get("acts", []) if isinstance(payload, dict) and "acts" in payload else (
        payload.values() if isinstance(payload, dict) else payload
    )
    for act in acts:
        if not isinstance(act, dict):
            continue
        title = flatten_text(act.get("act_title") or act.get("title"))
        source_url = flatten_text(act.get("source_url"))
        for section in act.get("sections", []) or []:
            content = flatten_text(section.get("section_content") or section.get("content"))
            section_title = flatten_text(section.get("section_title") or section.get("title"))
            if content:
                yield {
                    "question": f"{title}: {section_title}".strip(": "),
                    "answer": content[:500],
                    "context": content,
                    "title": title,
                    "publication_date": act.get("publication_date", ""),
                    "source_url": source_url,
                }


def reservoir(rows: Iterable, limit: int | None, seed: int) -> list:
    if not limit:
        return list(rows)
    rng = random.Random(seed)
    selected = []
    for index, row in enumerate(rows):
        if index < limit:
            selected.append(row)
        else:
            replacement = rng.randint(0, index)
            if replacement < limit:
                selected[replacement] = row
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize licensed Hugging Face data for Phase 2.")
    parser.add_argument("--dataset", action="append", choices=sorted(REGISTRY), default=[])
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/hf_phase2_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase2_data"))
    parser.add_argument("--max-rows-per-file", type=int, default=50_000)
    parser.add_argument("--max-file-mb", type=float, help="Skip larger remote files; useful for fast staged acquisition.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    selected = args.dataset or list(REGISTRY)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nli_path = args.output_dir / "hf_nli_pairs.jsonl"
    evidence_path = args.output_dir / "hf_evidence_corpus.jsonl"
    manifest = {"datasets": [], "nli_rows": 0, "evidence_rows": 0}
    nli_counts, evidence_counts = Counter(), Counter()

    with nli_path.open("w", encoding="utf-8") as nli_out, evidence_path.open("w", encoding="utf-8") as evidence_out:
        for name in selected:
            spec = REGISTRY[name]
            dataset_record = {"name": name, **spec, "downloaded_files": [], "rows": 0}
            for file_name in spec["files"]:
                expected_bytes = spec.get("file_bytes", {}).get(file_name)
                if args.max_file_mb and expected_bytes and expected_bytes > args.max_file_mb * 1024 * 1024:
                    dataset_record.setdefault("skipped_files", []).append({"file": file_name, "bytes": expected_bytes})
                    print("skip large", spec["repo"], file_name, expected_bytes, flush=True)
                    continue
                print("download", spec["repo"], file_name, flush=True)
                try:
                    local = Path(hf_hub_download(
                        repo_id=spec["repo"], repo_type="dataset", filename=file_name, cache_dir=args.cache_dir
                    ))
                except Exception as exc:
                    dataset_record.setdefault("failed_files", []).append(
                        {"file": file_name, "error": f"{type(exc).__name__}: {exc}"}
                    )
                    print("download failed", spec["repo"], file_name, type(exc).__name__, exc, flush=True)
                    continue
                dataset_record["downloaded_files"].append({"file": file_name, "bytes": local.stat().st_size})
                if spec["kind"] == "nli_json":
                    source_rows = iter_json(local, file_name)
                    normalized = (
                        nli_record(row, spec["repo"], file_name, spec["license"], split)
                        for row, split in source_rows
                    )
                    rows = reservoir((row for row in normalized if row), args.max_rows_per_file, args.seed)
                    for row in rows:
                        nli_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        nli_counts[row["nli_label"]] += 1
                elif spec["kind"] == "xnli_tar":
                    normalized = (
                        nli_record(row, spec["repo"], file_name, spec["license"], split)
                        for row, split in iter_xnli_tar(local)
                    )
                    rows = reservoir((row for row in normalized if row), args.max_rows_per_file, args.seed)
                    for row in rows:
                        nli_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        nli_counts[row["nli_label"]] += 1
                elif spec["kind"] == "legal_acts":
                    normalized = (
                        evidence_record(row, spec["repo"], file_name, spec["domain"], spec["license"])
                        for row in iter_legal_acts(local)
                    )
                    rows = reservoir((row for row in normalized if row), args.max_rows_per_file, args.seed)
                    for row in rows:
                        evidence_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        evidence_counts[row["domain"]] += 1
                else:
                    raw_rows = iter_jsonl(local) if spec["kind"] == "jsonl" else iter_table(local, file_name)
                    normalized = (
                        evidence_record(row, spec["repo"], file_name, spec["domain"], spec["license"])
                        for row in raw_rows
                    )
                    rows = reservoir((row for row in normalized if row), args.max_rows_per_file, args.seed)
                    for row in rows:
                        evidence_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        evidence_counts[row["domain"]] += 1
                dataset_record["rows"] += len(rows)
            manifest["datasets"].append(dataset_record)

    manifest["nli_rows"] = sum(nli_counts.values())
    manifest["nli_labels"] = dict(nli_counts)
    manifest["evidence_rows"] = sum(evidence_counts.values())
    manifest["evidence_domains"] = dict(evidence_counts)
    manifest["outputs"] = {"nli": str(nli_path), "evidence": str(evidence_path)}
    manifest_path = args.output_dir / "hf_phase2_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
