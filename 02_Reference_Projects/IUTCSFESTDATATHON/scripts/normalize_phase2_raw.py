from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from datathon_utils import clean_text


def chunks(text: str, max_chars: int, overlap: int) -> list[str]:
    value = clean_text(text)
    if len(value) <= max_chars:
        return [value] if value else []
    output, start = [], 0
    while start < len(value):
        stop = min(len(value), start + max_chars)
        if stop < len(value):
            boundary = max(value.rfind("।", start, stop), value.rfind("\n", start, stop))
            if boundary > start + max_chars // 2:
                stop = boundary + 1
        output.append(value[start:stop].strip())
        if stop >= len(value):
            break
        start = max(start + 1, stop - overlap)
    return [item for item in output if item]


def inferred_title(text: str) -> str:
    value = clean_text(text)
    sentence = re.split(r"[।\n]", value, maxsplit=1)[0]
    return sentence[:180]


def write_record(handle, *, title: str, context: str, source_repo: str, source_file: str,
                 domain: str, license_name: str, document_date: str = "", url: str = "") -> None:
    record = {
        "question": clean_text(title),
        "answer": clean_text(context)[:500],
        "context": clean_text(context),
        "title": clean_text(title),
        "source_repo": source_repo,
        "source_file": source_file,
        "domain": domain,
        "license": license_name,
        "document_date": clean_text(document_date),
        "url": str(url or ""),
    }
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_text_file(path: Path, handle, source_repo: str, source_file: str, domain: str,
                        license_name: str, max_rows: int, max_chars: int, overlap: int) -> int:
    count = 0
    if path.suffix == ".csv":
        iterator = (row for frame in pd.read_csv(path, chunksize=2000, low_memory=False) for row in frame.fillna("").to_dict("records"))
    else:
        parquet = pq.ParquetFile(path)
        iterator = (row for batch in parquet.iter_batches(batch_size=2000) for row in batch.to_pandas().fillna("").to_dict("records"))
    for row in iterator:
        text = str(row.get("text", ""))
        title = inferred_title(text)
        for index, chunk in enumerate(chunks(text, max_chars, overlap)):
            write_record(
                handle,
                title=title,
                context=chunk,
                source_repo=source_repo,
                source_file=f"{source_file}#row={count}#chunk={index}",
                domain=domain,
                license_name=license_name,
            )
        count += 1
        if max_rows and count >= max_rows:
            break
    return count


def normalize_news(path: Path, handle, max_rows: int, max_chars: int, overlap: int) -> int:
    parquet = pq.ParquetFile(path)
    count = 0
    for batch in parquet.iter_batches(batch_size=2000):
        for row in batch.to_pandas().fillna("").to_dict("records"):
            title, content = str(row.get("title", "")), str(row.get("content", ""))
            for index, chunk in enumerate(chunks(content, max_chars, overlap)):
                write_record(
                    handle,
                    title=title,
                    context=chunk,
                    source_repo="zabir-nabil/bangla_newspaper_dataset",
                    source_file=f"data/train-00000-of-00003.parquet#row={count}#chunk={index}",
                    domain="news",
                    license_name="mit",
                    document_date=str(row.get("published_date", "")),
                    url=str(row.get("url", "")),
                )
            count += 1
            if max_rows and count >= max_rows:
                return count
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize downloaded textbook, news, and Wikipedia corpora.")
    parser.add_argument("--raw-dir", type=Path, default=Path("outputs/hf_phase2_raw"))
    parser.add_argument("--output", type=Path, default=Path("outputs/phase2_data/hf_domain_evidence.jsonl"))
    parser.add_argument("--max-rows-per-domain", type=int, default=20_000)
    parser.add_argument("--max-chars", type=int, default=1800)
    parser.add_argument("--overlap", type=int, default=180)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with args.output.open("w", encoding="utf-8") as handle:
        wiki = args.raw_dir / "wikipedia_test.parquet"
        if wiki.exists():
            counts["wikipedia"] = normalize_text_file(
                wiki, handle, "rejauldu/bengali-wikipedia", "data/test-00000-of-00001.parquet",
                "wikipedia", "cc-by-3.0", args.max_rows_per_domain, args.max_chars, args.overlap,
            )
        textbook = args.raw_dir / "bangla_textbook.csv"
        if textbook.exists():
            counts["textbook"] = normalize_text_file(
                textbook, handle, "md-nishat-008/Bangla-TextBook", "bangla_textbook_128w_cleaned.csv",
                "textbook", "mit", args.max_rows_per_domain, args.max_chars, args.overlap,
            )
        news = args.raw_dir / "bangla_news_train.parquet"
        if news.exists():
            counts["news"] = normalize_news(news, handle, args.max_rows_per_domain, args.max_chars, args.overlap)
    summary = {"output": str(args.output), "source_rows": dict(counts), "max_rows_per_domain": args.max_rows_per_domain}
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
