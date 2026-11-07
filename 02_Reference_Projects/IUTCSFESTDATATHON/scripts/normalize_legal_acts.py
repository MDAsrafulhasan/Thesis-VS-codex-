from __future__ import annotations

import argparse
import json
from pathlib import Path

from hf_phase2_data import iter_legal_acts
from normalize_phase2_raw import chunks, write_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize the licensed Bangladesh Legal Acts HF corpus.")
    parser.add_argument("--input", type=Path, default=Path("outputs/hf_phase2_raw/bangladesh_legal_acts.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/phase2_data/hf_legal_acts_evidence.jsonl"))
    parser.add_argument("--max-sections", type=int, default=40_000)
    parser.add_argument("--max-chars", type=int, default=1800)
    parser.add_argument("--overlap", type=int, default=180)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sections = records = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for row in iter_legal_acts(args.input):
            for chunk_index, chunk in enumerate(chunks(row["context"], args.max_chars, args.overlap)):
                write_record(
                    handle, title=row["question"], context=chunk,
                    source_repo="sakhadib/Bangladesh-Legal-Acts-Dataset",
                    source_file=f"Contextualized_Bangladesh_Legal_Acts.json#section={sections}#chunk={chunk_index}",
                    domain="law", license_name="cc-by-4.0",
                    document_date=str(row.get("publication_date", "")), url=str(row.get("source_url", "")),
                )
                records += 1
            sections += 1
            if args.max_sections and sections >= args.max_sections:
                break
    summary = {"sections": sections, "records": records, "output": str(args.output), "license": "cc-by-4.0"}
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
