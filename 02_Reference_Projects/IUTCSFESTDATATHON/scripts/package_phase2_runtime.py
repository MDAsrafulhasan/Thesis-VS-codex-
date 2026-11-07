from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RUNTIME_FILES = [
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "licenses/Apache-2.0.txt",
    "licenses/GPL-3.0.txt",
    "resources/verified_fact_answers.json",
    "resources/typed_evidence_v1.json",
    "resources/legacy_law_acts.json",
    "outputs/source_answer_bank.jsonl",
    "outputs/source_answer_bank_complete_bcs.jsonl",
    "outputs/source_answer_bank_farhan_qdoc.jsonl",
    "outputs/source_answer_bank_bcs_all.jsonl",
    "outputs/lexical_evidence_corpus.jsonl",
    "outputs/source_cache/benhallueval_data/qa_gt_1000.csv",
    "outputs/source_cache/benhallueval_data/qa_4000.csv",
    "outputs/source_cache/bangla_bagdhara_lrec/bagdhara_export_done_2026-03-01_010356.json",
    "outputs/phase2_data/hf_legal_acts_evidence.jsonl",
    "scripts/apply_lexical_overrides.py",
    "scripts/birth_date_overrides.py",
    "scripts/bn_num2words.py",
    "scripts/context_numeric_overrides.py",
    "scripts/context_span_overrides.py",
    "scripts/datathon_utils.py",
    "scripts/establishment_overrides.py",
    "scripts/fuzzy_source_overrides.py",
    "scripts/gguf_selective_judge.py",
    "scripts/law_evidence_overrides.py",
    "scripts/phase2_pipeline.py",
    "scripts/source_answer_bank.py",
    "scripts/structured_overrides.py",
    "scripts/typed_evidence.py",
    "scripts/typed_idiom_matcher.py",
    "scripts/train_portable_verifier.py",
    "scripts/train_verifier.py",
    "scripts/weighted_source_overrides.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the public offline v91 Phase 2 router bundle.")
    parser.add_argument(
        "--archive",
        type=Path,
        default=ROOT / "outputs" / "kaggle_dataset_phase2_runtime_v91" / "kaggle_phase2_runtime_v91.tar.gz",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs" / "kaggle_dataset_phase2_runtime_v91" / "kaggle_phase2_runtime_v91.manifest.json",
    )
    parser.add_argument(
        "--dataset-metadata",
        type=Path,
        default=ROOT / "outputs" / "kaggle_dataset_phase2_runtime_v91" / "dataset-metadata.json",
    )
    args = parser.parse_args()

    missing = [relative for relative in RUNTIME_FILES if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing runtime files: {missing}")

    records = []
    for relative in RUNTIME_FILES:
        path = ROOT / relative
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    manifest = {
        "purpose": "Public Phase 2 offline raw-input deterministic router runtime for scored v91",
        "version": "v91",
        "reference_submission": 54836339,
        "reference_public_macro_f1": 0.967,
        "reference_output_sha256": "0fec5d610f4e7b94ab6c46e2094d58e940b0f36781e9fa98d5466d2135474c28",
        "contains_phase1_submission_labels": False,
        "files": records,
        "file_count": len(records),
        "total_uncompressed_bytes": sum(record["bytes"] for record in records),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    dataset_metadata = {
        "title": "Bengali Hallucination Phase 2 Runtime v91",
        "id": "seyamalam/bengali-hallucination-phase2-runtime-v91",
        "licenses": [{"name": "other"}],
        "subtitle": "Public offline deterministic router and evidence assets for Huntrix v91",
        "description": (
            "Pinned runtime bundle for the public v91 inference notebook. "
            "It contains deterministic Bengali task solvers, attributed answer/evidence banks, "
            "and no saved Phase 1 prediction vector. See THIRD_PARTY_NOTICES.md for source and "
            "license status, LICENSE for original Huntrix code, and the embedded manifest for hashes."
        ),
    }
    args.dataset_metadata.parent.mkdir(parents=True, exist_ok=True)
    args.dataset_metadata.write_text(
        json.dumps(dataset_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, "w:gz", compresslevel=6) as archive:
        for relative in RUNTIME_FILES:
            archive.add(ROOT / relative, arcname=relative, recursive=False)
        archive.add(args.manifest, arcname="runtime-manifest.json", recursive=False)

    print(
        json.dumps(
            {
                "archive": str(args.archive),
                "archive_bytes": args.archive.stat().st_size,
                "manifest": str(args.manifest),
                "dataset_metadata": str(args.dataset_metadata),
                **{key: manifest[key] for key in ("file_count", "total_uncompressed_bytes")},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
