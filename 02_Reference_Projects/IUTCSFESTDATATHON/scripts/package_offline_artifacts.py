from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from datathon_utils import OUTPUT_DIR, ROOT


REQUIRED_ARTIFACTS = [
    "resources/verified_fact_answers.json",
    "resources/legacy_law_acts.json",
    "outputs/source_answer_bank.jsonl",
    "outputs/source_cache/benhallueval_data/qa_gt_1000.csv",
    "outputs/source_cache/benhallueval_data/qa_4000.csv",
    "outputs/source_answer_bank_complete_bcs.jsonl",
    "outputs/source_answer_bank_farhan_qdoc.jsonl",
    "outputs/source_answer_bank_bcs_all.jsonl",
    "outputs/submission_v85_shakib_birth_repair_nosubmit.csv",
    "outputs/phase2_data/hf_legal_acts_evidence.jsonl",
    "outputs/evidence_index/lexical_char_tfidf/corpus.jsonl.gz",
    "outputs/evidence_index/lexical_char_tfidf/matrix.npz",
    "outputs/evidence_index/lexical_char_tfidf/metadata.json",
    "outputs/evidence_index/lexical_char_tfidf/vectorizer.joblib",
    "outputs/evidence_index/external_qa_char_tfidf/corpus.jsonl.gz",
    "outputs/evidence_index/external_qa_char_tfidf/matrix.npz",
    "outputs/evidence_index/external_qa_char_tfidf/metadata.json",
    "outputs/evidence_index/external_qa_char_tfidf/vectorizer.joblib",
    "outputs/evidence_index/wikidata_bangladesh_char_tfidf/corpus.jsonl.gz",
    "outputs/evidence_index/wikidata_bangladesh_char_tfidf/matrix.npz",
    "outputs/evidence_index/wikidata_bangladesh_char_tfidf/metadata.json",
    "outputs/evidence_index/wikidata_bangladesh_char_tfidf/vectorizer.joblib",
    "outputs/lexical_evidence_corpus.jsonl",
    "outputs/external_qa_corpus.jsonl",
    "outputs/wikidata_bangladesh_corpus.jsonl",
    "outputs/merged_evidence_corpus.jsonl",
    "outputs/lexical_controlled_pairs_80k.jsonl",
    "outputs/wikidata_bangladesh_pairs.jsonl",
    "outputs/external_context_contrastive_pairs.jsonl",
    "outputs/portable_verifier.joblib",
    "outputs/portable_verifier_metadata.json",
    "outputs/portable_verifier_externalqa.joblib",
    "outputs/portable_verifier_externalqa_metadata.json",
    "outputs/portable_verifier_externalqa_wikidata.joblib",
    "outputs/portable_verifier_externalqa_wikidata_metadata.json",
    "outputs/portable_verifier_context_contrastive.joblib",
    "outputs/portable_verifier_context_contrastive_metadata.json",
]


PHASE1_CANDIDATES = [
    "outputs/submission_v60_forensic_repairs_nosubmit.csv",
    "outputs/submission_v85_shakib_birth_repair_nosubmit.csv",
    "outputs/submission_mlx_qwen36_27b_noctx_fact_no_context.csv",
    "outputs/submission_mlx_qwen36_27b_noctx_fact_lexical_override.csv",
    "outputs/submission_mlx_qwen36_27b_noctx_fact_evidence025.csv",
    "outputs/submission_mlx_qwen36_27b_evidence025_lexical_override.csv",
    "outputs/submission_portable_verifier.csv",
    "outputs/submission_portable_verifier_externalqa.csv",
    "outputs/submission_portable_verifier_externalqa_wikidata.csv",
    "outputs/submission_portable_verifier_context_contrastive.csv",
    "outputs/submission_mlx_qwen36_27b_noctx_fact_portable_conf025.csv",
    "outputs/submission_mlx_qwen36_27b_lexical_portable_conf025.csv",
]

PHASE2_ARTIFACTS = [
    "resources/verified_fact_answers.json",
    "resources/typed_evidence_v1.json",
    "outputs/source_cache/bangla_bagdhara_lrec/bagdhara_export_done_2026-03-01_010356.json",
    "outputs/phase2_data/hf_nli_pairs.jsonl",
    "outputs/phase2_data/hf_legal_acts_evidence.jsonl",
    "outputs/evidence_index/phase2_authoritative_hybrid_v2/corpus.jsonl.gz",
    "outputs/evidence_index/phase2_authoritative_hybrid_v2/char_matrix.npz",
    "outputs/evidence_index/phase2_authoritative_hybrid_v2/word_matrix.npz",
    "outputs/evidence_index/phase2_authoritative_hybrid_v2/char_vectorizer.joblib",
    "outputs/evidence_index/phase2_authoritative_hybrid_v2/word_vectorizer.joblib",
    "outputs/evidence_index/phase2_authoritative_hybrid_v2/metadata.json",
]


REPRO_SCRIPTS = [
    "scripts/datathon_utils.py",
    "scripts/apply_lexical_overrides.py",
    "scripts/source_answer_bank.py",
    "scripts/fuzzy_source_overrides.py",
    "scripts/weighted_source_overrides.py",
    "scripts/structured_overrides.py",
    "scripts/typed_evidence.py",
    "scripts/typed_idiom_matcher.py",
    "scripts/birth_date_overrides.py",
    "scripts/context_numeric_overrides.py",
    "scripts/context_span_overrides.py",
    "scripts/establishment_overrides.py",
    "scripts/law_evidence_overrides.py",
    "scripts/evidence_index.py",
    "scripts/external_data.py",
    "scripts/lexical_data.py",
    "scripts/wikidata_facts.py",
    "scripts/context_contrastive_pairs.py",
    "scripts/apply_portable_verifier_blend.py",
    "scripts/combine_evidence_predictions.py",
    "scripts/evidence_controlled_pairs.py",
    "scripts/train_portable_verifier.py",
    "scripts/mlx_qwen_judge.py",
    "scripts/hf_phase2_data.py",
    "scripts/normalize_phase2_raw.py",
    "scripts/normalize_legal_acts.py",
    "scripts/build_phase2_training.py",
    "scripts/hybrid_evidence_index.py",
    "scripts/retrieval_consensus_overrides.py",
    "scripts/phase2_pipeline.py",
    "scripts/run_phase2_cascade.py",
    "scripts/train_nli_encoder.py",
    "scripts/selective_llm_judge.py",
    "scripts/benchmark_phase2_pipeline.py",
    "requirements.txt",
]


SOURCE_NOTES = [
    {
        "name": "Bagdhara Bangla Idioms Dataset",
        "locator": "kaggle:sakhadib/bagdhara-bangla-idioms-dataset",
        "usage": "lexical evidence for Bengali idiom figurative and literal meanings",
        "license": "Kaggle public dataset; license not exposed in downloaded files, cite dataset page and verify before Phase 2 package submission",
    },
    {
        "name": "BengaliDictionary",
        "locator": "github:MinhasKamal/BengaliDictionary",
        "usage": "English-Bengali lexical evidence and synonym-like Bengali entries",
        "license": "GPL-3.0",
    },
    {
        "name": "SumanMondal/bengali_qa",
        "locator": "huggingface:SumanMondal/bengali_qa",
        "usage": "external QA retrieval evidence and answer-swap contrastive pairs",
        "license": "Apache-2.0",
    },
    {
        "name": "Additional Hugging Face Bengali QA/literature datasets",
        "locator": "see outputs/external_qa_corpus source_repo/source_file columns",
        "usage": "external QA retrieval evidence, answer-swap pairs, and context-contrastive pairs",
        "license": "mixed: MIT, unknown, and dataset-card licenses requiring citation review",
    },
    {
        "name": "Wikidata Query Service",
        "locator": "https://query.wikidata.org/",
        "usage": "Bangladesh-specific public facts for retrieval evidence and controlled answer-swap pairs",
        "license": "CC0-1.0",
    },
]


REBUILD_COMMANDS = [
    ".venv/bin/python scripts/external_data.py --output-prefix outputs/external_qa_corpus --cache-dir outputs/hf_cache",
    ".venv/bin/python scripts/lexical_data.py --output-prefix outputs/lexical_evidence_corpus --cache-dir outputs/source_cache --merge-with outputs/external_qa_corpus.jsonl --merged-output-prefix outputs/merged_evidence_corpus",
    ".venv/bin/python scripts/wikidata_facts.py --limit 1200 --output-prefix outputs/wikidata_bangladesh_corpus --pairs-prefix outputs/wikidata_bangladesh_pairs --cache-dir outputs/source_cache/wikidata --sleep 0.5",
    ".venv/bin/python scripts/context_contrastive_pairs.py --corpus outputs/external_qa_corpus.jsonl --output-prefix outputs/external_context_contrastive_pairs --max-source-rows 120000 --max-per-label 60000",
    ".venv/bin/python scripts/evidence_index.py build --corpus outputs/external_qa_corpus.jsonl --index-dir outputs/evidence_index/external_qa_char_tfidf --max-features 200000 --min-df 2",
    ".venv/bin/python scripts/evidence_index.py build --corpus outputs/lexical_evidence_corpus.jsonl --index-dir outputs/evidence_index/lexical_char_tfidf --max-features 200000 --min-df 2",
    ".venv/bin/python scripts/evidence_index.py build --corpus outputs/wikidata_bangladesh_corpus.jsonl --index-dir outputs/evidence_index/wikidata_bangladesh_char_tfidf --max-features 120000 --min-df 1",
    ".venv/bin/python scripts/evidence_index.py retrieve --index-dir outputs/evidence_index/lexical_char_tfidf --split test --output outputs/evidence_test_noctx_lexical_exact.jsonl --top-k 5 --batch-size 64 --query-mode prompt --inject-exact-question",
    ".venv/bin/python scripts/evidence_index.py retrieve --index-dir outputs/evidence_index/wikidata_bangladesh_char_tfidf --split test --output outputs/evidence_test_noctx_wikidata_bangladesh.jsonl --top-k 5 --batch-size 64 --query-mode prompt",
    ".venv/bin/python scripts/evidence_controlled_pairs.py --lexical-corpus outputs/lexical_evidence_corpus.jsonl --output outputs/lexical_controlled_pairs_80k.jsonl --max-per-label 40000",
    ".venv/bin/python scripts/train_portable_verifier.py --max-rows-per-file 80000 --max-features 90000 --model-output outputs/portable_verifier.joblib --submission-output outputs/submission_portable_verifier.csv --metadata-output outputs/portable_verifier_metadata.json",
    ".venv/bin/python scripts/train_portable_verifier.py --train-file outputs/lexical_controlled_pairs_80k.jsonl --train-file outputs/external_qa_pairs.jsonl --train-file outputs/synthetic_plus_curated.jsonl --max-rows-per-file 50000 --max-features 100000 --skip-cv --fixed-threshold 0.58 --model-output outputs/portable_verifier_externalqa.joblib --submission-output outputs/submission_portable_verifier_externalqa.csv --metadata-output outputs/portable_verifier_externalqa_metadata.json",
    ".venv/bin/python scripts/train_portable_verifier.py --train-file outputs/lexical_controlled_pairs_80k.jsonl --train-file outputs/external_qa_pairs.jsonl --train-file outputs/wikidata_bangladesh_pairs.jsonl --train-file outputs/synthetic_plus_curated.jsonl --max-rows-per-file 50000 --max-features 110000 --skip-cv --fixed-threshold 0.58 --model-output outputs/portable_verifier_externalqa_wikidata.joblib --submission-output outputs/submission_portable_verifier_externalqa_wikidata.csv --metadata-output outputs/portable_verifier_externalqa_wikidata_metadata.json",
    ".venv/bin/python scripts/train_portable_verifier.py --train-file outputs/lexical_controlled_pairs_80k.jsonl --train-file outputs/external_qa_pairs.jsonl --train-file outputs/wikidata_bangladesh_pairs.jsonl --train-file outputs/external_context_contrastive_pairs.jsonl --train-file outputs/synthetic_plus_curated.jsonl --max-rows-per-file 50000 --max-features 120000 --skip-cv --fixed-threshold 0.57 --model-output outputs/portable_verifier_context_contrastive.joblib --submission-output outputs/submission_portable_verifier_context_contrastive.csv --metadata-output outputs/portable_verifier_context_contrastive_metadata.json",
    ".venv/bin/python scripts/apply_lexical_overrides.py --split test --evidence-jsonl outputs/evidence_test_noctx_lexical_exact.jsonl --base-submission outputs/submission_mlx_qwen36_27b_noctx_fact_no_context.csv --threshold 0.4 --output outputs/submission_mlx_qwen36_27b_noctx_fact_lexical_override.csv --trace-output outputs/lexical_overrides_test.json",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, root: Path) -> dict:
    rel = path.relative_to(root).as_posix()
    return {
        "path": rel,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def collect(paths: list[str], required: bool) -> tuple[list[dict], list[str]]:
    records = []
    missing = []
    for raw in paths:
        path = ROOT / raw
        if not path.exists():
            missing.append(raw)
            if required:
                continue
        else:
            records.append(file_record(path, ROOT))
    return records, missing


def write_archive(archive_path: Path, manifest_path: Path, records: list[dict]) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(manifest_path, arcname=manifest_path.relative_to(ROOT).as_posix())
        for record in records:
            tar.add(ROOT / record["path"], arcname=record["path"])
    print("wrote", archive_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(OUTPUT_DIR / "kaggle_offline_artifact_manifest.json"))
    parser.add_argument("--archive")
    parser.add_argument("--include-candidates", action="store_true")
    args = parser.parse_args()

    required_records, missing_required = collect(REQUIRED_ARTIFACTS, required=True)
    script_records, missing_scripts = collect(REPRO_SCRIPTS, required=True)
    candidate_records, missing_candidates = collect(PHASE1_CANDIDATES, required=False)
    phase2_records, missing_phase2 = collect(PHASE2_ARTIFACTS, required=False)
    if missing_required or missing_scripts:
        raise SystemExit(
            "missing required files:\n"
            + "\n".join(missing_required + missing_scripts)
        )

    archive_records = required_records + phase2_records + script_records
    if args.include_candidates:
        archive_records += candidate_records

    manifest = {
        "name": "bengali_hallucination_evidence_artifacts",
        "created_for": "Kaggle-offline packaging and Phase 1 candidate reproducibility",
        "phase2_constraints": {
            "offline": True,
            "open_weights_only": True,
            "target_gpu": "Kaggle 2xT4 or P100",
            "runtime_limit_hours": 9,
            "model_storage_limit_gb": 50,
        },
        "important_caveat": (
            "The deterministic router exactly reproduces v85 when v85 is supplied as the Phase 1 base. "
            "For unseen Phase 2 rows, unresolved decisions still require the packaged compact verifier "
            "and/or a Kaggle-compatible open-weight Qwen/Gemma selective judge."
        ),
        "sources": SOURCE_NOTES,
        "rebuild_commands": REBUILD_COMMANDS,
        "required_artifacts": required_records,
        "repro_scripts": script_records,
        "phase1_candidates": candidate_records,
        "phase2_artifacts": phase2_records,
        "missing_optional_phase2_artifacts": missing_phase2,
        "missing_optional_candidates": missing_candidates,
        "total_required_bytes": sum(item["bytes"] for item in required_records),
        "total_script_bytes": sum(item["bytes"] for item in script_records),
    }

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", manifest_path)
    print("required artifacts", len(required_records), "bytes", manifest["total_required_bytes"])
    print("scripts", len(script_records), "bytes", manifest["total_script_bytes"])
    print("candidates", len(candidate_records), "missing optional", len(missing_candidates))

    if args.archive:
        write_archive(Path(args.archive), manifest_path, archive_records)


if __name__ == "__main__":
    main()
