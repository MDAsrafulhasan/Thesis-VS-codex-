from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "kaggle_notebooks" / "phase2_v85_inference"
TARGET_DIR = ROOT / "kaggle_notebooks" / "phase2_v91_inference"
SOURCE_NOTEBOOK = SOURCE_DIR / "phase2_v85_inference.ipynb"
SOURCE_SCRIPT = SOURCE_DIR / "phase2_v85_inference.py"
TARGET_NOTEBOOK = TARGET_DIR / "phase2_v91_inference.ipynb"
TARGET_SCRIPT = TARGET_DIR / "phase2_v91_inference.py"

REFERENCE_SHA256 = "0fec5d610f4e7b94ab6c46e2094d58e940b0f36781e9fa98d5466d2135474c28"

REPLACEMENTS = (
    ("phase2-runtime-v85", "phase2-runtime-v91"),
    ("kaggle_phase2_runtime_v85", "kaggle_phase2_runtime_v91"),
    (
        "bengali-hallucination-llama-cpp-cuda-b9637",
        "bengali-hallucination-llama-cpp-cuda-b9637-public",
    ),
    ("phase2_v85_inference", "phase2_v91_inference"),
    ("v85 Compliance", "v91 Compliance"),
    ("v85 compliance", "v91 compliance"),
    ("v85 artifact", "v91 artifact"),
    ("v85 output", "v91 output"),
    ("v85", "v91"),
    ("0.966", "0.967"),
    ("label 0: 1,176 · label 1: 1,340", "label 0: 1,181 · label 1: 1,335"),
)

ARGUMENT_ANCHOR = (
    '    "--verified-facts", str(runtime / "resources/verified_fact_answers.json"),\n'
)
V91_ARGUMENTS = (
    '    "--typed-evidence", str(runtime / "resources/typed_evidence_v1.json"),\n'
    '    "--idiom-dataset", str(runtime / "outputs/source_cache/bangla_bagdhara_lrec/'
    'bagdhara_export_done_2026-03-01_010356.json"),\n'
)


def transform(text: str) -> str:
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    if ARGUMENT_ANCHOR not in text:
        raise ValueError("v85 router argument anchor is missing")
    return text.replace(ARGUMENT_ANCHOR, ARGUMENT_ANCHOR + V91_ARGUMENTS, 1)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    script = transform(SOURCE_SCRIPT.read_text(encoding="utf-8"))
    TARGET_SCRIPT.write_text(script, encoding="utf-8")

    notebook = nbformat.read(SOURCE_NOTEBOOK, as_version=4)
    inserted_arguments = False
    for cell in notebook.cells:
        for old, new in REPLACEMENTS:
            cell.source = cell.source.replace(old, new)
        if cell.cell_type == "code":
            if ARGUMENT_ANCHOR in cell.source and not inserted_arguments:
                cell.source = cell.source.replace(
                    ARGUMENT_ANCHOR,
                    ARGUMENT_ANCHOR + V91_ARGUMENTS,
                    1,
                )
                inserted_arguments = True
            cell.outputs = []
            cell.execution_count = None
    if not inserted_arguments:
        raise ValueError("v85 notebook router argument anchor is missing")
    notebook.metadata["huntrix_release"] = {
        "version": "v91",
        "reference_submission": 54836339,
        "reference_public_macro_f1": 0.967,
        "reference_output_sha256": REFERENCE_SHA256,
        "phase1_rows": 2516,
        "phase1_label_counts": {"0": 1181, "1": 1335},
    }
    nbformat.write(notebook, TARGET_NOTEBOOK)

    metadata = {
        "id": "seyamalam/bengali-hallucination-v91-compliance-inference",
        "title": "Bengali Hallucination v91 Compliance Inference",
        "code_file": TARGET_NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["bengali", "hallucination-detection", "reproducibility"],
        "dataset_sources": [
            "seyamalam/bengali-hallucination-phase2-runtime-v91",
            "seyamalam/bengali-hallucination-llama-cpp-cuda-b9637-public",
        ],
        "competition_sources": ["bengali-hallucination"],
        "model_sources": ["snowrab/qwen3-6-27b-q4-k-p/pyTorch/default/1"],
    }
    (TARGET_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "script": str(TARGET_SCRIPT),
                "script_sha256": sha256(TARGET_SCRIPT),
                "notebook": str(TARGET_NOTEBOOK),
                "notebook_sha256": sha256(TARGET_NOTEBOOK),
                "metadata": str(TARGET_DIR / "kernel-metadata.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
