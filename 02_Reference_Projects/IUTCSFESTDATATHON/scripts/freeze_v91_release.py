from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import nbformat
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = "0fec5d610f4e7b94ab6c46e2094d58e940b0f36781e9fa98d5466d2135474c28"
EXPECTED_COUNTS = {0: 1181, 1: 1335}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def executable_cells_sha256(path: Path) -> str:
    notebook = nbformat.read(path, as_version=4)
    source = "\n\n# --- CELL ---\n\n".join(
        cell.source.rstrip() for cell in notebook.cells if cell.cell_type == "code"
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def validate_run(path: Path) -> tuple[dict, dict]:
    submission_path = path / "submission.csv"
    report_path = path / "phase2_runtime_report.json"
    if not submission_path.is_file() or not report_path.is_file():
        raise FileNotFoundError(f"Missing Kaggle output in {path}")
    frame = pd.read_csv(submission_path)
    if list(frame.columns) != ["id", "label"] or len(frame) != 2516:
        raise ValueError(f"Invalid Phase 1 submission shape in {path}")
    if frame.id.nunique() != 2516 or not set(frame.label.astype(int)).issubset({0, 1}):
        raise ValueError(f"Invalid IDs or labels in {path}")
    counts = frame.label.astype(int).value_counts().sort_index().to_dict()
    digest = sha256_file(submission_path)
    if digest != EXPECTED_SHA256 or counts != EXPECTED_COUNTS:
        raise ValueError(
            f"Run {path} did not reproduce v91: sha={digest}, counts={counts}"
        )
    return json.loads(report_path.read_text(encoding="utf-8")), {
        "output_sha256": digest,
        "label_counts": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate two Kaggle runs and freeze the public v91 release.")
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--kaggle-notebook-version", type=int, required=True)
    parser.add_argument("--runtime-dataset-version", type=int, required=True)
    parser.add_argument("--cuda-dataset-version", type=int, required=True)
    args = parser.parse_args()
    if len(args.run) != 2:
        raise ValueError("Exactly two --run directories are required")

    reports, results = [], []
    for path in args.run:
        report, result = validate_run(path)
        reports.append(report)
        results.append(result)
    first_submission = args.run[0] / "submission.csv"
    second_submission = args.run[1] / "submission.csv"
    if first_submission.read_bytes() != second_submission.read_bytes():
        raise ValueError("The two Kaggle submission CSVs are not byte-identical")

    kernel_dir = ROOT / "kaggle_notebooks" / "phase2_v91_inference"
    notebook = kernel_dir / "phase2_v91_inference.ipynb"
    source = kernel_dir / "phase2_v91_inference.py"
    release_dir = ROOT / "release" / "phase2-v91-public-submission"
    runtime_dir = ROOT / "outputs" / "kaggle_dataset_phase2_runtime_v91"
    cuda_binary = (
        ROOT / "outputs" / "kaggle_dataset_llama_cpp_cuda_b9637_public" / "llama-server"
    )
    runtime_manifest = runtime_dir / "kaggle_phase2_runtime_v91.manifest.json"

    manifest = {
        "schema_version": 1,
        "frozen_at": datetime.now(ZoneInfo("Asia/Dhaka")).isoformat(timespec="seconds"),
        "freeze_policy": (
            "Do not update the notebook, runtime datasets, CUDA runtime, or model version "
            "after the organizer submission deadline."
        ),
        "source_repository": {
            "visibility": "private",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
        "public_submission_repository": {
            "repository": "Seyamalam/bengali-hallucination-v91-submission",
            "visibility": "public",
        },
        "kaggle_notebook": {
            "slug": "seyamalam/bengali-hallucination-v91-compliance-inference",
            "url": "https://www.kaggle.com/code/seyamalam/bengali-hallucination-v91-compliance-inference",
            "validated_versions": sorted(
                {1, 2, args.kaggle_notebook_version}
            ),
            "submitted_version": args.kaggle_notebook_version,
            "hardware": "Tesla T4 x2",
            "internet_enabled": False,
            "notebook_sha256": sha256_file(notebook),
            "canonical_executable_cells_sha256": executable_cells_sha256(notebook),
            "python_source_sha256": sha256_file(source),
        },
        "competition_input": {
            "literal_path": "/kaggle/input/competitions/bengali-hallucination/test set.csv",
            "dynamic_path_discovery": False,
        },
        "runtime_dataset": {
            "dataset_id": 11271419,
            "slug": "seyamalam/bengali-hallucination-phase2-runtime-v91",
            "url": "https://www.kaggle.com/datasets/seyamalam/bengali-hallucination-phase2-runtime-v91",
            "visibility": "public",
            "version": args.runtime_dataset_version,
            "content_manifest_sha256": sha256_file(runtime_manifest),
            "contains_phase1_prediction_vector": False,
        },
        "cuda_runtime_dataset": {
            "dataset_id": 11271407,
            "slug": "seyamalam/bengali-hallucination-llama-cpp-cuda-b9637-public",
            "url": (
                "https://www.kaggle.com/datasets/seyamalam/"
                "bengali-hallucination-llama-cpp-cuda-b9637-public"
            ),
            "visibility": "public",
            "version": args.cuda_dataset_version,
            "llama_cpp_release": "b9637",
            "llama_cpp_commit": "aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3",
            "binary_bytes": cuda_binary.stat().st_size,
            "binary_sha256": sha256_file(cuda_binary),
        },
        "model": {
            "slug": "snowrab/qwen3-6-27b-q4-k-p/pyTorch/default/1",
            "url": "https://www.kaggle.com/models/snowrab/qwen3-6-27b-q4-k-p/pyTorch/default/1",
            "version": 1,
            "visibility": "public",
            "license": (
                "Official Qwen3.6 base: Apache-2.0; Kaggle quantization "
                "publisher label: MIT"
            ),
            "file": "Qwen3.6-27B-Q4_K_P.gguf",
        },
        "reported_execution": {
            "hardware": "Tesla T4 x2",
            "runtime_hours_for_form": 0.17,
            "run_seconds": [float(report["total_seconds"]) for report in reports],
            "phase1_rows": 2516,
            "same_mix_5000_row_projection_seconds": [
                float(report["estimated_5000_seconds_at_same_mix"]) for report in reports
            ],
            "within_nine_hour_limit": all(
                float(report["estimated_5000_seconds_at_same_mix"]) < 9 * 3600
                for report in reports
            ),
        },
        "reproduction": {
            "reference_submission_id": 54836339,
            "reference_public_macro_f1": 0.967,
            "output_sha256": EXPECTED_SHA256,
            "rows": 2516,
            "label_counts": {"0": 1181, "1": 1335},
            "byte_identical_reruns": True,
        },
    }
    freeze_path = kernel_dir / "FREEZE_MANIFEST.json"
    freeze_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    release_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(notebook, release_dir / "inference.ipynb")
    shutil.copy2(freeze_path, release_dir / "FREEZE_MANIFEST.json")
    shutil.copy2(ROOT / "docs" / "DATA_SOURCES.md", release_dir / "DATA_SOURCES.md")
    shutil.copy2(ROOT / "LICENSE", release_dir / "LICENSE")
    shutil.copy2(
        ROOT / "THIRD_PARTY_NOTICES.md",
        release_dir / "THIRD_PARTY_NOTICES.md",
    )
    shutil.copy2(
        ROOT / "docs" / "COMPLIANCE_REVIEW.md",
        release_dir / "METHODOLOGY_DISCLOSURE.md",
    )
    shutil.copy2(
        ROOT / "output" / "pdf" / "Huntrix_Technical_Report.pdf",
        release_dir / "Huntrix_Technical_Report.pdf",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
