# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd
import torch


INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
TEST_PATH = Path("/kaggle/input/competitions/bengali-hallucination/test set.csv")
RUNTIME = Path(
    "/kaggle/input/datasets/seyamalam/bengali-hallucination-phase2-runtime-v85/"
    "kaggle_phase2_runtime_v85"
)
MODEL_PATH = Path(
    "/kaggle/input/models/snowrab/qwen3-6-27b-q4-k-p/pytorch/default/1/"
    "Qwen3.6-27B-Q4_K_P.gguf"
)
SERVER_PATH = Path(
    "/kaggle/input/datasets/seyamalam/bengali-hallucination-llama-cpp-cuda-b9637/"
    "llama-server"
)


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, check=True, text=True, **kwargs)


def wait_for_server(process: subprocess.Popen, timeout: float = 300.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited during startup with {process.returncode}")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(2)
    raise TimeoutError("llama-server did not become healthy")


started = time.perf_counter()
if torch.cuda.device_count() != 2 or not all("T4" in torch.cuda.get_device_name(i) for i in range(2)):
    raise RuntimeError(
        f"This submission is validated for T4 x2; observed "
        f"{[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}"
    )

test_path = TEST_PATH
test = pd.read_csv(TEST_PATH)
required = {"id", "context", "prompt_bn", "response_bn"}
if not required.issubset(test.columns):
    raise ValueError(f"Bad test file {test_path}: missing {sorted(required - set(test.columns))}")

runtime = RUNTIME
scripts = runtime / "scripts"
pipeline = scripts / "phase2_pipeline.py"
model = MODEL_PATH
server = WORK / "llama-server"
for required_asset in (pipeline, model, SERVER_PATH):
    if not required_asset.is_file():
        raise FileNotFoundError(f"Missing pinned runtime asset: {required_asset}")
shutil.copy2(SERVER_PATH, server)
server.chmod(0o755)

router_output = WORK / "router_submission.csv"
router_trace = WORK / "router_trace.json"
router_command = [
    sys.executable, str(pipeline),
    "--input", str(test_path),
    "--output", str(router_output),
    "--trace", str(router_trace),
    "--bank", str(runtime / "outputs/source_answer_bank_complete_bcs.jsonl"),
    "--bank", str(runtime / "outputs/source_answer_bank_farhan_qdoc.jsonl"),
    "--bank", str(runtime / "outputs/source_answer_bank_bcs_all.jsonl"),
    "--fuzzy-bank", str(runtime / "outputs/source_answer_bank.jsonl"),
    "--lexical-bank", str(runtime / "outputs/lexical_evidence_corpus.jsonl"),
    "--benhallu-gold", str(runtime / "outputs/source_cache/benhallueval_data/qa_gt_1000.csv"),
    "--benhallu-hallucinations", str(runtime / "outputs/source_cache/benhallueval_data/qa_4000.csv"),
    "--verified-facts", str(runtime / "resources/verified_fact_answers.json"),
    "--law-corpus", str(runtime / "outputs/phase2_data/hf_legal_acts_evidence.jsonl"),
    "--law-corpus", str(runtime / "resources/legacy_law_acts.json"),
]
environment = {**os.environ, "PYTHONPATH": str(scripts)}
run(router_command, env=environment)
router_seconds = time.perf_counter() - started

server_log_path = WORK / "llama_server.log"
server_log = server_log_path.open("w", encoding="utf-8")
server_command = [
    str(server),
    "--model", str(model),
    "--host", "127.0.0.1",
    "--port", "8080",
    "--n-gpu-layers", "99",
    "--split-mode", "layer",
    "--tensor-split", "1,1",
    "--ctx-size", "8192",
    "--parallel", "1",
    "--batch-size", "1024",
    "--ubatch-size", "256",
    "--flash-attn", "on",
    "--reasoning-budget", "0",
    "--chat-template-kwargs", '{"enable_thinking":false}',
    "--cache-type-k", "q8_0",
    "--cache-type-v", "q8_0",
    "--no-webui",
]
print("+", " ".join(server_command), flush=True)
server_process = subprocess.Popen(server_command, stdout=server_log, stderr=subprocess.STDOUT, text=True)
try:
    wait_for_server(server_process)
    server_ready_seconds = time.perf_counter() - started
    final_output = WORK / "submission.csv"
    judge_trace = WORK / "judge_trace.json"
    run(
        [
            sys.executable, str(scripts / "gguf_selective_judge.py"),
            "--input", str(test_path),
            "--base-submission", str(router_output),
            "--router-trace", str(router_trace),
            "--output", str(final_output),
            "--trace", str(judge_trace),
            "--workers", "1",
            "--row-batch-size", "4",
        ],
        env=environment,
    )
finally:
    server_process.terminate()
    try:
        server_process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        server_process.kill()
    server_log.close()

submission = pd.read_csv(WORK / "submission.csv")
if list(submission.columns) != ["id", "label"]:
    raise ValueError(f"Invalid output columns: {submission.columns.tolist()}")
if not submission.id.astype(int).equals(test.id.astype(int)):
    raise ValueError("Output IDs do not match test IDs in order")
if submission.label.isna().any() or not set(submission.label.astype(int)).issubset({0, 1}):
    raise ValueError("Output labels must be non-null binary integers")

router_report = json.loads(router_trace.read_text(encoding="utf-8"))
judge_report = json.loads((WORK / "judge_trace.json").read_text(encoding="utf-8"))
elapsed = time.perf_counter() - started
report = {
    "rows": len(test),
    "gpu_names": [torch.cuda.get_device_name(i) for i in range(2)],
    "model": str(model),
    "model_gib": round(model.stat().st_size / 1024**3, 3),
    "router_seconds": router_seconds,
    "model_ready_seconds": server_ready_seconds,
    "total_seconds": elapsed,
    "uncertain_rows": router_report["summary"]["uncertain_rows"],
    "judge_changed": judge_report["changed"],
    "label_counts": submission.label.astype(int).value_counts().sort_index().to_dict(),
    "estimated_5000_seconds_at_same_mix": elapsed * 5000 / len(test),
}
(WORK / "phase2_runtime_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(report, ensure_ascii=False, indent=2))
