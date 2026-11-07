from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path

import pandas as pd
import torch


def command_output(command: list[str]) -> str:
    try:
        return subprocess.run(command, check=False, capture_output=True, text=True).stdout.strip()
    except Exception as exc:  # pragma: no cover - diagnostics must keep running
        return f"{type(exc).__name__}: {exc}"


started = time.perf_counter()
input_root = Path("/kaggle/input")
csv_files = sorted(input_root.rglob("*.csv"))
gguf_files = sorted(input_root.rglob("*.gguf"))
test_candidates = [
    path
    for path in csv_files
    if "test" in path.name.lower() and "submission" not in path.name.lower()
]
if not test_candidates:
    raise FileNotFoundError(f"No competition test CSV found under {input_root}")

test_path = max(test_candidates, key=lambda path: path.stat().st_size)
test = pd.read_csv(test_path)
required = {"context", "prompt_bn", "response_bn"}
missing = required - set(test.columns)
if missing:
    raise ValueError(f"Selected {test_path} but required columns are missing: {sorted(missing)}")

repeats = max(1, (5000 + len(test) - 1) // len(test))
synthetic = pd.concat([test] * repeats, ignore_index=True).iloc[:5000].copy()
synthetic["_text_chars"] = (
    synthetic["context"].fillna("").astype(str).str.len()
    + synthetic["prompt_bn"].fillna("").astype(str).str.len()
    + synthetic["response_bn"].fillna("").astype(str).str.len()
)

gpu_rows = []
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    gpu_rows.append(
        {
            "index": index,
            "name": props.name,
            "total_memory_gib": round(props.total_memory / 1024**3, 3),
            "compute_capability": f"{props.major}.{props.minor}",
        }
    )

report = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
    "gpu_count": torch.cuda.device_count(),
    "gpus": gpu_rows,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "nvidia_smi": command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    ),
    "test_path": str(test_path),
    "test_rows": len(test),
    "test_columns": test.columns.tolist(),
    "synthetic_rows": len(synthetic),
    "synthetic_total_text_chars": int(synthetic["_text_chars"].sum()),
    "input_csv_files": [str(path) for path in csv_files],
    "gguf_files": [
        {"path": str(path), "size_gib": round(path.stat().st_size / 1024**3, 3)}
        for path in gguf_files
    ],
    "llama_cli": command_output(["bash", "-lc", "command -v llama-cli || command -v main || true"]),
    "llama_server": command_output(["bash", "-lc", "command -v llama-server || command -v server || true"]),
    "llama_cpp_import": None,
    "elapsed_seconds": round(time.perf_counter() - started, 4),
}

try:
    import llama_cpp

    report["llama_cpp_import"] = {
        "available": True,
        "version": getattr(llama_cpp, "__version__", "unknown"),
        "module": str(Path(llama_cpp.__file__).resolve()),
    }
except Exception as exc:
    report["llama_cpp_import"] = {
        "available": False,
        "error": f"{type(exc).__name__}: {exc}",
    }

if report["gpu_count"] != 2 or not all("T4" in gpu["name"] for gpu in gpu_rows):
    raise RuntimeError(f"Expected Kaggle T4 x2, observed: {gpu_rows}")
if len(gguf_files) != 1:
    raise RuntimeError(f"Expected exactly one attached GGUF model, observed: {gguf_files}")

output_path = Path("/kaggle/working/phase2_t4_smoke.json")
output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
