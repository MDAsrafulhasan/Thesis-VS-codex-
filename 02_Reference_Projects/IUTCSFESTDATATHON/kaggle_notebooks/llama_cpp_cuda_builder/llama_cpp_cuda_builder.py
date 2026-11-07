from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
import time
from pathlib import Path


LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp.git"
LLAMA_CPP_COMMIT = "aedb2a5"  # llama.cpp release b9637, 2026-06-14
WORK = Path("/kaggle/working")
# Keep the 200+ MB checkout out of saved notebook outputs, especially on failure.
SOURCE = Path("/tmp/llama.cpp")
PACKAGE = WORK / "llama_cpp_cuda_b9637"


def run(command: list[str], cwd: Path | None = None) -> str:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True)
    if completed.stdout:
        print(completed.stdout[-4000:], flush=True)
    if completed.stderr:
        print(completed.stderr[-4000:], flush=True)
    completed.check_returncode()
    return completed.stdout.strip()


started = time.perf_counter()
run(["git", "clone", "--filter=blob:none", LLAMA_CPP_REPO, str(SOURCE)])
run(["git", "checkout", LLAMA_CPP_COMMIT], cwd=SOURCE)
resolved_commit = run(["git", "rev-parse", "HEAD"], cwd=SOURCE)

build = SOURCE / "build"
# Kaggle's image has the runtime driver but omits the unversioned development
# symlink, so CMake's FindCUDAToolkit does not create CUDA::cuda_driver.
driver_candidates = [
    path for path in (
        Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1"),
        Path("/usr/local/cuda/compat/libcuda.so.1"),
    ) if path.exists()
]
if not driver_candidates:
    raise FileNotFoundError("libcuda.so.1")
cuda_driver = driver_candidates[0]
cmake_hook = Path("/tmp/kaggle_cuda_driver.cmake")
cmake_hook.write_text(
    "if(NOT TARGET CUDA::cuda_driver)\n"
    "  add_library(CUDA::cuda_driver SHARED IMPORTED GLOBAL)\n"
    f"  set_target_properties(CUDA::cuda_driver PROPERTIES IMPORTED_LOCATION \"{cuda_driver}\")\n"
    "endif()\n",
    encoding="utf-8",
)
run(
    [
        "cmake",
        "-S",
        str(SOURCE),
        "-B",
        str(build),
        "-DGGML_CUDA=ON",
        "-DGGML_CUDA_FA_ALL_QUANTS=ON",
        "-DCMAKE_CUDA_ARCHITECTURES=75",
        f"-DCMAKE_PROJECT_INCLUDE={cmake_hook}",
        "-DLLAMA_CURL=OFF",
        "-DLLAMA_BUILD_SERVER=ON",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
)
run(
    [
        "cmake",
        "--build",
        str(build),
        "--config",
        "Release",
        "--parallel",
        "4",
        "--target",
        "llama-cli",
        "llama-server",
        "llama-bench",
    ]
)

bin_dir = PACKAGE / "bin"
bin_dir.mkdir(parents=True, exist_ok=True)
binaries = {}
for name in ["llama-cli", "llama-server", "llama-bench"]:
    source = build / "bin" / name
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = bin_dir / name
    shutil.copy2(source, destination)
    destination.chmod(0o755)
    binaries[name] = {
        "bytes": destination.stat().st_size,
        "ldd": run(["ldd", str(destination)]),
    }

version = run([str(bin_dir / "llama-cli"), "--version"])
report = {
    "repository": LLAMA_CPP_REPO,
    "requested_commit": LLAMA_CPP_COMMIT,
    "resolved_commit": resolved_commit,
    "version": version,
    "binaries": binaries,
    "nvidia_smi": run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    ),
    "nvcc": run(["nvcc", "--version"]),
    "build_seconds": round(time.perf_counter() - started, 3),
}
(PACKAGE / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

archive_path = WORK / "llama_cpp_cuda_b9637.tar.gz"
with tarfile.open(archive_path, "w:gz", compresslevel=6) as archive:
    archive.add(PACKAGE, arcname=PACKAGE.name)

print(json.dumps({**report, "archive": str(archive_path), "archive_bytes": archive_path.stat().st_size}, indent=2))
