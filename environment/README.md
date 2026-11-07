# Evaluation environment

The active suite requires Python, NumPy, pandas, scikit-learn, and pytest. Its HTTP adapter and the workspace catalog use the Python standard library.

[requirements-evaluation.txt](requirements-evaluation.txt) records the direct package versions observed in the passing local environment: **Windows / Python 3.13.9**. It is not a complete transitive lockfile or a promise of bitwise GPU reproducibility. For a published run, additionally save `python --version` and `python -m pip freeze` in that experiment's record.

## Create an isolated environment

From the workspace root:

```sh
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r environment/requirements-evaluation.txt
.venv/Scripts/python.exe evaluation/test_pipeline.py -q
```

These explicit interpreter paths work on Windows without activating a PowerShell script. Installation requires package downloads unless dependencies are cached. On Linux/macOS use `.venv/bin/python` instead.

If the current interpreter already passes the suite, no environment installation is necessary just to organize, search, or inspect files:

```sh
python tools/workspace.py verify
python tools/workspace.py find "retrieval"
```

## Keep reference execution separate

- Do not install Huntrix's mixed GPU/Apple MLX requirements into the evaluation environment by default.
- The reference notebooks target Kaggle/Linux, dual T4 GPUs, external data, and model/runtime bundles that are not included locally.
- Use a separate environment or Kaggle session for each reference reproduction. Follow its original runbook and license terms, then save the exact resolved environment with that experiment.
- Merely opening a notebook is not a reason to run all cells. Inspect source and file dependencies first.
- API keys belong in environment variables, never in notebooks, manifests, source files, or experiment logs. Real remote inference may incur charges and sends prompt data to the configured endpoint.
