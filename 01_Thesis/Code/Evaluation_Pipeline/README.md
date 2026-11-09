# Evaluation pipeline guide

## Files

- `make_splits.py` normalizes the supplied raw Track-A/Track-B files, validates schemas and row counts, and writes deterministic stratified splits. It also generates exact-schema mock primary and dialect datasets.
- `eval.py` supports primary binary hallucination judging and dialect score evaluation. It provides deterministic decoding defaults, bilingual verdict parsing, repetition-collapse flags, per-row JSONL checkpoints, resume behavior, BHS weight sensitivity, task/type reports, CCC, and CBS.
- `test_pipeline.py` is the standalone end-to-end verification script.

## Splitter commands

Normalize the supplied data and rebuild the three primary splits:

```powershell
python "01_Thesis\Code\Evaluation_Pipeline\make_splits.py" `
  --input_csv "01_Thesis\Data\BenHalluEval\Raw" `
  --output_dir "01_Thesis\Data\BenHalluEval\Processed\Splits" `
  --seed 42 `
  --mode benhallu
```

Generate and split mock primary data:

```powershell
python "01_Thesis\Code\Evaluation_Pipeline\make_splits.py" `
  --generate_mock_data `
  --output_dir "01_Thesis\Data\BenHalluEval\Mock" `
  --seed 42 `
  --mode benhallu
```

The mock command writes canonical CSVs to `Mock/mock_data/` and the primary split files directly to `Mock/`.

## Evaluator backends

- `mock`: deterministic synthetic baselines (`mock`, `always_yes`, `always_no`, `mock_score`, `mock_collapse`).
- `openrouter`: remote inference against OpenRouter (supports free models like `meta-llama/llama-3.3-70b-instruct:free`).
- `openai`: remote inference against any OpenAI-compatible `/v1/chat/completions` endpoint.
- `ollama`: sends requests to a local Ollama endpoint; use `--base_url` when the endpoint differs from the default `http://localhost:11434/api/generate`.
- `transformers`: loads each value passed to `--model_name` or `--models` as a Hugging Face causal language model.

The splitter and mock evaluator use `numpy`, `pandas`, and `scikit-learn`. Transformers inference also needs `torch` and `transformers`; `accelerate` is optional and enables automatic multi-device placement. Without it, the model is loaded onto one CUDA device when available, otherwise CPU. An Ollama run requires an available local model and a running Ollama server. Cloud evaluation via OpenRouter requires an `OPENROUTER_API_KEY`.

Example using the mock backend:

```powershell
python "01_Thesis\Code\Evaluation_Pipeline\eval.py" `
  --input_csv "01_Thesis\Data\BenHalluEval\Processed\Splits\benhallu_micro_debug_120.csv" `
  --output_dir "01_Thesis\Results\BenHalluEval" `
  --backend mock `
  --models mock-baseline `
  --prompt_modes zero_shot cot
```

For a real local Transformers model, set `--backend transformers` and pass one or more Hugging Face model IDs or local model directories. For Ollama, pass the local model name and `--backend ollama`.

## Outputs and reporting

For each combination, the evaluator appends one JSON object per row to `OUTPUT/outputs/{model}_{prompt_mode}_{split}.jsonl`, flushes the record immediately, repairs a torn final line on restart, and resumes IDs with completed inference. Failed inference rows are recorded and retried on a later run.

Primary results include:

- `table1_overall_and_per_task_bhs.csv/.md` — overall and task metrics, collapse and parse-failure rates, latency, and support counts;
- `table2_granular_hallucination_types.csv/.md` — Track-B false-negative rates for all 12 types;
- `table3_cot_delta_ablation.csv/.md` — per-task `BHS_CoT - BHS_ZeroShot`;
- `table4_weight_sensitivity.csv/.md` — BHS for all nine `(w_a, w_b)` pairs and ranks.

Dialect input produces `table_dialect_ccc_cbs.csv/.md`, comparing human/primary scores with model scores and measuring the anchor-to-Chittagong performance drop.

## Updated five-stage detector

`update_evaluation.py` implements the detector architecture in the supplied diagrams. It removes every training row whose normalized question group appears in the holdout, then extracts claims, retrieves/reranks evidence, computes NLI and lexical signals, obtains Qwen first-token binary logits, and trains a grouped logistic-regression ensemble with training-only thresholds.

Run the 20-row 10+10 holdout against the leakage-filtered 10% training split:

```powershell
python "01_Thesis\Code\Evaluation_Pipeline\update_evaluation.py" `
  --train_csv "01_Thesis\Data\BenHalluEval\Processed\Splits\benhallu_stratified_10pct.csv" `
  --eval_csv "01_Thesis\Data\BenHalluEval\Processed\Splits\track_a_b_first10_each.csv" `
  --baseline_dir "01_Thesis\Results\BenHalluEval\track_a_b_first10_each_baseline" `
  --output_dir "01_Thesis\Results\BenHalluEval\update_pipeline_20row" `
  --cache_dir "$env:TEMP\benhallu_update_models\hub" `
  --ollama_url "http://127.0.0.1:11436" `
  --device cpu
```

The run downloads these public checkpoints into the selected Hugging Face cache: `intfloat/multilingual-e5-large`, `BAAI/bge-reranker-v2-m3`, `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`, and `FacebookAI/xlm-roberta-base`. Ollama must be serving `qwen3:1.7b` and `gemma3:1b` at the specified endpoint. Change the model flags or endpoint if your local names differ.

The main outputs are `update_pipeline_report.md`, `update_pipeline_bhs.csv`, `pre_post_matched_comparison.csv`, `meta_predictions.csv`, and `run_manifest.json`. The report separates the updated detector's full 20-row BHS from paired baseline comparisons over each baseline's successfully parsed IDs. The intermediate JSONL is appended after each row/stage and can resume after interruption. The diagram omits some implementation details, so the script records its explicit feature, claim, retrieval, and threshold defaults in the manifest. Its prediction is a detector verdict; it does not rewrite candidate answers.

For a quick pilot, add `--train_sample_size 20`. It deterministically selects 10 Track-A and 10 Track-B rows from the leakage-filtered training data, while retaining the same 20-row (10+10) evaluation holdout. Treat that score as preliminary; omit the option for the full 1,595-row training run.

## Corrected v2 evaluation (separate, experimental)

`update_pipeline_evaluation.py` is a new entry point; it does not change historical runs or `update_evaluation.py`. Its accompanying `pipeline_v2_math.py` implements corrected smooth extrema, conflict routing, polarity-specific RRF, BHS, finite-sample conformal sets, logit-level CAD and a grouped/nested lexical/meta learner. `pipeline_v2_features.py` supplies offline local E5/BM25, BGE and per-evidence mDeBERTa diagnostics.

Run from the workspace root (single-line command works in PowerShell or sh):

```sh
python 01_Thesis/Code/Evaluation_Pipeline/update_pipeline_evaluation.py run --profile neural-untrained --input-csv 01_Thesis/Data/BenHalluEval/Processed/Splits/track_a_b_first10_each.csv --output-dir 01_Thesis/Results/BenHalluEval/corrected_v2_20row_new_run --models bangla-ai-1.7b-q4 qwen3:1.7b qwen2.5:1.5b gemma3:1b
```

The default HF cache is the local temporary `benhallu_update_models/hub` directory; override `--cache-dir` if needed. Models must already exist. No checkpoint downloads, external evidence APIs or cloud inference are used. Ollama is explicitly CPU-only because the inspected GPU runtime had a CUDA/PTX incompatibility.

### Profiles and scope

- `judge-only`: ablated local model judging without neural diagnostics.
- `neural-untrained`: neural features and routing diagnostics, followed by each model's two-label score with a fixed 0.5 threshold. **Without `--fast-path`, features do not change verdicts.** No learned meta/calibration performance is claimed.
- `neural-untrained --fast-path`: exploratory rule-based cascade. Eligible NLI routes bypass judges; report fast-path coverage and remaining model-score coverage separately. This is not a trained meta-classifier.
- `trained`: requires a reviewed, source-disjoint artifact per model digest, fixed feature protocol/order, training/development/calibration provenance and explicit `--trust-local-artifacts` consent. Pickle loads executable code; never use untrusted files. The `fit` subcommand takes a JSON bundle with train/dev/calibration records (`features` in the documented 30-column order, `text`, `label`, `source_group`, plus development `regime`), global question/context `screening_keys` and `provenance` containing `source_lineage_review`, `model_digest`, and `neural_protocol` matching the run manifest. It writes a new artifact without overwriting existing artifacts. Source disjointness beyond recorded keys must be reviewed externally.

The 20 target labels and overlapping benchmark evaluation subsets must not be silently used for training/calibration. No approved train/dev/calibration partitions were supplied for the saved v2 pilot, so **the complete trained six-stage architecture has not been evaluated**. Semantic-equivalent transliteration/JSD, QA consistency, symbolic sandboxing and surgical repair are not claimed as executed. CAD exists as a tested mathematical primitive only.

### Artifacts and resumption

`--phase prepare` checkpoints claims/features; `--phase judge` resumes judging. A phase change alone does not change the experiment fingerprint. `outputs/shared_features.jsonl` stores label-free neural traces; `outputs/predictions.jsonl` stores exact raw model payloads; `results/` contains BHS CSV/JSON, per-row analysis, task/type breakdowns, weight sensitivity and `REPORT.md`. `run_manifest.json` records hashes, model digests, local revisions, runtime, commands and fixed settings. Interrupted runs retain completed rows and mark partial BHS explicitly. Resume refuses changed code/data/models/runtime; use a new directory after a substantive change. A killed process may leave `run.lock`; verify the recorded PID is no longer running before manually removing it.

When either label is missing from the first-token top-20, the runner records the missing score and may use an exact decoded label. It never invents a probability. Operational BHS penalizes null verdicts on their track; missing/unattempted rows cannot yield a final BHS.

Tests:

```sh
python -m pytest -q 01_Thesis/Code/Evaluation_Pipeline/test_pipeline_v2.py
python 01_Thesis/Code/Evaluation_Pipeline/test_pipeline.py -q
```

See `01_Thesis/Experiments/2026-11-08_corrected_v2_20row/PROTOCOL.md` and its `RESULTS.md` for the registered scope, deviations, exact commands and observed results. The exact executed sources are archived in that experiment's `executed_source/`; subsequent typing and checkpoint-safety hardening changes intentionally invalidate resuming completed runs with current code. Use a new run directory rather than modifying recorded manifests to bypass this guard.
