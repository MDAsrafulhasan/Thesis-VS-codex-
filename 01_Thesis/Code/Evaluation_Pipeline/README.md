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

- `mock`: deterministic always-No baseline for primary evaluation and score-7 baseline for dialect evaluation.
- `transformers`: loads each value passed to `--models` as a Hugging Face causal language model.
- `ollama`: sends requests to a local Ollama endpoint; use `--ollama_endpoint` when the endpoint differs from the default.

The splitter and mock evaluator use `numpy`, `pandas`, and `scikit-learn`. Transformers inference also needs `torch` and `transformers`; `accelerate` is optional and enables automatic multi-device placement. Without it, the model is loaded onto one CUDA device when available, otherwise CPU. An Ollama run requires an available local model and a running Ollama server.

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
