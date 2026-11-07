# Thesis workspace

## Contents

- `Data/BenHalluEval/Raw/` contains the original Track-A and Track-B CSVs.
- `Data/BenHalluEval/Processed/Splits/` contains the validated 120-row debug, 1,600-row stratified, and 16,000-row full datasets.
- `Code/Evaluation_Pipeline/` contains the dataset normalizer/splitter, model evaluation harness, and standalone verification script.
- `Results/` is the recommended output location for inference checkpoints and reports. The evaluator creates its `outputs/` and `results/` subfolders when run.

## Rebuild the splits

Run these commands from the workspace root in PowerShell:

```powershell
python "01_Thesis\Code\Evaluation_Pipeline\make_splits.py" `
  --input_csv "01_Thesis\Data\BenHalluEval\Raw" `
  --output_dir "01_Thesis\Data\BenHalluEval\Processed\Splits" `
  --seed 42 `
  --mode benhallu
```

The command validates the 16,000-row input before writing the three splits. It does not modify the raw CSVs.

## Verify the pipeline

```powershell
python "01_Thesis\Code\Evaluation_Pipeline\test_pipeline.py"
```

The standalone checks generate mock primary and dialect data in a temporary directory, verify split sizes and stratification, check BHS and CCC behavior, exercise English and Bengali repetition-collapse parsing, and check inference checkpoint recovery.

## Run an evaluation

For a deterministic wiring check using the built-in mock judge:

```powershell
python "01_Thesis\Code\Evaluation_Pipeline\eval.py" `
  --input_csv "01_Thesis\Data\BenHalluEval\Processed\Splits\benhallu_micro_debug_120.csv" `
  --output_dir "01_Thesis\Results\BenHalluEval" `
  --backend mock `
  --models mock-baseline `
  --prompt_modes zero_shot cot
```

For actual inference, select `--backend transformers` with Hugging Face model IDs, or `--backend ollama` with locally available Ollama model names. Use the 120-row split for a quick check, the 10% split for baselines/ablations, and the full split for the final run.

Each model/prompt/split combination gets a resumable JSONL checkpoint in `Results/BenHalluEval/outputs/`. CSV and Markdown tables are written to `Results/BenHalluEval/results/`.

## Data inventory

See [Data/BenHalluEval/README.md](Data/BenHalluEval/README.md) for the source-to-task mapping, schemas, row counts, and limitations.
