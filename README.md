# Bengali Hallucination Thesis Workspace

This workspace keeps the thesis data and evaluation pipeline separate from the two complete datathon research projects. Original reference-project files remain grouped together, and the thesis datasets are separated into raw inputs and reproducible derived splits.

## Start here

| Need | Go to |
|---|---|
| Thesis datasets and evaluation workflow | [01_Thesis/README.md](01_Thesis/README.md) |
| Raw and processed BenHalluEval data | [BenHalluEval data guide](01_Thesis/Data/BenHalluEval/README.md) |
| Splitter, evaluator, and verification script | [Evaluation pipeline guide](01_Thesis/Code/Evaluation_Pipeline/README.md) |
| Reference projects and recommended reading order | [02_Reference_Projects/README.md](02_Reference_Projects/README.md) |

## Workspace map

```text
01_Thesis/
  Data/BenHalluEval/
    Raw/                         Original Track-A and Track-B CSVs
    Processed/Splits/            Reproducible 120-row, 10%, and full splits
  Code/Evaluation_Pipeline/      make_splits.py, eval.py, test_pipeline.py
  Results/                       Created by eval.py when model runs are saved

02_Reference_Projects/
  IUT_Datathon_Hallu/             Report, presentation, and inference notebook
  IUTCSFESTDATATHON/              Complete 277-file research and competition project
```

The two reference projects are preserved as self-contained trees. The duplicate outer folder created during extraction was removed; files inside each project retain their relative layout. Their READMEs, licenses, data notes, logs, tests, releases, and notebooks remain with the projects.

## Recommended research workflow

1. Read the dataset guide and reference-project guide before selecting data or claims for the thesis.
2. Keep source CSVs in `01_Thesis/Data/BenHalluEval/Raw/` unchanged.
3. Regenerate derived splits with the command in the evaluation-pipeline guide. The default seed is 42.
4. Run the pipeline verification script after changing evaluation code.
5. Run model evaluations with an explicit backend, model ID, split, and output directory. Checkpoint JSONL files and summary tables are saved under `01_Thesis/Results/`.
6. Record thesis-specific experiment decisions and results separately from the two datathon projects’ historical `DECISION.log` and `EXPERIMENT.log` files.

## Reproducibility notes

- `01_Thesis/Data/BenHalluEval/SHA256_MANIFEST.csv` records checksums for the raw CSVs and generated splits.
- `02_Reference_Projects/REFERENCE_FILE_MANIFEST.csv` records checksums for every file imported in the two reference projects.
- Source IDs in the raw Track-A datasets are not globally unique. The normalizer creates deterministic task/source-row-qualified IDs in the processed splits so inference checkpoints can resume without skipping distinct examples.
- The supplied raw folder contains the primary BenHalluEval files. It does not contain a Dialectal_Bias source CSV; dialect examples can be generated for pipeline development with `--generate_mock_data`.
