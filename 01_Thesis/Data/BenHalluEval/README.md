# BenHalluEval data guide

## Data locations

- `Raw/Track-A/` and `Raw/Track-B/` contain the eight supplied source CSVs. Treat these files as read-only originals.
- `Processed/Splits/` contains outputs normalized to the thesis schema and split with seed 42.
- `SHA256_MANIFEST.csv` records SHA-256 and byte size for each raw and processed CSV.

## Raw-file map

| Track | Source file | Normalized task | Rows |
|---|---|---:|---:|
| A | `Track-A/banglahallueval_qa_1000.csv` | `bn_qa` | 1,000 |
| A | `Track-A/codemix_1000.csv` | `codemixed_qa` | 1,000 |
| A | `Track-A/banglahallueval_summarization_dataset_1000.csv` | `summarization` | 1,000 |
| A | `Track-A/somadhan_1000_main_ordered.csv` | `math` | 1,000 |
| B | `Track-B/qa_4000.csv` | `bn_qa` | 4,000 |
| B | `Track-B/codemix_4000.csv` | `codemixed_qa` | 4,000 |
| B | `Track-B/summarization_3000_corrected.csv` | `summarization` | 3,000 |
| B | `Track-B/reasoning_1000.csv` | `math` | 1,000 |

The normalizer maps raw Track-B `pattern`, `hallucination_type`, and `error_type` values into the 12 canonical labels in `make_splits.py`. Track A receives label 0 and hallucination type `None`; Track B receives label 1.

## Processed split files

| File | Total | Track A | Track B | Intended use |
|---|---:|---:|---:|---|
| `benhallu_micro_debug_120.csv` | 120 | 30 | 90 | Quick pipeline/debug checks |
| `benhallu_stratified_10pct.csv` | 1,600 | 400 | 1,200 | Baselines and ablations |
| `benhallu_full_16k.csv` | 16,000 | 4,000 | 12,000 | Final evaluation |

Each processed file has the columns `id`, `task`, `track`, `hallucination_type`, `question`, `context`, `candidate_answer`, and `ground_truth_label`. Raw Track-A IDs repeat across tasks and sometimes within a source file. Processed IDs therefore include the track, task, source filename, and source-row ordinal; this preserves unique examples for checkpointing.

## Secondary dialect benchmark

No Dialectal_Bias source file was included in the supplied data folders. The splitter accepts a CSV using the `Dialectal_Bias` schema, filters `Standard_Bengali`, `Tangail`, and `Chittagong`, then selects 100 samples per variant stratified by `question_type`. To generate a reproducible 4,000-row mock dialect dataset and its 300-row subset, run:

```powershell
python "01_Thesis\Code\Evaluation_Pipeline\make_splits.py" `
  --generate_mock_data `
  --output_dir "01_Thesis\Data\Dialectal_Bias\Mock" `
  --seed 42 `
  --mode dialect
```

