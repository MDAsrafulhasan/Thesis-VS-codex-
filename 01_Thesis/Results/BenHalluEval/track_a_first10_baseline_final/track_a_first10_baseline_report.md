# Track-A first-10 baseline report

## Run scope

- Dataset: `01_Thesis/Data/BenHalluEval/Processed/Splits/track_a_first10_full.csv`
- Rows: 10 real Track-A records; all are `bn_qa` with ground-truth label 0.
- Prompt mode: `zero_shot`
- Metric: Track-A false-positive rate (FPR), calculated over rows with a successfully parsed verdict. A `Yes` (`1`) is a false alarm. Parse failures are listed separately and are not guessed as verdicts.
- This Track-A-only sample cannot produce a complete BenHalluScore (BHS), because BHS also needs Track-B FNR.

## Results

| Model | False alarms / parsed | Track-A FPR | Parsed | Parse failures | Collapse rate | Average latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| `gemma3:1b` | 2 / 8 | 25.0% | 8 / 10 | 2 (20%) | 0% | 3,022.5 |
| `qwen3:1.7b` | 1 / 10 | 10.0% | 10 / 10 | 0 (0%) | 0% | 23,322.8 |
| `qwen2.5:1.5b` | 9 / 9 | 100.0% | 9 / 10 | 1 (10%) | 0% | 6,847.8 |
| `bangla-ai-1.7b-q4` | 3 / 8 | 37.5% | 8 / 10 | 2 (20%) | 20% | 36,966.3 |

These are exploratory measurements from only 10 Bengali QA items. The rates are sensitive to individual examples; they should not be treated as publication-level estimates or a broad ranking of model quality. The run also shows that parse coverage differs, so compare both FPR and parse-failure rate.

## Artifacts

- Per-row model responses and parsed verdicts: `outputs/*.jsonl`
- Full task summary: `results/table1_overall_and_per_task_bhs.csv` and `.md`
- Other generated tables are empty or not applicable to this Track-A-only, zero-shot run; in particular, BHS and Track-B hallucination-type rates require Track B.
- The Gemma checkpoint retains one initial failed inference attempt followed by its successful retry for the same ID. The summary uses the latest record per ID, so all metrics above use exactly 10 unique examples.

For the later pipeline comparison, run it on the same 10 IDs and the same model/prompt settings. Compare row-level verdicts and FPR alongside parse coverage and latency. Do not compare a changed question/context set as a before/after intervention.
