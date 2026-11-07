# BenHalluScore baseline: 10 rows per track

## Evaluation scope

This run evaluates 20 items total: the exact 10 Track-A items from `track_a_first10_full.csv` plus the first 10 Track-B rows in full-dataset order. All items are `bn_qa`; all Track-A labels are 0 and all Track-B labels are 1. The Track-B items cover four types: 3 Comprehension, 3 Factualness, 2 Inference, and 2 Specificity. The other eight hallucination types are not represented in this small sample.

The original Track-A model checkpoints were reused unchanged. Each model was run on the 10 added Track-B rows with zero-shot prompts, temperature 0, seed 42, and a 1,024-token generation limit.

## Results

BHS uses equal weights: `BHS = 0.5 × FPR_A + 0.5 × FNR_B`. FPR and FNR use successfully parsed verdicts in their respective tracks. Unparsed outputs are not assigned guessed labels; parse failures and collapse are reported separately.

| Model | Track-A false alarms / parsed | FPR_A | Track-B misses / parsed | FNR_B | BenHalluScore | Parse failures / 20 | Collapse rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bangla-ai-1.7b-q4` | 3 / 8 | 37.50% | 3 / 9 | 33.33% | **35.42%** | 3 (15%) | 15% |
| `qwen3:1.7b` | 1 / 10 | 10.00% | 8 / 10 | 80.00% | **45.00%** | 0 (0%) | 0% |
| `qwen2.5:1.5b` | 9 / 9 | 100.00% | 0 / 9 | 0.00% | **50.00%** | 2 (10%) | 0% |
| `gemma3:1b` | 2 / 8 | 25.00% | 4 / 5 | 80.00% | **52.50%** | 7 (35%) | 0% |

Lower BHS is better under this metric. For this sample, Bangla-AI has the lowest parsed-subset BHS (35.42%). Coverage differs substantially: Gemma has only 5 parsed Track-B verdicts, and Bangla-AI has one Track-B collapse. Treat the scores as results for the parseable subsets, not as fully observed scores over all 20 rows.

## Interpretation limits

This is a small, non-stratified sample from one task. The 10 Track-B rows cover only four of the 12 hallucination types, so it does not estimate performance across the full benchmark. The per-model denominators differ because some responses could not be parsed. Use the 16,000-row evaluation split, with all 12 Track-B types, for thesis-level aggregate claims.

## Files

- Combined input: `01_Thesis/Data/BenHalluEval/Processed/Splits/track_a_b_first10_each.csv`
- Detailed metrics: `results/table1_overall_and_per_task_bhs.csv` and `.md`
- Track-B type breakdown for the sampled types: `results/table2_granular_hallucination_types.csv` and `.md`
- Per-row prompts, raw responses, verdicts, and collapse flags: `outputs/*.jsonl`
- Reproduction settings and command: `run_manifest.md`
