# Updated BenHalluEval detector: 20-row Track-A/Track-B run

## Result

This run evaluates the same 20 input rows as the four-model baseline: 10 Track-A and 10 Track-B, all `bn_qa`. The updated detector predicted every row. The score uses the dataset label convention (0 = Track-A faithful, 1 = Track-B hallucinated):

| Rows | Track_A | Track_B | Track_A_false_alarms | Track_B_missed_hallucinations | FPR_A_pct | FNR_B_pct | BenHalluScore_pct |
| ---- | ------- | ------- | -------------------- | ----------------------------- | --------- | --------- | ----------------- |
| 20   | 10      | 10      | 3                    | 0                             | 30.00     | 0.00      | 15.00             |

Lower BenHalluScore is better. Track-A false alarms are predictions of hallucination on faithful examples. Track-B misses are predictions of faithful on hallucinated examples.

## Baseline comparison on identical parsed rows

The original judge JSONL files contain parse failures. To avoid assigning invented labels to those rows, each baseline BHS and its paired updated-detector BHS below use the exact same set of baseline-parsed IDs for that model. The full updated-detector result above still uses all 20 rows.

| Baseline_Model    | Baseline_Parsed_A | Baseline_Parsed_B | Baseline_Parse_Failures | Baseline_FPR_A_pct | Baseline_FNR_B_pct | Baseline_BHS_pct | Pipeline_FPR_A_on_same_parsed_rows_pct | Pipeline_FNR_B_on_same_parsed_rows_pct | Pipeline_BHS_on_same_parsed_rows_pct | Delta_BHS_pipeline_minus_baseline_same_rows |
| ----------------- | ----------------- | ----------------- | ----------------------- | ------------------ | ------------------ | ---------------- | -------------------------------------- | -------------------------------------- | ------------------------------------ | ------------------------------------------- |
| bangla-ai-1.7b-q4 | 8                 | 9                 | 3                       | 37.50              | 33.33              | 35.42            | 25.00                                  | 0.00                                   | 12.50                                | -22.92                                      |
| gemma3:1b         | 8                 | 5                 | 7                       | 25.00              | 80.00              | 52.50            | 25.00                                  | 0.00                                   | 12.50                                | -40.00                                      |
| qwen2.5:1.5b      | 9                 | 9                 | 2                       | 100.00             | 0.00               | 50.00            | 33.33                                  | 0.00                                   | 16.67                                | -33.33                                      |
| qwen3:1.7b        | 10                | 10                | 0                       | 10.00              | 80.00              | 45.00            | 30.00                                  | 0.00                                   | 15.00                                | -30.00                                      |

The `Delta_BHS_pipeline_minus_baseline_same_rows` column is updated detector BHS minus baseline BHS. Negative values mean the updated detector made fewer weighted errors on the matched rows; positive values mean more.

## Reproducible method choices

- Training data: 20 rows used from 1,595 eligible rows after removing every training row whose normalized question group appears in the evaluation set. An optional pilot sample is balanced by track with Track_A seed 42 and Track_B seed 43.
- Evaluation data: 20 rows; the normalized question groups are excluded from training.
- Retrieval: multilingual E5 dense retrieval plus in-file BM25, reciprocal-rank fusion with `k=60`, then BGE reranking. A supplied row context is used directly; rows without context retrieve from the filtered training-context corpus. Exact evaluation-answer hashes are purged from that corpus.
- Claim and evidence QA models: `qwen3:1.7b` and `gemma3:1b` by default. The first-token binary-logit judge defaults to `qwen3:1.7b`. Decoding uses temperature 0 and seed 42.
- Feature model defaults: `intfloat/multilingual-e5-large`, `BAAI/bge-reranker-v2-m3`, `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`, and `FacebookAI/xlm-roberta-base`. The exact resolved Hugging Face commits and Ollama digests are recorded in `run_manifest.json`.
- Stage 5: grouped five-fold logistic regression, five random seeds, `C` in `(0.1, 1.0, 10.0)`; model and open/closed-context thresholds are selected from training-only out-of-fold BHS.
- The pipeline is a hallucination detector. It classifies the existing answer; it does not rewrite or correct it. This result measures detector performance, not answer-quality improvement after correction.

## Audit and limitations

- Stage errors, claim-generation fallbacks, repetition-collapse counts/rates, and romanized-token audit are in `stage_failure_audit.json`.
- The diagrams do not define the 16 NLI features, 10 lexical features, the complete Banglish phoneme map, or the exact threshold-search objective. Their reproducible defaults and feature names are recorded in `run_manifest.json` and this script.
- Banglish normalization maps only reviewed words in the built-in/custom map; unknown Latin tokens are preserved and listed in the audit.
- The mDeBERTa/XLM-R signals are transfer features and are not Bengali-calibrated probabilities. The XLM-R feature is token-level max-cosine recall, an explicitly defined BERTScore-style proxy without baseline rescaling.
- This 20-row sample is exploratory and too small for a reliable general model ranking. Full thesis conclusions require the prescribed larger split and repeated evaluation.

## Files

- `update_pipeline_bhs.csv`: complete 20-row score and error counts.
- `pre_post_matched_comparison.csv`: per-baseline matched-row comparison.
- `meta_predictions.csv`: per-row labels, thresholds, and meta-classifier probabilities.
- `features_29_dense.csv`: extracted feature values for train and evaluation rows.
- `intermediate/five_stage_rows.jsonl`: resumable per-row stage outputs.
- `run_manifest.json`: data hashes, resolved model versions, model settings, and method details.
