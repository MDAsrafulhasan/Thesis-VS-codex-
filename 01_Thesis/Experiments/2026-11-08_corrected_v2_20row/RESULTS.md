# Corrected-v2 evaluation: completed 20-row results

## What was completed—and what was not

Fresh local inference was completed for all four requested models on the supplied **10 Track-A + 10 Track-B rows**. No historical prediction file was substituted and no mock output is included in these tables.

Two conditions were evaluated:

1. **Neural diagnostics + fixed model-score decisions:** claims, E5/BM25 retrieval, BGE reranking and per-evidence mDeBERTa NLI ran on all rows. Each model's point verdict used the same fixed 0.5 two-label-score threshold, with an explicitly disclosed decoded-label fallback where both label logits were unavailable. Neural diagnostics did not modify verdicts in this condition.
2. **Exploratory untrained NLI cascade:** the identical shared neural features and pre-existing routing thresholds selected 13 fast-path rows; each model freshly judged the remaining seven. No thresholds or rules were changed after viewing the first condition's results. This ablation was added after the first condition and is exploratory, not a confirmatory comparison.

**Neither condition is a full trained six-stage architecture result.** No approved source-disjoint training/development/conformal-calibration partitions were supplied. The 20 evaluation labels were used only for scoring. The new implementation includes a tested grouped lexical/meta learner and conformal primitives, but no fitted model or calibrated risk guarantee is invented. Semantic-equivalent script-view JSD, QA consistency, symbolic sandbox execution and surgical repair are not represented as executed.

## Primary results: fixed model-score decisions

BHS is a weighted error percentage, lower is better:

\[
\operatorname{BHS}=100\left(0.5\frac{W_A}{10}+0.5\frac{W_B}{10}\right).
\]

| Model | Track-A false alarms | Track-A FPR | Track-B misses | Track-B FNR | Valid verdicts | Operational BHS | Valid-only BHS |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bangla-ai-1.7b-q4` | 10/10 | 100% | 0/10 | 0% | 20/20 | **50.00%** | 50.00% |
| `qwen3:1.7b` | 2/10 | 20% | 6/10 | 60% | 20/20 | **40.00%** | 40.00% |
| `qwen2.5:1.5b` | 2/10 | 20% | 6/10 | 60% | 20/20 | **40.00%** | 40.00% |
| `gemma3:1b` | 6/10 | 60% | 0/10 | 0% | 20/20 | **30.00%** | 30.00% |

Example: Gemma has six false alarms and no misses, giving

\[
100\left(0.5\times0.6+0.5\times0\right)=30\%.
\]

### Coverage and interpretation

- 80/80 expected model–row decisions are present and unique.
- Zero missing verdicts or transport failures in this completed condition.
- Bangla-AI produced label `1` on every row. Its 50% BHS exposes uniform hallucinated prediction, not useful balanced discrimination.
- Bangla-AI lacked both selected labels together in the first-token top-20 for all 20 rows. Its probabilities remain **null**, and all 20 verdicts are explicitly marked decoded-label fallbacks.
- The other three models have genuine two-label scores on all 60 of their rows.
- Gemma has the lowest BHS **on this particular pilot**, but it falsely flags 60% of faithful answers. This does not establish broad superiority.
- Both Qwen models have the same aggregate BHS and error counts. Equal aggregate metrics alone do not prove identical row-level behavior.

Artifacts: `01_Thesis/Results/BenHalluEval/corrected_v2_20row_neural_untrained/`.

## Exploratory cascade results

Fixed routing thresholds: entailment/contradiction high thresholds 0.85, low contradiction ceiling 0.15, uniform-claim smooth temperature 0.15. Eligible rows require supplied context and a non-math task. Conflicts or incomplete coverage route to full verification.

Every model shares exactly the same 13 fast-path IDs and decisions: five faithful routes and eight hallucinated routes. Seven remaining rows per model were judged afresh, for **52 routed decisions + 28 new model calls**.

| Model | Track-A false alarms | Track-B misses | Valid verdicts | Cascade BHS | Fixed-score BHS | Cascade minus fixed-score |
|---|---:|---:|---:|---:|---:|---:|
| `bangla-ai-1.7b-q4` | 6/10 | 1/10 | 20/20 | **35.00%** | 50.00% | −15 percentage points |
| `qwen3:1.7b` | 4/10 | 5/10 | 20/20 | **45.00%** | 40.00% | +5 percentage points |
| `qwen2.5:1.5b` | 4/10 | 5/10 | 20/20 | **45.00%** | 40.00% | +5 percentage points |
| `gemma3:1b` | 5/10 | 1/10 | 20/20 | **30.00%** | 30.00% | 0 percentage points |

The cascade helped Bangla-AI, worsened both Qwen variants and left Gemma's aggregate BHS unchanged. Do not select and report only the most favorable condition. Early routing is not guaranteed to improve accuracy and is not equivalent to trained Stage-5 classification.

Artifacts: `01_Thesis/Results/BenHalluEval/corrected_v2_20row_cascade_exploratory/`.

## Shared neural-stage audit

- 20/20 successful claim-generation responses; zero whole-answer fallbacks.
- 21 atomic claims: 19 rows with one claim, one row with two. No claim-cap truncation occurred.
- Retrieval, reranking and NLI completed for every claim.
- Evidence is exclusively the current row's supplied context. No benchmark training contexts, other-row evidence or reference answer banks were loaded.
- Both positive and negative query retrieval/reranking metadata are retained for every claim.
- 64 selected claim–passage pairs; one to four passages per claim. NLI is evaluated per passage, not on a concatenated evidence string.
- Hard and smooth extrema independently recompute correctly. NLI scores are diagnostics, not calibrated Bengali truth probabilities.
- Whole-answer NLI is enabled; XLM-R proxy is disabled, and a skipped checkpoint phase is not evidence of model execution.
- QA consistency and script-view JSD remain missing, with explicit missingness indicators.
- Native/code text views are conservative identity-derived views; no invented transliterator or numerical zero “agreement” score is substituted.
- No inference record contains gold/track/type fields; track-bearing IDs are provenance only and are not sent in prompts.

## Data and sample limitations

Input:
`01_Thesis/Data/BenHalluEval/Processed/Splits/track_a_b_first10_each.csv`

SHA-256:
`3e335d3c30f80004fe02db111e623c281b953b22a875ad01da061098ee7497d4`

The sample contains only Bengali QA, ten distinct questions/contexts and repeated source derivatives. All ten hallucinated rows derive from just three question clusters. Track-B types are Comprehension (3), Factualness (3), Inference (2) and Specificity (2). It covers neither all tasks nor all twelve hallucination types.

No row-independent confidence interval is reported because source clustering violates an unqualified independent-row assumption. There is no real dialect evaluation or full-16k result. No model-pretraining independence claim is made; exact hash screening cannot certify source-lineage independence.

## Models, hardware and runtime

All inference used the installed local Ollama models. No model assets were downloaded and no data was sent to a cloud API. Initial GPU preflight failed with an unsupported CUDA/PTX toolchain; all actual scoring requests used CPU `num_gpu=0`, four threads, seed 42, temperature 0 and context length 4096.

| Requested model | Recorded digest |
|---|---|
| `bangla-ai-1.7b-q4` | `a052529b9ff537298f52236ab74933cdbd80c32f8c636ac6f850c219b6f86331` |
| `qwen3:1.7b` | `8f68893c685c3ddff2aa3fffce2aa60a30bb2da65ca488b61fff134a4d1730e7` |
| `qwen2.5:1.5b` | `65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b` |
| `gemma3:1b` | `97558784bdc9175075a72655bf360b15da17df6b3b9e5182af9f8c0c8e58ba7c` |

Cached HF revisions:

- E5: `intfloat/multilingual-e5-large`, `3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3`.
- BGE: `BAAI/bge-reranker-v2-m3`, `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`.
- NLI: `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`, `8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c`.

Package versions and exact decoding/prompt configuration are in each manifest. The fixed-score run's first manifest timestamp through completion spans approximately **575.83 seconds**. Its final judging session took **264.65 seconds**; this is **not total pipeline runtime**. Shared preparation lasted approximately 282 seconds, with additional startup/inter-phase overhead. The exploratory cascade reused the identical shared-feature journal; judge-only request timing excludes that preparation and cannot be advertised as end-to-end latency savings.

## Exact commands executed

Working directory: workspace root.

```sh
python -m pytest -q 01_Thesis/Code/Evaluation_Pipeline/test_pipeline_v2.py
python 01_Thesis/Code/Evaluation_Pipeline/test_pipeline.py -q

python 01_Thesis/Code/Evaluation_Pipeline/update_pipeline_evaluation.py run --profile neural-untrained --phase prepare --input-csv 01_Thesis/Data/BenHalluEval/Processed/Splits/track_a_b_first10_each.csv --output-dir 01_Thesis/Results/BenHalluEval/corrected_v2_20row_neural_untrained --models bangla-ai-1.7b-q4 qwen3:1.7b qwen2.5:1.5b gemma3:1b

python 01_Thesis/Code/Evaluation_Pipeline/update_pipeline_evaluation.py run --profile neural-untrained --phase judge --input-csv 01_Thesis/Data/BenHalluEval/Processed/Splits/track_a_b_first10_each.csv --output-dir 01_Thesis/Results/BenHalluEval/corrected_v2_20row_neural_untrained --models bangla-ai-1.7b-q4 qwen3:1.7b qwen2.5:1.5b gemma3:1b

python 01_Thesis/Code/Evaluation_Pipeline/update_pipeline_evaluation.py run --profile neural-untrained --phase judge --fast-path --input-csv 01_Thesis/Data/BenHalluEval/Processed/Splits/track_a_b_first10_each.csv --output-dir 01_Thesis/Results/BenHalluEval/corrected_v2_20row_cascade_exploratory --models bangla-ai-1.7b-q4 qwen3:1.7b qwen2.5:1.5b gemma3:1b

python -m pytest -q 01_Thesis/Code/Evaluation_Pipeline/test_pipeline_v2.py 01_Thesis/Code/Evaluation_Pipeline/test_pipeline.py
```

Before the exploratory command, only `outputs/shared_features.jsonl` was copied from the fixed-score run. No model prediction or gold label was imported.

## Validation, source preservation and deviations

- New synthetic/mock-only regressions: 119 passed before scoring.
- Existing active pipeline suite: 220 passed.
- Combined suite: 339 passed after typing cleanup. Mock tests are software validation, not thesis evidence.
- Independent read-only arithmetic/provenance audits reproduced all eight model-condition BHS values, all expected model–row keys, routing coverage and source snapshots.
- The initial `corrected_v2_20row_judge_only` attempt stopped after 32 decisions because Windows briefly locked an atomically replaced report. It is preserved as an interrupted, superseded attempt; it is not one of the completed result tables. Bounded replacement retries were added before the two completed runs.
- After scoring, executed source files were copied unchanged to `executed_source/` and verified against **all five source hashes** in both completed run manifests.
- The active modules subsequently received typing/behavior-equivalent narrowing/formatting cleanup and stricter shared-checkpoint checks plus canonical JSON comparison of trained-artifact provenance. These changes do not alter the recorded scoring rules. No holdout scores were tuned and no results/manifests were rewritten to conceal source drift.
- To inspect or reproduce the exact executed version, use the archived files with their original relative module layout. They are provenance snapshots, not standalone executable scripts in the archive directory: root-relative path resolution assumes the original pipeline location. Do not overwrite current code automatically; reconstruct the archived layout separately if needed.
- Current code intentionally refuses to resume the completed outputs after source changes. Use a **new run directory**. Never edit source hashes in a recorded manifest just to bypass the guard.
- Final combined regression run after checkpoint hardening: **339 passed in 48.08 seconds**. The three new production modules have no editor errors or warnings.
- Preservation verification still reports the pre-existing migration/path issue: **294 missing recorded paths, 0 changed, 0 unsafe, 0 errors**. The immutable baseline was not rewritten.
- Original data, imported research, prior results and the immutable preservation baseline were not altered.

## Next required experiment

Before calling this the complete six-stage architecture, supply an approved, source-disjoint training/development/conformal-calibration protocol and reviewed feature bundles for each judge digest; validate cross-script semantics and implement/validate the remaining QA, symbolic and repair components. Then lock the protocol and evaluate larger task/type-covered data. The reported pilot scores are useful error diagnostics, not final thesis conclusions.
