# Corrected-v2 local 20-row detection pilot

## Identity and intent

- Date: 2026-11-08; status: registered before new evaluation.
- Request: compare bangla-ai-1.7b-q4, qwen3:1.7b, qwen2.5:1.5b and gemma3:1b on the supplied first-10-per-track CSV.
- Primary metric: operational BHS, equal track weights, lower is better; include valid-only BHS, per-track errors, coverage, failures, per-row decisions and task/type breakdowns.
- Scope: new local-model pilot, no benchmark-derived answer-memory or label lookup. This small source-order sample is not an independent final thesis holdout and cannot support broad performance claims.

## Architecture and honest deviations

Specification: root `updated_benhallueval_v2_master_architecture.pdf` and `Update_formulation.md`.
New modules: `update_pipeline_evaluation.py`, `pipeline_v2_math.py`, `pipeline_v2_features.py`, `test_pipeline_v2.py` under `01_Thesis/Code/Evaluation_Pipeline/`.

Two bounded profiles are planned:

1. `judge-only`: four freshly executed model decisions, no neural diagnostics; a labeled ablation.
2. `neural-untrained`: new shared claim extraction, per-claim polarity-preserving E5/BM25 RRF, BGE reranking, per-evidence mDeBERTa NLI, smooth/hard pooling, routing diagnostics, plus the same four-model judging policy. Fast-path overrides are disabled to preserve model-comparison interpretability. Neural features do not change point predictions without an eligible fitted Stage-5 artifact.

No approved source-disjoint train/development/conformal-calibration partitions were supplied. Do NOT train on the 20 evaluation rows or silently reinterpret overlapping 10%/full benchmark subsets as independent training. The trained meta/conformal implementation is provided for future reviewed feature bundles but is not an empirical trained-pipeline result here.

Cross-script JSD remains unavailable without a validated meaning-equivalent transliterator. QA consistency proxies are not fabricated. Symbolic sandbox execution and surgical CAD repair are not executed; this is original-candidate detection only. The CAD transformation and conformal mathematics are unit-tested separately.

## Inference contract frozen before results

- Ollama loopback: `http://127.0.0.1:11434`; no cloud API calls or downloads.
- GPU preflight failed with unsupported PTX toolchain; all requests explicitly use CPU `num_gpu=0`.
- Seed 42, temperature 0, four threads, context 4096, 1 verdict token, constrained integer output grammar `{0,1}`, first-token top-20 log-probabilities.
- Decision: two-label score >= 0.5 -> hallucinated; if both labels absent from top-20, accept only a strictly parsed output label and disclose decoded fallback; never invent its probability.
- Transport/invalid verdict failures remain null and count as track errors. No answer-specific retry/prompts or post-result threshold changes.
- Claim generation: fixed qwen3:1.7b, maximum four claims, 256 tokens, question+candidate only. Whole-answer fallback is explicitly flagged and preserves candidate text.
- Retrieval: own row's supplied context only. No other-row contexts, saved predictions, reference answer banks or labels are evidence.
- Neural checkpoints loaded from existing local HF snapshots only; E5 large (not large-instruct), BGE-v2-m3 and mDeBERTa. No Bengali stemming claim. Model revisions captured in manifests.
- Raw Bengali and signs are preserved; normalized derived views are separate. Exact 13-token overlap is only an audit primitive; no corpus supplied means training-overlap status unknown, not clean.

## Data, provenance and artifacts

Input: `01_Thesis/Data/BenHalluEval/Processed/Splits/track_a_b_first10_each.csv`.
CSV labels: Track_A=0 faithful; Track_B=1 hallucinated. Only allowlisted question/context/candidate/task fields enter feature stages; judge sees no ID, gold, type or track. IDs encode tracks and are never sent to a model.

Run-specific manifest stores dataset/PDF/source hashes, model digests, neural snapshots, versions, exact prompt texts, CPU settings, commands and timestamps. Run-specific `outputs/` stores raw prediction/feature journals; matching `results/` stores derived tables.

Planned roots:
- `01_Thesis/Results/BenHalluEval/corrected_v2_20row_judge_only/`
- `01_Thesis/Results/BenHalluEval/corrected_v2_20row_neural_untrained/`

Resume rejects source/config/model/data/runtime mismatches. Incomplete rows cannot be presented as final BHS. Journals repair only torn final lines. Exclusive run lock prevents concurrent writes.

## Exploratory cascade ablation amendment

Added after the fixed-score pilot completed, before scoring the routed variant: run a separate `--fast-path` experiment with exactly the same code, thresholds, claims and neural features. The previously computed label-free features may be reused; no prediction labels or judge outputs are imported. Early-route thresholds stay gamma_e=gamma_c=0.85 and lambda_c=0.15; no holdout-based tuning is permitted. Five faithful and eight hallucinated routing recommendations are known from unscored NLI diagnostics; seven rows require judges. This is an exploratory, untrained cascade ablation, not confirmatory model selection or a trained/conformal architecture result. Store independently under `corrected_v2_20row_cascade_exploratory/`; judge the unresolved rows afresh for every requested model.

## Reuse and licensing

No imported reference source is copied or executed. Reuse is bounded to the active `update_evaluation.py` BM25/token helpers and NLI label mapping, plus existing strict judge parsing. Existing code, dataset and model licenses remain applicable; cached model availability does not grant redistribution permission. Raw payloads should remain local unless permissions are reviewed.

## Validation and interpretation

Run synthetic math/runner regressions and existing active pipeline tests separately. Never treat synthetic tests as empirical findings. Record commands/outcomes after execution. Scores are diagnostic QA-only pilot evidence, with clustered questions and only four Track-B error types; no dialect conclusion or full-16k ranking is supported.
