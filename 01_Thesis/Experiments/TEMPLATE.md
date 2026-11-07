# Experiment record

Complete this record before running a thesis experiment; use a unique directory/run ID. This template is not a completed experiment or empirical claim.

## Identity and intent

- Run ID / date / researcher:
- Status: planned, smoke-test, running, completed, failed, or superseded:
- Research question / hypothesis:
- Scope: independent evaluation **or** lineage-/lookup-assisted reproduction:
- Planned primary metric, slices, baseline, ablations, and success/failure criteria:

## Research reuse and provenance

- Original repository, authors, version/commit or preserved file SHA-256:
- Exact source files/functions/notebook cells and relevant tests:
- Source licenses, third-party terms, required attribution and citation:
- Files adapted here and precise changes from originals:
- Components intentionally excluded (especially label transfer, answer banks and target-specific facts):

## Data and contamination audit

- Raw input paths and SHA-256 hashes:
- Canonical dataset / split paths, sizes, hashes and seed:
- Label and probability conventions; tested conversion at boundaries:
- Summarization taxonomy mapping path, rationale and review:
- Dialect source/validation, annotation rubric, score provenance, judge identities:
- Source lineage, duplicates, derivative overlap, grouping and exclusion results:
- Separate development/calibration data and untouched final evaluation:
- Audit of retrieval, evidence banks, rules, training, synthetic data and demonstrations:

## Reproduction contract

- Python and resolved dependency versions / environment export:
- OS / hardware / accelerator / memory:
- Model/provider/runtime names and exact revisions; tokenizer/quantization:
- Prompt files and modes:
- Temperature / maximum tokens / seed / thresholds / batching:
- Checkpoint and resume policy; input/configuration fingerprint:
- API/data-sharing permissions, expected costs and external asset licenses:

## Commands and artifacts

Record exact executed commands, working directory, start/end times and exit status. Do not include credentials.

- Raw predictions: `outputs/<run_id>/`
- Derived reports: `results/<run_id>/`
- Logs, environment, protocol and adaptation notes: this experiment directory
- Experiment-specific files, source/data/config hashes and report manifests:

## Validation and observed results

- Tests/checks run, exact outcomes, failures/skips and fixes:
- Both-track counts, valid-parse coverage, failure/collapse rates, latency/runtime:
- Overall/per-task BHS, Track A FPR, Track B FNR and all fine-grained types:
- Paired zero-shot/CoT deltas and weight-sensitivity observations:
- Dialect CCC, valid-pair counts and explicitly defined CBS anchor/units:
- Uncertainty method, resampling unit, seed and dependence assumptions if used:
- Comparison against the registered hypothesis without inventing missing results:

## Interpretation, deviations and next decision

- Positive and negative findings, limitations and possible confounders:
- Deviations from the pre-registered protocol and their reasons:
- Whether the result is eligible for the main thesis or an assisted-reproduction appendix:
- Writing/table/figure paths backed by these results:
- Next bounded experiment or reason to stop:
