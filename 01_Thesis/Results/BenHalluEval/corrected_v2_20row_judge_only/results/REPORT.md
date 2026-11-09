# Corrected v2: local 20-row pilot

Profile: **judge-only**. Lower BHS is better. This is NOT a full 16,000-row benchmark evaluation.

## Scientific scope

- No approved train/dev/conformal-calibration data was supplied for this invocation unless an explicit trained artifact is loaded.
- neural-untrained uses shared neural diagnostics plus each model score at fixed threshold .5; no learned Stage-5 or conformal guarantee is claimed.
- judge-only disables neural stages and is a clearly labeled ablation, not the complete six-stage architecture.
- Native/code views are conservative identity-derived views; no validated semantic-equivalent transliterator is supplied, so cross-script JSD is unavailable.
- QA/QG consistency precision/recall proxies are not fully implemented; unavailable features are explicit, not invented.
- Symbolic execution is unavailable; mathematical rows are unresolved rather than approved by a timeout/error fallback.
- Stage 6 CAD math exists but surgical mitigation is not enabled; BHS scores the original candidate answers only.
- No benchmark label memory, other-row contexts, answer banks or held-out labels enter inference.
- Exact normalized question/context screening is not proof of complete source-lineage independence.
- Decoded label fallback is disclosed when either label is missing from top-20 logits; no probability is fabricated.
- Grammar-constrained single-token decisions intentionally exhaust one output token; this is not repetition collapse.

| Model | Valid / attempted | Two-label scores | Track-A errors | Track-B errors | Operational BHS % | Valid-only BHS % |
|---|---:|---:|---:|---:|---:|---:|
| bangla-ai-1.7b-q4 | 20/20 | 0 | 10/10 | 0/10 | 50.00 | 50.00 |
| qwen3:1.7b | 11/11 | 11 | 2/10 | 1/1 | NA | 60.00 |

## Reproducibility

Dataset SHA-256: `3e335d3c30f80004fe02db111e623c281b953b22a875ad01da061098ee7497d4`

Exact prompts, response payloads, statuses, timing and per-row decisions: `../outputs/predictions.jsonl`.
Shared claims/retrieval/NLI checkpoints: `../outputs/shared_features.jsonl`.
Model digests, protocol, source hashes, runtime and settings: `../run_manifest.json`.

BHS = 100 × (0.5 × Track-A error rate + 0.5 × Track-B error rate). Failures count as errors, not faithful labels.
Missing/unattempted rows make the final BHS undefined; partial recorded-row metrics are separately named.
No evaluation labels are passed to claims, retrieval, NLI or judge prompts.
The sample is QA-only, has repeated source questions, and cannot establish broad Bengali-task superiority.
