# BenHalluEval: Days 11–14 Research Design and Chapter 3 Draft

**Thesis title:** BenHalluEval: A Comprehensive Benchmark for Evaluating Hallucinations of Large Language Models on Low-Resource Bengali NLP Tasks.

**Prepared:** 2026-11-09. **Status:** research plan and unapproved thesis draft; no new tuning, training, inference, or ablation results are claimed in this document.

**Researcher / supervisor / protocol approval:** **[INSERT NAMES AND APPROVAL DATE]**.

## Executive assessment

The project evaluates whether existing candidate answers are hallucinated across Bengali question answering, code-mixed question answering, summarization, and mathematics. Its corrected-v2 detector combines claim extraction, context-only hybrid retrieval, evidence reranking, per-evidence natural-language inference (NLI), and local language-model judging. An optional conflict-aware cascade replaces some judge calls with NLI decisions. A source-grouped lexical/meta-classification implementation exists, but no approved source-disjoint training, development, and conformal-calibration partitions were supplied for the completed v2 pilot.

The current empirical contribution is therefore a **small, traceable detection pilot**, not an evaluated complete six-stage detection-and-mitigation architecture. The research title describes the intended comprehensive scope; current results do not establish that scope. The benchmark CSVs were supplied to the project. Dataset authorship, original collection, generation procedures, and the extent of this thesis's benchmark-construction contribution must be verified rather than inferred from the working title: **[INSERT VERIFIED DATASET PROVENANCE AND CONTRIBUTION STATEMENT]**.

This plan treats component value as a falsifiable question. Ablations cannot prove architectural necessity in advance. Negative and null results must be retained. No plausible-looking numerical placeholders are used: unmeasured quantities are marked **[INSERT VALUE]**, while known results are labeled observed.

## Navigation

1. Evidence and scope ledger.
2. Phase 1, Days 11–12: tuning and three ablations.
3. Phase 2, Days 13–14: error analysis, tables, and plotting specification.
4. Chapter 3 draft: methodology and experimental setup.
5. Execution checklist, provenance, and outstanding decisions.

# Evidence and scope ledger

## E1. Documents and implementation authority

Paths in this report are workspace-root-relative unless a Markdown link is supplied.

| Source | Role and limitation |
|---|---|
| `Update_formulation.md` | Corrected mathematical design; its historical implementation-status list predates parts of corrected v2 |
| `01_Thesis/Code/Evaluation_Pipeline/README.md` | Current profiles, limitations, artifact contracts, and commands |
| `01_Thesis/Code/Evaluation_Pipeline/pipeline_v2_math.py` | Mathematical primitives and optional grouped learner |
| `01_Thesis/Code/Evaluation_Pipeline/pipeline_v2_features.py` | Actual feature definitions and evidence protocol |
| `01_Thesis/Code/Evaluation_Pipeline/update_pipeline_evaluation.py` | Executed orchestration and decision policies |
| [Corrected-v2 protocol](../2026-11-08_corrected_v2_20row/PROTOCOL.md) | Pilot registration and exploratory amendment |
| [Corrected-v2 results](../2026-11-08_corrected_v2_20row/RESULTS.md) | Consolidated saved pilot results and deviations |
| `docs/DATA_GUIDE.md`, `docs/RESEARCH_REUSE.md` | Labels, overlap, licensing, and reference-reuse boundaries; some navigation paths are historical |

The existing `MASTER_ARCHITECTURE.md` and Chapter 4 draft contain older claims and formulations. They are not evidence of implemented or validated behavior. This report does not overwrite those documents or promote this draft to approved thesis prose.

## E2. Current empirical evidence

The completed pilot uses `01_Thesis/Data/BenHalluEval/Processed/Splits/track_a_b_first10_each.csv`, SHA-256 `3e335d3c30f80004fe02db111e623c281b953b22a875ad01da061098ee7497d4`. It contains ten faithful and ten hallucinated Bengali-QA rows, ten question/context groups, and only three question groups among hallucinated rows. It is development-exposed and must not become an untouched final test set.

**Table E1. Observed corrected-v2 pilot results, not a final benchmark comparison.** Track-specific errors are counts. BHS is an error percentage; lower is better.

| Model | Fixed-score A errors / 10 | Fixed-score B errors / 10 | Fixed-score BHS (%) | Cascade A errors / 10 | Cascade B errors / 10 | Cascade BHS (%) | Cascade minus fixed-score (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bangla-ai-1.7b-q4` | 10 | 0 | 50.00 | 6 | 1 | 35.00 | -15.00 |
| `qwen3:1.7b` | 2 | 6 | 40.00 | 4 | 5 | 45.00 | +5.00 |
| `qwen2.5:1.5b` | 2 | 6 | 40.00 | 4 | 5 | 45.00 | +5.00 |
| `gemma3:1b` | 6 | 0 | 30.00 | 5 | 1 | 30.00 | 0.00 |

Both completed conditions contain all expected 80 model–row verdicts. The fixed-score condition computes neural diagnostics but does not use them to change verdicts. The exploratory cascade uses the same 13 routed rows per model, comprising five faithful and eight hallucinated routes, and freshly judges seven rows per model. It therefore contains 52 routed decisions and 28 fresh judge decisions.

Bangla-AI's fixed-score outputs all use decoded-label fallback because a usable pair of label scores is unavailable; its probability remains missing. All other models have two-label scores on all 20 fixed-score rows. These are different measurement channels and must be disclosed. The initial `corrected_v2_20row_judge_only` run is interrupted at 32 journaled responses and is excluded from completed comparisons. Its generated report can lag the journal.

No confidence intervals or significance claims are inferred from these aggregates. In particular, a 20-row sample with shared source derivatives is not 20 independent observations. The observed cascade did not consistently improve BHS.

## E3. Claims that are and are not supported

| Claim | Current status |
|---|---|
| Local claims/retrieval/reranking/NLI can execute on the pilot | Observed engineering result |
| Cascade routing can reduce judge-call count | Observed on this pilot; not an end-to-end latency claim |
| Routing improves every judge's detection accuracy | Contradicted by the pilot aggregates |
| Smooth pooling improves accuracy over hard pooling | Untested controlled hypothesis |
| Contrastive retrieval improves over positive-only retrieval | Untested controlled hypothesis |
| Source-grouped meta-learning improves over judging | Not evaluated in corrected v2 |
| Split conformal bounds BHS at a chosen level | Incorrect interpretation; no such guarantee |
| Cross-script JSD, symbolic execution, and surgical repair ran | Not supported; these components were not executed |
| Real dialect robustness has been evaluated | Not supported; no real dialect benchmark supplied |
| The thesis is the first to introduce these techniques | **[INSERT VERIFIED LITERATURE COMPARISON]**; no novelty claim yet |

# Phase 1 — Days 11–12: Hyperparameter Tuning and Ablation Studies

## 1. Research questions and experimental units

- **RQ1:** How do judge-only and routed detectors trade Track-A false alarms against Track-B misses across the four tasks?
- **RQ2:** Do atomic decomposition, polarity-preserving evidence selection, and smooth pooling improve the error/cost trade-off under a locked protocol?
- **RQ3:** Which failure categories and evidence conditions explain remaining errors?
- **Conditional RQ4:** Given approved source-disjoint labeled data, does grouped lexical/meta-learning improve over the locked untrained detectors?

The scientific resampling and partition unit is a reviewed **source-lineage group**, not a row ID. Group all related questions, shared source passages, faithful answers, generated negatives, paraphrases, and other known derivatives before partitioning. A normalized question match is a screening key, not proof that all related sources were found. Document ambiguous links and connected-component grouping decisions.

The current first-10-per-track pilot has already influenced analysis. Do not tune on it and subsequently label it held-out. Its existing results serve only as exploratory motivation and implementation checks.

### 1.1 Proposed data partition contract — not yet executed

| Partition | Purpose | Proposed allocation by source groups | Required actual counts |
|---|---|---:|---|
| Training | Optional grouped lexical/meta fitting | 60% target | **[INSERT ROWS, GROUPS, TASK/TRACK COUNTS]** |
| Development | Architecture, hyperparameters, routing and point thresholds | 15% target | **[INSERT ROWS, GROUPS, TASK/TRACK COUNTS]** |
| Calibration | Conformal scores only after configuration is frozen | 10% target | **[INSERT ROWS, GROUPS, TASK/TRACK COUNTS]** |
| Final test | One locked evaluation of declared configurations | 15% target | **[INSERT ROWS, GROUPS, TASK/TRACK COUNTS]** |

These percentages are planning targets, not implemented splits or fixed row counts. Source grouping may make exact proportions impossible. Audit task, track, type, and open/closed-context coverage; preserve group integrity rather than forcing a nominal ratio. The trained implementation currently requires both regimes globally in development and sufficient class/group support for nested folds. If these requirements are infeasible, report that limitation rather than introducing row-wise fallback.

If data approval is delayed, Days 11–14 can still deliver the protocol, error audit, plots of existing results, and chapter draft. They cannot deliver valid fitted-pipeline or confirmatory tuning findings. Do not silently train on the existing micro/10%/full overlapping evaluation files. Selection of a new approved partition from their underlying sources requires explicit review and a new manifest.

## 2. Three targeted ablations

### 2.1 Shared parent configuration

The primary ablation family uses the **untrained conflict-aware cascade**, because its NLI features actually influence predictions. Fix the judge digest, prompt, decoding, context segmentation, candidate budget, NLI model, routing safeguards, and evaluation IDs. Compare all variants against the same full cascade parent on the same source groups.

The trained profile is a separate conditional extension, not an undisclosed change to the three primary experiments. A fixed-score profile with routing disabled is useful as a baseline, but cannot test the accuracy value of neural components: changing diagnostics alone leaves its predictions unchanged.

Use initial reference settings from the pilot before tuning: smooth temperature $\tau=0.15$, entailment and contradiction thresholds $\gamma_e=\gamma_c=0.85$, contradiction ceiling $\lambda_c=0.15$, and at most four selected evidence passages per claim. New variants require new source fingerprints and output directories. Several settings are source-level constants rather than existing CLI flags; implement and test configuration support separately before execution. No commands in this plan imply those flags already exist.

### 2.2 Ablation specification

**Table A1. Three controlled component-removal studies.** A directional hypothesis is a prediction to test, not a required outcome.

| ID | Component / intervention | Directional hypothesis | Controlled factors and measurements | Interpretation if not supported |
|---|---|---|---|---|
| A1 | Replace generated atomic claims with one unmodified whole-candidate hypothesis | Decomposition will reduce misses on multi-fact answers containing one unsupported detail; removing it is expected to raise Track-B FNR | Same judge, router, retrieval/reranker/NLI checkpoints and source split; report single- versus multi-claim slices, evidence pairs, routed coverage, total latency | Decomposition may omit facts, misresolve short answers, or add noise; retain simpler whole-answer verification if preferable |
| A2 | Replace positive/negative retrieval with positive-only evidence selection | Counterevidence-oriented selection will improve contradiction detection; removing it is expected to raise misses on negation and relation-conflict cases | Same maximum candidate and selected-evidence budgets, deduplication, original-claim NLI and decision policy; record actual unique evidence and scoring calls | The fixed English negation wrapper may be ineffective for Bengali, or row contexts may be too short for retrieval to matter |
| A3 | Replace normalized smooth pooling with hard minimum/maximum for routing; keep hard conflict guards | Smooth pooling will reduce sensitivity to a single noisy entailment score without materially increasing misses; hard-only routing is expected to worsen the balanced trade-off on multi-claim answers | Same claims and per-passage scores, eligibility checks and thresholds; no changes to gold access or context policy; report both track rates and routing errors | Hard pooling may be preferable, or smooth pooling may mask genuine unsupported claims; claim necessity is not established |

**Shared slice assignments:** Freeze single-/multi-fact membership before interventions using independently reviewed answer facts, or explicitly identified parent-extractor counts if review is unavailable. Use identical row membership for every variant; A1's operational claim count is always one and must not redefine the comparison strata. Store each variant's actual extraction count separately.

**A1 controls:** A whole-answer verifier typically uses fewer NLI pairs than an atomic verifier. The primary comparison should retain a practical per-claim evidence cap and disclose that computational difference. Add a matched-total-pair-budget analysis as a secondary diagnostic if feasible; do not pretend different cost budgets are identical. Inspect decomposition coverage independently: checking all extracted claims is not checking all answer claims. The cap of four is a particular risk for long summaries.

**A2 controls:** Retrieve positive-only candidates up to the same maximum union budget, and rerank the same number where enough unique passages exist. Remove negative-query-specific numeric inputs if a future trained version is evaluated. Do not replace missing negative features with observed zero. This intervention estimates the value of the contrastive evidence-selection package, not a pure negation-only effect isolated from all feature changes. Log actual operations because equal caps do not guarantee equal realized compute.

**A3 controls:** First run a locked-threshold mechanistic contrast. Separately report a fair best-development configuration for each pooling family using an equal tuning budget. Keep these two estimands distinct. With one claim, hard and normalized smooth pooling are equal: the pilot has only one multi-claim row, so it is largely uninformative for this hypothesis. The confirmatory set needs sufficient reviewed multi-claim examples: **[INSERT TARGET GROUP COUNT AND JUSTIFICATION]**.

### 2.3 Outcomes and falsification

The primary outcome is paired final-test operational BHS difference:

$$
\Delta\operatorname{BHS}=\operatorname{BHS}_{\mathrm{variant}}-\operatorname{BHS}_{\mathrm{parent}}.
$$

Positive values mean the variant is worse. Report Track-A false alarms, Track-B misses, verdict coverage, failure counts, routed coverage, routed error rates, judge-tail errors, and end-to-end latency beside BHS. Establish the development selection budget and any acceptable degradation margin before final testing: **[INSERT PREDECLARED MARGIN IN PERCENTAGE POINTS]**. Never infer non-inferiority merely from a nonsignificant result.

For uncertainty, propose a paired source-group bootstrap with 2,000 resamples and seed 42 on a sufficiently large final test set. Resample the same group indices for every variant and include all rows belonging to each selected group, preserving the row-weighted BHS estimand. Track undefined replicates missing a class; do not silently discard them. If this occurs materially, or independent groups are too few, report descriptive results and the limitation rather than an unreliable interval. Provide a separately declared group-weighted sensitivity analysis if long source families dominate. Do not perform a row-i.i.d. bootstrap or apply an unqualified row-level McNemar test to dependent derivatives. If formal tests are used for all three primary contrasts, preregister a source-group-aware procedure and multiplicity control: **[INSERT STATISTICAL TEST AND FAMILY-WISE POLICY]**.

### 2.4 Incremental architecture table — planned, not observed

**Table A2. Incremental detector construction.** Each row is a complete decision system. The progression is descriptive, not a substitute for full-parent removal studies. No missing entries are estimates.

| Configuration | Atomic claims | Contrastive selection | Pooling / routing | Learned meta | Dev BHS (%) | Final BHS (%) | FPR (%) / FNR (%) | End-to-end ms/example |
|---|---|---|---|---|---|---|---|---|
| C0: judge-only | No | No | No early routing | No | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUES]** | **[INSERT VALUE]** |
| C1: positive-only atomic cascade | Yes | No | Hard; guards retained | No | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUES]** | **[INSERT VALUE]** |
| C2: contrastive atomic cascade | Yes | Yes | Hard; guards retained | No | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUES]** | **[INSERT VALUE]** |
| C3: full untrained cascade parent | Yes | Yes | Smooth + hard guards | No | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUES]** | **[INSERT VALUE]** |
| C4: conditional trained detector | Yes | Yes | Explicit trained missingness protocol | Yes | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUES]** | **[INSERT VALUE]** |

C4 is not a completed six-stage architecture and is not required to finish the primary three ablations. To isolate a learned decision layer, its comparison must provide the same available feature computation to both decision policies; do not confound trained classification with different judge-call availability. Fit a distinct compatible artifact for each judge digest. Conformal sets, when available, are a separate uncertainty output.

**Table A3. Full-parent removal summary, for the locked final test.**

| Variant | Groups / rows | BHS (%) | Paired $\Delta$BHS (pp), 95% interval | FPR (%) | FNR (%) | Valid verdicts (%) | Fast-path (%) | E2E median / p95 (ms) |
|---|---|---|---|---|---|---|---|---|
| C3 parent | **[INSERT VALUES]** | **[INSERT VALUE]** | Reference | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUES]** |
| A1: whole-answer replacement | **[INSERT VALUES]** | **[INSERT VALUE]** | **[INSERT VALUES]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUES]** |
| A2: positive-only selection | **[INSERT VALUES]** | **[INSERT VALUE]** | **[INSERT VALUES]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUES]** |
| A3: hard-only pooling | **[INSERT VALUES]** | **[INSERT VALUE]** | **[INSERT VALUES]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUES]** |

Tables use simple cells and explicit units for conversion to LaTeX `booktabs`. Escape underscores in model IDs when converting to LaTeX; retain placeholders until measurements exist. Captions must include split hash, judge digest, coverage policy, resampling unit, and experiment ID.

## 3. Hyperparameter sensitivity

Tune the parameters that change this detector rather than borrowing learning-rate/dropout grids from an unrelated neural architecture. The pretrained encoders and judges are frozen in this plan.

### 3.1 Primary cascade grid

**Table H1. Proposed development-only grid.**

| Parameter | Reference | Grid | Rationale / recomputation |
|---|---:|---|---|
| Smooth temperature $\tau$ | 0.15 | $\{0.05,0.15,0.30\}$ | Controls aggregation across claims; recompute from cached per-claim scores; singleton claims cannot reveal sensitivity |
| Shared high threshold $\gamma=\gamma_e=\gamma_c$ | 0.85 | $\{0.80,0.85,0.90\}$ | Trades routing coverage against route errors; keep $\lambda_c=0.15$ fixed |
| Selected passages per claim $M$ | 4 | $\{2,4,8\}$ | Trades evidence coverage against NLI cost/noise; requires a new configurable selector and sufficient candidate/NLI traces |

This defines 27 development configurations per selected judge. Keep RRF constant 60, retrieval-union cap 20, claim cap four, model revisions, prompt, and decoding fixed. Do not tune BHS track weights to obtain a preferred outcome: equal weights define the primary metric; alternate weights are reporting sensitivity, not a model-selection opportunity.

**Efficient staged execution:**

1. Evaluate $M=4$ for all nine $(\tau,\gamma)$ combinations using a shared, label-free score cache. Judge all required development rows once under the fixed protocol and assemble simulated cascades from those responses; count simulated calls separately from measured deployment cost.
2. At the development-selected pair, evaluate $M=2$ and $M=8$ with new passage selection and NLI as required. This is 11 unique configurations, not an exhaustive 27-point search.
3. Confirm the selected region with the declared seed/timing repetitions. Only execute all 27 configurations if the resource budget is approved; report the actual search, including stopping rules.
4. Freeze selected values before calibration and final testing. Report all development configurations, not only the winning point.

Existing pilot journals contain only the selected evidence protocol and cannot automatically supply an eight-passage counterfactual. Reuse is permissible only for identical model/input/selection fingerprints. Changing $\tau$ changes both aggregation and routing features; changing $M$ changes evidence and can change whole-answer features. Record this dependency explicitly.

**Selection rule:** minimize development operational BHS, then prefer lower measured end-to-end cost for genuine ties, then the configuration closest to the reference settings. If an error/cost tolerance band is used instead of exact ties, preregister it: **[INSERT VALUE OR STATE EXACT-TIE POLICY]**. Report the searched budget and guard against unsupported rankings from very small group counts.

### 3.2 Conditional trained-learner sensitivity

Only after approved feature bundles exist, vary the inverse regularization strength $C\in\{0.1,1,10\}$ with fixed source-group folds. The implementation uses the same $C$ in lexical and meta logistic regressions; changing it therefore changes both fits unless separately parameterized and documented. Current `fit` orchestration constructs the default model, so exposing this grid requires a tested configuration change, not an assumed CLI flag.

Hold the fold target at five where feasible, solver at L-BFGS, and iteration cap at 2,000. Fold count is a support constraint, not a score-maximizing knob. Predefine three repeated training split seeds $\{42,123,2026\}$ for robustness if group support and budget permit. Do not select the best seed. The current code does not automatically create a multi-seed ensemble. Select point thresholds from development only; keep conformal significance at $\alpha=0.05$ unless a separate uncertainty-coverage study is explicitly registered. Never select $\alpha$ on test BHS.

### 3.3 Tuning report tables

**Table H2. One-dimensional summaries from the declared grid, not fabricated results.** Hold the other parameters at the stated reference or selected setting and state which.

| Parameter / value | Other settings | Dev BHS (%) | Dev FPR / FNR (%) | Route coverage (%) | Routed BHS (%) | E2E ms/example | Independent groups |
|---|---|---|---|---|---|---|---|
| $\tau=0.05$ | $\gamma=0.85,M=4$ | **[INSERT VALUE]** | **[INSERT VALUES]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** |
| $\tau=0.15$ | $\gamma=0.85,M=4$ | **[INSERT VALUE]** | **[INSERT VALUES]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** |
| $\tau=0.30$ | $\gamma=0.85,M=4$ | **[INSERT VALUE]** | **[INSERT VALUES]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** |
| $\gamma\in\{0.80,0.85,0.90\}$ | One row per tested value | **[INSERT VALUES]** | **[INSERT VALUES]** | **[INSERT VALUES]** | **[INSERT VALUES]** | **[INSERT VALUES]** | **[INSERT VALUES]** |
| $M\in\{2,4,8\}$ | One row per tested value | **[INSERT VALUES]** | **[INSERT VALUES]** | **[INSERT VALUES]** | **[INSERT VALUES]** | **[INSERT VALUES]** | **[INSERT VALUES]** |
| Conditional $C\in\{0.1,1,10\}$ | One row per tested value and seed | **[INSERT VALUES]** | **[INSERT VALUES]** | Not comparable without declared trained routing policy | Not applicable if disabled | **[INSERT VALUES]** | **[INSERT VALUES]** |

Undefined routed BHS when one track is absent must be shown as undefined, with available track-specific rates. Export one machine-readable row per actual configuration; the grouped rows above are compact planning notation, not completed sweep rows.

### 3.4 Academic ablation discussion

**Discussion supported by the existing pilot.** The exploratory cascade produced heterogeneous effects across judges. It reduced Bangla-AI's BHS from 50% to 35%, increased both Qwen variants from 40% to 45%, and left Gemma at 30%. These outcomes do not support an architecture-wide improvement claim. Because the pilot is restricted to clustered QA examples and the cascade was added after the fixed-score condition, the results motivate controlled evaluation rather than confirm architectural necessity. The absence of neural influence on fixed-score decisions further prevents that condition from identifying the benefit of retrieval or NLI.

**Fill-in discussion after the planned ablations.** Relative to the locked cascade parent, replacing atomic claims with the whole candidate changed BHS by **[INSERT DELTA AND INTERVAL]** percentage points, with the largest change in **[INSERT PRESPECIFIED SLICE]**. Positive-only selection changed Track-B FNR by **[INSERT DELTA]**, while hard-only pooling changed Track-A FPR by **[INSERT DELTA]**. The findings **[INSERT SUPPORTED / DID NOT SUPPORT / WERE INCONCLUSIVE FOR]** the prespecified hypotheses. Any gain was accompanied by **[INSERT MEASURED COST OR COVERAGE CHANGE]**. These conclusions are limited to the audited sources, tasks, judge digests, and protocol; a null or adverse component effect favors simplification or further validation rather than selective omission.

# Phase 2 — Days 13–14: Error Analysis and Publication Assets

## 4. Quantitative error-analysis framework

The detector predicts binary hallucination labels. The twelve hallucination types are gold evaluation slices, not twelve predicted classes. A 12-class confusion matrix would therefore misrepresent the task.

### 4.1 Minimal analysis record

Produce a run-specific derived analysis CSV joined by validated IDs, never by row order. Keep inference traces label-free and perform this gold join only for scoring/analysis. Retain:

- Run/configuration ID, row ID, reviewed source-group ID, task, track, canonical type, context regime, and original raw text reference.
- Gold label; point verdict or explicit failure; decision source; score availability; two-label score if observed; fallback reason.
- Extracted claims; manual claim-coverage assessment; evidence IDs and positive/negative ranks; per-passage NLI; hard/smooth extrema; routing eligibility and route.
- Claim count, candidate/context length, actual tokenizer fertility, numeric/negation markers, context truncation indicators where measurable.
- End-to-end timing, per-stage timing, judge-call status, runtime state, and raw payload/checkpoint references.
- Human error-category labels, annotator IDs, adjudication status, and evidence-backed rationale.

Existing journals do not contain every proposed field. New instrumentation and reviewed source groups are required: **[INSERT ANALYSIS SCHEMA VERSION AND EXTRACTOR HASH]**. Missing fields stay missing; do not reconstruct reliable timing from unrecorded phases or assign source groups from IDs alone.

### 4.2 Failure taxonomy

**Table E2. Investigator-coded error categories, separate from dataset gold types.** Categories may overlap; percentages must identify their denominator.

| Code | Category | Operational diagnostic | Examples of audit questions |
|---|---|---|---|
| FP-P | Faithful paraphrase rejected | Gold 0, verdict 1 | Was a correct short name/inflection treated as insufficient support? |
| FP-C | Conflicting or noisy evidence | Gold 0, verdict 1, support and contradiction signals | Did unrelated passages or NLI label maxima create apparent conflict? |
| FN-F | Unsupported/fabricated detail accepted | Gold 1, verdict 0 | Did a mostly faithful answer hide one false detail? |
| FN-R | Relation, date, entity, or numeric confusion | Gold 1, verdict 0 | Was birth confused with death, or a signed number/range altered? |
| CLM | Claim-extraction omission/distortion | Manual mismatch between answer facts and extracted claims | Did the four-claim cap omit a decisive detail? Was a qualifier changed? |
| RET | Retrieval/reranking failure | Relevant row-context evidence present but not selected | Was counterevidence missed, or did candidate echo dominate? |
| NLI | Verification failure | Appropriate evidence selected but NLI disagrees with reviewed relation | Is Bengali negation, scope, coreference, or multi-step inference involved? |
| ROUTE | Incorrect fast-path commitment | Routed verdict wrong against gold | Was a hard guard insufficient? Were extracted-claim and answer coverage conflated? |
| JUDGE | Judge-tail misclassification | Wrong valid verdict after full verification | Was the score extreme or near the decision threshold? |
| OPS | Transport, invalid output, truncation, or missing score | Explicit stage/status failure or fallback | Was missingness handled honestly? Is failure distinct from a decoded fallback? |
| DATA | Potential ambiguity or incorrect benchmark annotation | Reviewed disagreement with source evidence | Preserve original label; use a separate adjudication record, not silent relabeling |

A high-score error is not automatically a calibrated high-confidence error. Define an **extreme-score diagnostic** at $p\geq0.90$ or $p\leq0.10$ for rows with observed scores only. State that this cutoff is descriptive and not calibrated. Bangla-AI fallback rows have no such score and must be excluded from this diagnostic with their count reported.

### 4.3 Required quantitative slices

Report pooled BHS and task-specific BHS with actual supports. Also report macro-task BHS, averaging only a prespecified set of tasks with both tracks available; never silently drop unsupported tasks. Provide Track-B miss rates for all twelve canonical types, including undefined rows for absent types. Summarization type mapping must be reviewed before these subtype claims are publishable.

Additional prespecified slices are open/closed context, single-/multi-claim answers, short/long candidates using training/development-defined cutoffs, numeric/negated answers, fast path versus judge tail, and scored versus decoded-fallback decisions. Script comparisons require reviewed meaning-equivalent material; comparing existing code-mixed and native tasks alone does not isolate a script effect.

**Table E3. Publication-ready task/type error template.**

| Slice | Source groups | $N_A$ | $N_B$ | FP + failures A | FN + failures B | FPR (%) | FNR (%) | Operational BHS (%) | Valid coverage (%) |
|---|---|---|---|---|---|---|---|---|---|
| Overall | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** |
| Bengali QA | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** |
| Code-mixed QA | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** |
| Summarization | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** |
| Mathematics | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** |
| Each Track-B type; expand to twelve rows | **[INSERT VALUE]** | 0 | **[INSERT VALUE]** | Not applicable | **[INSERT VALUE]** | Undefined | **[INSERT VALUE]** | Undefined | **[INSERT VALUE]** |

**Table E4. Routed-versus-judge-tail audit.** Report conditional errors without presenting an easier tail as evidence that the judge improved.

| Configuration | Eligible / all rows | Routed faithful / hallucinated | Routed FP / FN | Tail FP / FN | Failures | Actual judge calls | Shared prep time (s) | Full run wall time (s) |
|---|---|---|---|---|---|---|---|---|
| Locked parent | **[INSERT VALUES]** | **[INSERT VALUES]** | **[INSERT VALUES]** | **[INSERT VALUES]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** |
| Each ablation | **[INSERT VALUES]** | **[INSERT VALUES]** | **[INSERT VALUES]** | **[INSERT VALUES]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** | **[INSERT VALUE]** |

## 5. Qualitative error audit

For the existing pilot, inspect every unique erroneous row across all four judges and both completed conditions; preserve model-specific decisions but count unique sources and model–row errors separately. For the larger study, preregister a stratified audit sample: all operational failures plus up to 20 false positives, 20 false negatives, and 20 correct controls per task, sampled with seed 42 and reduced to the available count. Report the selected count and unique source groups; do not fill empty strata with invented examples.

Two Bengali-proficient annotators should independently review gold-blinded source evidence and candidate claims before seeing model scores or component attribution. For mathematical cases, add a reviewer qualified to verify the derivation. Then reveal traces for causal-stage coding. Adjudicate disagreements, distinguish label ambiguity from detector error, and report raw agreement plus an appropriately specified agreement statistic when support permits: **[INSERT ANNOTATOR QUALIFICATIONS, RUBRIC, CONSENT/ACCESS POLICY, AND AGREEMENT METHOD]**. Existing benchmark labels are not silently replaced.

**Table E5. Qualitative case-study format.** No actual excerpts are invented below.

| Case ID / source group | Task / gold / prediction | Bengali candidate excerpt + verified translation | Decisive context evidence | Trace diagnosis | Error code / adjudication |
|---|---|---|---|---|---|
| **[INSERT ID]** | **[INSERT VALUES]** | **[INSERT PERMISSION-CHECKED EXCERPT]** | **[INSERT EVIDENCE AND SOURCE OFFSET]** | **[INSERT CLAIM, RETRIEVAL, NLI, ROUTE, OR JUDGE FAILURE]** | **[INSERT VALUES]** |

Use short licensed excerpts, preserve Bengali and mathematical signs, and have translations reviewed. A plausible counterfactual explanation is not proof of causation; validate stage attribution with the corresponding controlled intervention where possible.

## 6. Publication plot specifications and Python template

The companion [plots.py](plots.py) is an offline Matplotlib script with an optional seaborn style. It reads actual observations only, validates inputs, refuses existing output directories, and writes vector PDF, 300-dpi PNG, byte-exact source CSV, metrics CSV, and JSON provenance. [test_plots.py](test_plots.py) contains synthetic software checks; those fixtures are not thesis evidence.

### 6.1 Figure specifications

| Figure | Axes / encoding | Scientific conditions | Caption requirements |
|---|---|---|---|
| F1: optimization diagnostic | Observed optimizer iteration vs training and development loss; separate seed trajectories | Optional grouped logistic learner only; new instrumented logs required; no existing epoch curves | Exact objective, iteration convention, fold, regularization, seed; development data is monitored, never fitted |
| F2: row-normalized confusion | Truth rows 0/1; prediction columns 0/1/failure; count and row percentage | All expected rows retained; compare models/configurations on identical IDs | $N_A,N_B$, source groups, operational BHS, valid coverage, failure policy |
| F3: efficiency frontier | End-to-end latency in ms/example vs balanced accuracy (%); true nondominated frontier | Complete valid verdicts, equal track weights, identical data/hardware/timing policy; no incomparable judge-only phase timings | Cold/warm definition, preparation allocation, repetitions, runtime/model digest, accuracy definition |
| F4: recommended additional sensitivity figure | $\tau$ vs $\gamma$ heatmap of dev BHS, faceted by $M$ | Only measured configurations; missing cells visibly unmeasured | Supports, fixed values, search budget; not a test-set selection plot |

Use readable single-/double-column sizes (approximately 3.4/7.0 inches), consistent fonts, colorblind-safe choices, distinguishable line styles, explicit units, and no decorative 3D effects. Do not add uncertainty bands without independent repeated measurements or an appropriate group-aware interval. Choose a locally available Bengali-capable font for excerpt figures and record it: **[INSERT FONT NAME AND VERSION]**. The provided aggregate plots use English axis labels and do not assume such a font is installed.

**Loss-curve limitation:** the current L-BFGS classifier exposes a fitted solution and iteration count, not saved train/development trajectories. Do not relabel an iteration cap as epochs, fabricate smooth curves, or claim neural fine-tuning. A new instrumented optimizer diagnostic must preserve the same objective and training-only preprocessing; otherwise label it a separate experiment. If unavailable, omit F1 with an explicit explanation.

**Latency limitation:** the existing cascade reuses shared features. Its judge request timing excludes preparation and is not comparable to a cold end-to-end baseline. Measure both deployment conditions afresh under one policy. Report cold-start and warm steady-state separately; for warm repeated trials, preregister five repeats per configuration if feasible, randomize configuration order, and record thermal/load conditions. Do not derive per-example p95 from five total-run averages. Per-request journal times are phase-specific, not necessarily complete per-example pipeline latency.

### 6.2 Exact CSV contracts

```text
# loss input; optional seed column
iteration,train_loss,val_loss,seed
# confusion input; blank prediction denotes FAILURE
id,source_group,y_true,prediction
# Pareto input; coverage must equal 1 and protocol_id must match across rows
configuration,latency_ms,balanced_accuracy_pct,coverage,protocol_id
```

These are schemas, not sample observations. Remove explanatory comment lines when preparing CSVs. For confusion, externally verify membership against the expected ID manifest before plotting. `--expected-rows` validates cardinality, not identity. The plotting script can visualize a complete expected roster with failed/unattempted rows encoded as failures; this conservative visualization must not relabel an interrupted experiment as a completed final run. The runner's completion-aware BHS remains authoritative.

Use one confusion input per model/configuration and all expected rows. Ground truth is joined only in the derived scoring file, not fed back into inference. Preserve missing probabilities for decoded fallbacks. A source-group assignment must come from the reviewed lineage manifest, not a guessed ID prefix.

For complete valid binary predictions with equal track weights,

$$
\operatorname{BalancedAccuracy}(\%)=100-\operatorname{BHS}.
$$

This complement is labeled balanced accuracy, never BHS. It is not raw accuracy on the imbalanced benchmark. The Pareto script rejects incomplete verdict coverage; compare systems with failures in a separate operational-BHS-versus-latency analysis rather than silently dropping those failures.

### 6.3 Commands

Run from the workspace root after preparing the named, real CSVs. The paths below define proposed new artifacts, not existing measurements. Replace each bold placeholder in the prose with the actual input/output paths before execution; do not run literal nonexistent inputs.

```sh
python 01_Thesis/Experiments/2026-11-09_days11_14_design/plots.py --help
python 01_Thesis/Experiments/2026-11-09_days11_14_design/plots.py loss --help
python 01_Thesis/Experiments/2026-11-09_days11_14_design/plots.py confusion --help
python 01_Thesis/Experiments/2026-11-09_days11_14_design/plots.py pareto --help
python -m unittest discover -s 01_Thesis/Experiments/2026-11-09_days11_14_design -p test_plots.py -v
```

Execution templates (replace capitalized path arguments; the plotting commands do not launch inference):

```sh
python 01_Thesis/Experiments/2026-11-09_days11_14_design/plots.py loss --input ACTUAL_LOSS_CSV --output-dir NEW_LOSS_FIGURE_DIRECTORY
python 01_Thesis/Experiments/2026-11-09_days11_14_design/plots.py confusion --input ACTUAL_SINGLE_MODEL_ANALYSIS_CSV --expected-rows ACTUAL_EXPECTED_COUNT --output-dir NEW_CONFUSION_FIGURE_DIRECTORY
python 01_Thesis/Experiments/2026-11-09_days11_14_design/plots.py pareto --input ACTUAL_MATCHED_TIMING_CSV --latency-policy "End-to-end wall time divided by the same evaluation count, including amortized preparation; fixed cold/warm policy" --output-dir NEW_PARETO_FIGURE_DIRECTORY
```

Record actual paths: **[INSERT LOSS CSV OR OMIT F1]**, **[INSERT CONFUSION CSV AND EXPECTED-ID MANIFEST]**, **[INSERT MATCHED TIMING CSV]**. Add `--style seaborn` only if installed; Matplotlib is sufficient. No dependencies are installed by the script. Store generated figures under a matching run's `results/` directory, then copy approved publication assets with provenance into `01_Thesis/Writing/figures/`.

# Chapter 3 Draft — Methodology and Experimental Setup

**Draft status:** original, unapproved prose. Present-tense descriptions refer to implemented mechanisms. Conditional or future-tense descriptions identify unexecuted extensions. Replace all marked fields and insert verified primary-source citations before submission. This chapter structure follows the requested combined methodology/setup chapter and does not silently renumber the existing thesis outline.

## 3.1 System Overview

This study investigates hallucination detection in candidate answers associated with low-resource Bengali natural-language-processing tasks. The input to the detector is

$$
x_i=(q_i,r_i,c_i,t_i),
$$

where $q_i$ is the question or task instruction, $r_i$ is the candidate answer, $c_i$ is optional grounding context, and $t_i$ denotes task identity. Task identity controls eligibility for mechanisms such as non-mathematical early routing; gold labels, tracks, and hallucination types are unavailable to inference. The binary target follows

$$
y_i=0\;\Longleftrightarrow\;\text{faithful},\qquad y_i=1\;\Longleftrightarrow\;\text{hallucinated}.
$$

The system evaluates the original candidate rather than generating a replacement. The corrected-v2 pipeline preserves raw text, extracts candidate claims, selects evidence from the current row's context, computes per-passage NLI features, and obtains a local judge verdict when required. In the fixed-score profile, NLI is diagnostic and the judge determines the point prediction. In the exploratory cascade profile, eligible unambiguous NLI outcomes bypass the judge. An optional trained profile combines numeric and lexical features through source-grouped logistic models; it was not fitted in the completed pilot.

The implemented data flow is summarized as

$$
(q_i,r_i,c_i)\rightarrow\{a_{ik}\}\rightarrow\{\mathcal E_{ik}\}\rightarrow\mathbf z_i\rightarrow\{\text{route},\text{judge}\}\rightarrow\hat y_i.
$$

The broader six-stage specification additionally proposes calibrated set prediction and optional evidence-grounded repair. Those proposals are distinguished from the executed detection profiles. Cross-script JSD, symbolic execution, and repair are not reported as executed components. A rigorous contribution statement therefore concerns reproducible task-aware evaluation and tested detector mechanisms, subject to comparison with verified prior work: **[INSERT VERIFIED RELATED-WORK AND CONTRIBUTION CITATIONS]**.

## 3.2 Architectural Components

### 3.2.1 Raw-text preservation and derived views

Raw Bengali strings and mathematical notation are retained unchanged. A separate canonicalization function $\kappa(\cdot)$ applies conservative Unicode/whitespace normalization for derived views and screening keys. Because normalization is not necessarily lossless, raw and canonical values are stored separately. No validated meaning-equivalent transliterator is supplied; the present native/code views are therefore conservative identity-derived views, not evidence of cross-script robustness.

For tokenizer $\mathcal T$ and nonempty word sequence $s$, token fertility is

$$
F_{\mathcal T}(s)=\frac{|\mathcal T(s)|}{W(s)},\qquad W(s)>0.
$$

Empty-denominator measurements remain undefined. The tokenizer, special-token policy, and word segmentation are recorded with the measurement. Fertility is a diagnostic feature, not a tokenizer optimization or a guaranteed explanation for errors.

Exact normalized 13-token hashes can support overlap screening, but only when a comparison corpus exists. The pilot records hash counts without a training-corpus comparison; overlap status therefore remains unknown. Hash screening does not establish source-lineage independence or model-pretraining independence.

### 3.2.2 Atomic claims and evidence selection

A fixed local claim extractor maps the question and candidate answer to at most four claims:

$$
\mathcal A_i=G(q_i,r_i)=\{a_{i1},\ldots,a_{iK_i}\},\qquad 1\leq K_i\leq4.
$$

The extractor may use the question to resolve short answers but receives neither gold labels nor source-context facts for constructing the claim text. Invalid extraction falls back to the unmodified whole candidate and records the failure. The four-claim cap and generation errors can omit content; successful extraction is not equivalent to complete semantic coverage.

For each claim, the implementation constructs positive and negative query forms:

$$
u_{ik}^{+}=q_i\mathbin\Vert a_{ik},\qquad
u_{ik}^{-}=q_i\mathbin\Vert\text{``Is it false that''}\mathbin\Vert a_{ik}.
$$

Here $\Vert$ denotes text concatenation. The negative hypothesis uses the fixed wrapper “It is not true that”. These literal English wrappers are an implementation choice, not a verified Bengali logical-negation operator. Their behavior is evaluated through the contrastive-selection ablation.

Only passages segmented from the same row's supplied context form the evidence pool $\mathcal D_i$. No other-row contexts, answer banks, benchmark labels, or generated benchmark negatives are added. Dense multilingual-E5 and sparse BM25 rankings are fused independently for each polarity:

$$
\operatorname{RRF}^{s}_{ik}(d)=\sum_{m\in\{\mathrm{dense},\mathrm{sparse}\}}\frac{1}{60+\operatorname{rank}^{s}_{m,ik}(d)},\qquad s\in\{+,-\}.
$$

Only lists containing $d$ contribute. Rank positions start at one. Balanced merging retains polarity provenance while producing at most 20 unique candidates. BGE reranks candidates against both query forms; balanced selection retains at most four unique passages per claim in the pilot. Negative-query evidence is potential counterevidence, not automatically support for the original claim. Closed-context rows without supplied evidence do not receive fabricated retrieval results.

### 3.2.3 Per-evidence NLI and smooth aggregation

For every selected passage $d\in\mathcal E_{ik}$, an NLI encoder evaluates the original claim. Define

$$
e_{ik}=\max_{d\in\mathcal E_{ik}}P_{\mathrm{NLI}}(\mathrm{entailment}\mid d,a_{ik}),\qquad
b_{ik}=\max_{d\in\mathcal E_{ik}}P_{\mathrm{NLI}}(\mathrm{contradiction}\mid d,a_{ik}).
$$

NLI operates on each passage separately, not a concatenation of retrieved evidence. Because label-wise maxima can come from different passages, their exported values need not sum to one and are not a calibrated truth distribution. Negated-claim and optional whole-answer NLI are recorded as additional diagnostics. QA consistency remains missing, and the pilot disables the optional XLM-R proxy.

For nonnegative normalized weights $w_{ik}$ summing to one and temperature $\tau>0$, smooth extrema are

$$
E_i^{(\tau)}=-\tau\log\sum_{k=1}^{K_i}w_{ik}\exp(-e_{ik}/\tau),\qquad
B_i^{(\tau)}=\tau\log\sum_{k=1}^{K_i}w_{ik}\exp(b_{ik}/\tau).
$$

The pilot uses uniform weights. Hard guards retain

$$
E_i^{\min}=\min_k e_{ik},\qquad B_i^{\max}=\max_k b_{ik}.
$$

As $\tau$ approaches zero, the smooth values approach the corresponding extrema over positive-weight claims. For singleton claims, smooth and hard values coincide. Incomplete extracted-claim evidence coverage disables early routing rather than averaging only the easy claims. This check does not guarantee completeness of the extractor itself.

Let $H_i$ indicate supplied context, a non-mathematical task, and complete usable NLI coverage. Define

$$
R_i^{0}=H_i\land(E_i^{(\tau)}\geq\gamma_e)\land(B_i^{\max}\leq\lambda_c),
$$

$$
R_i^{1}=H_i\land(B_i^{(\tau)}\geq\gamma_c)\land(E_i^{\min}<\gamma_e).
$$

Exactly one true condition permits the corresponding fast-path label; otherwise the row receives full verification. Routing is active only in the explicit cascade condition. Mathematical rows are ineligible for this NLI fast path; they can still receive ordinary judge verdicts, but no symbolic verification is claimed.

### 3.2.4 Local first-token judging

The judge receives only question, candidate answer, and optional context as untrusted data and is requested to output exactly one integer. When exact label-token log scores $\ell_0,\ell_1$ are available, the hallucination score is

$$
s_i=\frac{\exp(\ell_1)}{\exp(\ell_0)+\exp(\ell_1)}=\sigma(\ell_1-\ell_0),
$$

with point decision $\hat y_i=\mathbb I[s_i\geq0.5]$ in the untrained fixed-score condition. This is a two-label normalized first-token score, not automatically a calibrated full-vocabulary probability. Labels must be exact usable tokens; scores are missing if the required pair is absent from the available top-20 entries.

When a score is missing, the untrained condition may retain a strictly parsed decoded verdict and explicitly mark the fallback. It never fabricates $s_i$. If neither channel is valid, the outcome is $\bot$, an operational failure. One-token constrained output is not itself evidence of repetition collapse. Score-derived confidence analyses report their available denominator separately.

### 3.2.5 Optional grouped lexical/meta classification

The optional trained model uses 30 ordered numeric inputs: 16 semantic features, four judge features, and ten surface features. Unavailable QA, JSD, or other measurements remain missing. Character TF-IDF, capped at 4,000 features with word-boundary character $2$–$5$-grams and sublinear term frequency, feeds a lexical logistic classifier producing one scalar score.

Nested source-grouped cross-validation supplies lexical out-of-fold scores to each outer-training fold. Each fold fits its own median imputer and standardizer using training data only. The numeric and lexical logical inputs are augmented with 30 explicit numeric missingness indicators, yielding 61 internal meta-classifier inputs. All-missing training columns remain represented. Development, calibration, and test rows do not fit vocabularies, imputers, scalers, or classifiers.

For outer-fold model $f_b$, the ensemble estimate is

$$
\hat p_i=\frac{1}{B}\sum_{b=1}^{B}\sigma(\boldsymbol\beta_b^{\top}\widetilde{\mathbf z}_{ib}+\beta_{0b}),
$$

where $\widetilde{\mathbf z}_{ib}$ is that fold's transformed design. The fold count targets five and may decrease only when necessary for valid nested class/group support; infeasible grouped fitting fails explicitly. This is a fold ensemble, not a trained multi-seed ensemble. This component was not fitted for the completed corrected-v2 pilot.

### 3.2.6 Optional conformal sets and excluded mitigation

After model and hyperparameter selection, a separate calibration partition supplies scores

$$
a_j=1-\hat p(y_j\mid x_j).
$$

For $n_{\mathrm{cal}}$ calibration observations and chosen $\alpha$, let

$$
k=\left\lceil(n_{\mathrm{cal}}+1)(1-\alpha)\right\rceil,\qquad
q_\alpha=\begin{cases}a_{(k)},&k\leq n_{\mathrm{cal}},\\+\infty,&k>n_{\mathrm{cal}}.\end{cases}
$$

The prediction set is

$$
\Gamma_\alpha(x)=\{y\in\{0,1\}:1-\hat p(y\mid x)\leq q_\alpha\}.
$$

Under the required exchangeability assumptions and a fixed fitted scoring function, standard split conformal concerns marginal label coverage, not BHS, conditional accuracy, or correctness after singleton selection. Group-disjoint partitions alone do not make correlated derivative rows exchangeable. Before claiming coverage, specify a defensible calibration/test unit, such as one prespecified representative per independent source group, or implement and validate a separately justified group-level procedure: **[INSERT CONFORMAL UNIT, ASSUMPTIONS, AND VERIFIED CITATION]**. The current primitive does not implement such a group-level extension.

The runner can record prediction sets, but it does not yet operationalize abstention/escalation for empty or two-label sets. That policy and its cost must be implemented and registered before selective-system claims. No conformal performance is claimed for this pilot.

For completeness, the design's CAD primitive transforms aligned context-conditioned and prior logits as

$$
\widetilde{\boldsymbol\ell}=(1+\eta)\boldsymbol\ell_{\mathrm{context}}-\eta\boldsymbol\ell_{\mathrm{prior}},\qquad
P_{\mathrm{CAD}}=\operatorname{softmax}(\widetilde{\boldsymbol\ell}/T),\quad\eta\geq0,\;T>0.
$$

This mathematical primitive is not integrated into surgical repair. No repaired answers are evaluated, and original candidate labels would not be valid gold labels for newly generated repairs.

## 3.3 Objective Function and Optimization

The executed untrained profiles do not optimize trainable model weights. E5, BGE, mDeBERTa, and local judges are used as fixed pretrained components. A learning rate, dropout schedule, batch-training epoch count, or neural weight-decay sweep is therefore not applicable to the completed pilot.

For the conditional lexical and meta learners, define $p_i=\sigma(\boldsymbol\beta^{\top}\mathbf u_i+\beta_0)$ and binary cross-entropy

$$
\mathcal L_{\mathrm{BCE}}=-\frac{1}{n}\sum_{i=1}^{n}\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right].
$$

For unweighted observations, an equivalent scaled representation of the regularized objective is

$$
\mathcal J(\boldsymbol\beta,\beta_0)=\mathcal L_{\mathrm{BCE}}+\frac{1}{2Cn}\|\boldsymbol\beta\|_2^2.
$$

The implementation delegates objective conventions to the recorded scikit-learn version, uses L2-regularized `LogisticRegression` with L-BFGS, $C=1$ by default, no class weighting, and a maximum of 2,000 iterations. The intercept is not included in the displayed penalty. Record convergence warnings and actual iteration counts; the cap is not evidence of convergence. Exact optimizer tolerances and package defaults must be exported from the fitted estimator: **[INSERT RESOLVED ESTIMATOR PARAMETERS]**.

The point-threshold objective differs from the differentiable fitting loss. With context regime $r\in\{\mathrm{open},\mathrm{closed}\}$, choose development thresholds by

$$
(\theta_o^*,\theta_c^*)=\arg\min_{\theta_o,\theta_c}100\left[\frac{0.5}{N_A^{\mathrm{dev}}}\sum_{i\in A_{\mathrm{dev}}}\mathbb I(\hat p_i\geq\theta_{r_i})+\frac{0.5}{N_B^{\mathrm{dev}}}\sum_{i\in B_{\mathrm{dev}}}\mathbb I(\hat p_i<\theta_{r_i})\right].
$$

Use global development track denominators, not separate regime-local F1 objectives. The implementation searches regime-specific score breakpoints, includes all-positive/all-negative candidates, and resolves exact ties by proximity to 0.5 and then lower threshold. Calibration labels are reserved for conformal scores; test labels are reserved for final reporting. Neither BHS weights nor point thresholds are tuned on test outcomes.

**Training algorithm for the conditional extension:** (1) audit and freeze source groups; (2) extract label-free features; (3) generate inner grouped lexical out-of-fold predictions; (4) fit outer-fold preprocessing and meta classifiers; (5) ensemble development predictions and select thresholds/hyperparameters under the preregistered budget; (6) compute calibration scores after selection; (7) freeze the artifact, provenance, and feature protocol; (8) evaluate the untouched final test. No such trained artifact is claimed in the completed pilot.

## 3.4 Experimental Setup

### 3.4.1 Dataset and preprocessing

The supplied BenHalluEval CSVs comprise 16,000 rows, with 4,000 faithful Track-A examples and 12,000 hallucinated Track-B examples.

| Task | Track A | Track B | Total |
|---|---:|---:|---:|
| Bengali QA | 1,000 | 4,000 | 5,000 |
| Code-mixed QA | 1,000 | 4,000 | 5,000 |
| Summarization | 1,000 | 3,000 | 4,000 |
| Mathematics/reasoning | 1,000 | 1,000 | 2,000 |
| Total | 4,000 | 12,000 | 16,000 |

Original CSV bytes are preserved. Canonical records contain ID, task, track, type, question, context, candidate answer, and gold label. Code-mixed tasks use their code-mixed fields; summarization uses the source passage as context and a summarization instruction as the question. The benchmark's mathematical candidate construction must preserve both erroneous reasoning and final answer where supplied. Original IDs are not globally unique; deterministic qualified IDs support checkpointing but do not establish source independence.

The raw summarization labels `Intrinsic`, `Non-factual`, and `Factual Contradiction` require an explicit reviewed mapping before canonical subtype findings. Current documentation differs on how automatically this mapping is performed; a loader mapping is not evidence of scientific review. Supply **[INSERT APPROVED MAPPING FILE, HASH, REVIEWER, DEFINITIONS, AND AMBIGUITY POLICY]**. Verify source licensing and dataset creation metadata: **[INSERT VERIFIED DATASET CITATION AND TERMS]**.

Existing 120-row, 1,600-row, and 16,000-row files are overlapping evaluation subsets, not training/development/test partitions. The completed 20-row source-order pilot is QA-only. Planned partition counts and hashes follow the approved lineage-group protocol in Section 1.1 and must be supplied before training or confirmatory evaluation. No augmentation is used in the pilot. Any future synthetic/paraphrase augmentation must occur after group assignment, remain inside training sources, and receive its own provenance and validation protocol. No real dialect dataset or human dialect scores are available, so dialect conclusions are outside the present experiment.

### 3.4.2 Baselines and controlled comparisons

The baseline family comprises the four local judges in Table E1 under identical prompts, constrained labels, decoding, data, and failure policy. The fixed-score neural profile is a diagnostic condition, not an accuracy-enhanced model: it uses the same decision rule as judging unless explicit routing is enabled. The interrupted judge-only run is not a completed baseline experiment; a new output directory is required for a complete matched run under current code.

Comparisons include the locked cascade and the three full-parent ablations. A conditional trained comparison is permitted only after data approval and per-judge feature/artifact compatibility review. No benchmark-derived label memory, answer bank, gold-answer retrieval, or reference-project label transfer enters independent inference. Reference competition F1 results are not comparable to thesis BHS. Constant classifiers may be included as sanity baselines; their outcomes are analytically determined, not new empirical model findings.

### 3.4.3 Implementation details and reproducibility

The completed pilot used local CPU inference because GPU preflight failed with a CUDA/PTX incompatibility. The manifest records Windows 11, Python 3.13.9, NumPy 2.3.4, scikit-learn 1.7.2, PyTorch 2.14.1, Transformers 5.20.1, sentence-transformers 6.1.0, and Ollama 0.40.15. These are historical manifest entries, not the currently selected editor interpreter, and should be reproduced from a resolved environment export rather than assumed available. Record physical CPU, RAM, and any accelerator explicitly: **[INSERT HARDWARE MODEL, RAM, AND ENVIRONMENT EXPORT HASH]**.

| Setting | Recorded pilot value / scope |
|---|---|
| Judges | `bangla-ai-1.7b-q4`, `qwen3:1.7b`, `qwen2.5:1.5b`, `gemma3:1b`; exact digests in run manifests |
| Claim extractor | `qwen3:1.7b`; maximum four claims; 256 output tokens |
| Dense encoder | `intfloat/multilingual-e5-large`, revision `3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3` |
| Reranker | `BAAI/bge-reranker-v2-m3`, revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` |
| NLI encoder | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`, revision `8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c` |
| Local generation | Seed 42; temperature 0; context length 4,096; four CPU threads; GPU layers 0 |
| Verdict generation | One token; integer grammar; top-20 first-token log scores requested |
| Neural feature inference | Float32; batch size two; maximum sequence length 512; no neural training |
| Pooling / routing | $\tau=0.15$, $\gamma_e=\gamma_c=0.85$, $\lambda_c=0.15$ |
| Optional logistic training | L-BFGS, L2, $C=1$, at most 2,000 iterations; not run in pilot |
| Neural epochs / training learning rate | Not applicable |

Verify truncation behavior and report how many claim/evidence pairs exceed model sequence limits: **[INSERT TOKEN-LENGTH AUDIT]**. Cached checkpoints were loaded locally; no cloud inference or downloads were used for the recorded pilot. Model identity, tokenizer/quantization details, and prompts remain pinned in each manifest. Cached availability does not grant redistribution rights: **[INSERT VERIFIED MODEL/CODE LICENSE AND CITATION REGISTER]**.

Each run records data hashes, source hashes, model digests/revisions, decoding, commands, timestamps, runtime, and raw responses. Raw predictions reside in run-specific `outputs/`, derived reports in matching `results/`, and protocols in `Experiments/`. The original executed sources were archived under the corrected-v2 experiment and match all five source hashes in both completed manifests. Subsequent active-code cleanup/hardening invalidates resumption under those manifests; new runs must use new directories rather than editing recorded hashes. Exact seeds do not guarantee cross-version or cross-hardware determinism.

The pilot record reports 339 passing regression tests after checkpoint hardening. These are software checks, not performance evidence, and are historical results rather than a new run in this report. The unresolved preservation migration issue must remain disclosed until reconciled independently; the immutable original-file baseline must not be rewritten.

### 3.4.4 Evaluation metrics and failure policy

Let $A=\{i:y_i=0\}$ and $B=\{i:y_i=1\}$ with supports $N_A,N_B>0$. For valid binary outcomes,

$$
\operatorname{FPR}_A=\frac{\operatorname{FP}}{N_A},\qquad
\operatorname{FNR}_B=\frac{\operatorname{FN}}{N_B},
$$

and the primary metric is

$$
\operatorname{BHS}=100\left(w_A\operatorname{FPR}_A+w_B\operatorname{FNR}_B\right),\qquad w_A=w_B=0.5.
$$

For operational outcome $\widetilde y_i\in\{0,1,\bot\}$, failures count against their respective tracks:

$$
\operatorname{BHS}_{\mathrm{op}}=100\left[\frac{w_A}{N_A}\sum_{i\in A}\mathbb I(\widetilde y_i\in\{1,\bot\})+\frac{w_B}{N_B}\sum_{i\in B}\mathbb I(\widetilde y_i\in\{0,\bot\})\right].
$$

Unattempted expected rows prevent a final completed-run BHS in the runner; distinguish those rows from attempted failures. Valid-only BHS uses the valid support of each track and is undefined if either valid track is absent. Overall BHS is undefined when either required track is absent. Report partial-run metrics explicitly as partial rather than final.

Verdict coverage is

$$
\operatorname{Coverage}=\frac{\#\{i:\widetilde y_i\in\{0,1\}\}}{N_A+N_B}.
$$

Also report the fraction with usable two-label scores and the decoded-fallback fraction. Point-verdict coverage and probability-score coverage are not the same quantity. A fallback may be a valid verdict without a usable probability.

For valid binary predictions, secondary measures include

$$
\operatorname{Precision}_1=\frac{\operatorname{TP}}{\operatorname{TP}+\operatorname{FP}},\quad
\operatorname{Recall}_1=\frac{\operatorname{TP}}{\operatorname{TP}+\operatorname{FN}},\quad
F_{1,1}=\frac{2\operatorname{TP}}{2\operatorname{TP}+\operatorname{FP}+\operatorname{FN}},
$$

$$
\operatorname{Accuracy}=\frac{\operatorname{TP}+\operatorname{TN}}{N_{\mathrm{valid}}},\qquad
\operatorname{BalancedAccuracy}=1-\frac{\operatorname{FPR}_A+\operatorname{FNR}_B}{2}.
$$

The displayed secondary accuracy/precision/recall use the declared valid-only roster; report its coverage. They do not silently replace operational BHS. Undefined denominators remain explicit. Macro-F1, if reported, averages defined class-specific F1 values only under a prespecified undefined-class policy; no automatic missing-class zero is introduced. On complete valid equal-weight predictions, BHS equals 100 times one minus balanced accuracy, not one minus ordinary accuracy on the imbalanced dataset.

Per-type Track-B miss rates use each type's actual support. Pooled BHS and macro-task BHS are distinct aggregations and must be named separately. Ranking under alternative BHS weights is a sensitivity analysis using fixed predictions, not a new selected headline metric.

For complete comparable runs, end-to-end amortized latency is

$$
L_{\mathrm{E2E}}=\frac{T_{\mathrm{loading}}+T_{\mathrm{claims}}+T_{\mathrm{retrieval/rerank/NLI}}+T_{\mathrm{judging}}+T_{\mathrm{decision/reporting}}}{N},
$$

when those phases are sequential and measured without overlap; otherwise measure total wall time directly rather than summing overlapping phase times. Report warm versus cold policy, measured repetitions, throughput, and per-example median/p95 only where complete per-example timings exist. Multi-model shared-feature amortization is a different deployment scenario from a single-model system and must be labeled separately. Judge-call reduction does not establish equal latency reduction.

**Deferred dialect agreement measures.** If a real, licensed dialect dataset and independent human/primary scores are later supplied, the evaluator additionally supports Lin's concordance correlation coefficient. For valid score pairs $(h_i,s_i)$, define population means, variances, and covariance on the same retained pairs:

$$
\operatorname{CCC}(h,s)=\frac{2\operatorname{Cov}_{\mathrm{pop}}(h,s)}{\operatorname{Var}_{\mathrm{pop}}(h)+\operatorname{Var}_{\mathrm{pop}}(s)+(\bar h-\bar s)^2}.
$$

No arbitrary epsilon is added. Zero-denominator cases, including identical constant arrays, remain undefined; missing/invalid pairs and valid-pair coverage are reported. The Chittagong bias-sensitivity comparison is

$$
\operatorname{CBS}=\operatorname{CCC}_{\mathrm{anchor}}-\operatorname{CCC}_{\mathrm{Chittagong}}.
$$

Preregister whether the anchor is Standard Bengali or Tangail and define matched semantic content, annotation provenance, and valid support: **[INSERT DIALECT SOURCE, ANCHOR, ANNOTATION PROTOCOL, AND VERIFIED METRIC CITATION]**. CBS is measured in CCC units, not accuracy percentage points. Neither CCC nor CBS is empirically evaluated in the current thesis pilot because no real dialect data or human scores are supplied.

### 3.4.5 Statistical analysis and validity threats

Primary comparisons are paired at identical source groups and use the fixed metrics and uncertainty plan of Section 2.3. The final test is inspected only after prompts, configurations, thresholds, and analysis choices are frozen. Error analysis may motivate a subsequent experiment, but cannot retroactively convert the already inspected test into a fresh holdout.

Principal validity threats are clustered source derivatives; unreviewed summarization taxonomy; limited pilot task coverage; claim truncation or semantic omissions; imperfect Bengali negation wrappers and NLI transfer; score-channel differences across judges; operational truncation; pretrained-model contamination that cannot be certified from local files; and incomplete hardware/runtime comparability. No evidence supports a broad script-bias or dialect-bias conclusion yet. Reference competition results and lineage-assisted lookups remain separate from independent thesis evidence.

# 7. Days 11–14 execution and decision checklist

| Day | Work | Concrete deliverable | Stop/go gate |
|---|---|---|---|
| 11 | Audit lineage, taxonomy, partitions; register hypotheses and instrument new settings | Signed protocol, source-group manifest, grid/config schemas, tested variant code | No approved splits: stop fitting and label work design-only |
| 12 | Execute declared development grid and three controlled ablations; lock decisions | Full development sweep CSV, cost ledger, selected configuration with rationale | Never tune on final-test outcomes; record null/adverse effects |
| 13 | Run locked evaluation if gates pass; audit failures and derive figures/tables | Complete prediction roster, paired group analysis, error-code records, figures with provenance | Incomplete runs remain incomplete; missing evidence stays explicit |
| 14 | Complete the Chapter 3 draft and review claims against evidence | Reviewed methodology/setup prose, final caption/table checklist, next-experiment decision | No unsupported claims or unverified citations promoted to thesis findings |

The schedule is conditional on CPU runtime and data review. It is not a promise that new experiments can complete in four calendar days. Estimate a development run from measured stage costs first: **[INSERT COMPUTE BUDGET, EXPECTED WALL TIME, AND STOPPING RULE]**.

## 7.1 Protocol registration fields before new execution

- Identity: **[INSERT NEW RUN ID, RESEARCHER, STATUS, APPROVAL]**.
- Intended scope: independent candidate-answer detection; no answer-bank/label-memory assistance.
- Data: **[INSERT SPLIT PATHS, SHA-256 VALUES, GROUP COUNTS, OVERLAP EXCLUSIONS]**.
- Mapping and licenses: **[INSERT REVIEWED TAXONOMY FILE, TERMS, ATTRIBUTION]**.
- Sources adapted: active corrected-v2 modules only unless separately authorized; no imported notebook execution.
- Code/config: **[INSERT SOURCE HASHES, VARIANT DIFF, SETTINGS, FEATURE ORDER]**.
- Runtime: **[INSERT ENVIRONMENT EXPORT, HARDWARE, MODEL DIGESTS, TOKENIZERS]**.
- Randomness: seed 42 for initial split/sampling; additional seeds only as explicitly declared above.
- Metrics: operational BHS primary; per-track errors, verdict/score coverage, source-group uncertainty, and E2E cost secondary.
- Commands: **[INSERT EXACT EXECUTED COMMANDS, START/END TIMES, EXIT STATUS]**.
- Artifacts: new `01_Thesis/Results/BenHalluEval/` run directories with `outputs/` and `results/`; protocol under the matching `Experiments/` directory.
- Deviations and next decision: **[INSERT ALL CHANGES, FAILED RUNS, AND STOPPING DECISIONS]**.

## 7.2 Preparation validation — not model evaluation

The report preparation includes read-only recomputation of the eight saved BHS values, 80 unique records per completed condition, the 32-response interrupted attempt, byte-identical shared-feature journals, the pilot dataset hash, and the five executed-source hashes against both completed manifests. These checks support consistency of saved artifacts, not scientific independence or final benchmark validity.

The plotting template's synthetic unit tests validate CSV handling, confusion/failure metrics, Pareto dominance, rendering, provenance, and overwrite protection. They do not create empirical curves, fit models, contact APIs, or establish detection quality. Fresh preparation validation on 2026-11-09:

- `python -m unittest discover -s 01_Thesis/Experiments/2026-11-09_days11_14_design -p test_plots.py -v`: **17 tests passed**, including actual temporary PDF/PNG rendering. Synthetic data and figures were confined to temporary test directories.
- Editor diagnostics for `plots.py`: no errors or warnings.
- `python tools/workspace.py catalog`: completed; derived catalogs refreshed. The initial regeneration cataloged 467 files; rerun after final document edits to refresh their hashes.
- `python tools/workspace.py verify`: exited with status 1 for the existing layout migration problem: 294 missing recorded paths, zero changed, zero unsafe, zero errors. This is not a successful preservation verification. Original inputs and the immutable baseline were not rewritten.
- The historical 339-test active-pipeline result is not rerun here. This preparation adds planning documents and plotting utilities only; no new model evaluation was executed.

## 7.3 Submission-readiness checklist

- [ ] Verified thesis contribution and dataset authorship statement, not inferred benchmark ownership.
- [ ] Source-disjoint partition audit and reviewed summarization mapping.
- [ ] New configurable ablation implementations tested against their declared controls.
- [ ] Full sweep logs and negative results retained; no winner-only reporting.
- [ ] Final test untouched until configuration freeze.
- [ ] Every table includes source groups, actual track support, coverage, and experiment identity.
- [ ] Every plot has input hashes, generation command, versions, and appropriate timing policy.
- [ ] No invented loss curves, confidence bounds, transliteration measurements, human ratings, or repairs.
- [ ] Conformal claims withheld until unit/exchangeability assumptions and operational set handling are justified.
- [ ] Citations and licenses verified from original sources; unresolved fields remain visibly marked.
- [ ] Supervisor approves this draft before it is copied into approved thesis writing.
