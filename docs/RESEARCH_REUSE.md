# Research reuse map: Huntrix and CUET_AL_Masaar

## Scope and preservation
This is a static asset map, not a reproduction or permission to run code. Preserve both repositories intact; adapt selected assets outside them, never refactor reference files.
Links use the preserved locations relative to this document:
- `../references/IUTCSFESTDATATHON-main/` — Huntrix.
- `../references/IUT_DATATHON_HALLU-main/` — CUET_AL_Masaar.

No reference code, import, notebook, test, build, download, or installation was executed for this organization task. Stored notebook outputs are historical author artifacts.

## Critical label adapter
**Both references: `0 = hallucinated`, `1 = faithful`. Thesis: `1 = hallucinated` (Track_B), `0 = faithful` (Track_A).**
After validating a verdict as a non-null binary integer, explicitly convert:
```python
thesis_label = 1 - reference_label
```
Validate before subtraction: reject malformed text, missing values, nonbinary numbers, and ambiguous outputs; do not coerce them to zero or invert twice.
Preserve the raw verdict and record adapter/version. Test both labels, invalid inputs, probability orientation, and ID alignment.
Invert probabilities only after identifying their event: `p_thesis_hallucinated = 1 - p_reference_faithful`; CUET's `p_hallucinated` needs no inversion.
Map `question`/`candidate_answer` to `prompt_bn`/`response_bn` explicitly. Preserve raw text and IDs; CUET sorts IDs, so output alignment cannot be assumed.

## Reuse statuses and scientific boundary
| Status | Meaning here | Independent evaluation eligibility |
|---|---|---|
| `adapted-with-tests` | Candidate to port with attribution and new tests; not already adapted | Only after removing gold access and validating on separate development data |
| `inspect-only` | Architecture, failure analysis, or deployment reference; not a drop-in dependency | Inspection establishes no performance claim |
| `lineage-assisted` | Uses benchmark/source answers, generated negatives, or label lookup | Separate disclosed reproduction/ablation, never silently part of the independent baseline |

Neither full pipeline is clean for independent BenHallu evaluation as shipped.
Huntrix `load_benhallu_lineage()` maps benchmark `correct_answer` to 1 and `hallucinated_answer` to 0 using normalized context/question/response keys.
The final v91 wrapper explicitly supplies `qa_gt_1000.csv` and `qa_4000.csv`. CUET also builds keys from released BenHallu gold/generated answers and QA sources.
Public availability, exact matching, and absence of hidden competition labels do not make benchmark-target recovery independent. Never silently load benchmark gold into retrieval.
For a clean independent experiment:
- Exclude gold/generated benchmark records, derivative answer banks, label transfers, saved predictions, and target-derived synthetic variants from inference/training/evidence.
- Audit TyDiQA, Suman QA, exam, idiom, and pattern corpora for shared lineage; exclude target-prompt-specific verified facts from the primary experiment.
- Split source questions, contexts, entities/headwords, and generation seeds before augmentation; row-random splits can leak related examples.
- Freeze prompts, rules, evidence, thresholds, and calibration using separate development data; evaluation labels belong in scoring only.
- Trace evidence provenance; report clean and lineage-assisted results separately, including coverage, abstention, and judge-tail performance.

## Read order and artifact authority
1. Huntrix [AGENTS.md][h-agents], [README][h-readme], then [data register][h-data].
2. Latest [v91 Python entrypoint][h-v91] and sibling notebook/manifest; [final freeze][h-freeze] pins notebook v3, runtime v4, CUDA v2, model v1.
3. [Technical companion][h-paper], [bottleneck analysis][h-bottleneck], and disclosures; bottleneck narrative retains older research-candidate status.
4. [Frozen v85 source][h-v85] as earlier fallback, not latest inference; [archive notes][h-archive] last, because their “current v85” wording is stale.
5. CUET [README][c-readme], actual [six-cell notebook][c-notebook], then [report][c-report].
CUET's actual file is `CUET_AL_Masaar_inference_notebook.ipynb`; README's `notebook.ipynb` name is wrong.
Huntrix [final release package][h-release] contains `Huntrix_inference_notebook.ipynb`, `FREEZE_MANIFEST.json`, `RUNBOOK.md`, disclosures, and scholarly artifacts.
`release/phase2-public-submission/` is older than `release/phase2-v91-public-submission/`. Current `scripts/` are development sources, not proof of byte identity with pinned external runtime scripts.

## Huntrix: exact reusable assets
| Status | Source | Symbols / practical boundary |
|---|---|---|
| `adapted-with-tests` | [datathon_utils.py][h-utils] | `clean_context`, `clean_text`, `normalize_bn`, `bn_tokens`, `prepare_frame`; schema/null handling |
| `adapted-with-tests` | [source_answer_bank.py][h-bank] | `canonical_answer`, `structural_answer`, `strip_inline_choices`, `number_signature`; isolate from bank loading |
| `adapted-with-tests` | [context_numeric_overrides.py][h-numeric] | `numeric_components`, `word_numeric_components`, `numeric_equivalent`, `relation_aligned`, `split_sentences`, `decide` |
| `adapted-with-tests` | [context_span_overrides.py][h-span] | `content_tokens`, `context_span_decision`; clause-bound positive support, otherwise abstain |
| `adapted-with-tests` | [fuzzy_source_overrides.py][h-fuzzy] | `relation_signature`, `relation_compatible`, `match_features`, `eligible`; relation-preserving match gates |
| `adapted-with-tests` | [structured_overrides.py][h-math] | `solve_math`, `solve_samas`, `solve_sandhi`, `solve_prefix`, `answer_matches`, `apply_structured_overrides` |
| `adapted-with-tests` | [typed_evidence.py][h-typed] | `infer_intent`, `load_typed_evidence`, `typed_evidence_decision`; independently sourced records only |
| `adapted-with-tests` | [typed_idiom_matcher.py][h-idiom] | `extract_typed_idiom`, `TypedIdiomMatcher.decide`; headword/alias priority and conflict abstention |
| `adapted-with-tests` | [normalize_phase2_raw.py][h-normalize] | `chunks`, `write_record`, `normalize_text_file`, `normalize_news`; provenance/license-aware conversion |
| `adapted-with-tests` | [evidence_index.py][h-index] | `evidence_doc`, `query_text`, `build_index`, `load_index`, `topk_indices`; CPU sparse retrieval |
| `adapted-with-tests` | [hybrid_evidence_index.py][h-hybrid] | `relation_types`, `source_relations`, `read_corpus`, `retrieval_queries`, `build`, `retrieve`; char/word TF-IDF |
| `adapted-with-tests` | [retrieval_consensus_overrides.py][h-consensus] | `cluster_votes`, `decide`; relation/source-count gates, not proof of source independence |
| `adapted-with-tests` | [datathon_utils.py][h-utils] | `metric_row`, `print_metrics`, `evaluate_by_slice`; adapt positive class, not thesis BHS/CCC definitions |
| `inspect-only` | [train_portable_verifier.py][h-calib] | `tune_threshold`, `cross_validate`; development scaffold, not transferable calibration |
| `inspect-only` | [phase2_pipeline.py][h-router] | `load_input`, `context_decision`, stage traces, uncertainty queue; separate from gold lookup |
| `inspect-only` | [gguf_selective_judge.py][h-judge] | `parse_binary`, `stable_seed`, `make_prompt`, `judge_row`; constrained GPU judging |
| `lineage-assisted` | [phase2_pipeline.py][h-router] | `load_benhallu_lineage`, `benhallu_decision`; benchmark answer/generated-negative lookup |
| `lineage-assisted` | [structured_overrides.py][h-math] | `apply_sample_transfer`, `apply_external_gold`; standalone CLI enables sample transfer unless disabled |

`run_phase2_cascade.py` calls a Transformers/bitsandbytes judge, not the frozen GGUF route.
`package_phase2_runtime.py` defines public runtime closure without saved submissions; `package_offline_artifacts.py` also lists historical predictions/training artifacts.

## CUET: useful cell functions
All cell references are in the actual [CUET notebook][c-notebook]. There is no separate test suite in that repository; adapt Huntrix analogues and add new tests.
| Cell | Status | Functions / boundary |
|---|---|---|
| 1 | `adapted-with-tests` | `norm_df`: null/context schema; preserve IDs/raw text and avoid implicit row reordering |
| 2 | `adapted-with-tests` | `bn_norm`, `qn2`, `deep_norm`, `numvals`: isolate lexical keys from signed mathematical structure |
| 2 | `lineage-assisted` | `goldkey`, `stack3`: gold-key/rule joins; exclude attached benchmark keys from clean evaluation |
| 3 | `inspect-only` | `judge_margin`, `run_margins`, `LogisticRegression`: judge/calibration design; identity-safe checkpoints needed |
| 4 | `adapted-with-tests` | `solve_math`, `solve_math2`, `repair_general`: templates/operand repair; test false positives and abstention |
| 5 | `adapted-with-tests` | `retrieve_entity_bio`: local E5/Wikipedia retrieval idea; rebuild licensed, lineage-audited evidence |
| 6 | `inspect-only` | `build`: rules/judge/math/consensus composition; trace and test precedence conflicts |

Cell 3 uses `logit("0") - logit("1")`: hallucinated-minus-faithful margin.
`LogisticRegression` fits 299 anchors against `(y == 0)`; sigmoid output is `p_hallucinated`, and reference verdicts are `(p_hallucinated < 0.340).astype(int)`.
Do not copy threshold 0.340 or treat anchor-fit performance as independent calibration. Refit and assess calibration/thresholds on separate held-out development data.
Label-token code takes the first encoded token; validate distinct single-token encodings before reuse.
Fixed `/kaggle/working/` caches `margins_anchor.npy`, `ok_anchor.npy`, `margins_test.npy`, `ok_test.npy` do not verify dataset hash, IDs/order, model, or prompt identity.
Changing `TEST_PATH` alone is unsafe with stale caches; filenames are not identity checks. Repeated OOM may return margin 0/status 0: retain failure status, not a valid verdict.

## Testing analogues and pitfalls
| Existing test analogue | Required thesis regression cases |
|---|---|
| [test_datathon_utils.py][t-utils] | Digit forms, nulls, label inversion, invalid labels, raw-text preservation |
| [test_context_numeric_overrides.py][t-numeric] | Wrong relation/year, ranges, compound number words, units, contradiction/abstention |
| [test_context_span_overrides.py][t-span] | Semicolon clauses, incidental copied answers, numeric/date exclusion |
| [test_structured_overrides.py][t-math] | Signs, fractions, intervals, distractor numbers, malformed operands, no label transfer |
| [test_fuzzy_source_overrides.py][t-fuzzy] | Birth/death, source/confluence, writing/publication near-duplicate swaps |
| [test_hybrid_evidence_index.py][t-hybrid] | Relation metadata, corpus provenance, prompt-only vs response-conditioned retrieval |
| [test_gguf_selective_judge.py][t-judge] | Invalid verdicts, stable ordering, failure handling, label/probability orientation |
| [test_phase2_v91_kaggle_inference.py][t-v91] | Paths/settings/raw-input contract; static checks are not GPU reproduction |

Huntrix `parse_label` silently defaults malformed output to 0; never adopt that fallback. `parse_binary` raises on no label but searches text; active code should validate the entire verdict.
`normalize_bn` strips hyphens/punctuation: aggressive mathematical normalization can erase signs/ranges. Use structural representations and semantic-distinction tests.
`apply_context_rule` is substring presence/absence, not general entailment. Huntrix threshold tuning and heuristic confidence are not probability calibration.
CUET anchor assertions and Huntrix controlled-pair precision are development checks, not generalization proof. Distinct URLs/repositories do not establish independent corroboration.
Candidate-response-conditioned retrieval and indexes containing test contexts can introduce confirmation bias/transduction; declare the retrieval protocol explicitly.

## Runtime and missing local resources
Organization does not install dependencies, attach Kaggle inputs, or run notebook cells.
Huntrix final execution expects offline Kaggle T4 x2, Qwen3.6-27B Q4_K_P GGUF, public runtime v91/v4, and llama.cpp CUDA b9637/v2; the wrapper rejects other GPU layouts.
CUET expects T4 x2, Qwen2.5-14B-Instruct fp16 with CPU spill, and `bhd-all-inputs`, including local multilingual-e5-base, Wikipedia passages/embeddings, and rule corpora.
Those external weights, indices, and bundles are not supplied by the local CUET repository.
Huntrix `outputs/` is absent: answer banks, lexical corpus, BenHallu CSVs, idiom export, legal evidence, verifier models, and retrieval indices are not locally available.
`output/` (singular) contains reports/presentations, not inference assets. No local GGUF or llama-server binary was found; Kaggle metadata is not a runtime.
The cited organizer-run `huntrix-1.ipynb` is absent. Packaging tests require missing assets; idiom tests can skip without the corpus.
Huntrix requirements are unpinned, mix CUDA/Apple MLX dependencies, and omit test dependency `nbformat`; `/Users/seyam/...` and `.venv/bin/python` commands are not Windows instructions.

## Active thesis workflow and scholarly use
- `evaluation/`: new adapters, metrics, tests, extraction provenance, and attribution; no imports that execute reference notebooks.
- `data/raw/benhallu/`: immutable benchmark inputs, never a reference knowledge bank; scoring labels stay outside inference.
- `outputs/`: generated traces/checkpoints; `results/`: scored tables/manifests; `experiments/`: frozen configurations and leakage ablations.
- `writing/`: chapters, figures, tables, citations. These are active root paths, not relocation targets for selected reference files.
The existing `eval.py`, `make_splits.py`, and `test_pipeline.py` were moved together into `evaluation/` without byte changes. Run them from the workspace root as `python evaluation/<script>.py`; thesis scoring semantics are unchanged.
Read Huntrix [PDF][h-pdf]/[LaTeX][h-tex] and CUET [report][c-report]/[presentation][c-slides] as methodological sources, not automatically validated evidence.
Record source version/hash, license, borrowed symbols, changes, and leakage exclusions; cite benchmark design sources and teams, distinguishing their claims from our runs.
Huntrix reports Phase-1 private macro-F1 0.982 and Phase-2 score 0.90144; CUET reports Phase-1 private score 0.937. All are author-reported, not verified here.
Different splits, source access, label orientation, and metric descriptions prevent direct comparison. Competition F1/ranks are not thesis BHS/CCC; never substitute them into thesis result tables.

## Copyright and license verification
Preserve [Huntrix MIT][h-license] and [CUET MIT][c-license] notices/attribution. Original-code MIT does not relicense models, benchmark data, evidence, or third-party report content.
Review [third-party notices][h-notices] and [data register][h-data]: recorded terms include Apache-2.0, CC BY/CC BY-SA, GPL-3.0, noncommercial datasets, and undeclared licenses.
Public accessibility is not redistribution permission. BenHallu licensing is unresolved in the Huntrix register; verify upstream terms and individual CUET bundle/index/model licenses.
Huntrix [bn_num2words.py][h-numwords] attributes its table/grouping to LGPL-2.1 num2words. A dedicated LGPL notice is absent from the inspected license folder/notices.
Verify upstream attribution, applicable LGPL obligations/license text, and evidence-source permissions before extraction or redistribution; do not treat this dependency as blanket MIT.
Keep competition files private as required. Preserving a repository does not grant permission to publish its data.

[h-agents]: ../references/IUTCSFESTDATATHON-main/AGENTS.md
[h-readme]: ../references/IUTCSFESTDATATHON-main/README.md
[h-data]: ../references/IUTCSFESTDATATHON-main/docs/DATA_SOURCES.md
[h-v91]: ../references/IUTCSFESTDATATHON-main/kaggle_notebooks/phase2_v91_inference/phase2_v91_inference.py
[h-v85]: ../references/IUTCSFESTDATATHON-main/kaggle_notebooks/phase2_v85_inference/phase2_v85_inference.py
[h-freeze]: ../references/IUTCSFESTDATATHON-main/docs/FINAL_SUBMISSION_FREEZE.md
[h-paper]: ../references/IUTCSFESTDATATHON-main/docs/TECHNICAL_PAPER.md
[h-bottleneck]: ../references/IUTCSFESTDATATHON-main/docs/BOTTLENECK.md
[h-archive]: ../references/IUTCSFESTDATATHON-main/archive/README.md
[h-release]: ../references/IUTCSFESTDATATHON-main/release/phase2-v91-public-submission/README.md
[h-utils]: ../references/IUTCSFESTDATATHON-main/scripts/datathon_utils.py
[h-bank]: ../references/IUTCSFESTDATATHON-main/scripts/source_answer_bank.py
[h-numeric]: ../references/IUTCSFESTDATATHON-main/scripts/context_numeric_overrides.py
[h-span]: ../references/IUTCSFESTDATATHON-main/scripts/context_span_overrides.py
[h-fuzzy]: ../references/IUTCSFESTDATATHON-main/scripts/fuzzy_source_overrides.py
[h-math]: ../references/IUTCSFESTDATATHON-main/scripts/structured_overrides.py
[h-typed]: ../references/IUTCSFESTDATATHON-main/scripts/typed_evidence.py
[h-idiom]: ../references/IUTCSFESTDATATHON-main/scripts/typed_idiom_matcher.py
[h-normalize]: ../references/IUTCSFESTDATATHON-main/scripts/normalize_phase2_raw.py
[h-index]: ../references/IUTCSFESTDATATHON-main/scripts/evidence_index.py
[h-hybrid]: ../references/IUTCSFESTDATATHON-main/scripts/hybrid_evidence_index.py
[h-consensus]: ../references/IUTCSFESTDATATHON-main/scripts/retrieval_consensus_overrides.py
[h-calib]: ../references/IUTCSFESTDATATHON-main/scripts/train_portable_verifier.py
[h-router]: ../references/IUTCSFESTDATATHON-main/scripts/phase2_pipeline.py
[h-judge]: ../references/IUTCSFESTDATATHON-main/scripts/gguf_selective_judge.py
[t-utils]: ../references/IUTCSFESTDATATHON-main/tests/test_datathon_utils.py
[t-numeric]: ../references/IUTCSFESTDATATHON-main/tests/test_context_numeric_overrides.py
[t-span]: ../references/IUTCSFESTDATATHON-main/tests/test_context_span_overrides.py
[t-math]: ../references/IUTCSFESTDATATHON-main/tests/test_structured_overrides.py
[t-fuzzy]: ../references/IUTCSFESTDATATHON-main/tests/test_fuzzy_source_overrides.py
[t-hybrid]: ../references/IUTCSFESTDATATHON-main/tests/test_hybrid_evidence_index.py
[t-judge]: ../references/IUTCSFESTDATATHON-main/tests/test_gguf_selective_judge.py
[t-v91]: ../references/IUTCSFESTDATATHON-main/tests/test_phase2_v91_kaggle_inference.py
[h-pdf]: ../references/IUTCSFESTDATATHON-main/output/pdf/Huntrix_Technical_Report.pdf
[h-tex]: ../references/IUTCSFESTDATATHON-main/docs/latex/Huntrix_Technical_Report.tex
[h-license]: ../references/IUTCSFESTDATATHON-main/LICENSE
[h-notices]: ../references/IUTCSFESTDATATHON-main/THIRD_PARTY_NOTICES.md
[h-numwords]: ../references/IUTCSFESTDATATHON-main/scripts/bn_num2words.py
[c-readme]: ../references/IUT_DATATHON_HALLU-main/README.md
[c-notebook]: ../references/IUT_DATATHON_HALLU-main/CUET_AL_Masaar_inference_notebook.ipynb
[c-report]: ../references/IUT_DATATHON_HALLU-main/CUET_AL_Masaar_report.pdf
[c-slides]: ../references/IUT_DATATHON_HALLU-main/CUET_AL_Masaar_presentation.pdf
[c-license]: ../references/IUT_DATATHON_HALLU-main/LICENSE
