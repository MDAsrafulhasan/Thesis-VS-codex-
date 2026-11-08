# Data guide and integration boundaries

## 1. Available original data

The original `All data/` directory is now [`data/raw/benhallu/`](../data/raw/benhallu/). All eight CSVs and their bytes are preserved. Do not normalize or rewrite these originals in place.

| Raw file | Rows | Thesis task | Track / thesis label | Candidate field |
|---|---:|---|---|---|
| [Track-A/banglahallueval_qa_1000.csv](../data/raw/benhallu/Track-A/banglahallueval_qa_1000.csv) | 1,000 | `bn_qa` | A / `0` | `correct_answer` |
| [Track-A/codemix_1000.csv](../data/raw/benhallu/Track-A/codemix_1000.csv) | 1,000 | `codemixed_qa` | A / `0` | `codemix_answer` |
| [Track-A/banglahallueval_summarization_dataset_1000.csv](../data/raw/benhallu/Track-A/banglahallueval_summarization_dataset_1000.csv) | 1,000 | `summarization` | A / `0` | `summary` |
| [Track-A/somadhan_1000_main_ordered.csv](../data/raw/benhallu/Track-A/somadhan_1000_main_ordered.csv) | 1,000 | `math` | A / `0` | `answer` |
| [Track-B/qa_4000.csv](../data/raw/benhallu/Track-B/qa_4000.csv) | 4,000 | `bn_qa` | B / `1` | `hallucinated_answer` |
| [Track-B/codemix_4000.csv](../data/raw/benhallu/Track-B/codemix_4000.csv) | 4,000 | `codemixed_qa` | B / `1` | `hallucinated_answer` |
| [Track-B/summarization_3000_corrected.csv](../data/raw/benhallu/Track-B/summarization_3000_corrected.csv) | 3,000 | `summarization` | B / `1` | `hallucinated_summary` |
| [Track-B/reasoning_1000.csv](../data/raw/benhallu/Track-B/reasoning_1000.csv) | 1,000 | `math` | B / `1` | Corrupted `hallucinated_chain` **and** `hallucinated_answer` |
| **Total** | **16,000** | Four tasks | **4,000 A + 12,000 B** | |

Code-mixed QA uses `codemix_context` and `codemix_question`, not the parallel Bengali fields. Summarization source text becomes `context` with a summarization instruction as `question`. Math without a supplied passage uses empty context. Raw correctness fields are not extra judge evidence.

Raw IDs are not globally unique across tasks, tracks, or duplicate records. The loader creates task-/track-prefixed canonical IDs and deterministic duplicate suffixes without dropping originals. **A new ID is not proof of independent source lineage.**

## 2. Canonical thesis schemas

### BenHallu

```text
id, task, track, hallucination_type, question, context,
candidate_answer, ground_truth_label
```

- `task`: `bn_qa`, `codemixed_qa`, `summarization`, `math`.
- `track`: `Track_A`, `Track_B`.
- `ground_truth_label`: `0` for A, `1` for B.
- Track A `hallucination_type`: canonical string `None` (also tolerates missing input values during validation).
- Track B types: `Factualness`, `Comprehension`, `Specificity`, `Inference`, `Fabricated_Content`, `Non_Factual_Addition`, `Direct_Contradiction`, `Arithmetic_Slip`, `Formula_Misapplication`, `Invalid_Deduction`, `Hallucinated_Intermediate_Fact`, `Semantic_Drift`.
- Preserve raw Bengali text, signs, and candidate reasoning. Normalized lookup keys must remain separate from scientific inputs.

### Dialectal robustness

```text
id, dialect, question_type, question, response, human_or_primary_score
```

Scores are finite values in `[0, 10]`; question types are `definitional` and `factual`. The 300-row anchor subset takes 100 each from `Standard_Bengali`, `Tangail`, and `Chittagong`, stratified by question type. Additional dialects may exist in the full dataset.

**No real dialect dataset is currently supplied.** [`data/raw/dialect/`](../data/raw/dialect/) is reserved for a properly sourced dataset, not invented observations. Synthetic prompts are not validated dialect resources; mock scores are not human ratings.

Before collection, document speaker/source provenance, consent and licensing where relevant, dialect identification, shared semantic content, annotator rubric, judge identity, missing values, and inter-rater checks. Decide whether comparisons concern agreement, answer quality, or both before evaluating.

## 3. Required summarization taxonomy review

The raw summarization Track B file contains:

| Raw `pattern` | Count | Status |
|---|---:|---|
| `Intrinsic` | 1,000 | Needs reviewed mapping |
| `Non-factual` | 1,000 | Needs reviewed mapping |
| `Factual Contradiction` | 1,000 | Needs reviewed mapping |

These names do not establish equivalence with the three thesis categories by themselves. Review the dataset documentation and actual examples; record definitions, rationale, reviewer, ambiguous cases, and any exclusions. Do not choose a mapping merely to satisfy the expected counts.

The splitter accepts `--input_csv data/raw/benhallu` and **requires `--type_mapping_json`** for the unknown raw labels. The mapping is a flat JSON object from raw labels to valid canonical type names. Store the approved JSON with the experiment protocol; supply that actual file to the command. No unreviewed mapping file has been created.

Canonical CSVs that already satisfy the schema can be passed directly via `--input_csv` without a raw-directory mapping.

## 4. Label inversion: never concatenate blindly

| System | `0` | `1` |
|---|---|---|
| **Active thesis evaluation** | Non-hallucinated | Hallucinated |
| **CUET_AL_Masaar reference** | Hallucinated | Faithful |
| **Huntrix reference** | Hallucinated | Faithful |

After validating that an imported prediction is binary and follows the reference convention, translate explicitly: `thesis_label = 1 - reference_label`. Translate probability semantics too: a faithfulness probability and a hallucination probability are not the same quantity. Do not invert raw thesis Track labels, scores, margins, or arbitrary numeric columns without identifying their meaning.

For an integration, add tests containing a known faithful example, a known hallucinated example, malformed output, and any threshold/tie policy. Save the original label column alongside its translated derivative in the experiment artifact.

## 5. Splits, overlap, and contamination

Generated files belong under [`data/splits/`](../data/splits/) in a dataset-/seed-specific directory:

| File | Size | Purpose |
|---|---:|---|
| `benhallu_micro_debug_120.csv` | 120: 30 A + 90 B | Debugging |
| `benhallu_stratified_10pct.csv` | 1,600: 400 A + 1,200 B | Baseline/ablation evaluation |
| `benhallu_full_16k.csv` | 16,000 | Full evaluation |
| `dialect_ccc_subset_300.csv` | 300 | Anchor-variant agreement comparison |

- The seed defaults to 42, IDs are sorted before sampling, and the composite BenHallu stratum includes task, track, and type.
- Micro and 10% are sampled from full and may overlap. **They are not train/dev/test partitions.**
- Stratification follows sklearn integer allocation. If rare strata disappear after rounding, the splitter fails rather than silently claiming complete coverage.
- Future training/calibration splits must group source questions/documents and their generated derivatives before splitting. Audit source overlap, not just IDs.
- Reference projects' answer banks explicitly incorporate BenHallu or related QA sources. Do not use these banks, source answers, saved labels, or target-specific facts for independent inference, retrieval, calibration, or demonstrations.
- Treat the competition's own sample/test CSVs under `references/` as a distinct benchmark with its own permissions, label convention, and development history.

## 6. Licensing and evidence status

The available CSVs do not establish a redistribution license on their own. Consult original dataset terms before publishing raw data or large excerpts. Repository MIT licenses cover their authors' code, not every external corpus, model, or figure. See [research reuse and licensing](RESEARCH_REUSE.md).

Use `python tools/workspace.py verify` to check original bytes and `python tools/workspace.py catalog` to refresh derived inventories. The catalog describes local files; it does not certify dataset correctness or the scientific independence of a model.
