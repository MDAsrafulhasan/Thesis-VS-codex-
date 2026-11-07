# অলীকবচন — Bengali LLM Hallucination Detection (IUT 12th ICT Fest Datathon 2026)

TEAM: CUET_AL_Masaar (Kaggle: [rafiurrahman01](https://www.kaggle.com/rafiurrahman01)).

Binary classification of Bengali (context, prompt, response) triples: **1 = faithful, 0 = hallucinated**.
Scored by F1 on the hallucinated class (`pos_label=0`). Phase-1 private LB score of this pipeline: **0.937**.

## Repository contents

- `notebook.ipynb` — the complete, self-contained inference notebook (6 cells). All model
  inference, rule construction, and label generation happen inside this notebook. There is no
  offline step: rerunning it end-to-end regenerates `submission.csv` from raw test inputs.

## How to run on Kaggle

1. **Open the notebook on Kaggle** (or upload `notebook.ipynb` as a new Kaggle notebook).
2. **Attach the two inputs** via *Add Input*:
   - Dataset (all data files, ~2.5 GB): https://www.kaggle.com/datasets/rafiurrahman01/bhd-all-inputs
   - Model (judge weights): https://www.kaggle.com/models/dipamc77/qwen2.5-14b-instruct
     (Transformers, V1, default variant — fp16, **not** AWQ)
3. **Accelerator: GPU T4 ×2. Internet: OFF.** The notebook performs no network access.
4. **Set the test file path** in Cell 1 (`TEST_PATH`, the only configurable line). To evaluate
   a different test file (e.g. a held-out fold), replace the file at that path or point
   `TEST_PATH` at the new CSV. The file must have columns `id, context, prompt_bn, response_bn`.
5. **Save & Run All.** Output is written to `/kaggle/working/submission.csv`
   (columns `id,label`).

Cell 1 audits all required inputs and prints OK/MISSING for each before anything runs, so a
misattached input fails immediately with a clear message.

## Phase-2 / held-out evaluation (for organizers)

The same notebook handles both runs unchanged — only the test file differs.

1. **Reproduction run:** rerun the notebook as-is (Save & Run All, GPU T4 ×2, internet OFF). It
   reads the Phase-1 `test set.csv` from the attached bundle and regenerates the submitted
   `submission.csv` exactly.
2. **Held-out run:** upload the held-out CSV (columns `id, context, prompt_bn, response_bn`) as
   an attached dataset, then edit one line — `TEST_PATH` at the top of Cell 1 — to point at it,
   and Save & Run All again. Nothing else needs changing.

The pipeline is size-agnostic: no assertion, threshold, or rule depends on row count or ids. A
~5,000-row set completes in roughly 4–4.5 h; judge margins checkpoint every 200 rows and resume
automatically if the session restarts.

## Runtime

- ~1.5–3.5 h on T4 ×2 for the 2,516-row Phase-1 test set (the 14B judge pass dominates,
  ~2.1 s/row; margins are checkpointed every 200 rows and resume automatically).
- Scales linearly with test size: a ~5,000-row set completes in roughly 4–4.5 h.

## Reproducibility

- Fully deterministic: logit-margin scoring (no sampling), fixed threshold, fixed rule joins.
  A clean offline rerun reproduces the submitted Phase-1 predictions exactly (zero row diff).
- No hardcoded test IDs or answer maps anywhere; all labels derive from raw test inputs via
  normalized joins and solvers.
- No external APIs; all inference is local to the Kaggle session.

## Method overview

A hybrid rule-plus-judge pipeline, applied in a fixed order:

1. **Deterministic rule exoskeleton** — gold-answer key joins built from public Bengali QA
   datasets (BanglaHalluEval, TyDiQA-BN, Bangla-MMLU), an idiom/বাগধারা meaning pool, and a
   grammar solver for সমাস / সন্ধি questions. All rules are joins from attached files;
   ambiguous and internally contradictory keys are excluded automatically.
2. **14B logit-margin judge** — Qwen2.5-14B-Instruct scores each row by the logit margin
   between the "0" and "1" verdict tokens, Platt-calibrated on a 299-row labeled anchor set,
   thresholded at 0.340. Rows not covered by rules fall back to the judge.
3. **Math solver layers** — closed-form templates (work/rate, ratio, interest, percentage,
   LCM, combinatorics, sympy-based algebra), weekday arithmetic, and a corruption-aware
   operand-repair layer that recovers comma-truncated numbers via bisection.
4. **Fact solver layers** — date verification from context, prime/BCE checks, and a
   closed-book biography lookup over Bengali Wikipedia using multilingual-e5-base embeddings
   (precomputed passage index attached; query encoding runs locally in-notebook).
5. **Consistency reverts** — corpus-consensus and context-grounding guards that undo unsafe
   flips before the final label is written.

Every layer is validated against the 299-row anchor inside the notebook (assertion gates);
the run aborts rather than silently degrading if any layer regresses.

## Data and model sources

| Resource | Link |
|---|---|
| All input data (bundle) | https://www.kaggle.com/datasets/rafiurrahman01/bhd-all-inputs |
| Judge model (Qwen2.5-14B-Instruct fp16) | https://www.kaggle.com/models/dipamc77/qwen2.5-14b-instruct |
| Embedding model (multilingual-e5-base, included in bundle)(no need to give input for running the notebook, it is already included on All input data(bundle)) | https://huggingface.co/intfloat/multilingual-e5-base |

The bundle contains the public external datasets used for rule construction
(BanglaHalluEval, TyDiQA-BN GoldP, Bangla-MMLU, idiom pool, pattern corpus), the
precomputed Bengali-Wikipedia E5 passage index, and the local copy of multilingual-e5-base.

## Dataset sources & transparency

All external data is public, declared to the organizers, and none of it is derived from the
competition test set. What each file in the bundle is and where it comes from:

- **`banglahallueval` QA files (`qa_4000.csv`, `qa_gt_1000.csv`, `banglahallueval_qa_dataset.csv`)** —
  Bengali QA pairs with correct and hallucinated answers from the public BenHalluEval
  benchmark (arXiv:2605.31483); used to build exact-match gold/hallucination answer keys.
- **`tydiqa_bn_goldp.parquet`** — the Bengali GoldP split of Google's public TyDiQA dataset;
  used as a gold-answer key for context-grounded factual questions.
- **`bangla_mmlu_all.parquet`** — the public Bangla-MMLU multiple-choice benchmark; correct
  options serve as gold answers, wrong options as known-distractor keys.
- **`bagdhara_pool.json`** — a pool of Bengali idioms (বাগধারা) and their meanings compiled
  from public educational sources; used for idiom-meaning verification.
- **`merged_pattern_dataset.csv`** — a public Bengali prompt/response pattern corpus; only
  rows with fully consistent labels across duplicates are used, as consensus lookups.
- **`passages.parquet` + `bnwiki_e5_emb.npy`** — passages from the public Bengali Wikipedia
  dump with precomputed multilingual-e5-base embeddings (204k passages); used for closed-book
  biography date verification.
- **`e5_plain`** — an unmodified local copy of `intfloat/multilingual-e5-base` (MIT license,
  Hugging Face), included so the notebook runs with internet off.

No dataset was manually labeled or filtered against the test set; all keys are joined to test
rows at runtime via normalized text matching.

## Known limitations

- The judge inherits Qwen2.5-14B's knowledge gaps on Bengali-specific facts; rule and solver
  layers correct only the row families they cover.
- Gold-key rules depend on the correctness of their public source datasets; keys where
  sources contradict each other are detected and excluded rather than trusted.
