# Huntrix — Bengali Hallucination Detection Submission

## Result

Our strongest audited offline Phase 1 candidate is:

- Upload file: `submission.csv`
- Reproducible source file: `outputs/submission_v23_antonym_consensus.csv`
- Public F1 (`label=0`): **0.945**
- Current rank after scoring: **4th**
- Previous scored anchor: v15, public F1 **0.934**
- Previous pre-source-bank best: **0.797**
- v23 difference from scored v15: **33 rows** (`25` from 1→0, `8` from 0→1)
- SHA-256: `c555910b51afebcd3ee3d5bcdf5f025b4f4bc12328e82de101ae3c86f93f0f89`
- Prediction counts: `{0: 1144, 1: 1372}`

The public leaderboard evaluates only 30% of the test set. We therefore selected the final system using the 299 officially labeled samples, source reliability, and abstention behavior—not by optimizing individual public-leaderboard rows.

## Submission command after the quota resets

```bash
.venv/bin/python -m kaggle competitions submit \
  -c bengali-hallucination \
  -f submission.csv \
  -m "v23 source-bank + digit fix + public-law verifier + 3-view Bengali Wikipedia fact consensus + 2-source antonym consensus"
```

## Leaderboard progression

| Iteration | Main change | Public F1 |
|---|---|---:|
| Initial Qwen/lexical system | Closed-book Qwen judge plus lexical rules | 0.797 |
| v6 | Public answer bank, deterministic math/grammar, conservative fuzzy matching | 0.903 |
| v9 | Added MIT LiveMCQ source, semantic numeric equivalence | 0.924 |
| v11c | Cross-source conflict abstention and QDoc/BCS evidence | 0.928 |
| **v15** | Bengali Unicode normalization and reliability-weighted source voting | **0.934** |
| v21 | Digit fix, public-law sections, three-view exact-page factual consensus | Not submitted |
| **v23** | v21 plus two-source Bengali antonym consensus | **0.945** |

## Core idea

A generic LLM judge was not enough. Bengali factual and linguistic questions contain exact answer-key structure, orthographic variation, culturally specific facts, idioms, and deliberately swapped values. Our strongest system is therefore a routed hybrid:

1. Recover relevant answers from public Bengali QA, MCQ, dictionary, and idiom sources.
2. Normalize Bengali orthography and numeric forms before matching.
3. Use deterministic solvers for arithmetic and structured grammar.
4. Compare the proposed response with independently recovered answers.
5. Abstain when reliable sources disagree.
6. Use a deterministic Qwen 27B judge only where structured/source rules do not resolve the case.

## Method

### 1. Public-source answer bank

We normalized 194,718 answer-key entries from public sources, including:

- `hishab/bangla-mmlu`
- `SumanMondal/bengali_qa`
- `minhaj1403/bluck`
- `farihashifa/SOMAJGYAAN`
- `mojnupust/api.farhan-mcq` / LiveMCQ-derived BCS data
- `Nh-em0n/QDoc`
- two additional public BCS/job-exam mirrors

The bank stores the question, accepted answer, MCQ choices, correct option, source, file, license metadata, and task type.

Exact source recovery is the main improvement. After Bengali Unicode and digit normalization, the final bank matched 1,669 of 2,516 test questions. It made 1,406 deterministic decisions and passed the remaining source conflicts to an abstaining reliability layer.

### 2. Bengali Unicode and numeric normalization

Visually identical Bengali forms were initially treated as different strings. For example:

- `হয়` versus `হয়`
- `কোথায়` versus `কোথায়`

We added NFKC normalization before punctuation and corrected a mixed Bengali/Persian digit table that had produced partially normalized years. We also canonicalized equivalent numeric forms, such as:

- `১২.৫০%` = `12.5%`
- Bengali and ASCII digits
- year suffixes such as `সালে`, `সন`, and `খ্রিষ্টাব্দে`

This converted many near-duplicates into exact matches and prevented correct numeric answers from being marked hallucinated.

### 3. High-precision deterministic comparators

For MCQ records, the verifier checks:

- exact option identity;
- normalized number equality;
- uniquely matching numeric options;
- conservative containment and fuzzy-option similarity.

For open QA, it checks:

- exact normalized answers;
- numeric contradictions;
- high-confidence textual contradictions;
- cross-script cases, where Bengali/English translation cannot be judged lexically, are left unresolved.

### 4. Conflict abstention and source reliability

Different public answer banks sometimes contain conflicting keys. We do not force a decision when sources disagree.

Source tiers were calibrated only on the official 299 labeled rows:

- The high-reliability tier—Suman Bengali QA, Hishab Bangla-MMLU, LiveMCQ/Farhan, and QDoc—had 100% precision on its confident released-sample decisions.
- The `80-20` mirror scored 97.4% on its confident released-sample decisions.
- `job_mcq` scored 78.3% and therefore receives only tie-breaking weight.

The final reliability-weighted layer covers deterministic abstentions only when the winning source score and margin both clear conservative thresholds. On the official released data it resolved **21/21** such rows correctly.

### 5. Deterministic task solvers

We route structured tasks to reproducible solvers rather than asking an LLM to recall an answer:

- generated arithmetic and word problems;
- ratio, work-rate, interest, average, speed, combination, and calendar problems;
- Bengali সমাস from the supplied ব্যাসবাক্য;
- সন্ধি composition;
- Bengali, Sanskrit/tatsama, and foreign prefix classification;
- exact-headword Bengali idiom evidence.

### 6. Open-weight LLM fallback

The fallback is `Qwen3.6-27B-MLX-4bit`, run deterministically with short outputs. It handles unresolved no-context factuality and structured language cases. The final system does not let the LLM override a high-confidence source or deterministic solution.

We also tested 9B judges, generic evidence prompts, and context-only LLM verification. They were weaker and were rejected rather than blended into the final candidate.

### 7. Public-law section verifier

Fourteen law questions are resolved directly from the public `lawsofbangladesh/laws` text. The verifier extracts the requested section, distinguishes maximum from minimum imprisonment/fines, normalizes Bengali number words, and compares the response. This changed five v15 labels across DNA Act penalties, child-marriage ages, Right to Information commencement dates, and repeated-offence trademark penalties.

### 8. Exact-page factual consensus

The recurring fact block contains deliberate relation swaps: publication versus writing date, construction start versus completion, UNESCO versus Ramsar year, and book parts versus chapters. A response merely occurring on the same page is therefore insufficient.

We cached 27 full Bengali Wikipedia articles for 28 detected topics and ran three evidence views:

1. short API sentence chunks;
2. full-HTML sentence chunks;
3. a title-selected full-page view.

A factual decision is accepted only when at least two views agree at confidence `>=0.85` and no confident view disagrees. This accepted 57/63 rows and changed 24 labels. The disagreement rule caught and rejected a real retrieval error where the 1680 Lalbagh restart date was confused with the original 1678 construction date.

### 9. Two-source antonym consensus

Antonym rows are changed only when two public Bengali dictionary/exam sources agree. Four rows reached the threshold and three changed: `ভূত → বর্তমান` was rejected, `তীক্ষ্ণ → ভোঁতা` was accepted, and `স্বাধীন → বন্দী` was rejected in favor of the standard `পরাধীন` pair.

## Official-sample checks

Only `dataset samples.json` is official labeled development data. All other QA/MCQ corpora are external evidence created or downloaded by our pipeline.

| Check | Coverage | Result |
|---|---:|---:|
| Final deterministic source decisions | 166 rows | 166/166 correct |
| Reliability-weighted conflict decisions | 21 rows | 21/21 correct |
| Conservative fuzzy source matching, earlier calibration | 29 rows | 29/29 correct |
| Strict cross-format context-number equivalence | 4 rows | 4/4 correct; zero test overrides |
| Unit tests | 55 tests | all passing |

These are covered-slice precision checks, not an inflated claim of 100% accuracy on all 299 rows.

## Reproduction

Run tests:

```bash
PYTHONPATH=scripts .venv/bin/python -m unittest discover -s tests -q
```

Build the extended source bank after obtaining the public repositories:

```bash
PYTHONPATH=scripts .venv/bin/python -u scripts/bcs_answer_bank.py \
  --base-bank outputs/source_answer_bank.jsonl \
  --farhan-dir /tmp/api-farhan-mcq/questions \
  --qdoc-dir /tmp/QDoc \
  --exam-8020-dir /tmp/80-20-exam/public \
  --job-mcq-dir /tmp/job_mcq/assets \
  --output outputs/source_answer_bank_complete_bcs.jsonl
```

Audit against the official labeled sample:

```bash
PYTHONPATH=scripts .venv/bin/python -u scripts/source_answer_bank.py audit \
  --bank outputs/source_answer_bank_complete_bcs.jsonl \
  --split sample \
  --trace outputs/source_answer_bank_unicode_sample_audit.json

PYTHONPATH=scripts .venv/bin/python -u scripts/weighted_source_overrides.py \
  --split sample \
  --trace outputs/weighted_source_unicode_sample_audit.json
```

Reproduce the scored v15 refinement from the conflict-aware v11c anchor, then run the v21/v23 stages recorded in `EXPERIMENT.log`:

```bash
PYTHONPATH=scripts .venv/bin/python -u scripts/source_answer_bank.py apply \
  --bank outputs/source_answer_bank_complete_bcs.jsonl \
  --base-submission outputs/submission_source_bank_complete_bcs_conflict_v11c.csv \
  --output outputs/submission_source_bank_unicode_v14.csv \
  --trace outputs/source_answer_bank_unicode_v14_trace.json

PYTHONPATH=scripts .venv/bin/python -u scripts/weighted_source_overrides.py \
  --split test \
  --base-submission outputs/submission_source_bank_unicode_v14.csv \
  --output outputs/submission_source_bank_unicode_weighted_v15.csv \
  --trace outputs/weighted_source_unicode_v15_trace.json
```

Verify the exact artifact:

```bash
shasum -a 256 submission.csv
```

Expected hash:

```text
c555910b51afebcd3ee3d5bcdf5f025b4f4bc12328e82de101ae3c86f93f0f89
```

## What did not work

- Character n-gram and compact supervised baselines were too weak for factuality.
- Prompt ensembles overfit the 299-row sample and transferred poorly.
- A generic Qwen evidence prompt often trusted related but wrong evidence.
- A global “response appears in context” rule failed on question/answer-type mismatches and structured grammar contexts.
- Qwen 9B was not competitive with the source-recovery system.
- A 27B semantic judge over source abstentions reached only 30/33 on the official sample after adding context, so it was rejected.
- Open-QA fuzzy matching reached only 37/40 and was rejected.
- A broad context-number layer passed 13/13 covered sample rows but made 22 unsafe test changes by confusing multiple dates or quantities inside one sentence. It was rejected; the strict version makes zero test changes.

## Current limitations and next step

The current scored candidate is 0.945, while first place is 0.976. Because the public board contains only 30% of test rows, further candidates must add independently evidenced changes rather than tune individual rows against the leaderboard.

The next high-value experiment is Phase-2 packaging and a targeted verifier for unresolved relation/presupposition errors. The source corpora and deterministic layers already run offline; the three-view MLX factual judge must be translated to a Kaggle-compatible CUDA quantized model or distilled verifier and benchmarked on `2xT4` for approximately 5,000 questions.
