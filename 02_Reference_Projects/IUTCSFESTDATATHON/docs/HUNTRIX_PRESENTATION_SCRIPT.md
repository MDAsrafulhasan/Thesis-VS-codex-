# Team Huntrix Final Presentation Script

**Hard limit:** 10 minutes  
**Rehearsed target:** 9 minutes 15 seconds  
**Format:** two presenters and one handoff. Slides 1-15 form the timed talk; slides 16-20 are appendix-only for Q&A.  
**Deck:** `Huntrix_Final_Presentation_20_Slides.pptx`

The presentation uses short numbered source markers so citations remain readable on screen. The complete source map is at the end of this script.

## Presenter allocation

| Presenter | Slides | Approximate time |
|---|---:|---:|
| Touhidul Alam Seyam | 1-7 | 4 min 15 sec |
| MD. Abtahee Kabir | 8-15 | 5 min 00 sec |

The individual slide targets add up to 9 minutes 15 seconds. This leaves 45 seconds for the handoff, natural pauses, or a delayed start without crossing the 10-minute limit.

### Clock checkpoints

| After slide | Elapsed target |
|---:|---:|
| 3 | 1:45 |
| 7 and handoff | 4:15 |
| 10 | 6:10 |
| 13 | 8:05 |
| 15 | 9:15 |

If the talk reaches slide 13 after 8:15, Abtahee should state only the headline result on slides 13 and 14, then move directly to the conclusion.

## Slide 1: Evidence-first selective inference for Bengali hallucination detection

**Presenter:** Touhidul Alam Seyam  
**Target:** 30 seconds

Good morning. We are Team Huntrix. Our question was simple: when is a Bengali hallucination decision exact enough for a deterministic method, and when should the system abstain to a language-model judge? We will show the architecture that emerged, the experiments that changed it, and what the organizer-run Phase 2 evaluation revealed about generalization and runtime.

## Slide 2: One binary label hid many different reasoning tasks

**Presenter:** Touhidul Alam Seyam  
**Target:** 35 seconds

The output was binary, but the reasoning was not. Some rows required context entailment; others required arithmetic, option parsing, morphology, idioms, law, history, or literature. A single generic prompt applied one evidence policy to every task. It produced a useful baseline, but it also confused a related fact with an answer to the requested relation. That observation led us from uniform judging to task routing.

## Slide 3: The working hypothesis became a selective cascade

**Presenter:** Touhidul Alam Seyam  
**Target:** 40 seconds

This is the final selective cascade, matching Figure 1 in our paper. We normalize the row, then try exact lineage, structured solvers, context verification, and typed evidence. Each deterministic branch may return zero, one, or abstain. A label is emitted only when the source and requested relation align. Unresolved rows reach the fixed Qwen judge. The important design choice is that abstention is a valid outcome, not a failure.

## Slide 4: Routing made abstention an explicit state

**Presenter:** Touhidul Alam Seyam  
**Target:** 40 seconds

The router first separates answer-bearing context from no-context tasks. Context rows go to claim verification. No-context rows may enter numeric, multiple-choice, morphology, lineage, or typed-evidence paths. Every route uses the same contract: return a label with provenance, or return null. Approximate retrieval can propose a candidate, but it cannot force a verdict. Ambiguity or equal-trust conflict therefore moves forward instead of contaminating deterministic decisions.

## Slide 5: Structured solvers shared one decision contract

**Presenter:** Touhidul Alam Seyam  
**Target:** 35 seconds

The deterministic layer was a set of small task-specific parsers, not one keyword system. Numeric rules normalized Bengali digits, units, dates, and ranges. Multiple-choice rules mapped answer aliases. Morphology rules checked the requested grammatical form, while idiom rules separated literal and figurative meaning. Despite different parsing logic, every solver returned the same fields: entity, predicate, value, source, and verdict. Missing or conflicting fields produced abstention.

## Slide 6: A correct fact can still answer the wrong relation

**Presenter:** Touhidul Alam Seyam  
**Target:** 35 seconds

This example explains the predicate guard. The passage says that *Runway* was released in 2010. The prompt asks for the release year, but the response gives the film title. The response is related to the correct entity, yet its semantic type is wrong: a title is not a year. Similarity-based retrieval may accept it; our relation check emits label zero.

## Slide 7: Context rows were reduced to claim-level entailment tests

**Presenter:** Touhidul Alam Seyam  
**Target:** 40 seconds

For context rows, we split the passage into clauses and the response into atomic claims. Entity, number, negation, and predicate anchors select the applicable clause. Each claim becomes supported, contradicted, or unknown. One contradiction is enough for label zero. Label one requires every factual claim to be supported with no unsupported addition. Everything between those cases abstains. This catches responses that copy most of the passage but alter one decisive fact.

**Handoff:** Abtahee will cover the fallback judge, evidence gates, experiment history, and Phase 2 result.

## Slide 8: Qwen judged 158 rows, not the whole test set

**Presenter:** MD. Abtahee Kabir  
**Target:** 35 seconds

The Phase 1 fallback used quantized Qwen3.6-27B across two T4 GPUs. Context rows received one entailment view. No-context rows received up to three short checks and required unanimous positive votes for label one. Thinking was disabled and generation was constrained to one binary token. Fixed row order, prompt order, seeds, and batches made the judge deterministic. In Phase 1, only 158 unresolved rows reached it.

## Slide 9: A source record was accepted only after four fields aligned

**Presenter:** MD. Abtahee Kabir  
**Target:** 40 seconds

This is Figure 2b from the paper. A candidate fact carried its provenance, entity, predicate, and value. It could decide a row only if four gates passed: the entity matched, the requested predicate matched, the value was equivalent, and no equal-trust source contradicted it. Any failed gate returned abstention. This prevented a relevant source from answering a neighboring question and gave every deterministic verdict an inspectable decision trace.

## Slide 10: Versioned comparisons locate the main design pivots

**Presenter:** MD. Abtahee Kabir  
**Target:** 40 seconds

The left curve shows public macro-F1 across selected versions; the right chart shows the shrinking judge queue. A uniform Qwen judge reached 0.797. The largest jump came with source and structured routing in v6, which reached 0.903. Later provenance and relation guards raised the score more gradually while reducing the Phase 1 judge queue from 620 early rows to 158 in v91. These are cumulative versions, not controlled single-factor ablations.

## Slide 11: Decision logs turned failed runs into routing rules

**Presenter:** MD. Abtahee Kabir  
**Target:** 35 seconds

Failed runs changed the next hypothesis. Prompt voting was unstable, so prompt variation was not treated as independent evidence. TF-IDF missed cultural facts, so it remained only a fallback. Raw retrieval repeatedly found the correct entity but the wrong predicate, so it was restricted to candidate generation. When v87 changed plausible facts and reduced the public score, we reverted it and added the source-plus-predicate admission rule.

## Slide 12: Reproducibility was treated as an evaluation result

**Presenter:** MD. Abtahee Kabir  
**Target:** 40 seconds

The final notebook used pinned model, runtime, and evidence assets with a fixed schedule. The codebase had 249 automated tests, including 225 deterministic controls. Three full Phase 1 runs produced byte-identical CSVs, and the frozen output matched the scored submission. The notebook preserved incoming IDs and wrote only `id,label`. The organizer then ran the same pipeline on 5,000 Phase 2 rows without schema, ordering, or runtime failure.

## Slide 13: Phase 2 ranked 7/32 with an official score of 0.90144

**Presenter:** MD. Abtahee Kabir  
**Target:** 40 seconds

On the organizer-run 5,000-row set, the router resolved 207 rows and sent 4,793 to Qwen. End-to-end inference took 3 hours, 52 minutes, and 1.91 seconds on two T4 GPUs, using about 43 percent of the nine-hour budget. Output validation passed for all 5,000 ordered IDs. The official Phase 2 score was 0.90144, ranking seventh among 32 teams.

## Slide 14: Phase 2 source shift collapsed deterministic coverage

**Presenter:** MD. Abtahee Kabir  
**Target:** 35 seconds

The broader Phase 2 sources changed system behavior sharply. Deterministic coverage fell from 93.72 percent in Phase 1 to 4.14 percent. The router did not compensate by forcing approximate evidence; it abstained, so 95.86 percent of rows reached Qwen. Runtime increased but remained comfortably compliant. Because only the aggregate score was released, we cannot yet attribute the remaining errors to individual sources or branches.

## Slide 15: What the experiments support—and what remains open

**Presenter:** MD. Abtahee Kabir  
**Target:** 35 seconds

Our experiments support three conclusions. Different task families benefit from different evidence policies. Predicate alignment prevents relation swaps. Explicit abstention protects the pipeline under source shift. The limitations are 299 released labels and no row-level Phase 2 feedback. With those labels, our next step would be to analyse the judged tail and expand licensed civic and news evidence. Thank you. We are ready for your questions.

## Likely questions

### Why did you not fine-tune a classifier or the judge?

The released set had only 299 labeled rows, while Phase 2 deliberately changed the source distribution. Fine-tuning on that small set risked learning Phase 1 artifacts. Routing produced a larger observed gain, preserved an interpretable abstention path, and kept inference inside the Kaggle limit.

### Why Qwen3.6-27B?

It is an open multilingual model that handled Bengali factual and entailment prompts well in our checks. The 16.332 GiB Q4_K_P checkpoint fits the two-T4 environment, and constrained one-token output keeps the fallback practical.

### Did exact-source matching amount to manually labeling the test set?

No. We did not manually assign labels to test rows, use hidden labels, or call paid or closed APIs. A source record could decide only when its entity, value, and requested predicate agreed; otherwise the system abstained.

### Did the judge process 158 requests?

It processed 158 unique Phase 1 rows. Context rows used one view. Some no-context rows used two or three short views with early exit on the first zero, so the number of model requests was higher than the number of rows.

### How did the system handle unseen Phase 2 sources?

Unknown sources did not receive approximate deterministic labels. Context checks and general structured solvers still ran, but lineage rules abstained unless their exact conditions were met. The unresolved row then reached the fixed LLM judge.

### Why are the public and private Phase 1 scores different?

The public leaderboard evaluated only part of the test set. The private score came from the remaining held-out portion. We report both aggregate scores, but we do not infer row-level labels or error patterns from them.

## Source map

1. Bengali LLM Hallucination Detection Challenge, data description, leaderboard, and rules: <https://www.kaggle.com/competitions/bengali-hallucination>
2. Organizer Phase 2 source preview: <https://www.kaggle.com/competitions/bengali-hallucination/discussion/719835>
3. *BenHalluEval: A Multi-Task Hallucination Evaluation Framework for Large Language Models on Bengali*: <https://arxiv.org/abs/2605.31483>
4. *When Words Don't Mean What They Say: Figurative Understanding in Bengali Idioms*: <https://arxiv.org/abs/2602.12921>
5. Qwen3.6-27B model card: <https://huggingface.co/Qwen/Qwen3.6-27B-FP8>
6. llama.cpp: <https://github.com/ggml-org/llama.cpp>
7. Team Huntrix experiment log, frozen manifest, deterministic trace, Kaggle runtime records, and final leaderboard records in this repository.
8. Organizer-executed Phase 2 notebook `huntrix-1.ipynb` and its final runtime report: 5,000 rows on Tesla T4 x2, completed July 22, 2026.
