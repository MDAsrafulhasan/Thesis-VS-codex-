# Phase 2 compliance review

Audit date: 2026-07-20, before the Phase 2 package deadline.

## Confirmed requirements

- Official Kaggle team size: four members, within the 1–4 member limit.
- Raw input: literal
  `/kaggle/input/competitions/bengali-hallucination/test set.csv`.
- Output: exactly `id,label`, ordered IDs, non-null binary labels.
- Hardware: Kaggle Tesla T4 ×2.
- Internet: disabled.
- Model: attached public open-weight Qwen3.6-27B quantization, 16.332 GiB.
- Runtime: 527.03 s and 593.43 s for 2,516 rows, far below nine hours.
- Stress bound: approximately 5.2–5.5 hours if all 5,000 held-out rows require
  all three judge views.
- Reproduction: two notebook runs are byte-identical to scored v91,
  SHA-256 `0fec5d610f4e7b94ab6c46e2094d58e940b0f36781e9fa98d5466d2135474c28`.
- No paid API, closed inference service, saved Phase 1 vector, dynamic test-path
  discovery, or held-out Phase 2 access.
- Public notebook, runtime, CUDA binary, model, and immutable GitHub package.

## Organizer clarification

Kaggle's Foundational Rule 4(b) prohibits incorporating hand labeling or human
prediction of test records. The competition-specific rules also allow public
external data and require raw-input reproducibility.

The v91 router contains citation-backed answer records created by researching
unlabeled prompts against public sources:

- 96 of 98 `verified_fact_answers.json` prompts exactly match Phase 1 prompts;
- both typed-evidence prompts exactly match Phase 1 prompts;
- the verified-fact stage owns 95 router decisions;
- removing the manually researched files changes 42 router labels before the
  selective judge and increases uncertainty from 158 to 251 rows.

No hidden test labels were available or used. The team disclosed the exact
prompt/source matching and public-source verification process to the
organizers. They replied that Kaggle foundational rules were not imposed by the
organizers and that a solution without manually labeled rows, API calls, or
other rulebook violations would not be disqualified.

The notebook still assigns every label through the frozen router and judge. It
does not contain a hand-written label vector or a Phase 1 dataset switch.

## Third-party licensing review

- Original Huntrix code now carries an explicit MIT license.
- llama.cpp's complete MIT notice accompanies the redistributed CUDA binary.
- The runtime includes a source register, license status, attribution links,
  and copies of applicable GPL/Apache notices.
- Three public QA repositories have no declared license. Their use and
  redistribution status are explicitly disclosed in `THIRD_PARTY_NOTICES.md`.
  They must be removed or replaced if the organizers or authors require it.
- The official Qwen3.6 base model is Apache-2.0. The third-party Kaggle
  quantization page labels its upload MIT; both facts are recorded without
  claiming ownership of the weights.

## Paper timing

The Kaggle rules describe a four-page Phase 2 report, while the latest
organizer notice says the report and presentation are requested only from the
final top 15 after July 23. Huntrix prepared the four-page PDF in advance and
asked the organizers to confirm the operative timing.

## Freeze policy

After July 21 at 12:00 AM Bangladesh time, do not update or replace:

- the submitted GitHub notebook commit;
- the submitted Kaggle notebook version;
- attached runtime or CUDA dataset versions;
- the model checkpoint/version.

Documentation should reference immutable commits and exact Kaggle versions.
