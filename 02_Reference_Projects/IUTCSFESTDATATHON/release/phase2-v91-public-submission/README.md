# Huntrix — Olikbochon Onsite Submission

This public repository is the single submission package for Team Huntrix.

## Required onsite files

| Deliverable | File |
|---|---|
| Technical report | [`Huntrix_report.pdf`](Huntrix_report.pdf) |
| Presentation | [`Huntrix_presentation.pdf`](Huntrix_presentation.pdf) |
| Final inference notebook | [`Huntrix_inference_notebook.ipynb`](Huntrix_inference_notebook.ipynb) |
| Training notebook | Not applicable; no model training or fine-tuning was used |
| Form values | [`form.md`](form.md) |

## Team

| Member | Role | Affiliation |
|---|---|---|
| Touhidul Alam Seyam | Team leader | BGC Trust University Bangladesh |
| MD. Abtahee Kabir | Member | Chittagong University of Engineering & Technology |
| Joyeta Barua Moni | Member | Chittagong University of Engineering & Technology |
| Noore Tamanna Orny | Member | Chittagong University of Engineering & Technology |

## Frozen configuration

| Field | Value |
|---|---|
| Kaggle notebook | [`seyamalam/bengali-hallucination-v91-compliance-inference`, version 3](https://www.kaggle.com/code/seyamalam/bengali-hallucination-v91-compliance-inference/versions/3) |
| Hardware | **Tesla T4 ×2** |
| Runtime reported in organizer form | **0.17 hours = 10 minutes 12 seconds = 612 seconds** |
| Slowest measured full run | **9 minutes 53.43 seconds = 593.43 seconds for all 2,516 rows** |
| Internet | Disabled |
| Model | [`Qwen3.6-27B-Q4_K_P`, version 1](https://www.kaggle.com/models/snowrab/qwen3-6-27b-q4-k-p/pyTorch/default/1) |
| Public knowledge/runtime dataset | [`bengali-hallucination-phase2-runtime-v91`, version 4](https://www.kaggle.com/datasets/seyamalam/bengali-hallucination-phase2-runtime-v91/versions/4) |
| Public CUDA runtime | [`llama.cpp CUDA b9637`, version 2](https://www.kaggle.com/datasets/seyamalam/bengali-hallucination-llama-cpp-cuda-b9637-public/versions/2) |
| Competition input | `/kaggle/input/competitions/bengali-hallucination/test set.csv` |

The notebook is offline and produces `submission.csv` directly from the raw
competition input. It never loads a saved Phase 1 prediction vector and does
not depend on the Phase 1 row count.

Version 3 completed in `440.8734s` on T4×2 and produced a CSV byte-identical
to the scored v91 submission.

## Measured Phase 2 execution

The organizer-executed notebook completed the 5,000-row held-out input on
Tesla T4 x2 in `13,921.9134s`, or **3 hours 52 minutes 1.91 seconds**. The
router resolved 207 rows (4.14%) and sent 4,793 rows (95.86%) to Qwen. Output
validation completed successfully. The official Phase 2 result was **0.90144,
rank 7 of 32 teams**; no row-level labels were released. See
`PHASE2_RUNTIME_EVIDENCE.md` for the complete measured breakdown.

## Scored reference

- Kaggle competition submission: `54836339`
- Public leaderboard: **rank #5, macro-F1 0.967**
- Private leaderboard: **rank #1, macro-F1 0.982**
- Phase 2 private result: **rank #7 of 32, score 0.90144**
- Phase 1 rows: `2,516`
- Label counts: `{0: 1181, 1: 1335}`
- Expected output SHA-256:
  `0fec5d610f4e7b94ab6c46e2094d58e940b0f36781e9fa98d5466d2135474c28`

## Files

- `Huntrix_inference_notebook.ipynb` — frozen raw-input inference notebook
  submitted during the previous phase.
- `FREEZE_MANIFEST.json` — exact versions, hashes, hardware, runtime, and
  reproduction result.
- `CITATIONS.md` — model, runtime, benchmark, and evidence citations.
- `DATA_SOURCES.md` — external-data register with links and license-review
  status.
- `THIRD_PARTY_NOTICES.md` — upstream licenses, attribution, and unresolved
  redistribution status.
- `METHODOLOGY_DISCLOSURE.md` — rule audit, public-source verification method,
  and organizer clarification.
- `Huntrix_report.pdf` — ACL-style paper with four pages of main
  content plus references in the same PDF.
- `report/Huntrix_Technical_Report.tex` — ACL-style LaTeX source with native
  TikZ and PGFPlots figures; the pinned official style and bibliography files
  are included beside it.
- `presentation/Huntrix_Final_Presentation_20_Slides.pptx` — light-themed
  scientific deck with a 15-slide timed talk and five technical appendix slides
  for Q&A, plus detailed methodology diagrams and speaker notes.
- `Huntrix_presentation.pdf` — final onsite presentation in the required
  submission format.
- `presentation/HUNTRIX_PRESENTATION_SCRIPT.md` — timed rehearsal script,
  handoffs, likely questions, and numbered source map.
- `presentation/Huntrix_Presentation_Script.docx` — presenter-ready rehearsal
  edition with the run of show, slide-by-slide scripts, timing, and handoffs.
- `presentation/Huntrix_Presentation_Script.pdf` — fixed-layout 13-page copy of
  the presenter-ready rehearsal edition.
- `PHASE2_RUNTIME_EVIDENCE.md` — measured organizer-run Phase 2 timing and
  routing distribution.
- `SUBMISSION_LINKS.md` — organizer form, competition pages, and exact public
  artifact versions.
- `RUNBOOK.md` — copy-and-run instructions and expected checks.
- `FORM_VALUES.md` — the exact non-sensitive values for the organizer form.
- `form.md` — copy-ready values for the onsite submission form.
- `LICENSE` — MIT license for original Huntrix code only.

The scored router includes short, citation-backed factual answer records
researched from public sources. We disclosed the construction to the
organizers. They replied that a solution without manually labeled rows, API
calls, or other rulebook violations would not be disqualified. No hidden label
was available or used, and the notebook computes every output from raw rows and
pinned evidence.

## Phase 2 replacement

The organizers replace only the file mounted at:

```python
TEST_PATH = Path(
    "/kaggle/input/competitions/bengali-hallucination/test set.csv"
)
```

No notebook variable needs to be edited for the held-out set. The notebook
infers `N` from that file, preserves its IDs in order, and writes exactly
`id,label`.
