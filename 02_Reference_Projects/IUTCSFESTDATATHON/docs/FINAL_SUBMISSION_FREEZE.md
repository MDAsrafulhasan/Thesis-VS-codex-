# Final submission freeze

Prepared on 2026-07-20 before the organizer package deadline. The form is
filled but deliberately not submitted at the user's request.

## Form values

| Form field | Frozen value |
|---|---|
| Team | Huntrix |
| Registration ID | 03-74970 |
| Team leader | Touhidul Alam Seyam |
| Private rank | **1** |
| Public rank | **5** |
| GPU | **Tesla T4 ×2** |
| Runtime in hours | **0.17** |
| Inference notebook (GitHub) | [`inference.ipynb` at immutable commit `26fefdb`](https://github.com/Seyamalam/Huntrix_Olikbochon_Onsite_Submission/blob/26fefdbbd054d0069914bf66dcb79aa59d64fbc0/inference.ipynb) |
| Model/checkpoint platform | Kaggle Model |
| Model/checkpoint | [`snowrab/qwen3-6-27b-q4-k-p/pyTorch/default/1`](https://www.kaggle.com/models/snowrab/qwen3-6-27b-q4-k-p/pyTorch/default/1) |
| Kaggle inference notebook | [`seyamalam/bengali-hallucination-v91-compliance-inference`, version 3](https://www.kaggle.com/code/seyamalam/bengali-hallucination-v91-compliance-inference/versions/3) |
| Public runtime dataset | [`seyamalam/bengali-hallucination-phase2-runtime-v91`, version 4](https://www.kaggle.com/datasets/seyamalam/bengali-hallucination-phase2-runtime-v91/versions/4) |
| Public CUDA runtime | [`llama.cpp CUDA b9637`, version 2](https://www.kaggle.com/datasets/seyamalam/bengali-hallucination-llama-cpp-cuda-b9637-public/versions/2) |

## Team roster

| Member | Role | Institution |
|---|---|---|
| Touhidul Alam Seyam | Team leader | BGC Trust University Bangladesh |
| MD. Abtahee Kabir | Member | Chittagong University of Engineering & Technology |
| Joyeta Barua Moni | Member | Chittagong University of Engineering & Technology |
| Noore Tamanna Orny | Member | Chittagong University of Engineering & Technology |

Emails, phone numbers, and student IDs belong only in the organizer's private
form and are not stored in the public submission repository.

The three measured full 2,516-row T4×2 runtimes are `440.8734`, `527.0319`,
and `593.4301` seconds. The slowest run is 9 minutes 53.43 seconds. Reporting
`0.17` hours means 10 minutes 12 seconds, or 612 seconds, and is a conservative
round-up.

## Scored reproduction

- Scored submission: `54836339`.
- Public macro-F1: `0.967`.
- Final public rank: `#5`.
- Final private macro-F1: `0.982`, private rank `#1`.
- Output rows: `2,516`.
- Label counts: `{0: 1181, 1: 1335}`.
- Output SHA-256:
  `0fec5d610f4e7b94ab6c46e2094d58e940b0f36781e9fa98d5466d2135474c28`.
- Three T4×2 notebook runs are byte-identical to the scored CSV.

## Frozen evaluation assets

- Notebook: `seyamalam/bengali-hallucination-v91-compliance-inference`,
  submitted version `3`.
- Runtime dataset: ID `11271419`, version `4`.
- CUDA dataset: ID `11271407`, version `2`.
- Model version: `snowrab/qwen3-6-27b-q4-k-p/pyTorch/default/1`.
- Model file: `Qwen3.6-27B-Q4_K_P.gguf`.
- llama.cpp: release `b9637`, commit
  `aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3`.
- Competition input path:
  `/kaggle/input/competitions/bengali-hallucination/test set.csv`.

## Repository and disclosure policy

The full project repository remains private because it contains
competition-provided files. The isolated public submission repository contains
only the notebook, manifest, README, citations, source register, license and
third-party notices, methodology disclosure, and technical report.

The citation-backed public-source method was disclosed to the organizers. They
replied that a solution without manually labeled rows, API calls, or other
rulebook violations would not be disqualified. No hidden label was available
or used.

After the submission deadline, do not update or replace the submitted GitHub
commit, Kaggle notebook version, runtime dataset, CUDA runtime, or model
checkpoint.
