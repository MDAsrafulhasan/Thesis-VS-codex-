# Third-party notices

This repository's original code is licensed under the MIT License. Models,
datasets, factual sources, and bundled runtime components retain their
respective upstream licenses. Nothing in the Huntrix license relicenses those
third-party materials.

## Inference model and runtime

| Component | Frozen artifact | Upstream license |
|---|---|---|
| Qwen3.6-27B Q4_K_P | [`snowrab/qwen3-6-27b-q4-k-p`, version 1](https://www.kaggle.com/models/snowrab/qwen3-6-27b-q4-k-p/pyTorch/default/1) | The Kaggle publisher labels the quantization MIT; the official [`Qwen/Qwen3.6-27B-FP8`](https://huggingface.co/Qwen/Qwen3.6-27B-FP8) base is Apache-2.0. |
| llama.cpp | release `b9637`, commit `aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3` | MIT; the complete notice accompanies the public CUDA dataset. |

The Kaggle model is loaded as an attached public input. No model weights are
copied into this Git repository.

## Data present in the public v91 runtime

| Source | Upstream location | Recorded license/status |
|---|---|---|
| Suman Bengali QA | [`SumanMondal/bengali_qa`](https://huggingface.co/datasets/SumanMondal/bengali_qa) | Apache-2.0 |
| SOMAJGYAAN | [`farihashifa/SOMAJGYAAN`](https://huggingface.co/datasets/farihashifa/SOMAJGYAAN) | MIT |
| Bangla MMLU | [`hishab/bangla-mmlu`](https://huggingface.co/datasets/hishab/bangla-mmlu) | Apache-2.0 in the downloaded dataset metadata |
| BLUCK | [`minhaj1403/bluck`](https://github.com/minhaj1403/bluck) | CC BY-SA 4.0 |
| Bangla Bagdhara | [`sakhadib/bangla-bagdhara`](https://www.kaggle.com/datasets/sakhadib/bangla-bagdhara) | CC BY-SA 4.0 |
| BengaliDictionary | [`MinhasKamal/BengaliDictionary`](https://github.com/MinhasKamal/BengaliDictionary) | GPL-3.0; a copy of GPL-3.0 accompanies the runtime bundle |
| Bangladesh Legal Acts | [`sakhadib/Bangladesh-Legal-Acts-Dataset`](https://huggingface.co/datasets/sakhadib/Bangladesh-Legal-Acts-Dataset) | CC BY 4.0 |
| Farhan MCQ API | [`mojnupust/api.farhan-mcq`](https://github.com/mojnupust/api.farhan-mcq) | MIT |
| Job MCQ | [`NewtonGain/job_mcq`](https://github.com/NewtonGain/job_mcq) | Public repository; no declared license found |
| QDoc | [`Nh-em0n/QDoc`](https://github.com/Nh-em0n/QDoc) | Public repository; no declared license found |
| 80-20 exam bank | [`tanvirmahfuz100/80-20-exam`](https://github.com/tanvirmahfuz100/80-20-exam) | Public repository; no declared license found |
| BenHalluEval-derived released files | Organizer-cited benchmark lineage | License must be read from the upstream release |

The three repositories without declared licenses were publicly accessible and
were used as factual question-answer evidence under the competition's external
data rule. Public accessibility is not asserted to transfer copyright or grant
a general redistribution license. Their authors retain all rights. This status
is disclosed so the organizers can determine whether those records may remain
in the evaluation bundle; the files will be removed or replaced if requested.

## Manually verified factual sources

`resources/verified_fact_answers.json` records the exact public URLs used for
each accepted answer set. Those pages retain their own terms. The runtime stores
short factual answer strings and citations, not copies of the cited pages.

## Competition data

No competition test rows, hidden labels, or generated Phase 1 prediction vector
are included in the public Git repository or public v91 runtime dataset. The
notebook reads the competition input only from the organizer-mandated mounted
path at execution time.
