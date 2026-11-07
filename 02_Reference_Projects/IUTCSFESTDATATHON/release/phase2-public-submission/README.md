# Huntrix · Frozen Phase 2 inference

This is the minimal reproducibility package for Team Huntrix’s submission to the Bengali LLM Hallucination Detection Challenge.

## Frozen configuration

| Field | Value |
|---|---|
| Kaggle notebook | [`seyamalam/bengali-hallucination-v85-compliance-inference`](https://www.kaggle.com/code/seyamalam/bengali-hallucination-v85-compliance-inference) |
| Frozen notebook version | **4** |
| Hardware | **Tesla T4 ×2** |
| Reported runtime | **0.15 hours** |
| Measured conservative run | 508.185 seconds (`0.1412` hours) |
| Internet | Disabled |
| Model | [`Qwen3.6-27B-Q4_K_P`, version 1](https://www.kaggle.com/models/snowrab/qwen3-6-27b-q4-k-p/pyTorch/default/1) |
| Runtime dataset | [`bengali-hallucination-phase2-runtime-v85`](https://www.kaggle.com/datasets/seyamalam/bengali-hallucination-phase2-runtime-v85) |
| CUDA runtime | [`llama.cpp CUDA b9637`](https://www.kaggle.com/datasets/seyamalam/bengali-hallucination-llama-cpp-cuda-b9637) |
| Competition input | `/kaggle/input/competitions/bengali-hallucination/test set.csv` |

The notebook is offline and produces `submission.csv` directly from the raw competition input. It does not load a saved Phase 1 prediction vector.

## Reproduction result

Two consecutive frozen T4×2 runs reproduced the scored Phase 1 v85 output byte-for-byte:

- 2,516 ordered rows with columns exactly `id,label`;
- label counts `{0: 1176, 1: 1340}`;
- output SHA-256 `19c5880021d6e3e9fd47c567d7b45898fbaee09d5f59009e96cf818c3737cb31`;
- Kaggle public macro-F1 `0.966`;
- measured runtimes `473.564s` and `508.185s`.

## Files

- `inference.ipynb` — frozen raw-input inference notebook.
- `FREEZE_MANIFEST.json` — exact versions, hashes, hardware, runtime, and asset links.
- `CITATIONS.md` — model, runtime, benchmark, and evidence citations.

No competition data, generated predictions, private credentials, model weights, or runtime corpora are stored in this repository.

## Integrity

Do not modify the notebook, model version, runtime datasets, or manifest after the submission deadline. Verify the notebook with:

```bash
shasum -a 256 inference.ipynb
```

Expected SHA-256 for this documented notebook revision:

```text
52fd4dcdcd53539847a06df207a065d64f379f0fd7491b06f480a0c43431c9a2
```

This revision adds Markdown documentation only. Its 13 executable cells are
byte-for-byte identical to the frozen Kaggle v4 notebook. Canonical executable
code-cell SHA-256:

```text
944f80f84f589946dfd840f9fed23fa0b64f71beb02a25d943d3bf612c925b26
```
