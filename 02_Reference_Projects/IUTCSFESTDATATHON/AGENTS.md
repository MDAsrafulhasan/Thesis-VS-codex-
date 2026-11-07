# Agent Workflow

This repo is for the Bengali hallucination detection datathon. Follow this loop.

## Experiment Loop

1. Read `Problem.md`, `README.md`, `DECISION.log`, and `EXPERIMENT.log` before making modeling changes.
2. Keep Phase 2 constraints in mind: offline Kaggle, open weights, target `2xT4`, under 9 hours, under 50 GB model storage.
3. Start with the smallest credible experiment that can produce a submission.
4. Use the 299 labeled rows for validation and error analysis, not as evidence that a large fine-tune will generalize.
5. Evaluate context and no-context rows separately.
6. Track hallucinated-class performance (`label=0`) in addition to macro-F1.
7. When generating a submission, write the exact command, model, validation score, output path, runtime, and Kaggle score into `EXPERIMENT.log`.
8. Prefer reproducible scripts over one-off notebook state.
9. Do not use paid APIs or closed models for any candidate intended for Phase 2.
10. Before increasing model size, check whether encoders, rerankers, overlap features, calibration, or better prompting can achieve the same gain faster.

## Model Notes

- Gemma 4 12B QAT is a priority small/mid LLM candidate because QAT reduces memory while preserving quality.
- Qwen3.5/Qwen3.6 small and mid variants are priority multilingual judge candidates.
- MTP is allowed as a speed optimization for GGUF inference, but must be benchmarked on target hardware.
- For bulk LLM judging, prefer compact deterministic JSON outputs and short max-token limits.

