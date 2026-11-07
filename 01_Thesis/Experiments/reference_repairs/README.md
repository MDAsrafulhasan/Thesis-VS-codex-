# Reference Repairs and Adaptations

This directory contains repaired adaptations of code and notebooks from imported reference repositories (`references/`).

## Preservation and Integrity Safeguards

As mandated by [`AGENTS.md`](../../AGENTS.md):
- Files in `references/` and `data/raw/` are **strictly read-only baselines** representing original datathon submissions and research.
- Any bug fixes, adaptations, or refactorings are performed on separate copies in `experiments/reference_repairs/` rather than modifying original files.
- Adaptations preserve attribution and original licenses.

## Subdirectories

- **`huntrix/`**: Repaired adaptation of the Huntrix submission (`references/IUTCSFESTDATATHON-main/`):
  - `best_gemma12_qat_submission.repaired.ipynb`: Fixed R1 syntax errors (unterminated strings in cells 3, 4, and 6) so all code cells compile cleanly under Python 3.
  - `structured_overrides.py`: Fixed R2 numeric extraction issue where unsigned regex extracted digits out of fractions (`-1/2` leaking candidates `1` and `2`, and `-1` matching positive `1`). Signed numbers and indivisible fractions are now parsed atomically via `SIGNED_TOKEN_RE`.
  - `datathon_utils.py`: Preserved utility module for reference inspection.
  - `LICENSE`: Preserved original MIT License.
  - See [`huntrix/README.md`](huntrix/README.md).

- **`cuet/`**: Preserved adaptation of the CUET_AL_Masaar project (`references/IUT_DATATHON_HALLU-main/`):
  - `CUET_AL_Masaar_inference_notebook.repaired.ipynb`: Reference inference notebook. Note scientific limitations documented in `docs/CODE_REVIEW.md` (R3 vowel-sign normalization, R4 operand reverse-solving, R5 sandhi logic, R6 cache collision).
  - `LICENSE`: Preserved original MIT License.
  - See [`cuet/README.md`](cuet/README.md).
