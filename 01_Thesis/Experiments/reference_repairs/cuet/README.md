# CUET_AL_Masaar Reference Adaptation

## Origin & Attribution
- Source: `references/IUT_DATATHON_HALLU-main/`
- Authors: CUET_AL_Masaar team (IUT Datathon 2024)
- License: MIT License (see `LICENSE`)

## Research Notes & Documented Limitations
As documented in [`docs/CODE_REVIEW.md`](../../../docs/CODE_REVIEW.md):
- **R3 (Vowel Signs):** `bn_norm` strips non-alphanumeric characters, discarding dependent vowel signs (`কালি` vs `কুলি` collapsing).
- **R4 (Operand Reverse-Solving):** `repair_general` reverse-solves operands from candidate answers without verifying corruptions.
- **R5 (Sandhi Rule):** The rule tests difference from naive concatenation rather than matching the expected sandhi result.
- **R6 (Cache & OOM):** Static file names for anchor/test margin caches do not validate dataset identity or configuration.

Per thesis guidelines, this code serves as historical reference and is not used for independent thesis evaluations.
