# Huntrix Reference Adaptation & Repairs

## Origin & Attribution
- Source: `references/IUTCSFESTDATATHON-main/`
- Authors: Huntrix team (IUT CS Fest Datathon 2024, Phase 2 Public Submission)
- License: MIT License (see `LICENSE`)

## Applied Repairs
1. **R1 — Syntax Errors in Legacy Notebook (`best_gemma12_qat_submission.repaired.ipynb`):**
   - **Cell 3:** Fixed unterminated raw regex string containing smart quotes `r"[...,;:!?()\[\]{}"“”‘’...` by enclosing in single quotes and properly escaping the double quote.
   - **Cell 4:** Fixed split newline literal `return "\n\n".join(parts)` where the newline was split across two broken lines.
   - **Cell 6:** Fixed JSON record write where `"\n"` was split across lines.
   - *Result:* All 11 code cells now compile successfully under Python 3.

2. **R2 — Indivisible Fraction & Signed Number Parsing (`structured_overrides.py`):**
   - Original `numbers()` extracted unsigned digits alongside fractions, causing `-1/2` to leak `1` and `2`, and `-1` to match positive `1`.
   - Replaced with `SIGNED_TOKEN_RE` which extracts signed fractions and signed decimals/integers as indivisible units, preventing sign flipping and fraction component splitting.
