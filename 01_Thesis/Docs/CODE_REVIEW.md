# Code review: evaluation, workspace tools, and preserved research

## Status and scope

> [!NOTE]
> For the latest evaluation pipeline status, blocker analysis, benchmarked overhead, and optimization roadmap, see the follow-up [Evaluation Pipeline & Optimization Audit Report](AUDIT_REPORT.md).

**Review findings, not a repair release.** No Python source, original notebook, raw input, or preservation-baseline entry was changed. This report and the regenerated derived catalog are the only intentional workspace changes from this review.

The review combined executable offline checks of active code, static checks across the entire Python/notebook inventory, and focused source inspection of reference runtime paths. It is not a claim that every branch of every imported module has been exercised.

### Observed validation

| Check | Observed result |
|---|---|
| `python -m pytest -q` | **202 passed**; default collection is the active evaluation suite |
| `python tools/test_workspace.py -q` | **12 passed** |
| `python tools/workspace.py verify` | **294 original files matched; 0 missing, 0 changed** |
| Standalone Python syntax/compilation, without execution | **155 files passed**, including 150 reference files |
| Notebook JSON inspection | **7 notebooks readable** |
| Nonempty notebook code-cell compilation, without execution | **81 cells inspected: 78 passed, 3 failed**, all failures in one archived notebook |
| Targeted editor diagnostics | No errors/warnings in active `eval.py`, `make_splits.py`, or `tools/workspace.py` |

The existing passing tests do **not** cover the bugs below. Additional active-code reproductions used temporary directories, synthetic callbacks, and mocked HTTP; no real API calls, downloads, GPU runs, reference imports, or notebook execution occurred. Reference behavior findings are static traces, not independently reproduced model results.

### Correct behavior reconfirmed

- BHS is the weighted **error** percentage, lower is better; Track A truth is 0 and Track B truth is 1.
- Lin CCC uses population moments without epsilon; identical constant arrays remain undefined.
- CBS uses anchor CCC minus Chittagong CCC, in CCC units.
- Ordinary failed parses are retained, penalized in operational BHS, and reported separately with valid-only metrics/coverage.
- Normal resume, torn trailing writes, ordinary configuration mismatches, duplicate IDs, and concurrent checkpoint locks pass their existing tests.
- Normal mock split quotas and stratification pass. Rare-stratum loss is rejected rather than silently published.
- Workspace integrity/catalog tests pass; no new high-priority defect was reproduced in those tools during this review.

## Active pipeline: high-priority findings

### A1 — P1: cross-origin redirects can expose an API bearer token

**Location:** [evaluation/eval.py](../evaluation/eval.py), lines 306–314.

`Authorization` is added as a normal urllib request header; the default redirect handler forwards it on a cross-origin redirect. A synthetic HTTPS transport returned a 302 from the configured provider to another origin and observed the same synthetic bearer token on the redirected request. No actual credential or network destination was used.

**Impact:** A malicious/misconfigured provider redirect can disclose the configured credential. The adapter also accepts the redirected response as inference output.

**Repair direction:** Reject cross-origin redirects or explicitly strip credentials and validate origins before following. Add a transport-level redirect test; replacing `urlopen` directly does not test its redirect handler.

### A2 — P1: quoted examples and nested malformed JSON are accepted as judgments

**Location:** [evaluation/eval.py](../evaluation/eval.py), `_extract`, lines 197–222.

Reproduced outcomes:

```text
Input:  The candidate says "<verdict>No</verdict>". I cannot determine correctness.
Actual: parsed_verdict=0, parse_failure=False

Input:  {broken: {"hallucinated": true}}
Actual: parsed_verdict=1, parse_failure=False
```

The score parser likewise accepts a quoted/example `<score>10</score>` when the surrounding text explicitly declines to assign a score. Searching every opening brace also promotes a nested object from an invalid outer document.

**Impact:** Candidate echoes or reasoning examples can become false successful predictions and distort scores without increasing parse-failure counts.

**Repair direction:** Define an authoritative final-output grammar: an explicit terminal verdict/score, or a complete top-level JSON object. Do not promote quoted examples or arbitrary nested objects. Retain conservative conflict handling and bilingual support; add these counterexamples as tests.

### A3 — P1: different custom judges can silently share a resumed checkpoint

**Location:** [evaluation/eval.py](../evaluation/eval.py), configuration construction, lines 518–525.

Without explicit `backend_identity` or a `.identity` attribute, `type(judge).__qualname__` is used. Every ordinary function becomes `function`; instances of one callable class can likewise hide different configurations.

**Reproduction:** A Yes-returning function saves one row and interrupts; a different No-returning function resumes the same two-row dataset. The checkpoint contains one Yes and one No prediction with no mismatch error. Existing test helpers normally supply an explicit identity and therefore mask the fallback.

**Repair direction:** Require an explicit stable, configuration-aware identity for opaque callables rather than pretending their type identifies the judge. Test differing implementations and differing callable-instance parameters.

## Active pipeline: additional reproducible findings

### A4 — P2: malformed response metadata can prevent durable recording

**Location:** [evaluation/eval.py](../evaluation/eval.py), lines 330–340 and 560–580.

Offline cases included valid verdict text with `usage=["bad"]` (uncaught `AttributeError`), string-valued completion counts, negative token counts, deeply nested JSON (`RecursionError`), and an unpaired surrogate (`UnicodeEncodeError`). Processing fails before a durable JSONL record is written. Some cases repeatedly reissue the same inference on restart.

**Repair direction:** Validate adapter schemas explicitly and journal received evidence with an explicit failure state even when parsing, metadata validation, or UTF-8 serialization fails. Preserve malformed metadata separately rather than silently treating it as valid usage. Do not discard meaningful text just to satisfy the normal-record schema.

### A5 — P2: truncated partial revisions evade collapse/failure flags

**Location:** [evaluation/eval.py](../evaluation/eval.py), lines 153–174 and 209–213.

With `completion_tokens=1024` and `finish_reason='length'`, both examples retain the earlier Yes verdict and report neither collapse nor parse failure:

```text
<verdict>Yes</verdict>
<verd

<verdict>Yes</verdict>
{"hallucinated": false
```

Equivalent score examples also reproduce the problem. Existing tests cover a complete opening tag followed by truncation, not a partial tag name or unfinished JSON revision.

**Repair direction:** Detect unfinished trailing output structures/revisions, and do not let any earlier balanced tag certify that the final revision completed.

### A6 — P2: danda-separated Bengali repetition can be missed

**Location:** [evaluation/eval.py](../evaluation/eval.py), `_repetition_collapse`, lines 137–149.

Four copies of `এই উত্তর শুধু একই কথা বলে`, joined by `।` without spaces, return `is_collapsed=False`. Joining the same copies by `। ` returns true. Whitespace-only splitting followed by punctuation deletion fuses boundary words.

**Repair direction:** Treat sentence punctuation as word boundaries while preserving Bengali combining marks. Test Bengali/English punctuation with and without following spaces.

### A7 — P2: collapse detection has cubic worst-case work on unbounded input

**Location:** [evaluation/eval.py](../evaluation/eval.py), lines 137–149; records are reparsed at lines 496–498.

Enumerating block lengths and starting positions while slicing variable-length lists gives O(W³) worst-case work. Offline nonrepeating-word probes took approximately 0.03 s at 512 words, 0.20 s at 1,024, 1.05 s at 2,048, and 6.4 s at 4,096; these are environment-specific engineering observations, not model performance results. Larger combined probes reached their process timeout.

The requested token limit does not enforce a local bound on text returned by a noncompliant endpoint or read from a checkpoint.

**Repair direction:** Use a more efficient exact repetition algorithm and an explicit response/resource policy. Avoid silently truncating text or changing the required repetition definition. Do not parse the same fresh response twice unnecessarily.

### A8 — P2: response-body transport failures bypass configured retries

**Location:** [evaluation/eval.py](../evaluation/eval.py), lines 311–327.

Synthetic response `.read()` failures `http.client.IncompleteRead` and `ConnectionResetError` caused one attempt and no retry despite `retries=2`. `IncompleteRead` can also escape the CLI's controlled error handling.

**Repair direction:** Cover transient connection/body-read failures with bounded retries and a consistent final exception. Do not retry permanent schema errors or unsupported API parameters indiscriminately.

### A9 — P2: floating-point noise breaks mathematically exact ranking ties

**Location:** [evaluation/eval.py](../evaluation/eval.py), lines 738–744.

With 30 A and 90 B examples:

- Model A: 0 A errors and 1 B error.
- Model B: 3 A errors and 0 B errors.

At weights `(0.1, 0.9)`, both exact BHS values are **1%**, but floating results are `1.0` and `1.0000000000000002`. The ranker gives ranks 1 and 2, affecting stability calculations.

**Repair direction:** Rank using exact count/rational comparisons for the nine rational weight pairs, not arbitrary score rounding or epsilon. Specify the policy for changes between ties and strict orderings.

### A10 — P2: CoT comparisons ignore known returned model-revision differences

**Location:** [evaluation/eval.py](../evaluation/eval.py), lines 712–724.

Two complete mocked runs used the same requested alias/backend/protocol but reported different resolved models in `provider_metadata.model`. The report still emitted a complete paired delta (one probe produced 100 to 0, delta −100) as though only prompting changed.

**Repair direction:** Validate known resolved-model identities across a run and comparison; expose provider/fingerprint drift and unknown identity. Reject or visibly qualify incompatible revisions instead of presenting an unqualified prompt-only ablation. Missing metadata must not be represented as verified identity.

### A11 — P2: undefined task rankings become false instability results

**Location:** [evaluation/eval.py](../evaluation/eval.py), lines 733–744.

Two complete valid runs containing only `bn_qa` generate absent-task entries with `BHS=NaN`, `Rank=NaN`, but `Ranking_Stable=False`.

**Repair direction:** Keep stability undefined when a ranking cannot be computed; do not compare NaN-filled rank dictionaries as if they were empirical orderings.

### A12 — P2: canonical CSV reload loses synthetic-data provenance

**Location:** [evaluation/make_splits.py](../evaluation/make_splits.py), lines 436–441 and 573–575.

A generated mock DataFrame has `attrs['mock']=True`. Saving it to CSV and loading via `load_dataset` resets that flag to false. Re-splitting then writes `mock:false` into the new manifest. This was reproduced entirely with synthetic files in a temporary directory.

**Impact:** The normal CSV handoff can turn known synthetic input into a manifest that looks non-mock. Evaluator configurations also do not carry an explicit input-kind declaration.

**Repair direction:** Carry hash-validated provenance through a sidecar/data register or an explicit input-kind contract; distinguish real, synthetic and unknown. Do not infer authenticity merely from a filename or default unknown input to scientifically real.

### A13 — P2: contradictory duplicate CSV headers bypass schema validation

**Location:** [evaluation/make_splits.py](../evaluation/make_splits.py), `_csv_read` lines 281–283, column check lines 117–124, and projection lines 182–183.

A canonical CSV with two `ground_truth_label` headers containing opposite values is accepted. pandas renames the second header to `ground_truth_label.1` before the duplicate-column check, and projection silently discards it.

**Repair direction:** Validate the raw CSV header before pandas' automatic name mangling. Reject duplicate required headers explicitly instead of silently selecting one conflicting label column.

## Imported research: do not modify originals or reuse unchecked

These findings concern preserved reference code, not execution of that code in the thesis pipeline. The reference labels remain inverted relative to the thesis. Benchmark-answer lookup remains ineligible for the independent baseline regardless of these code defects.

### R1 — Archived notebook contains three syntax errors

**File:** [best_gemma12_qat_submission.ipynb](../references/IUTCSFESTDATATHON-main/archive/legacy_notebooks/best_gemma12_qat_submission.ipynb).

Source-cell locations are 1-based; they are not lines in the JSON container:

| Cell | Source line | Compilation failure |
|---|---:|---|
| 3 | 19 | Invalid smart-quote character after an incorrectly terminated regex string |
| 4 | 43 | Unterminated string literal |
| 6 | 49 | Unterminated string literal |

These original bytes still match the pre-organization baseline, so the failures were not introduced by relocation. Treat this notebook as historical material, not a runnable entrypoint. A repaired adaptation would need a new file and explicit provenance.

### R2 — Huntrix structured matching accepts incorrect signs/fraction components

**Location:** [structured_overrides.py](../references/IUTCSFESTDATATHON-main/scripts/structured_overrides.py), `NUMBER_RE` line 18, `numbers` lines 31–52, `answer_matches` lines 526–530.

The parser adds a signed fraction, then scans the same text using an unsigned number regex. Thus `-1` contributes candidate `1`, and `1/2` contributes candidates `1/2`, `1`, and `2`. Acceptance of any candidate lets an expected result of 1 match an incorrect negative/fraction response. These rules can resolve a row before the fallback judge.

This is a direct static consequence of the implementation, not a GPU/model reproduction. Any thesis adaptation needs signed-number and indivisible-fraction regression tests.

### R3 — CUET normalization destroys Bengali vowel-sign distinctions

**Location:** [CUET notebook](../references/IUT_DATATHON_HALLU-main/CUET_AL_Masaar_inference_notebook.ipynb), cell 2 `bn_norm`, source lines 12–17; lookup construction lines 133–144.

Keeping only `isalnum()`/whitespace excludes dependent vowel signs and other combining marks. For example, `কালি` and `কুলি` both normalize to `ক ল`. Question/headword lookup keys and answer comparisons use this representation, allowing semantically different entries to collide.

Preserve Bengali marks and raw text in any independent adaptation; test actual Bengali minimal pairs, not just punctuation.

### R4 — CUET operand repair can justify a wrong answer by changing the question

**Location:** CUET notebook, cell 4 `_trunc_ok` lines 218–219 and `repair_general` lines 234–259; cell 6 final repair application lines 86–88.

The final repair layer reverse-solves operands from a candidate, then accepts prefix/suffix-compatible reconstructions without proving corruption occurred. For workers taking 2 and 3 days, the true joint time is 1.2 days; candidate 2.4 can be justified by reconstructing the first duration as 12 because `12×3/(12+3)=2.4` and `"12".endswith("2")`.

This is a static false-positive path. Do not adopt candidate-conditioned question repair as independent factual validation without a separately justified corruption protocol and false-positive tests.

### R5 — CUET sandhi rule accepts unrelated non-concatenated answers

**Location:** CUET notebook, cell 2 `grammar`, source lines 335–341, and `stack3`, lines 344–346.

For a recognized sandhi question without the special unchanged-context branch, the rule returns `int(not is_naive)`. It checks that the response is not naive concatenation, not that it equals the actual sandhi result. An unrelated response can therefore be labeled faithful.

A thesis adaptation must verify the expected answer or abstain, not equate 'different from one wrong answer' with correctness.

### R6 — CUET cache identity and OOM failure handling remain unsafe

**Location:** CUET notebook, cell 3 `run_margins`, source lines 47–65, and `judge_margin`, lines 31–45.

Fixed `anchor`/`test` filenames do not validate dataset text, IDs/order, model or configuration. A same-length new dataset can reuse stale margins. Repeated OOM returns margin zero/status zero; failed rows can be counted as completed and later classified from ordinary margins.

These are previously documented hazards reconfirmed by source inspection. Use identity-validated checkpoints and explicit inference-failure states in any adaptation.

## Recommended repair order

1. Fix **A1–A3** before using secrets, real-model outputs, or custom resumable judges.
2. Add regression cases for A4–A8; harden evidence preservation, parsing and transport behavior without weakening the required mathematical definitions.
3. Fix A9–A13 before producing publication tables or reusing generated datasets as evidence.
4. Keep original reference snapshots untouched. Repair only separately attributed adaptations, with tests covering R1–R6 and independent source-lineage checks.
5. Rerun active tests plus all new reproductions, refresh only derived catalogs, and keep any authorized original-code migration explicit. Do not alter baseline hashes to conceal changed files.

The preserved reference systems' reported competition scores were not independently reproduced, and no dialect benchmark or summarization mapping decision was invented by this review.
