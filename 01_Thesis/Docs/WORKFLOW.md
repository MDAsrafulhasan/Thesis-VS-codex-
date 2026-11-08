# Reproducible thesis workflow

Use the preserved research as a reference library—not as a hidden extension of the test set. Work from the workspace root unless a specific reference runbook requires Kaggle or its own repository directory.

## 1. Discover before implementing

1. Read [README](../README.md), [DATA_GUIDE](DATA_GUIDE.md), and [RESEARCH_REUSE](RESEARCH_REUSE.md).
2. Search: `python tools/workspace.py find "QUERY"`. Try `normaliz`, `math`, `retrieval`, `judge`, `calibr`, `paper`, or `presentation`.
3. Read the selected source, its tests, license and original documentation. Check whether the file is a current artifact, frozen release, archive, or duplicated packaging copy.
4. Decide whether to reuse an idea, adapt a bounded component, or reproduce the original system. Do not import the whole pipeline by default.
5. Check preservation: `python tools/workspace.py verify`.

The catalog excludes its own generated files and cache/environment trees. A catalog is navigation, not a substitute for reviewing implementation and scientific assumptions.

## 2. Register a protocol before looking at results

Create a uniquely named experiment directory under `experiments/`, for example `experiments/2026-11-06_context_numeric_pilot/`, and start its record from [TEMPLATE.md](../experiments/TEMPLATE.md). This is an illustrative naming convention, not a completed experiment.

Record:

- hypothesis and what would count as success/failure;
- independent evaluation **or** explicitly lineage-/lookup-assisted reproduction;
- exact source paths, authors, licenses, selected functions, and intended changes;
- dataset identity, raw/canonical hashes, label convention, approved taxonomy mapping;
- source-lineage overlap audit, grouping, exclusion rules and calibration data;
- prompt files, decoding parameters, seed, models/versions and hardware;
- primary metrics, coverage/failure policy, planned slices, comparison and ablations.

Keep an untouched final evaluation set. Do not repeatedly alter prompts, evidence, rules or thresholds in response to its outcomes and still describe the same set as independent.

## 3. Isolate environment and adaptation

Use the [evaluation environment](../environment/README.md) for the active suite. Reference GPU stacks belong in separate environments or Kaggle sessions. Record resolved versions; do not assume an unpinned reference requirements file reproduces the authors' runtime.

Adapt only the chosen component into an appropriate active module or the named experiment. Retain attribution and applicable license notices. Keep the source reference tree unchanged. Write a short source-to-adaptation change note.

Add targeted tests for:

- label/probability orientation (`1 = hallucinated` in the thesis);
- Bengali numerals, combining marks and relevant mathematical signs;
- missing context, invalid output, conflicting evidence and abstention;
- task distinctions and dialect-preservation behavior;
- positive and negative cases, not only the component's expected successes.

A reference source's existing tests are useful evidence, not proof of transfer to this benchmark. Inspect dependencies before running them explicitly; root pytest intentionally collects only active tests.

## 4. Validate on synthetic/debug data first

```sh
python evaluation/test_pipeline.py -q
python -m pytest -q
```

Use the offline commands in the [evaluation guide](../evaluation/README.md) before loading a large real split. Synthetic data checks schemas, parsers, metrics and resume behavior; it does not demonstrate Bengali capability or genuine dialect robustness.

Before each real run, verify input hashes and source-lineage disjointness. Do not use the micro/10%/full overlapping subsets as training/calibration/test partitions. Freeze any summarization mapping after review.

## 5. Run once with traceable artifacts

Use the same run ID in these locations:

```text
experiments/<run_id>/    Protocol, commands, environment, adaptation and decisions
outputs/<run_id>/        Raw per-response JSONL and checkpoint metadata
results/<run_id>/        Derived CSV/Markdown tables and report manifests
```

The angle-bracket notation describes a naming convention; substitute the actual registered run ID when running commands.

- Keep exact CLI commands and prompt content, not only a notebook screenshot.
- Record the model revision, provider/runtime, hardware, seed, temperature, token limit, dataset hashes, elapsed time and costs where relevant.
- Never place API keys or credentials in source files, notebooks, logs or manifests.
- Use `--prompt_mode both` for a paired zero-shot/CoT comparison. Check identical inputs, model revision and protocol; do not compare unmatched subsets silently.
- The active harness checkpoints each response immediately with flush/fsync and validates resume configuration. Keep the `.meta.json` sidecars with the JSONL.
- If a configuration changes, use a new run directory. Do not delete sidecars or edit checkpoint IDs to bypass a mismatch.
- A crash can require repeating a remote request that completed before its response was durably written. A seed alone cannot guarantee identical output across provider/hardware changes.

Raw inference payloads can contain substantial copyrighted or sensitive text. Review privacy and distribution rules before uploading them to Git or sending them to a remote API.

## 6. Interpret errors, coverage, and agreement honestly

- **BHS:** weighted error percentage; lower is better. Default weights are 0.5/0.5. Always report Track A FPR and Track B FNR as well.
- **Failed parses:** retained, counted as errors in operational BHS, and separately counted; valid-only metrics do not replace coverage disclosure.
- **Collapse:** report model × prompt-mode counts and rates, even when a collapsed response still contains a usable verdict.
- **Ablation:** `Delta_BHS = BHS_CoT - BHS_ZeroShot`; negative means improved error, positive means worse.
- **Cost sensitivity:** inspect all nine weight pairs. Ranking stability is a measured property, not a result to assume in advance.
- **CCC:** population moments, no arbitrary epsilon. Identical constant arrays have undefined CCC, not forced perfect agreement. Show valid-pair coverage.
- **CBS:** anchor CCC minus Chittagong CCC, in CCC units. State whether the anchor is Tangail or Standard_Bengali. Do not call score differences, CCC differences and F1 differences the same metric.
- **Undefined slices:** leave them explicit; absence of examples does not mean zero error.

Report per-task and per-type results before broad claims. Group repeated source questions/documents when estimating uncertainty; do not fabricate confidence intervals or perform row-i.i.d. bootstrapping without checking dependence. If uncertainty analysis is added, record the estimator, resampling unit and seed.

A reference leaderboard score is author-reported until reproduced. A competition result, a lineage-assisted result, and an independent thesis evaluation answer different questions.

## 7. Turn validated experiments into writing

1. Complete the experiment record, including negative results and deviations.
2. Keep raw outputs intact; regenerate tables rather than manually changing scores.
3. Put approved text in `writing/chapters/`, figures in `writing/figures/`, and curated tables in `writing/tables/`.
4. Every table/figure needs its experiment ID, source report path, dataset hash, model/protocol identity and generation command or script.
5. Verify original bibliographic metadata before adding citations. Code, datasets, models and papers can require separate citations and licenses.
6. Do not copy reference prose or figures as your own. Acknowledge adaptation and check permission/attribution terms.

Keep references' existing presentation/report artifacts as study material. Their LaTeX or slide organization can inspire your structure, but their empirical results are not your findings.

## 8. Maintain the workspace

```sh
python tools/workspace.py catalog
python tools/workspace.py verify
```

`catalog` rebuilds derived indexes and duplicate listings. It never refreshes the immutable original-file manifest. Keep all duplicate originals even if their hashes match: they preserve historical/release context.

Check [the original-file manifest](catalog/original_files.json) for old → new paths. If a future authorized change modifies an original, preserve a separate untouched copy and record the explicit migration rather than rewriting baseline hashes to make a failing verification disappear.

Back up original inputs, experiment records, resolved environments, checkpoints and writing separately. The folder layout, hashes and Git are not substitutes for a tested backup.
