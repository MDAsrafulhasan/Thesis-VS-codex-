# Thesis workspace instructions

## Read before acting

1. Read `README.md`, `docs/DATA_GUIDE.md`, and `docs/RESEARCH_REUSE.md` before modeling, data conversion, or reusing a reference implementation.
2. Search `docs/catalog/FILE_INDEX.md` or run `python tools/workspace.py find "QUERY"` before adding duplicate implementations or asking the user for material already present.
3. Read the relevant original README, license, source implementation, and tests before adapting a component. In `references/IUTCSFESTDATATHON-main/`, its own `AGENTS.md` also applies to that subtree. Competition constraints there are not automatic requirements for the independent thesis pipeline.

## Ownership and preservation

- `evaluation/` is the active thesis pipeline. `references/` contains imported research, not active thesis modules.
- Treat `references/` and original files in `data/raw/` as read-only unless the user explicitly authorizes changes. Preserve internal directory structures, attribution, licenses, notebooks, historical artifacts, and all duplicate copies.
- Adapt selected components into a separately documented experiment or an appropriate active module; do not silently import an entire notebook pipeline or replace existing thesis code.
- `docs/catalog/original_files.json` is the immutable pre-organization baseline. Never regenerate it, overwrite it, or update hashes merely to make verification pass.
- Rebuild derived catalogs with `python tools/workspace.py catalog` after adding files. Verify preservation with `python tools/workspace.py verify`.

## Research correctness

- Thesis labels: `0 = non-hallucinated`, `1 = hallucinated`; Track_A=0, Track_B=1. Both reference projects invert this: 0=hallucinated, 1=faithful. Validate and explicitly translate at integration boundaries; add tests.
- Thesis BHS is a weighted error percentage, lower is better. Do not report its complement as BHS. CCC uses population moments without arbitrary epsilon; undefined results remain explicit.
- Keep independent experiments separate from lineage-/lookup-assisted reproduction. Do not allow benchmark gold answers, generated hallucinations, saved labels, manually test-targeted facts, or source-overlapping answer banks into independent inference, retrieval, training, calibration, or few-shot demonstrations.
- Check source lineage and derivative overlap, not just row IDs. Existing micro/10%/full evaluation subsets overlap intentionally; they are not train/dev/test partitions.
- Preserve raw Bengali alongside any normalized keys. Do not discard mathematical signs, dialectal variation, or task distinctions without evaluating the effect.
- The raw summarization taxonomy requires an explicit reviewed mapping. No dialect benchmark is supplied yet. Do not invent labels, mapping decisions, human scores, bibliographic metadata, empirical findings, or missing datasets.
- Reference scores are author-reported until independently reproduced. Competition F1 results are not comparable directly to thesis BHS or CCC.
- Review original and third-party licenses before copying or redistributing code, models, datasets, or figures.

## Workflow and validation

- Run commands from the workspace root unless a reference runbook explicitly requires its own directory or Kaggle environment.
- Use `python evaluation/test_pipeline.py -q` or `python -m pytest -q` for active pipeline tests. Root pytest configuration excludes imported reference tests; run them explicitly and separately only after checking their dependencies.
- Keep evaluation dependencies in a separate environment; do not install mixed CUDA/MLX reference requirements into it by default.
- Do not execute imported notebooks, install GPU stacks, download external assets, or send data to APIs merely to inspect or organize this workspace.
- Record changes, source paths, versions, commands, seeds, decoding, dataset hashes, overlap checks, runtime, and results using `experiments/TEMPLATE.md` and `docs/WORKFLOW.md`.
- Write raw predictions to run-specific `outputs/` directories and derived reports to matching `results/` directories. Keep mock outputs explicitly marked; never present them as thesis evidence.
- Keep approved thesis prose, citations, figures, and tables in `writing/`, with provenance back to validated experiments.
