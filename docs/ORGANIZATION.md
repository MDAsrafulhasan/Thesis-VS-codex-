# Workspace organization record

## What was preserved

Before moving anything, a SHA-256 inventory captured **294 original files**, totaling **40,585,529 bytes**:

- CUET_AL_Masaar research repository: 6 files.
- Huntrix research repository: 277 files.
- Original Track-A/Track-B CSVs: 8 files.
- Existing evaluation suite: 3 Python files.

The immutable [original-file manifest](catalog/original_files.json) records each original/current relative path, byte count and fingerprint. `python tools/workspace.py verify` compares the actual files against it. This detects accidental loss/change but is not an authenticated signature or a backup.

## Moves

| Original root | Current root |
|---|---|
| `IUT_DATATHON_HALLU-main/` | `references/IUT_DATATHON_HALLU-main/` |
| `IUTCSFESTDATATHON-main/` | `references/IUTCSFESTDATATHON-main/` |
| `All data/Track-A/` | `data/raw/benhallu/Track-A/` |
| `All data/Track-B/` | `data/raw/benhallu/Track-B/` |
| Root evaluation Python files | `evaluation/` |

Only the now-empty `All data/` container was removed. Both imported trees retain internal names, hierarchy, duplicate release copies, notebook cells/outputs, artifacts, licenses and scoped instructions. The evaluation scripts retain their original bytes; use their new root-relative command paths from [evaluation/README.md](../evaluation/README.md).

## New organizational material

- Root navigation and persistent `AGENTS.md` discovery/preservation rules.
- [Research reuse map](RESEARCH_REUSE.md) based on source/README/license/test inspection, including scientific limitations and missing external assets.
- [Data guide](DATA_GUIDE.md), [workflow](WORKFLOW.md), environment instructions and experiment template.
- Separate directories for generated data, raw predictions, derived reports and thesis writing.
- Root pytest configuration limiting default collection to the active suite; reference tests remain preserved for deliberate separate runs.
- Standard-library catalog/search/verification tool, regression tests and derived file/duplicate indexes.

No dataset rows, human ratings, taxonomy mapping decisions, bibliographic records or empirical thesis findings were invented. No reference notebook or GPU pipeline was executed; no external assets were downloaded and no API inference was performed as part of organization. Model weights and many original runtime assets are still external dependencies.

## Validation observed after moving

- `python tools/workspace.py verify`: all **294** original files matched their pre-move fingerprints; none missing or changed.
- `python -m pytest -q`: **202 active evaluation tests passed** from the workspace root with reference tests excluded.
- `python tools/test_workspace.py -q`: **12 isolated utility tests passed**, including changed/missing files, path traversal, duplicate preservation, nonexecuting notebook metadata and live search after catalog changes.
- Workspace documentation links were checked against the relocated local files. Original reference documentation remains untouched, including its historical inconsistencies.

## Ongoing maintenance

```sh
python tools/workspace.py catalog
python tools/workspace.py verify
python evaluation/test_pipeline.py -q
python tools/test_workspace.py -q
```

Keep a separate backup. Search and inspect existing research before adapting a component, retain provenance and licenses, test label translation and behavior, and keep independent evaluation separate from benchmark-lookup-assisted reproduction. Rebuilding derived catalogs must never overwrite the original preservation baseline.
