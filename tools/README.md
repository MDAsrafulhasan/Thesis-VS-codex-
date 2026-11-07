# Workspace navigation and integrity tools

[`workspace.py`](workspace.py) requires only the Python standard library. It reads local metadata; it does not execute notebooks, import reference modules, install packages, download assets, or send data to an API.

From the workspace root:

```sh
python tools/workspace.py catalog
python tools/workspace.py find "judge"
python tools/workspace.py find "normaliz" --group Huntrix
python tools/workspace.py find "math" --kind notebook
python tools/workspace.py verify
python tools/test_workspace.py -q
```

- **`catalog`** generates `docs/catalog/FILE_INDEX.md`, `file_catalog.csv`, `file_catalog.json`, and `duplicate_files.csv`. It retains all duplicate copies; matching hashes are a discovery aid, not permission to delete releases/history.
- **`find`** searches live paths, titles and top-level Python/notebook symbols, case-insensitively. Add `--limit`, `--group`, or `--kind` to narrow results. Hash-validated cached metadata may be reused. It is not full-text PDF search.
- **`verify`** checks all original paths, sizes and SHA-256 values recorded before reorganization; missing or changed files produce a nonzero exit status.
- **`test_workspace.py`** exercises the utility in temporary fixture directories; it never changes real original files.

Catalogs exclude their own generated directory, environments/caches/VCS metadata, and symbolic links/junctions. Original files in reference `output/`, `release/`, and `archive/` trees are included. Notebook outputs and CSV record values are not copied into metadata; full-file SHA-256 fingerprints cover their bytes. CSV row counts/headers and code symbols are metadata only.

**The pre-organization baseline `docs/catalog/original_files.json` is immutable.** `catalog` never regenerates it. Do not change hashes to conceal a preservation failure. Intentional future changes to original material need separate preservation and migration records.

The script locates the workspace relative to its own file, so navigation commands are not dependent on the shell's current directory. An explicit `--root` exists for isolated tests and other deliberately selected workspaces; it is not needed for normal use.
