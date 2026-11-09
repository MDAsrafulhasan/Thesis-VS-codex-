---
name: graphify
description: "Map a repository into a local Graphify knowledge graph and query code relationships. Use when exploring architecture, tracing dependencies, explaining symbols, finding paths between concepts, or generating a visual code map."
argument-hint: "Repository path or Graphify query"
user-invocable: true
disable-model-invocation: false
---

# Graphify

Use Graphify when a repository is large or its relationships are easier to inspect as a graph than by reading files one at a time. Graphify builds a local knowledge graph and produces an interactive HTML map, a report, and JSON that can be queried later.

## Prerequisites

- Python 3.10 or newer
- `uv` or `pipx`
- A supported AI coding assistant with the Graphify integration installed

On Windows PowerShell, install the CLI with one of these commands:

```powershell
uv tool install graphifyy
# Or:
pipx install graphifyy
```

The package is named `graphifyy`; the installed executable is `graphify`.

## Install the assistant skill

Run this once from the repository or workspace:

```powershell
graphify install
```

This detects the available assistant and registers Graphify's native skill. If PowerShell cannot find `graphify`, reopen the terminal after installing the CLI and verify it with `graphify --help`.

## Build a graph

Run the registered command against the repository:

```text
/graphify .
```

Use a relative or absolute path when targeting a different repository:

```text
/graphify C:\path\to\repository
```

The default output directory is `graphify-out\` and contains:

- `graph.html` — interactive graph for browsing, filtering, and searching
- `GRAPH_REPORT.md` — generated highlights and suggested questions
- `graph.json` — persisted graph for subsequent queries

Keep generated output local unless the user explicitly asks to commit or share it. Do not expose source files, credentials, or other sensitive material outside the local workspace.

## Query an existing graph

Prefer graph queries for relationship questions after a graph has been built:

```powershell
graphify explain "SymbolOrConcept"
graphify path "ConceptA" "ConceptB"
graphify query "How does the authentication flow reach the database?"
```

- `explain` shows a node and its connections.
- `path` traces the shortest relationship path between two nodes.
- `query` returns a scoped subgraph for a natural-language question.

Treat `EXTRACTED` edges as explicit source relationships and `INFERRED` edges as relationships resolved by Graphify. Report that distinction when explaining results.

## Troubleshooting

1. If Python is missing or too old, install Python 3.10+ and reopen PowerShell.
2. If `uv` or `pipx` is unavailable, install one of them before installing `graphifyy`.
3. If `graphify` is not recognized after installation, reopen PowerShell and run `graphify --help`.
4. If assistant registration is missing, rerun `graphify install` from the target workspace.
5. If a query has no useful result, rebuild the graph for the correct repository path and check `graphify-out\GRAPH_REPORT.md`.
