"""Metadata-only workspace catalog. Never executes notebooks or updates the baseline."""

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys
import tempfile
from urllib.parse import quote

EXCLUDED = {".git", ".venv", "venv", "__pycache__", ".pytest_cache",
            ".mypy_cache", ".ruff_cache", "node_modules"}
GROUPS = {"references/IUTCSFESTDATATHON-main": "Huntrix",
          "references/IUT_DATATHON_HALLU-main": "CUET_AL_Masaar",
          "data/raw/benhallu": "BenHallu_raw", "evaluation": "Thesis_evaluation"}
FIELDS = ["path", "group", "kind", "title", "bytes", "sha256", "symbols",
          "csv_row_count", "csv_headers", "metadata_error"]
NOTES = ["Metadata only: no code execution, notebook outputs, or data row values.",
         "Titles use filenames or the first Markdown heading, not README claims; "
         "slide counts, versions, and other README drift are not inferred.",
         "All copies are retained, including outputs, releases, archives, and split data.",
         "docs/catalog, symlinks/junctions, and standard VCS/environment/cache directories "
         "are excluded. No model-size or output-directory exclusions are applied.",
         "find scans live paths and hashes; matching cached metadata may be reused. "
         "Run catalog after changes to refresh the published indexes."]


def is_link(path):
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def safe_path(root, relative):
    """Reject traversal and links before resolving any manifest/output path."""
    if not isinstance(relative, str) or not relative:
        raise ValueError("empty or non-string relative path")
    posix, windows = PurePosixPath(relative), PureWindowsPath(relative)
    if (posix.is_absolute() or windows.drive or windows.root or "\\" in relative
            or ":" in relative or any(p in {"", ".", ".."} for p in relative.split("/"))):
        raise ValueError("unsafe relative path")
    current = root
    for part in posix.parts:
        current = current / part
        if is_link(current):
            raise ValueError("symlink/junction path rejected")
    resolved = current.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("path escapes root")
    return resolved


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def kind_of(path):
    name, suffix = path.name.casefold(), path.suffix.casefold()
    words = path.as_posix().casefold()
    if name.startswith(("license", "licence", "copying", "notice")):
        return "license"
    if suffix == ".ipynb":
        return "notebook"
    if suffix in {".py", ".js", ".ts", ".sh", ".ps1", ".r"}:
        return "test" if name.startswith("test") or "/tests/" in words else "sourcecode"
    if suffix in {".ppt", ".pptx", ".odp"} or any(w in words for w in ("presentation", "slides")):
        return "presentation"
    if suffix in {".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp", ".eps"} or "/figures/" in words:
        return "figure"
    if suffix in {".pdf", ".tex", ".bib"}:
        return "paper"
    if suffix in {".csv", ".tsv", ".jsonl", ".parquet", ".arrow", ".xlsx", ".zip", ".gz", ".pkl", ".pt", ".bin"}:
        return "data"
    if (name.startswith(".") or suffix in {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".lock"}
            or name.startswith(("requirements", "dockerfile", "makefile"))):
        return "config"
    return "documentation"


def heading(text):
    match = re.search(r"^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$", text, re.MULTILINE)
    return match.group(1) if match else None


def symbols(source):
    try:
        return [node.name for node in ast.parse(source).body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    except (SyntaxError, ValueError):
        return []  # Notebook magic and incomplete source are never executed.


def metadata(path, relative, sha):
    row = {"path": relative, "group": next((group for prefix, group in GROUPS.items()
            if relative.startswith(prefix + "/")), "Workspace"), "kind": kind_of(PurePosixPath(relative)),
           "title": path.name, "bytes": path.stat().st_size, "sha256": sha, "symbols": []}
    try:
        suffix = path.suffix.casefold()
        if suffix == ".py":
            row["symbols"] = symbols(path.read_text(encoding="utf-8-sig"))
        elif suffix in {".md", ".markdown"}:
            row["title"] = heading(path.read_text(encoding="utf-8-sig")) or path.name
        elif suffix == ".ipynb":
            notebook = json.loads(path.read_text(encoding="utf-8-sig"))
            title = None
            for cell in notebook.get("cells", []):
                source = cell.get("source", "")
                source = source if isinstance(source, str) else "".join(source)
                if cell.get("cell_type") == "markdown" and title is None:
                    title = heading(source)
                elif cell.get("cell_type") == "code":
                    row["symbols"].extend(symbols(source))
            row["title"] = title or path.name
            row["symbols"] = list(dict.fromkeys(row["symbols"]))
        elif suffix == ".csv":
            csv.field_size_limit(2**31 - 1)
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream)
                row["csv_headers"] = next(reader, [])
                row["csv_row_count"] = sum(1 for _ in reader)
    except (OSError, UnicodeError, ValueError, csv.Error, TypeError, AttributeError, RecursionError) as exc:
        row["metadata_error"] = type(exc).__name__  # No source or data values in errors.
    return row


def inventory(root, cached=()):
    def fail(exc):
        raise exc

    cache = {r["path"]: r for r in cached
             if isinstance(r, dict) and isinstance(r.get("path"), str)}
    rows = []
    for directory, dirs, names in os.walk(root, followlinks=False, onerror=fail):
        base = Path(directory)
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED and not is_link(base / d)
                         and (base / d).relative_to(root).as_posix() != "docs/catalog")
        for name in sorted(names):
            path = base / name
            if is_link(path) or not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            sha = digest(path)
            old = cache.get(relative, {})
            if old.get("sha256") == sha and all(k in old for k in FIELDS[:7]):
                rows.append({k: old[k] for k in FIELDS if k in old})
            else:
                rows.append(metadata(path, relative, sha))
    return sorted(rows, key=lambda row: row["path"])


def csv_text(rows, fields):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
                         for k, v in row.items() if k in fields})
    return buffer.getvalue()


def atomic_write(root, relative, text):
    path = safe_path(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def catalog(root):
    rows = inventory(root)
    generated = datetime.now(timezone.utc).isoformat()
    document = {"generated_utc": generated, "notes": NOTES, "files": rows}
    groups = {}
    for row in rows:
        groups.setdefault(row["sha256"], []).append(row)
    duplicates = [row for copies in groups.values() if len(copies) > 1 for row in copies]
    lines = ["# Workspace file index", "", f"Generated UTC: {generated}", "",
             f"Files: {len(rows)}; bytes: {sum(r['bytes'] for r in rows)}.", ""]
    lines.extend(f"- {note}" for note in NOTES)
    for group in sorted({r["group"] for r in rows}):
        lines.extend(["", f"## {group}"])
        for kind in sorted({r["kind"] for r in rows if r["group"] == group}):
            lines.extend(["", f"### {kind}", ""])
            for row in rows:
                if row["group"] == group and row["kind"] == kind:
                    label = row["path"].replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
                    title = " ".join(row["title"].split()).replace("[", "\\[").replace("]", "\\]")
                    lines.append(f"- [{label}](../../{quote(row['path'], safe='/')}) — "
                                 f"{title}; {row['bytes']} bytes")
    outputs = {"FILE_INDEX.md": "\n".join(lines) + "\n",
               "file_catalog.json": json.dumps(document, ensure_ascii=False, indent=2) + "\n",
               "file_catalog.csv": csv_text(rows, FIELDS),
               "duplicate_files.csv": csv_text(duplicates, ["sha256", "bytes", "path", "group"])}
    for name, text in outputs.items():
        atomic_write(root, "docs/catalog/" + name, text)
    print(f"Cataloged {len(rows)} files; {len(duplicates)} copies in duplicate groups.")
    return 0


def find(root, args):
    cached = []
    path = safe_path(root, "docs/catalog/file_catalog.json")
    if path.is_file():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))["files"]
            if not isinstance(cached, list):
                cached = []
        except (ValueError, KeyError, TypeError):
            pass
    query = args.query.casefold()
    matches = [r for r in inventory(root, cached)
               if (not args.kind or r["kind"].casefold() == args.kind.casefold())
               and (not args.group or r["group"].casefold() == args.group.casefold())
               and query in " ".join([r["path"], r["title"], *r["symbols"]]).casefold()]
    matches.sort(key=lambda row: (
        query not in PurePosixPath(row['path']).name.casefold(),
        query not in row['title'].casefold(), row['path']))
    for row in matches[:args.limit]:
        print(f"[{row['group']} / {row['kind']}] {row['title']}")
        print(f"  {row['path']}\n  {root.joinpath(row['path']).as_uri()}")
        if row["symbols"]:
            relevant = [name for name in row["symbols"] if query in name.casefold()]
            preview = relevant or row["symbols"]
            remainder = len(preview) - 8
            suffix = f" (+{remainder} more)" if remainder > 0 else ""
            print("  Symbols: " + ", ".join(preview[:8]) + suffix)
    print(f"{len(matches)} matches; showing {min(args.limit, len(matches))}.")
    return 0


def preservation_migrations(root, entries, relative=None):
    """Load explicit location changes, never replacement sizes or fingerprints."""
    path = safe_path(root, relative if relative is not None
                     else "docs/catalog/preservation_migrations.json")
    if relative is None and not path.exists():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(document, dict) or type(document.get("schema_version")) is not int
            or document["schema_version"] != 1 or not isinstance(document.get("migrations"), list)):
        raise ValueError("invalid migrations schema; expected schema_version=1 and migrations list")
    baseline_paths = {}
    for row in entries:
        try:
            baseline_paths[row["path"]] = safe_path(root, row.get("path"))
        except (KeyError, ValueError):
            continue  # verify reports unsafe baseline entries independently.
    reserved = set(baseline_paths.values()) | {safe_path(root, "docs/catalog/original_files.json")}
    migrations, sources, targets = {}, set(), set()
    for record in document["migrations"]:
        if not isinstance(record, dict):
            raise ValueError("migration record must be an object")
        baseline, preserved = record.get("baseline_path"), record.get("preserved_path")
        source, target = safe_path(root, baseline), safe_path(root, preserved)
        if baseline not in baseline_paths:
            raise ValueError(f"unknown baseline entry: {baseline}")
        if source == target:
            raise ValueError("migration paths must differ")
        if source in sources:
            raise ValueError("duplicate migration baseline target")
        if target in targets:
            raise ValueError("duplicate migration preserved target")
        if target in reserved:
            raise ValueError("migration would overwrite a baseline location")
        if "reason" in record and (not isinstance(record["reason"], str) or not record["reason"].strip()):
            raise ValueError("migration reason must be a nonempty string when supplied")
        sources.add(source)
        targets.add(target)
        migrations[baseline] = preserved
    return migrations


def verify(root, migrations=None):
    manifest = json.loads(safe_path(root, "docs/catalog/original_files.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("baseline must be an object")
    entries = manifest.get("files")
    if (manifest.get("schema_version") != 1 or not isinstance(entries, list)
            or any(not isinstance(r, dict) or not isinstance(r.get("bytes"), int)
                   or r["bytes"] < 0 or not isinstance(r.get("sha256"), str)
                   or not re.fullmatch(r"[0-9a-f]{64}", r["sha256"]) for r in entries)
            or manifest.get("file_count") != len(entries)
            or manifest.get("total_bytes") != sum(r["bytes"] for r in entries)):
        raise ValueError("invalid baseline schema/count/total")
    try:
        relocated = preservation_migrations(root, entries, migrations)
    except (OSError, ValueError) as exc:
        detail = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        print(f"MIGRATION ERROR: {detail}")
        return 1
    counts = {"ok": 0, "missing": 0, "changed": 0, "unsafe": 0, "errors": 0, "migrated": 0}
    seen = set()
    for row in entries:
        relative = row.get("path")
        try:
            path = safe_path(root, relative)
            if relative in seen:
                raise ValueError("duplicate baseline path")
            seen.add(relative)
            preserved = relocated.get(relative)
            if preserved is not None:
                path = safe_path(root, preserved)
        except ValueError:
            counts["unsafe"] += 1
            print("UNSAFE: rejected baseline path")
            continue
        location = f"{relative} -> {preserved} (preserved archive)" if preserved else relative
        try:
            if not path.is_file():
                counts["missing"] += 1
                print(f"MISSING: {location}")
            else:
                size, sha = path.stat().st_size, digest(path)
                if size != row["bytes"] or sha != row["sha256"]:
                    counts["changed"] += 1
                    print(f"CHANGED: {location}")
                else:
                    counts["ok"] += 1
                    if preserved is not None:
                        counts["migrated"] += 1
                        print(f"ORIGINAL_LOCATION_CHANGED: {relative} -> {preserved} "
                              "(baseline bytes verified; active path not compared)")
        except OSError as exc:
            counts["errors"] += 1
            print(f"ERROR: {location} ({type(exc).__name__})")
    print("Verify: " + ", ".join(f"{key}={value}" for key, value in counts.items())
          + f"; original_location_changed={counts['migrated']}; preserved={counts['ok']}/{len(entries)}"
          + f"; baseline files={len(entries)}.")
    return int(counts["ok"] != len(entries) or counts["unsafe"] != 0)


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("limit must be positive")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="workspace root (default: script's parent workspace)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("catalog", help="atomically regenerate four metadata indexes; never the baseline")
    search = commands.add_parser("find", help="search live metadata, using hash-validated catalog cache")
    search.add_argument("query", metavar="QUERY")
    search.add_argument("--limit", type=positive, default=30)
    search.add_argument("--kind", help="exact kind, case-insensitive")
    search.add_argument("--group", help="exact group, case-insensitive")
    verification = commands.add_parser("verify", help="verify immutable baseline sizes and SHA-256, "
                                       "including explicitly migrated originals")
    verification.add_argument("--migrations", metavar="PATH", help="root-relative migrations JSON "
                              "(default: docs/catalog/preservation_migrations.json, if present)")
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("root is not a directory")
        if args.command == "find":
            return find(root, args)
        if args.command == "verify":
            return verify(root, args.migrations)
        return catalog(root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"{args.command} failed: {type(exc).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
