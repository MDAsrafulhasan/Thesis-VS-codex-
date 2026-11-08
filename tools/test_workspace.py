"""Isolated regression tests: python tools/test_workspace.py -q."""

from contextlib import redirect_stderr, redirect_stdout
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
_bytecode_setting = sys.dont_write_bytecode
sys.dont_write_bytecode = True
try:
    import workspace
finally:
    sys.dont_write_bytecode = _bytecode_setting


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="workspace-tests-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = "evaluation/sample.py"
        self.put(self.source, "def CatalogNeedle():\n    return 1\n")
        self.put("data/raw/benhallu/Track-A/sample.csv", "question,answer\na,b\n")
        entries = []
        for name in (self.source, "data/raw/benhallu/Track-A/sample.csv"):
            content = (self.root / name).read_bytes()
            entries.append({"original_path": "old/" + name, "path": name,
                            "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
        self.baseline = {"schema_version": 1, "moves": {}, "file_count": len(entries),
                         "total_bytes": sum(r["bytes"] for r in entries), "files": entries}
        self.manifest = self.put("docs/catalog/original_files.json", json.dumps(self.baseline))

    def put(self, relative, text):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="")
        return path

    def cli(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = workspace.main(["--root", str(self.root), *args])
        self.assertEqual(stderr.getvalue(), "")
        return status, stdout.getvalue()

    def catalog_rows(self):
        self.assertEqual(self.cli("catalog")[0], 0)
        return json.loads((self.root / "docs/catalog/file_catalog.json").read_text(encoding="utf-8"))["files"]

    def test_verify_correct_baseline(self):
        status, output = self.cli("verify")
        self.assertEqual(status, 0)
        self.assertIn("ok=2", output)
        self.assertIn("baseline files=2", output)

    def test_verify_same_size_change(self):
        path = self.root / self.source
        original = path.read_bytes()
        path.write_bytes(original.replace(b"return 1", b"return 2"))
        self.assertEqual(path.stat().st_size, len(original))
        status, output = self.cli("verify")
        self.assertNotEqual(status, 0)
        self.assertIn("changed=1", output)
        self.assertIn("CHANGED: " + self.source, output)

    def test_verify_missing_file(self):
        (self.root / self.source).unlink()
        status, output = self.cli("verify")
        self.assertNotEqual(status, 0)
        self.assertIn("missing=1", output)

    def test_verify_rejects_traversal_and_absolute_paths(self):
        for unsafe in ("../escape", "evaluation/../../escape", "/absolute", "C:/absolute", "x\\escape"):
            with self.subTest(path=unsafe):
                self.baseline["files"][0]["path"] = unsafe
                self.manifest.write_text(json.dumps(self.baseline), encoding="utf-8")
                status, output = self.cli("verify")
                self.assertNotEqual(status, 0)
                self.assertIn("unsafe=1", output)

    def migration_record(self):
        preserved = "experiments/original_evaluation/sample.py"
        archive = self.root / preserved
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes((self.root / self.source).read_bytes())
        self.put(self.source, "def CatalogNeedle():\n    return 2\n")
        return {"baseline_path": self.source, "preserved_path": preserved,
                "reason": "Fixture: original bytes retained"}

    def write_migrations(self, records, relative="docs/catalog/preservation_migrations.json"):
        return self.put(relative, json.dumps({"schema_version": 1, "migrations": records}))

    def test_verify_authorized_migration_preserves_changed_active_file(self):
        original_manifest = self.manifest.read_bytes()
        record = self.migration_record()
        self.write_migrations([record])
        status, output = self.cli("verify")
        self.assertEqual(status, 0)
        self.assertIn("ok=2", output)
        self.assertIn("migrated=1", output)
        self.assertIn("original_location_changed=1", output)
        self.assertIn("preserved=2/2", output)
        self.assertIn("ORIGINAL_LOCATION_CHANGED: " + self.source + " -> " + record["preserved_path"], output)
        self.assertIn("active path not compared", output)
        self.assertNotEqual(workspace.digest(self.root / self.source), self.baseline["files"][0]["sha256"])
        self.assertEqual(self.manifest.read_bytes(), original_manifest)

    def test_verify_explicit_migrations_and_optional_reason(self):
        record = self.migration_record()
        del record["reason"]
        relative = "experiments/migrations.json"
        self.write_migrations([record], relative)
        self.assertNotEqual(self.cli("verify")[0], 0)
        status, output = self.cli("verify", "--migrations", relative)
        self.assertEqual(status, 0)
        self.assertIn("migrated=1", output)

    def test_verify_missing_explicit_migrations_fails(self):
        status, output = self.cli("verify", "--migrations", "experiments/missing.json")
        self.assertNotEqual(status, 0)
        self.assertIn("MIGRATION ERROR", output)

    def test_verify_migration_missing_archive_does_not_fall_back_to_active(self):
        record = self.migration_record()
        archive = self.root / record["preserved_path"]
        (self.root / self.source).write_bytes(archive.read_bytes())
        archive.unlink()
        self.write_migrations([record])
        status, output = self.cli("verify")
        self.assertNotEqual(status, 0)
        self.assertIn("missing=1", output)
        self.assertIn("migrated=0", output)
        self.assertIn("MISSING: " + self.source + " -> " + record["preserved_path"], output)

    def test_verify_migration_archive_hash_must_match_immutable_baseline(self):
        record = self.migration_record()
        archive = self.root / record["preserved_path"]
        original = archive.read_bytes()
        archive.write_bytes(original.replace(b"return 1", b"return 3"))
        self.assertEqual(archive.stat().st_size, self.baseline["files"][0]["bytes"])
        record.update({"sha256": workspace.digest(archive), "bytes": archive.stat().st_size})
        self.write_migrations([record])
        status, output = self.cli("verify")
        self.assertNotEqual(status, 0)
        self.assertIn("changed=1", output)
        self.assertIn("migrated=0", output)
        self.assertIn("CHANGED: " + self.source + " -> " + record["preserved_path"], output)

    def test_verify_migration_archive_size_must_match_baseline(self):
        record = self.migration_record()
        (self.root / record["preserved_path"]).write_bytes(b"short archive")
        self.write_migrations([record])
        status, output = self.cli("verify")
        self.assertNotEqual(status, 0)
        self.assertIn("changed=1", output)
        self.assertIn("migrated=0", output)

    def test_verify_migrations_leave_unmigrated_baseline_entries_enforced(self):
        record = self.migration_record()
        self.write_migrations([record])
        self.put(self.baseline["files"][1]["path"], "changed raw fixture")
        status, output = self.cli("verify")
        self.assertNotEqual(status, 0)
        self.assertIn("changed=1", output)
        self.assertIn("migrated=1", output)
        self.assertIn("CHANGED: " + self.baseline["files"][1]["path"], output)

    def test_verify_rejects_unknown_migration_baseline(self):
        record = self.migration_record()
        record["baseline_path"] = "evaluation/unknown.py"
        self.write_migrations([record])
        status, output = self.cli("verify")
        self.assertNotEqual(status, 0)
        self.assertIn("unknown baseline entry", output)

    def test_verify_rejects_unsafe_migration_paths(self):
        record = self.migration_record()
        for field in ("baseline_path", "preserved_path"):
            for unsafe in ("../escape", "evaluation/../../escape", "/absolute", "C:/absolute",
                           "x\\escape", "x//escape", "./escape", "", None):
                with self.subTest(field=field, path=unsafe):
                    self.write_migrations([{**record, field: unsafe}])
                    status, output = self.cli("verify")
                    self.assertNotEqual(status, 0)
                    self.assertIn("MIGRATION ERROR", output)
        for unsafe in ("../escape.json", "/absolute.json", "C:/absolute.json", "x\\escape.json"):
            with self.subTest(manifest=unsafe):
                status, output = self.cli("verify", "--migrations", unsafe)
                self.assertNotEqual(status, 0)
                self.assertIn("MIGRATION ERROR", output)

    def test_verify_rejects_duplicate_migration_baseline_targets(self):
        record = self.migration_record()
        self.write_migrations([record, {**record, "preserved_path": "experiments/other.py"}])
        status, output = self.cli("verify")
        self.assertNotEqual(status, 0)
        self.assertIn("duplicate migration baseline target", output)

    def test_verify_rejects_duplicate_migration_preserved_targets(self):
        record = self.migration_record()
        self.write_migrations([record, {**record, "baseline_path": self.baseline["files"][1]["path"]}])
        status, output = self.cli("verify")
        self.assertNotEqual(status, 0)
        self.assertIn("duplicate migration preserved target", output)

    def test_verify_rejects_equal_paths_and_baseline_overwrites(self):
        record = self.migration_record()
        for target in (self.source, self.baseline["files"][1]["path"], "docs/catalog/original_files.json"):
            with self.subTest(target=target):
                self.write_migrations([{**record, "preserved_path": target}])
                status, output = self.cli("verify")
                self.assertNotEqual(status, 0)
                self.assertIn("MIGRATION ERROR", output)
                self.assertIn("paths must differ" if target == self.source else "overwrite a baseline", output)

    def test_verify_rejects_invalid_migrations_schema(self):
        record = self.migration_record()
        documents = [[], {"schema_version": 2, "migrations": [record]},
                     {"schema_version": True, "migrations": [record]},
                     {"schema_version": 1, "migrations": {}}, {"schema_version": 1},
                     {"schema_version": 1, "migrations": [None]},
                     {"schema_version": 1, "migrations": [{}]},
                     {"schema_version": 1, "migrations": [{**record, "reason": ""}]},
                     {"schema_version": 1, "migrations": [{**record, "reason": 12}]}]
        for document in documents:
            with self.subTest(document=document):
                self.put("docs/catalog/preservation_migrations.json", json.dumps(document))
                status, output = self.cli("verify")
                self.assertNotEqual(status, 0)
                self.assertIn("MIGRATION ERROR", output)
        self.put("docs/catalog/preservation_migrations.json", "{invalid JSON")
        self.assertNotEqual(self.cli("verify")[0], 0)

    def test_catalog_leaves_baseline_and_migrations_bytes_unchanged(self):
        record = self.migration_record()
        migrations = self.write_migrations([record])
        original_manifest, original_migrations = self.manifest.read_bytes(), migrations.read_bytes()
        for _ in range(2):
            self.catalog_rows()
            self.assertEqual(self.manifest.read_bytes(), original_manifest)
            self.assertEqual(migrations.read_bytes(), original_migrations)
        self.assertEqual(self.cli("verify")[0], 0)

    def test_inventory_and_duplicate_catalog_keep_all_copies(self):
        paths = {"references/IUTCSFESTDATATHON-main/output/report one.pdf",
                 "references/IUTCSFESTDATATHON-main/release/archive/report copy.pdf"}
        for name in paths:
            self.put(name, "duplicate PDF fixture")
        copies = [r for r in workspace.inventory(self.root) if r["path"] in paths]
        self.assertEqual({r["path"] for r in copies}, paths)
        self.assertEqual(len({r["sha256"] for r in copies}), 1)
        self.catalog_rows()
        with (self.root / "docs/catalog/duplicate_files.csv").open(encoding="utf-8", newline="") as stream:
            self.assertEqual({r["path"] for r in csv.DictReader(stream)}, paths)
        markdown = (self.root / "docs/catalog/FILE_INDEX.md").read_text(encoding="utf-8")
        self.assertIn("../../references/IUTCSFESTDATATHON-main/output/report%20one.pdf", markdown)

    def test_catalog_excludes_indexes_and_caches_not_original_assets(self):
        included = {"references/IUTCSFESTDATATHON-main/.gitignore",
                    "references/IUTCSFESTDATATHON-main/LICENSE",
                    "references/IUTCSFESTDATATHON-main/output/report.pdf",
                    "references/IUTCSFESTDATATHON-main/release/archive/model.py",
                    "outputs/later.json", "data/splits/train.csv"}
        for name in included:
            self.put(name, "asset: " + name)
        self.put("docs/catalog/stale/nested.md", "excluded")
        for cache in (".git", ".venv", "venv", "__pycache__", ".pytest_cache",
                      ".mypy_cache", ".ruff_cache", "node_modules"):
            self.put("nested/" + cache + "/ignored.txt", "excluded")
        expected = included | {entry["path"] for entry in self.baseline["files"]}
        for _ in range(2):
            rows = self.catalog_rows()
            self.assertEqual({r["path"] for r in rows}, expected)
            by_path = {r["path"]: r for r in rows}
            self.assertEqual(by_path["references/IUTCSFESTDATATHON-main/LICENSE"]["kind"], "license")
            self.assertEqual(by_path["references/IUTCSFESTDATATHON-main/.gitignore"]["group"], "Huntrix")

    def test_catalog_leaves_baseline_bytes_unchanged(self):
        original = self.manifest.read_bytes()
        self.catalog_rows()
        self.catalog_rows()
        self.assertEqual(self.manifest.read_bytes(), original)
        self.assertEqual(self.cli("verify")[0], 0)

    def test_notebook_metadata_without_execution_or_outputs(self):
        notebook = {"cells": [
            {"cell_type": "markdown", "source": ["# Notebook fixture\n"]},
            {"cell_type": "code", "source": ["raise RuntimeError('MUST NOT EXECUTE')\n",
             "def notebook_function(): pass\n", "async def async_function(): pass\n",
             "class NotebookClass: pass\n"], "outputs": [{"text": "PRIVATE_OUTPUT"}]}]}
        name = "references/IUT_DATATHON_HALLU-main/example.ipynb"
        self.put(name, json.dumps(notebook))
        row = next(r for r in self.catalog_rows() if r["path"] == name)
        self.assertEqual(row["symbols"], ["notebook_function", "async_function", "NotebookClass"])
        self.assertEqual(row["title"], "Notebook fixture")
        self.assertEqual(row["group"], "CUET_AL_Masaar")
        self.assertEqual(row["kind"], "notebook")
        self.assertNotIn("PRIVATE_OUTPUT", json.dumps(row))
        self.assertNotIn("metadata_error", row)

    def test_csv_unicode_bom_headers_and_multiline_row_count(self):
        name = "data/raw/benhallu/Track-A/sample.csv"
        self.put(name, '\ufeffপ্রশ্ন,উত্তর\n"প্রথম\nলাইন",সত্য\nদ্বিতীয়,মিথ্যা\n')
        row = next(r for r in self.catalog_rows() if r["path"] == name)
        self.assertEqual(row["csv_headers"], ["প্রশ্ন", "উত্তর"])
        self.assertEqual(row["csv_row_count"], 2)
        self.assertNotIn("সত্য", json.dumps(row, ensure_ascii=False))

    def test_find_casefold_symbol_search_and_filters(self):
        status, output = self.cli("find", "cAtAlOgNeEdLe", "--kind", "SOURCECODE",
                                  "--group", "thesis_evaluation", "--limit", "1")
        self.assertEqual(status, 0)
        self.assertIn(self.source, output)
        self.assertIn("CatalogNeedle", output)
        self.assertIn((self.root / self.source).as_uri(), output)
        self.assertIn("1 matches; showing 1", output)

    def test_find_add_remove_after_cached_catalog(self):
        self.catalog_rows()
        path = self.put("evaluation/new.py", "def FreshSymbol(): pass\n")
        self.assertIn("1 matches", self.cli("find", "freshsymbol")[1])
        self.catalog_rows()
        path.unlink()
        self.assertIn("0 matches", self.cli("find", "freshsymbol")[1])

    def test_find_changed_symbols_after_cached_catalog(self):
        self.catalog_rows()
        self.put(self.source, "def ChangedSymbol(): pass\n")
        self.assertIn("1 matches", self.cli("find", "changedsymbol")[1])
        self.assertIn("0 matches", self.cli("find", "catalogneedle")[1])


if __name__ == "__main__":
    unittest.main()
