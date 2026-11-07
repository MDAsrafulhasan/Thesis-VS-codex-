from pathlib import Path
import unittest

from package_phase2_runtime import ROOT, RUNTIME_FILES


class Phase2RuntimePackageTests(unittest.TestCase):
    def test_runtime_files_exist_and_exclude_saved_predictions(self) -> None:
        missing = [relative for relative in RUNTIME_FILES if not (ROOT / relative).is_file()]
        self.assertEqual(missing, [])
        self.assertFalse(any("submission" in Path(relative).name for relative in RUNTIME_FILES))

    def test_direct_router_dependencies_are_packaged(self) -> None:
        required = {
            "LICENSE",
            "THIRD_PARTY_NOTICES.md",
            "licenses/Apache-2.0.txt",
            "licenses/GPL-3.0.txt",
            "scripts/phase2_pipeline.py",
            "scripts/datathon_utils.py",
            "scripts/train_portable_verifier.py",
            "scripts/train_verifier.py",
            "scripts/source_answer_bank.py",
            "scripts/structured_overrides.py",
            "scripts/context_numeric_overrides.py",
            "scripts/context_span_overrides.py",
            "scripts/typed_evidence.py",
            "scripts/typed_idiom_matcher.py",
            "resources/typed_evidence_v1.json",
            "outputs/source_cache/bangla_bagdhara_lrec/bagdhara_export_done_2026-03-01_010356.json",
        }
        self.assertTrue(required.issubset(RUNTIME_FILES))


if __name__ == "__main__":
    unittest.main()
