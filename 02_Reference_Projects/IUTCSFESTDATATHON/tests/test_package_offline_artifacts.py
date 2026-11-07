import unittest

from package_offline_artifacts import PHASE2_ARTIFACTS, REQUIRED_ARTIFACTS, REPRO_SCRIPTS


class OfflinePackageClosureTests(unittest.TestCase):
    def test_current_candidate_and_router_banks_are_required(self):
        expected = {
            "resources/verified_fact_answers.json",
            "resources/legacy_law_acts.json",
            "outputs/source_answer_bank_complete_bcs.jsonl",
            "outputs/source_cache/benhallueval_data/qa_gt_1000.csv",
            "outputs/source_cache/benhallueval_data/qa_4000.csv",
            "outputs/source_answer_bank_farhan_qdoc.jsonl",
            "outputs/source_answer_bank_bcs_all.jsonl",
            "outputs/submission_v85_shakib_birth_repair_nosubmit.csv",
            "outputs/phase2_data/hf_legal_acts_evidence.jsonl",
        }
        self.assertTrue(expected.issubset(set(REQUIRED_ARTIFACTS)))

    def test_phase2_pipeline_direct_dependencies_are_packaged(self):
        expected = {
            "scripts/datathon_utils.py",
            "scripts/train_portable_verifier.py",
            "scripts/source_answer_bank.py",
            "scripts/fuzzy_source_overrides.py",
            "scripts/weighted_source_overrides.py",
            "scripts/structured_overrides.py",
            "scripts/typed_evidence.py",
            "scripts/typed_idiom_matcher.py",
            "scripts/birth_date_overrides.py",
            "scripts/context_numeric_overrides.py",
            "scripts/context_span_overrides.py",
            "scripts/establishment_overrides.py",
            "scripts/law_evidence_overrides.py",
            "scripts/run_phase2_cascade.py",
        }
        self.assertTrue(expected.issubset(set(REPRO_SCRIPTS)))

    def test_optional_typed_idiom_corpus_is_packaged(self):
        self.assertIn(
            "outputs/source_cache/bangla_bagdhara_lrec/"
            "bagdhara_export_done_2026-03-01_010356.json",
            PHASE2_ARTIFACTS,
        )


if __name__ == "__main__":
    unittest.main()
