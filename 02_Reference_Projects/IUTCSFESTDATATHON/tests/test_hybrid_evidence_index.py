import unittest

import pandas as pd

from hybrid_evidence_index import (
    relation_type,
    retrieval_queries,
    source_relation,
    source_relations,
)


class HybridEvidenceTests(unittest.TestCase):
    def test_relation_router(self):
        self.assertEqual(relation_type("তিনি কবে জন্মগ্রহণ করেন?"), "birth")
        self.assertEqual(relation_type("আইনে সর্বোচ্চ কারাদণ্ড কত?"), "law_penalty")
        self.assertEqual(relation_type("প্রতিষ্ঠানটি কোথায় অবস্থিত?"), "location")

    def test_specific_event_relations_precede_broad_relations(self):
        self.assertEqual(relation_type("নির্মাণ কাজ কত সালে শুরু হয়?"), "construction_start")
        self.assertEqual(relation_type("নির্মাণ কাজ ১৯৮০ সালে সম্পন্ন হয়।"), "construction_completion")
        self.assertEqual(relation_type("বইটি কত সালে রচনা করেন?"), "writing")
        self.assertEqual(relation_type("কবে বিশ্ব ঐতিহ্যের স্বীকৃতি পায়?"), "recognition")
        self.assertEqual(relation_type("মোট কত বছর কারাদণ্ড হতে পারে?"), "law_penalty")

    def test_source_question_beats_incidental_biography_context(self):
        self.assertEqual(
            source_relation(
                "তিনি কবে মৃত্যুবরণ করেন?",
                "তিনি ১৯২০ সালে জন্মগ্রহণ করেন এবং ১৯৭৫ সালে মৃত্যুবরণ করেন।",
            ),
            "death",
        )

    def test_document_retains_all_explicit_relations(self):
        self.assertEqual(
            source_relations(
                "একজন লেখকের জীবনী",
                "তিনি ১৯২০ সালে জন্মগ্রহণ করেন এবং ১৯৭৫ সালে মৃত্যুবরণ করেন।",
                source_repo="rejauldu/bengali-wikipedia",
            ),
            ("birth", "death"),
        )

    def test_known_qa_source_uses_requested_relation(self):
        self.assertEqual(
            source_relations(
                "তিনি কবে মৃত্যুবরণ করেন?",
                "তিনি ১৯২০ সালে জন্মগ্রহণ করেন এবং ১৯৭৫ সালে মৃত্যুবরণ করেন।",
                source_repo="bipinsaha/bangla-law-qna",
            ),
            ("death",),
        )

    def test_prompt_only_retrieval_does_not_leak_candidate_answer(self):
        frame = pd.DataFrame(
            [{"prompt_bn": "নদীর দৈর্ঘ্য কত?", "response_bn": "ভুল ৯৯৯ কিলোমিটার"}]
        )
        prompt_query = retrieval_queries(frame, "prompt")[0]
        combined_query = retrieval_queries(frame, "prompt_response")[0]
        self.assertNotIn("999", prompt_query)
        self.assertIn("999", combined_query)


if __name__ == "__main__":
    unittest.main()
