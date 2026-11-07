import unittest

import pandas as pd

from audit_test_derived_pseudolabels import recover_seed_labels


class TestDerivedPseudolabelAuditTests(unittest.TestCase):
    def test_recovers_only_exact_weight_variants(self) -> None:
        rows = []
        for seed, label in [(0, 1), (1, 0)]:
            rows.extend(
                {"seed_sample_index": seed, "sample_weight": 1.0, "label": label}
                for _ in range(3)
            )
            rows.append({"seed_sample_index": seed, "sample_weight": 0.5, "label": 1 - label})
        result = recover_seed_labels(pd.DataFrame(rows), variants_per_seed=3)
        self.assertEqual(result.to_dict("records"), [{"id": 1, "pseudo_label": 1}, {"id": 2, "pseudo_label": 0}])

    def test_rejects_conflicting_seed_labels(self) -> None:
        frame = pd.DataFrame(
            [
                {"seed_sample_index": 0, "sample_weight": 1.0, "label": 1},
                {"seed_sample_index": 0, "sample_weight": 1.0, "label": 0},
            ]
        )
        with self.assertRaisesRegex(ValueError, "conflicting pseudo-labels"):
            recover_seed_labels(frame, variants_per_seed=2)


if __name__ == "__main__":
    unittest.main()
