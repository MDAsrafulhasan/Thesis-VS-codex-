import unittest

import pandas as pd

from released_set_diagnostic import build_released_lookup, diagnostic_predictions


class ReleasedSetDiagnosticTests(unittest.TestCase):
    def test_exact_released_rows_fit_but_unseen_rows_are_refused(self) -> None:
        released = pd.DataFrame([
            {"context": "[NULL]", "prompt_bn": "ক?", "response_bn": "খ", "label": 1},
            {"context": "গ", "prompt_bn": "ঘ?", "response_bn": "ঙ", "label": 0},
        ])
        lookup = build_released_lookup(released)
        self.assertEqual(diagnostic_predictions(released, lookup), [1, 0])
        unseen = pd.DataFrame([{"context": "", "prompt_bn": "নতুন?", "response_bn": "উত্তর"}])
        with self.assertRaises(ValueError):
            diagnostic_predictions(unseen, lookup)


if __name__ == "__main__":
    unittest.main()
