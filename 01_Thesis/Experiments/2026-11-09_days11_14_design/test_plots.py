"""Synthetic software tests only; all fixtures/artifacts live in temporary folders.

Run from the workspace root, without pytest or package installation:
  python -B 01_Thesis/Experiments/2026-11-09_days11_14_design/test_plots.py -v

Rendering checks skip explicitly if matplotlib is unavailable. No test observations
or rendered figures are empirical thesis evidence or persisted in the workspace.
"""

import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).with_name("plots.py")
SPEC = importlib.util.spec_from_file_location("thesis_publication_plots", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
plots = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plots)
HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None

LOSS = "iteration,train_loss,val_loss,seed\n0,2,3,7\n1,1,2,7\n0,3,4,8\n1,2,3,8\n"
CONFUSION = ("id,source_group,y_true,prediction\n"
             "a,g1,0,0\nb,g1,0,1\nc,g2,0,\nd,g3,1,1\ne,g4,1,0\nf,g4,1,\n")
PARETO = ("configuration,latency_ms,balanced_accuracy_pct,coverage,protocol_id\n"
          "fast,10,70,1,synthetic-test\nslow,20,80,1,synthetic-test\n"
          "dominated,30,75,1,synthetic-test\ntied,20,80,1,synthetic-test\n"
          "same-latency-worse,20,72,1,synthetic-test\nsame-accuracy-slower,25,80,1,synthetic-test\n")


def read(text, kind):
    return plots.read_csv(text.encode("utf-8"), kind)


class ContractTests(unittest.TestCase):
    def test_loss_preserves_each_seed_and_observation(self):
        groups, metrics, summary = plots.loss_metrics(read(LOSS, "loss"))
        self.assertEqual(list(groups), ["7", "8"])
        self.assertEqual(groups["7"], [(0, 2, 3), (1, 1, 2)])
        self.assertEqual(metrics[1]["final_val_loss"], 3)
        self.assertIn("none", summary["aggregation"])
        unseeded, _, _ = plots.loss_metrics(read("iteration,train_loss,val_loss\n4,1,2\n", "loss"))
        self.assertEqual(list(unseeded), ["unseeded"])

    def test_loss_rejects_invalid_or_reordered_observations(self):
        for text in (LOSS.replace("1,1,2,7", "0,1,2,7"), LOSS.replace("1,1,2,7", "-1,1,2,7"),
                     LOSS.replace("0,2,3,7", "0,nan,3,7"), LOSS.replace("0,2,3,7", "0,2,inf,7"),
                     LOSS.replace("0,2,3,7", "0,-2,3,7"), LOSS.replace(",7", ", ")):
            with self.subTest(text=text), self.assertRaises(ValueError):
                plots.loss_metrics(read(text, "loss"))

    def test_csv_rejects_missing_duplicate_unknown_or_ragged_fields(self):
        invalid = ("iteration,train_loss\n0,1\n", "iteration,train_loss,train_loss\n0,1,2\n",
                   "iteration,train_loss,val_loss,typo\n0,1,2,x\n", "iteration,train_loss,val_loss\n0,1\n",
                   "iteration,train_loss,val_loss\n0,1,2,3\n", "iteration,train_loss,val_loss\n",
                   "iteration,train_loss,val_loss\n\n0,1,2\n")
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                read(text, "loss")

    def test_bom_and_quoted_multiline_source_group_are_preserved(self):
        rows = read('\ufeffid,source_group,y_true,prediction\na,"বাংলা\nsource",0,0\nb,g,1,1\n', "confusion")
        self.assertEqual(rows[0]["source_group"], "বাংলা\nsource")
        self.assertEqual(plots.confusion_metrics(rows, 2)[2]["operational_bhs_pct"], 0)

    def test_confusion_orientation_failure_column_and_normalization(self):
        counts, cells, summary = plots.confusion_metrics(read(CONFUSION, "confusion"), 6)
        self.assertEqual(counts, [[1, 1, 1], [1, 1, 1]])
        self.assertAlmostEqual(summary["operational_bhs_pct"], 200 / 3)
        self.assertEqual(summary["valid_only_bhs_pct"], 50)
        self.assertAlmostEqual(summary["coverage"], 2 / 3)
        for truth in ("0", "1"):
            self.assertAlmostEqual(sum(cell["row_percentage"] for cell in cells if cell["y_true"] == truth), 100)
        self.assertEqual(cells[2]["prediction"], "FAILURE")

    def test_operational_bhs_uses_equal_class_weights_not_row_error(self):
        rows = read("id,source_group,y_true,prediction\na,g,0,1\nb,h,1,1\nc,i,1,1\nd,j,1,1\n", "confusion")
        self.assertEqual(plots.confusion_metrics(rows, 4)[2]["operational_bhs_pct"], 50)

    def test_all_failures_are_retained_and_valid_only_is_undefined(self):
        rows = read("id,source_group,y_true,prediction\na,g,0,\nb,h,1,\n", "confusion")
        _, cells, summary = plots.confusion_metrics(rows, 2)
        self.assertEqual(summary["operational_bhs_pct"], 100)
        self.assertIsNone(summary["valid_only_bhs_pct"])
        self.assertEqual(summary["coverage"], 0)
        self.assertEqual(sum(cell["count"] for cell in cells), 2)

    def test_one_class_without_valid_verdicts_is_undefined(self):
        rows = read("id,source_group,y_true,prediction\na,g,0,\nb,h,1,1\n", "confusion")
        self.assertIsNone(plots.confusion_metrics(rows, 2)[2]["valid_only_bhs_pct"])

    def test_confusion_rejects_bad_labels_ids_and_source_groups(self):
        for field, values in {"y_true": ("", "2", "0.0", " 0", "True"),
                              "prediction": ("FAILURE", "null", "NaN", " ", "1.0", "-1"),
                              "id": ("", "b"), "source_group": ("", " ")}.items():
            for value in values:
                rows = read(CONFUSION, "confusion")
                rows[0][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    plots.confusion_metrics(rows, 6)

    def test_missing_expected_rows_or_truth_class_rejected(self):
        rows = read(CONFUSION, "confusion")
        for count in (0, 5, 7):
            with self.subTest(count=count), self.assertRaises(ValueError):
                plots.confusion_metrics(rows, count)
        with self.assertRaises(ValueError):
            plots.confusion_metrics(rows[:3], 3)

    def test_true_pareto_dominance_including_ties(self):
        points, _, summary = plots.pareto_metrics(read(PARETO, "pareto"))
        self.assertEqual({p["configuration"] for p in points if p["nondominated"]}, {"fast", "slow", "tied"})
        self.assertEqual(summary["protocol_id"], "synthetic-test")

    def test_pareto_rejects_incomparability_missingness_and_nonfinite_values(self):
        cases = {"coverage": ("0.99", "0", "nan", "1.01"), "latency_ms": ("0", "-1", "inf", ""),
                 "balanced_accuracy_pct": ("-1", "101", "nan", ""),
                 "configuration": ("", "slow"), "protocol_id": ("", "other-protocol")}
        for field, values in cases.items():
            for value in values:
                rows = read(PARETO, "pareto")
                rows[0][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    plots.pareto_metrics(rows)


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-plot-software-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def arguments(self, kind, text):
        source = self.root / f"{kind}.csv"
        source.write_bytes(text.encode("utf-8"))
        args = [kind, "--input", str(source), "--output-dir", str(self.root / kind)]
        if kind == "confusion":
            args += ["--expected-rows", "6"]
        if kind == "pareto":
            args += ["--latency-policy", "Synthetic software test only; identical preparation/amortization"]
        return args

    def assert_cli_error(self, args):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            plots.main(args)
        self.assertEqual(error.exception.code, 2)

    def test_existing_empty_directory_refused_without_plot_import(self):
        args = self.arguments("loss", LOSS)
        (self.root / "loss").mkdir()
        with mock.patch.object(plots, "render", side_effect=AssertionError("must not render")):
            self.assert_cli_error(args)
        self.assertEqual(list((self.root / "loss").iterdir()), [])

    def test_invalid_input_and_dimensions_leave_no_output(self):
        args = self.arguments("loss", LOSS.replace("0,2,3,7", "0,nan,3,7"))
        self.assert_cli_error(args)
        self.assertFalse((self.root / "loss").exists())
        args = self.arguments("loss", LOSS)
        for dimension in ("0", "nan", "inf", "-1"):
            self.assert_cli_error(args + ["--width", dimension])
        self.assertFalse((self.root / "loss").exists())

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib unavailable; rendering not tested")
    def test_all_commands_write_pdf_300dpi_png_metrics_and_provenance(self):
        for kind, text in (("loss", LOSS), ("confusion", CONFUSION), ("pareto", PARETO)):
            with self.subTest(kind=kind):
                args = self.arguments(kind, text)
                self.assertEqual(plots.main(args), 0)
                output = self.root / kind
                self.assertEqual({p.name for p in output.iterdir()},
                                 {"figure.pdf", "figure.png", "source.csv", "metrics.csv", "provenance.json"})
                self.assertEqual((output / "source.csv").read_bytes(), text.encode("utf-8"))
                self.assertTrue((output / "figure.pdf").read_bytes().startswith(b"%PDF"))
                png = (output / "figure.png").read_bytes()
                self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
                density = png.index(b"pHYs") + 4
                xppm, yppm, unit = struct.unpack(">IIB", png[density:density + 9])
                self.assertEqual(unit, 1)
                self.assertAlmostEqual(xppm * 0.0254, 300, places=1)
                self.assertEqual(xppm, yppm)
                provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
                self.assertEqual(provenance["command_argv"][2:], args)
                self.assertEqual(provenance["source"]["sha256"], hashlib.sha256(text.encode()).hexdigest())
                self.assertEqual(provenance["script_sha256"], hashlib.sha256(SCRIPT.read_bytes()).hexdigest())
                self.assertEqual(provenance["plot_settings"]["png_dpi"], 300)
                for name, digest in provenance["artifact_sha256"].items():
                    self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
                with (output / "metrics.csv").open(encoding="utf-8", newline="") as handle:
                    self.assertTrue(list(csv.DictReader(handle)))
                before = {p.name: p.read_bytes() for p in output.iterdir()}
                self.assert_cli_error(args)
                self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})
        plt = import_module("matplotlib.pyplot")
        self.assertEqual(plt.get_fignums(), [])

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib unavailable")
    def test_missing_optional_seaborn_is_actionable_no_silent_fallback(self):
        args = self.arguments("loss", LOSS) + ["--style", "seaborn"]
        with mock.patch.dict(sys.modules, {"seaborn": None}):
            self.assert_cli_error(args)
        self.assertFalse((self.root / "loss").exists())

    def test_real_cli_help_works_without_inference_imports(self):
        result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--help"],
                                capture_output=True, text=True, timeout=20, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("{loss,confusion,pareto}", result.stdout)


if __name__ == "__main__":
    unittest.main()
