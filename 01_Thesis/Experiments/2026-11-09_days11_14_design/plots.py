"""Offline publication plots from actual CSV observations; no inference or API imports.

Run from the workspace root (paths below name user-supplied, not bundled data):
  python 01_Thesis/Experiments/2026-11-09_days11_14_design/plots.py loss --input logs.csv --output-dir results/loss_run
  python 01_Thesis/Experiments/2026-11-09_days11_14_design/plots.py confusion --input predictions.csv --expected-rows 16000 --output-dir results/confusion_run
  python 01_Thesis/Experiments/2026-11-09_days11_14_design/plots.py pareto --input timings.csv --latency-policy "End-to-end per example, including preparation amortized over the same evaluation set" --output-dir results/pareto_run

Exact CSV headers (UTF-8, optional BOM; header order is unrestricted):
* loss: iteration,train_loss,val_loss[,seed]. Finite nonnegative observations;
  iteration strictly increases within each seed in input order. Seeds are opaque
  nonblank identifiers. Plot each seed separately, without averaging, smoothing,
  interpolation of missing observations, or invented uncertainty. LBFGS has no
  historical per-epoch logs here: use only actual newly instrumented diagnostics,
  identifying their iteration convention in the title/protocol, not fake epochs.
* confusion: id,source_group,y_true,prediction. Unique nonblank IDs and nonblank
  source groups; exact labels '0' (faithful) and '1' (hallucinated). Only an empty
  prediction cell means FAILURE (not a dropped row). Both truth classes required.
  Preprocessing MUST supply every expected evaluation ID, including unattempted
  rows as failures, and externally verify ID/lineage membership. --expected-rows
  checks cardinality, NOT identity or source disjointness. Row-normalized 2x3
  matrix includes failures. Operational BHS = 50*(FP+failureA)/nA +
  50*(FN+failureB)/nB. Valid-only BHS excludes failures from each denominator and
  is null/explicitly undefined if either class has no valid verdicts.
* pareto: configuration,latency_ms,balanced_accuracy_pct,coverage,protocol_id.
  Unique configurations, positive finite latency, accuracy in [0,100], coverage
  exactly 1, one nonblank shared protocol_id. Accuracy is 100-BHS ONLY for complete
  valid binary verdicts with both classes and equal track weights. Upstream must
  verify this and identical dataset/hardware/timing/preparation/amortization
  policy; --latency-policy records that policy, not proof of comparability.
  True Pareto dominance minimizes latency and maximizes balanced accuracy;
  identical nondominated points are all retained. No confidence bounds inferred.

All commands require a NEW --output-dir, even if an existing directory is empty.
Outputs: figure.pdf, figure.png (300 dpi), source.csv (byte-exact input snapshot),
metrics.csv, provenance.json (written last as the completion marker). Interrupted
runs may leave partial artifacts: use a new directory, never overwrite them.
Optional --title, --width/--height (inches), --style matplotlib|seaborn.
Matplotlib is required for rendering; seaborn only for explicit --style seaborn.
Standard-library validation/metrics and unittest tests need no plotting packages.
No packages are installed, datasets generated, or empirical evidence bundled.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import sys
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path

LABELS = ("Faithful (0)", "Hallucinated (1)", "FAILURE")


def number(value, name, minimum=0, maximum=math.inf):
    """Reject missing/nonfinite/out-of-range values rather than dropping rows."""
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name}: expected a finite number, got {value!r}") from exc
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name}: out of range or nonfinite: {value!r}")
    return result


def read_csv(data, kind):
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""), strict=True)
    required = {
        "loss": {"iteration", "train_loss", "val_loss"},
        "confusion": {"id", "source_group", "y_true", "prediction"},
        "pareto": {"configuration", "latency_ms", "balanced_accuracy_pct", "coverage", "protocol_id"},
    }[kind]
    fields = reader.fieldnames or []
    optional = {"seed"} if kind == "loss" else set()
    if len(fields) != len(set(fields)) or not required <= set(fields) or set(fields) - required - optional:
        raise ValueError(f"{kind}: expected columns {sorted(required)}; optional {sorted(optional)}")
    rows = []
    for row in reader:
        if None in row or None in row.values():
            raise ValueError(f"CSV line {reader.line_num}: wrong number of fields")
        rows.append(row)
    # DictReader ignores blank physical records: refuse rather than silently drop.
    records = list(csv.reader(io.StringIO(data.decode("utf-8-sig"), newline=""), strict=True))
    if any(not record for record in records):
        raise ValueError("CSV contains an empty record")
    if not rows:
        raise ValueError("CSV has no observations")
    return rows


def loss_metrics(rows):
    groups = {}
    for row in rows:
        seed = row.get("seed", "unseeded")
        if not seed.strip():
            raise ValueError("seed must be nonblank when supplied")
        point = tuple(number(row[key], key) for key in ("iteration", "train_loss", "val_loss"))
        points = groups.setdefault(seed, [])
        if points and point[0] <= points[-1][0]:
            raise ValueError(f"seed {seed!r}: iterations must strictly increase")
        points.append(point)
    metrics = [{"seed": seed, "observations": len(points), "final_iteration": points[-1][0],
                "final_train_loss": points[-1][1], "final_val_loss": points[-1][2]}
               for seed, points in groups.items()]
    return groups, metrics, {"seeds": list(groups), "aggregation": "none; separate observed trajectories"}


def confusion_metrics(rows, expected_rows):
    if expected_rows <= 0 or len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, received {len(rows)}; include all failures")
    counts, seen = [[0, 0, 0], [0, 0, 0]], set()
    for row in rows:
        if not row["id"].strip() or row["id"] in seen:
            raise ValueError("IDs must be nonblank and unique")
        seen.add(row["id"])
        if not row["source_group"].strip():
            raise ValueError("source_group must be nonblank")
        if row["y_true"] not in ("0", "1") or row["prediction"] not in ("0", "1", ""):
            raise ValueError("labels must be exact 0/1; only an empty prediction denotes FAILURE")
        counts[int(row["y_true"])][2 if row["prediction"] == "" else int(row["prediction"])] += 1
    support = [sum(row) for row in counts]
    if not all(support):
        raise ValueError("both truth classes are required; absent-class BHS is undefined")
    valid = [sum(row[:2]) for row in counts]
    summary = {
        "operational_bhs_pct": 50 * (counts[0][1] + counts[0][2]) / support[0]
                               + 50 * (counts[1][0] + counts[1][2]) / support[1],
        "valid_only_bhs_pct": (50 * counts[0][1] / valid[0] + 50 * counts[1][0] / valid[1]) if all(valid) else None,
        "coverage": sum(valid) / sum(support), "expected_rows": expected_rows,
        "support_A": support[0], "support_B": support[1],
        "failure_A": counts[0][2], "failure_B": counts[1][2],
        "completeness_check": "row count only; expected ID membership must be checked upstream",
    }
    metrics = [{"y_true": str(i), "prediction": str(j) if j < 2 else "FAILURE",
                "count": counts[i][j], "row_percentage": 100 * counts[i][j] / support[i],
                "operational_bhs_pct": summary["operational_bhs_pct"],
                "valid_only_bhs_pct": summary["valid_only_bhs_pct"], "coverage": summary["coverage"]}
               for i in range(2) for j in range(3)]
    return counts, metrics, summary


def pareto_metrics(rows):
    points, seen, protocols = [], set(), set()
    for row in rows:
        name, protocol = row["configuration"], row["protocol_id"]
        if not name.strip() or name in seen or not protocol.strip():
            raise ValueError("configurations must be nonblank/unique; protocol_id must be nonblank")
        seen.add(name)
        protocols.add(protocol)
        latency = number(row["latency_ms"], "latency_ms")
        if latency <= 0 or number(row["coverage"], "coverage", 0, 1) != 1:
            raise ValueError("Pareto requires positive latency and coverage exactly 1 (no failed verdicts)")
        points.append({"configuration": name, "latency_ms": latency,
                       "balanced_accuracy_pct": number(row["balanced_accuracy_pct"], "balanced_accuracy_pct", 0, 100),
                       "coverage": 1, "protocol_id": protocol})
    if len(protocols) != 1:
        raise ValueError("Pareto comparisons require one shared protocol_id")
    for point in points:
        x, y = point["latency_ms"], point["balanced_accuracy_pct"]
        point["nondominated"] = not any(
            other["latency_ms"] <= x and other["balanced_accuracy_pct"] >= y
            and (other["latency_ms"] < x or other["balanced_accuracy_pct"] > y)
            for other in points)
    return points, points, {"protocol_id": next(iter(protocols)), "coverage": 1,
                            "dominance": "minimize latency_ms; maximize balanced_accuracy_pct"}


def render(kind, data, summary, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ValueError("Rendering requires matplotlib in the selected environment; nothing was installed") from exc
    versions = {"matplotlib": matplotlib.__version__, "seaborn": None}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}):
        if args.style == "seaborn":
            try:
                sns = import_module("seaborn")
            except ImportError as exc:
                raise ValueError("--style seaborn requires seaborn; use --style matplotlib in this environment") from exc
            sns.set_theme(context="paper", style="whitegrid", font="DejaVu Sans", font_scale=1)
            versions["seaborn"] = getattr(sns, "__version__", "unknown")
        fig, ax = plt.subplots(figsize=(args.width, args.height), layout="constrained")
        try:
            if kind == "loss":
                for index, (seed, points) in enumerate(data.items()):
                    x, train, val = zip(*points)
                    color = plt.get_cmap("tab10")(index % 10)
                    ax.plot(x, train, color=color, marker="o", markersize=3, label=f"Train / {seed}")
                    ax.plot(x, val, color=color, linestyle="--", marker="s", markersize=3, label=f"Validation / {seed}")
                ax.set(xlabel="Observed iteration", ylabel="Loss")
                ax.legend(frameon=False)
            elif kind == "confusion":
                normalized = [[count / sum(row) for count in row] for row in data]
                ax.grid(False)
                image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1, aspect="auto")
                fig.colorbar(image, ax=ax, label="Fraction of truth class")
                for i in range(2):
                    for j in range(3):
                        ax.text(j, i, f"{data[i][j]}\n{normalized[i][j]:.1%}", ha="center", va="center",
                                color="white" if normalized[i][j] > 0.55 else "black")
                valid = summary["valid_only_bhs_pct"]
                valid_text = "undefined" if valid is None else f"{valid:.2f}%"
                ax.set(xticks=range(3), xticklabels=LABELS, yticks=range(2), yticklabels=LABELS[:2],
                       xlabel=f"Prediction\nOperational BHS: {summary['operational_bhs_pct']:.2f}% | "
                              f"Valid-only BHS: {valid_text}\nCoverage: {summary['coverage']:.1%}", ylabel="Truth")
            else:
                ax.scatter([p["latency_ms"] for p in data], [p["balanced_accuracy_pct"] for p in data],
                           color="#0072B2", s=35, label="Complete valid verdicts")
                frontier = sorted({(p["latency_ms"], p["balanced_accuracy_pct"]) for p in data if p["nondominated"]})
                ax.plot(*zip(*frontier), color="#D55E00", marker="o", linestyle="--", label="Nondominated frontier")
                for point in data:
                    ax.annotate(point["configuration"], (point["latency_ms"], point["balanced_accuracy_pct"]),
                                xytext=(4, 5), textcoords="offset points", fontsize=8)
                ax.set(xlabel="Latency (ms/example; lower is better)",
                       ylabel="Balanced accuracy (%; higher is better)", ylim=(0, 105), xlim=(0, None))
                ax.legend(frameon=False)
            ax.set_title(args.title)
            if kind != "confusion":
                ax.grid(axis="y", alpha=0.25)
            return fig, plt, versions
        except BaseException:
            plt.close(fig)
            raise


def digest(data):
    return hashlib.sha256(data).hexdigest()


def parser():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = cli.add_subparsers(dest="kind", required=True)
    titles = {"loss": "Observed training and validation loss", "confusion": "Row-normalized verdict confusion",
              "pareto": "Latency–accuracy trade-off (complete coverage)"}
    for kind, title in titles.items():
        command = sub.add_parser(kind, help=title)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True, help="New directory only; never overwritten")
        command.add_argument("--title", default=title)
        command.add_argument("--width", type=float, default=7.2)
        command.add_argument("--height", type=float, default=4.8)
        command.add_argument("--style", choices=("matplotlib", "seaborn"), default="matplotlib")
        if kind == "confusion":
            command.add_argument("--expected-rows", type=int, required=True)
        if kind == "pareto":
            command.add_argument("--latency-policy", required=True, help="Shared preparation/amortization/timing policy")
    return cli


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cli = parser()
    args = cli.parse_args(argv)
    fig = plt = None
    try:
        if args.output_dir.exists() or args.output_dir.is_symlink():
            raise ValueError(f"Refusing overwrite: {args.output_dir} already exists")
        if number(args.width, "width") <= 0 or number(args.height, "height") <= 0:
            raise ValueError("figure dimensions must be positive")
        if args.kind == "pareto" and not args.latency_policy.strip():
            raise ValueError("latency policy must be nonblank")
        raw = args.input.read_bytes()
        rows = read_csv(raw, args.kind)
        if args.kind == "loss":
            data, metrics, summary = loss_metrics(rows)
        elif args.kind == "confusion":
            data, metrics, summary = confusion_metrics(rows, args.expected_rows)
        else:
            data, metrics, summary = pareto_metrics(rows)
            summary["latency_policy"] = args.latency_policy
        fig, plt, versions = render(args.kind, data, summary, args)
        # Exclusive directory reservation protects even empty artifacts and races.
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "source.csv").write_bytes(raw)
        with (args.output_dir / "metrics.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows([{k: "undefined" if v is None else v for k, v in row.items()} for row in metrics])
        fig.savefig(args.output_dir / "figure.pdf", format="pdf", bbox_inches="tight")
        fig.savefig(args.output_dir / "figure.png", format="png", dpi=300, bbox_inches="tight")
        provenance = {
            "created_utc": datetime.now(UTC).isoformat(), "kind": args.kind,
            "command_argv": [sys.executable, str(Path(__file__).resolve()), *argv],
            "source": {"path": str(args.input.resolve()), "sha256": digest(raw), "rows": len(rows)},
            "script_sha256": digest(Path(__file__).read_bytes()), "python": platform.python_version(),
            "platform": platform.platform(), "versions": versions,
            "plot_settings": {"title": args.title, "width_inches": args.width, "height_inches": args.height,
                              "style": args.style, "png_dpi": 300, "pdf_fonttype": 42,
                              "label_order": list(LABELS), "smoothing": False, "uncertainty": "not inferred"},
            "summary": summary,
            "artifact_sha256": {name: digest((args.output_dir / name).read_bytes())
                                for name in ("source.csv", "metrics.csv", "figure.pdf", "figure.png")},
        }
        with (args.output_dir / "provenance.json").open("x", encoding="utf-8") as handle:
            json.dump(provenance, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        return 0
    except (ValueError, OSError, csv.Error, UnicodeError) as exc:
        cli.error(str(exc))
    finally:
        if fig is not None and plt is not None:
            plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
