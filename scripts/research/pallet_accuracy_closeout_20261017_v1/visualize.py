#!/usr/bin/env python3
"""Render the frozen six-arm PRIMARY192 comparison from its two public CSVs.

This is a presentation-only program. It does not load images, predictors, point
arrays, reference poses, or evaluators and does not calculate accuracy scores.
Existing summary metrics and per-image errors are rendered without filtering
success/failure or tail values. Both predeclared PoseFix+SubPix anchors remain.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import platform
import resource
import time

# Set these before importing any plotting library.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter

ARMS = (
    "R0", "N0", "E0", "POSEFIX_RAW",
    "POSEFIX_SUBPIX_PF_ANCHOR", "POSEFIX_SUBPIX_YOLO_ANCHOR",
)
LABELS = {
    "R0": "R0",
    "N0": "N0 (N3 + SubPix)",
    "E0": "E0 (statewise)",
    "POSEFIX_RAW": "PoseFix RAW",
    "POSEFIX_SUBPIX_PF_ANCHOR": "PoseFix + SubPix\nPF anchor (fixed)",
    "POSEFIX_SUBPIX_YOLO_ANCHOR": "PoseFix + SubPix\nYOLO anchor (fixed)",
}
COLORS = ("#777777", "#009E73", "#D55E00", "#CC79A7", "#0072B2", "#7B52AB")
DENOMINATOR = 192


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_value(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Expected existing finite nonnegative {field}: {row['arm']} {row['id']}")
    return value


def load_inputs(public_dir: Path):
    summary_path = public_dir / "FAIR_192_COMPARISON.csv"
    rows_path = public_dir / "FAIR_ROW_METRICS.csv"
    summary_rows = [r for r in read_csv(summary_path) if r["population"] == "PRIMARY192"]
    summary = {r["arm"]: r for r in summary_rows}
    if len(summary_rows) != len(ARMS) or set(summary) != set(ARMS):
        raise ValueError("Expected exactly the frozen six PRIMARY192 summary arms")
    primary = [r for r in read_csv(rows_path) if r["partition"] in {"TEST", "REMAINDER"}]
    by_arm = {arm: [r for r in primary if r["arm"] == arm] for arm in ARMS}
    id_sets = []
    for arm in ARMS:
        group = by_arm[arm]
        ids = {r["id"] for r in group}
        if len(group) != DENOMINATOR or len(ids) != DENOMINATOR:
            raise ValueError(f"Expected exactly 192 unique PRIMARY images: {arm}")
        if int(summary[arm]["denominator"]) != DENOMINATOR:
            raise ValueError(f"Incorrect existing denominator: {arm}")
        if int(summary[arm]["no_pose"]) != 0 or any(r["pose_available"] != "True" for r in group):
            raise ValueError("These figures require the frozen all-available six-arm result; no missing rows may be dropped")
        for field in ("translation_cm", "rotation_deg"):
            values = [finite_value(r, field) for r in group]
            if int(summary[arm][f"{field}_n"]) != DENOMINATOR:
                raise ValueError(f"Incorrect existing continuous-error denominator: {arm} {field}")
            if max(values) != float(summary[arm][f"{field}_max"]):
                raise ValueError(f"Existing maximum does not match the supplied rows: {arm} {field}")
        id_sets.append(ids)
    if any(ids != id_sets[0] for ids in id_sets[1:]):
        raise ValueError("Methods do not have identical PRIMARY192 image IDs")
    return summary, by_arm, (summary_path, rows_path)


def success_figure(summary: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.8, 5.8), layout="constrained")
    counts = [int(summary[arm]["successes"]) for arm in ARMS]
    bars = ax.bar(range(len(ARMS)), counts, color=COLORS, width=0.70)
    ax.set_ylim(0, DENOMINATOR)
    ax.set_yticks([0, 24, 48, 72, 96, 120, 144, 168, 192])
    ax.set_ylabel("Successful images (of the same 192)")
    ax.set_xticks(range(len(ARMS)), [LABELS[arm] for arm in ARMS], fontsize=9)
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.7)
    ax.set_axisbelow(True)
    for bar, arm, count in zip(bars, ARMS, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 3,
                f"{count} / 192\n{100 * float(summary[arm]['success_rate']):.2f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("PRIMARY192: joint 5 cm / 5 deg success", loc="left", fontsize=15, pad=30)
    ax.text(0, 1.025, "Translation <= 5 cm AND physical rotation <= 5 deg on the same final pose",
            transform=ax.transAxes, fontsize=10, color="#444444")
    fig.supxlabel("C2 I/Y (180 deg) only; no 90 deg width/depth equivalence. Both fixed anchors are shown.\n"
                  "All 192 images were previously exposed during development; this is not an independent test.",
                  fontsize=9)
    fig.savefig(output, dpi=160, facecolor="white", metadata={"Software": "Frozen CSV visualization"})
    plt.close(fig)


def error_figure(by_arm: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.0))
    fields = ("translation_cm", "rotation_deg")
    xlabels = ("Per-image center translation error (cm)", "Per-image C2 physical rotation error (deg)")
    titles = ("Center translation: full range", "Physical rotation: full range")
    maximums = {field: max(finite_value(r, field) for arm in ARMS for r in by_arm[arm]) for field in fields}
    for ax, field, xlabel, title in zip(axes, fields, xlabels, titles):
        for arm, color in zip(ARMS, COLORS):
            values = sorted(finite_value(r, field) for r in by_arm[arm])
            x_values = [0.0] + values
            fractions = [0.0] + [i / DENOMINATOR for i in range(1, DENOMINATOR + 1)]
            ax.step(x_values, fractions, where="post", label=LABELS[arm].replace("\n", " "),
                    color=color, linewidth=1.65, alpha=0.95)
            ax.plot(values[-1], 1.0, marker="o", ms=3, color=color)
        ax.axvline(5, color="#222222", linestyle="--", linewidth=1.0)
        ax.text(5 + maximums[field] * 0.014, 0.065, "5 cm" if field == "translation_cm" else "5 deg",
                fontsize=9, color="#333333")
        ax.set_xlim(0, maximums[field] * 1.045)
        ax.set_ylim(0, 1.025)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=12, loc="left", pad=10)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.grid(color="#E0E0E0", linewidth=0.65)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Fraction of all 192 images at or below x")
    axes[1].tick_params(labelleft=False)
    fig.suptitle("PRIMARY192: per-image error distributions, all six methods", x=0.07, ha="left", fontsize=15, y=0.98)
    fig.text(0.07, 0.915,
             "192 / 192 poses available per method. Every failure and every tail value is retained; linear axes start at zero.",
             fontsize=9.5, color="#444444")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=3, frameon=False, fontsize=9)
    fig.text(0.5, 0.016,
             "Marginal error ECDFs do not show joint success. Reference poses are existing geometric reconstructions, not independent physical GT.",
             ha="center", fontsize=8.5, color="#444444")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.28, wspace=0.12)
    fig.savefig(output, dpi=160, facecolor="white", metadata={"Software": "Frozen CSV visualization"})
    plt.close(fig)


def write_index(summary: dict, output: Path) -> None:
    def v(arm: str, field: str, digits: int = 2) -> str:
        return f"{float(summary[arm][field]):.{digits}f}"
    table_rows = []
    for arm in ARMS:
        row = summary[arm]
        table_rows.append(
            "<tr><th scope='row'>" + html.escape(LABELS[arm]).replace("\n", "<br>") + "</th>"
            f"<td>{row['successes']}/192</td>"
            f"<td>{v(arm, 'translation_cm_mean')} &plusmn; {v(arm, 'translation_cm_sample_std')}</td>"
            f"<td>{v(arm, 'translation_cm_P95')} / {v(arm, 'translation_cm_max')}</td>"
            f"<td>{v(arm, 'rotation_deg_mean')} &plusmn; {v(arm, 'rotation_deg_sample_std')}</td>"
            f"<td>{v(arm, 'rotation_deg_P95')} / {v(arm, 'rotation_deg_max')}</td>"
            f"<td>{row['T_gt50cm']} / {row['T_gt100cm']} / {row['R_gt45deg']}</td></tr>"
        )
    document = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PRIMARY192 frozen six-method comparison</title>
<style>
body{font:16px/1.55 system-ui,sans-serif;color:#20262e;background:#f7f8fa;margin:0}main{max-width:1120px;margin:32px auto;padding:0 22px 40px}h1{font-size:29px;line-height:1.2}h2{font-size:21px;margin-top:30px}p{max-width:1000px}figure{margin:24px 0;background:white;border:1px solid #dce1e6;padding:12px;border-radius:8px}img{display:block;width:100%;height:auto}figcaption{font-size:14px;color:#495460;padding:8px}a{color:#075caa}.notice{background:#fff5da;border-left:4px solid #b16b00;padding:10px 16px}.table-wrap{overflow-x:auto;background:white}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;border-bottom:1px solid #dde2e7;text-align:right;vertical-align:middle}th:first-child{text-align:left}thead th{background:#edf1f5}small{font-size:13px;color:#495460}
</style><main>
<h1>PRIMARY192: frozen six-method comparison</h1>
<p class="notice"><strong>Exploratory comparison on the same 192 images.</strong> All images were previously exposed during development. This is not an independent test; existing geometric reconstruction poses are the reference, not independent physical ground truth.</p>
<p>E0 has 114 successes; the two predeclared fixed PoseFix + SubPix anchors have 112 and 111. E0 has higher translation/rotation means than both anchors. The report's paired cluster intervals include zero. These descriptive results establish neither statistical superiority nor equivalence, and no single anchor was selected as a representative winner.</p>
<p><strong>Success:</strong> center translation &le;5 cm <em>and</em> physical rotation &le;5&deg; on one final pose. Rotation allows only C2 I/Y (180&deg;); a 90&deg; width/depth swap is not equivalent. PRIMARY192 = TEST49 + REMAINDER143; CAL46 is excluded. All six methods have 192/192 available poses and zero missing poses.</p>
<figure><a href="success_primary192.png"><img src="success_primary192.png" alt="Six-method joint success counts on the same 192 images, vertical axis zero to 192: R0 66, N0 108, E0 114, PoseFix RAW 80, fixed PF anchor 112, fixed YOLO anchor 111."></a><figcaption>All-image success count. Both predeclared anchor settings are shown. N0 is N3 + SubPix; E0 is the existing statewise method. Click for the standalone PNG.</figcaption></figure>
<figure><a href="error_ecdf_primary192.png"><img src="error_ecdf_primary192.png" alt="Full-range linear empirical cumulative distributions of center translation and physical rotation errors. All 192 per-image errors per method are retained, including failures and tails."></a><figcaption>Each ECDF includes all 192 existing per-image errors per method, including all failures. Full-range linear axes preserve maximum errors. The marginal curves do not encode joint success. Click for the standalone PNG.</figcaption></figure>
<h2>Count and error tradeoffs on the identical 192 images</h2>
<div class="table-wrap"><table><thead><tr><th scope="col">Method</th><th scope="col">Success</th><th scope="col">T mean &plusmn; sample SD (cm)</th><th scope="col">T P95 / max (cm)</th><th scope="col">R mean &plusmn; sample SD (&deg;)</th><th scope="col">R P95 / max (&deg;)</th><th scope="col">T&gt;50 / T&gt;100 / R&gt;45 counts</th></tr></thead><tbody>
""" + "\n".join(table_rows) + """
</tbody></table></div>
<p><small>Numbers are copied from the supplied PRIMARY192 summary CSV; SD uses ddof=1. No outlier filtering, method selection, new scoring, or inference is performed by the renderer. PF anchor limits SubPix motion relative to the PF output; YOLO anchor limits total PF + SubPix motion relative to original YOLO. Both use the predeclared 1% diagonal cap.</small></p>
<p><a href="../FAIR_COMPARISON_REPORT_KO.md">Comparison report (Korean)</a> &middot; <a href="../FAIR_192_COMPARISON.csv">Summary CSV</a> &middot; <a href="../FAIR_ROW_METRICS.csv">All row metrics CSV</a> &middot; <a href="VISUALIZATION_RECEIPT.json">Renderer provenance and resource receipt</a> &middot; <a href="SHA256SUMS.txt">Artifact SHA-256</a></p>
<p><small>Static local research artifact. Source: <code>scripts/research/pallet_accuracy_closeout_20261017_v1/visualize.py</code>. CSV-only presentation: new neural network forwards, F/PnP calls, training calls and operational changes are all zero.</small></p>
</main></html>
"""
    output.write_text(document, encoding="utf-8")


def main() -> None:
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-dir", type=Path, default=Path(__file__).resolve().parents[3] / "_docs/experiments/pallet_accuracy_closeout_20261017_v1")
    args = parser.parse_args()
    public_dir = args.public_dir.resolve()
    out_dir = public_dir / "gallery"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, by_arm, input_paths = load_inputs(public_dir)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight"})
    outputs = [out_dir / name for name in ("success_primary192.png", "error_ecdf_primary192.png", "index.html")]
    success_figure(summary, outputs[0])
    error_figure(by_arm, outputs[1])
    write_index(summary, outputs[2])
    source = Path(__file__).resolve()
    receipt = {
        "schema": "frozen_csv_visualization_v1",
        "population": "PRIMARY192 = TEST49 + REMAINDER143; CAL46 excluded",
        "same_image_ids_per_arm": 192,
        "arms": list(ARMS),
        "predeclared_anchors": "Both fixed PF and YOLO anchors are shown; no representative anchor selected",
        "success_rule": "Existing CSV success: center translation <=5 cm AND physical C2 rotation <=5 deg on the same final pose; no 90 deg equivalence",
        "scope": "CSV-only rendering; existing summary values and all existing per-image errors; no accuracy score recomputation",
        "source": {"path": str(source), "sha256": sha256(source)},
        "inputs": [{"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size} for p in input_paths],
        "outputs": [{"file": p.name, "sha256": sha256(p), "bytes": p.stat().st_size} for p in outputs],
        "coverage": {arm: {"images": len(by_arm[arm]), "successes_from_summary": int(summary[arm]["successes"]),
                           "pose_available": 192, "failures_excluded": 0, "tail_values_excluded": 0} for arm in ARMS},
        "new_calls": {"NN_forward": 0, "F_or_PnP": 0, "training": 0, "pose_or_accuracy_scoring": 0},
        "operational_changes": 0,
        "resources": {"wall_seconds_through_artifact_write": time.perf_counter() - wall_start,
                      "cpu_seconds_through_artifact_write": time.process_time() - cpu_start,
                      "process_peak_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                      "measurement_scope": "renderer main through PNG and HTML writes; peak RSS is full process high-water"},
        "environment": {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
                        "BLAS_and_OpenMP_threads": {k: os.environ[k] for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}},
        "limitations": ["All images were previously exposed during development; not an independent test",
                        "Reference poses are existing geometric reconstructions, not independent physical ground truth",
                        "Marginal ECDFs do not encode joint success", "No statistical superiority or equivalence claim"],
    }
    receipt_path = out_dir / "VISUALIZATION_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    checksummed = [*outputs, receipt_path]
    (out_dir / "SHA256SUMS.txt").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in checksummed), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "files": [p.name for p in checksummed] + ["SHA256SUMS.txt"],
                      "resources": receipt["resources"], "source_sha256": receipt["source"]["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
