"""Read the screen JSONs and produce the two tables the brief asks for.

Section 23's comparison table and section 17's complementarity table are built
here rather than during training, so the strata can be recut without retraining
and so the gate arithmetic is in one readable place.

The complementarity numbers use ground truth to decide which branch was closer
on each corner.  That is a diagnostic and it is labelled as one everywhere it
appears; nothing here selects a branch at inference.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mh_data as MD                                            # noqa: E402
import mh_arms as MH                                            # noqa: E402

OUT = MD.OUT
TIE_BAND_CELLS = 0.10       # below this the two branches are called equal
POPULATION = "D2_MH_DEV512"


def load(arm: str, seed: int = 1) -> dict:
    suffix = "" if seed == 1 else f"_seed{seed}"
    path = OUT / f"mh_screen_{arm}{suffix}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def strata() -> dict:
    return {row["stem"]: row for row in MD.load_split()}


def _subsets(rows, meta):
    """The four axes section 17 names, plus the whole population."""
    def keep(predicate):
        return [r for r in rows if r["stem"] in meta and predicate(meta[r["stem"]])]
    return {"ALL": rows,
            "V=8": keep(lambda m: m["v"] == 8),
            "V<8": keep(lambda m: m["v"] < 8),
            "low-angle": keep(lambda m: m["elev"] < 15.0),
            "near/large": keep(lambda m: m["size"] >= 0.40)}


def _corner_pairs(rows, only=None):
    """(direct, CIGM) per corner, optionally restricted by in-grid status.

    `only=True` keeps corners whose ground truth is inside the belief grid,
    `only=False` keeps the ones outside it.  Pooling the two hides the single
    asymmetry worth measuring: a heatmap has no cell to peak in when the corner
    is off the grid, so its error there is bounded by the grid, while CIGM is
    free to place the intersection anywhere.
    """
    direct, cigm = [], []
    for row in rows:
        flags = row.get("in_grid", [True] * len(row["cigm_cell"]))
        for index, (d, c) in enumerate(zip(row.get("direct_cell", []),
                                           row["cigm_cell"])):
            if only is None or flags[index] is only:
                direct.append(d)
                cigm.append(c)
    return np.asarray(direct), np.asarray(cigm)


def _median(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else None


def _fmt(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def _pose(entry, key, field):
    block = entry.get(key)
    return None if not block else block.get(field)


# --------------------------------------------------------------------------


def comparison(step: str, arms: dict, meta: dict) -> str:
    lines = []
    header = f"{'':<26}" + "".join(f"{a.split('_')[0]:>12}" for a in arms)
    lines.append(header)
    lines.append("-" * len(header))

    def row(label, getter, digits=3):
        cells = []
        for entry in arms.values():
            try:
                cells.append(_fmt(getter(entry), digits))
            except (KeyError, TypeError):
                cells.append("N/A")
        lines.append(f"{label:<26}" + "".join(f"{c:>12}" for c in cells))

    row("line angle med (deg)", lambda e: e["line"]["angle_median"])
    row("line angle p90", lambda e: e["line"]["angle_p90"])
    row("line offset med (cell)", lambda e: e["line"]["offset_median"])
    row("line offset p90", lambda e: e["line"]["offset_p90"])
    row("CIGM corner med (cell)", lambda e: e["corner"]["cigm_cell_median"])
    row("direct corner med (cell)", lambda e: e["corner"]["direct_cell_median"])
    row("centroid med (cell)", lambda e: e["corner"]["centroid_cell_median"])
    row("PnP solve PATH-L", lambda e: e["pose_L"]["solve_rate"])
    row("PnP solve PATH-C", lambda e: e["pose_C"]["solve_rate"])
    row("R err PATH-L (deg)", lambda e: e["pose_L"]["R_deg_median"], 2)
    row("R err PATH-C (deg)", lambda e: e["pose_C"]["R_deg_median"], 2)
    row("t err PATH-L (m)", lambda e: e["pose_L"]["t_m_median"])
    row("t err PATH-C (m)", lambda e: e["pose_C"]["t_m_median"])
    row("5cm5deg PATH-L", lambda e: e["pose_L"]["success_5cm5deg"])
    row("5cm5deg PATH-C", lambda e: e["pose_C"]["success_5cm5deg"])
    row("mask IoU med", lambda e: e["mask"]["iou_median"])

    lines.append("")
    lines.append("line angle median by subset")
    for name in ("V=8", "V<8", "low-angle", "near/large"):
        cells = []
        for entry in arms.values():
            subset = _subsets(entry["rows"], meta)[name]
            angles = [a for r in subset
                      for a, s in zip(r["angle"], r["support"]) if s]
            cells.append(_fmt(_median(angles)))
        lines.append(f"  {name:<24}" + "".join(f"{c:>12}" for c in cells))

    lines.append("")
    lines.append("CIGM corner median by subset")
    for name in ("V=8", "V<8", "low-angle", "near/large"):
        cells = []
        for entry in arms.values():
            subset = _subsets(entry["rows"], meta)[name]
            cells.append(_fmt(_median([v for r in subset for v in r["cigm_cell"]])))
        lines.append(f"  {name:<24}" + "".join(f"{c:>12}" for c in cells))

    lines.append("")
    lines.append("direct corner median by subset")
    for name in ("V=8", "V<8", "low-angle", "near/large"):
        cells = []
        for entry in arms.values():
            subset = _subsets(entry["rows"], meta)[name]
            values = [v for r in subset for v in r.get("direct_cell", [])]
            cells.append(_fmt(_median(values)) if values else "N/A")
        lines.append(f"  {name:<24}" + "".join(f"{c:>12}" for c in cells))
    return "\n".join(lines)


def complementarity(entry: dict, meta: dict) -> str:
    """Which branch is closer, where, and how much a perfect chooser would buy.

    `oracle-min` takes the better of the two per corner using ground truth.  It
    is an upper bound on any fusion rule and is reported only to size the
    headroom -- a small gap over the better branch means fusion has little to
    win no matter how it is built.
    """
    rows = entry["rows"]
    if not rows or "direct_cell" not in rows[0]:
        return "no corner branch in this arm"
    lines = []
    header = (f"{'subset':<14}{'n':>6}{'direct':>9}{'CIGM':>9}{'tie':>7}"
              f"{'direct med':>12}{'CIGM med':>10}{'oracle':>9}{'gain%':>8}")
    lines.append(header)
    lines.append("-" * len(header))
    for name, subset in _subsets(rows, meta).items():
        direct = np.array([v for r in subset for v in r["direct_cell"]])
        cigm = np.array([v for r in subset for v in r["cigm_cell"]])
        if direct.size == 0:
            continue
        gap = direct - cigm
        tie = np.abs(gap) < TIE_BAND_CELLS
        oracle = np.minimum(direct, cigm)
        better = min(np.median(direct), np.median(cigm))
        gain = 100.0 * (better - np.median(oracle)) / max(better, 1e-9)
        lines.append(f"{name:<14}{direct.size:>6}"
                     f"{100 * float(((gap < 0) & ~tie).mean()):>8.1f}%"
                     f"{100 * float(((gap > 0) & ~tie).mean()):>8.1f}%"
                     f"{100 * float(tie.mean()):>6.1f}%"
                     f"{np.median(direct):>12.3f}{np.median(cigm):>10.3f}"
                     f"{np.median(oracle):>9.3f}{gain:>7.1f}%")

    lines.append("")
    lines.append("split by whether the GT corner is inside the belief grid")
    lines.append(f"{'':<14}{'n':>6}{'direct':>9}{'CIGM':>9}{'tie':>7}"
                 f"{'direct med':>12}{'CIGM med':>10}{'oracle':>9}")
    for name, flag in (("in grid", True), ("off grid", False)):
        direct, cigm = _corner_pairs(rows, only=flag)
        if direct.size == 0:
            lines.append(f"{name:<14}{0:>6}   none")
            continue
        gap = direct - cigm
        tie = np.abs(gap) < TIE_BAND_CELLS
        lines.append(f"{name:<14}{direct.size:>6}"
                     f"{100 * float(((gap < 0) & ~tie).mean()):>8.1f}%"
                     f"{100 * float(((gap > 0) & ~tie).mean()):>8.1f}%"
                     f"{100 * float(tie.mean()):>6.1f}%"
                     f"{np.median(direct):>12.3f}{np.median(cigm):>10.3f}"
                     f"{np.median(np.minimum(direct, cigm)):>9.3f}")

    lines.append("")
    lines.append("per-corner win map (whole population)")
    lines.append(f"{'corner':<8}{'direct med':>12}{'CIGM med':>11}"
                 f"{'direct wins':>13}{'oracle':>9}")
    for corner in range(8):
        direct = np.array([r["direct_cell"][corner] for r in rows])
        cigm = np.array([r["cigm_cell"][corner] for r in rows])
        tie = np.abs(direct - cigm) < TIE_BAND_CELLS
        lines.append(f"{corner:<8}{np.median(direct):>12.3f}{np.median(cigm):>11.3f}"
                     f"{100 * float(((direct < cigm) & ~tie).mean()):>12.1f}%"
                     f"{np.median(np.minimum(direct, cigm)):>9.3f}")
    return "\n".join(lines)


def verdict(step: str, arms: dict, meta: dict) -> dict:
    """The pre-registered gates, applied as written in SCREEN_LOCK.md."""
    def line_primary(entry):
        return (entry["line"]["angle_median"], entry["line"]["offset_median"])

    def subset_line(entry, name):
        subset = _subsets(entry["rows"], meta)[name]
        return _median([a for r in subset
                        for a, s in zip(r["angle"], r["support"]) if s])

    out = {"step": step}
    a0, a1 = arms.get("A0_LINE_ONLY"), arms.get("A1_CORNER_LINE")
    a2 = arms.get("A2_CORNER_LINE_MASK")
    if a0 and a1:
        angle0, offset0 = line_primary(a0)
        angle1, offset1 = line_primary(a1)
        trunc0, trunc1 = subset_line(a0, "V<8"), subset_line(a1, "V<8")
        full0, full1 = subset_line(a0, "V=8"), subset_line(a1, "V=8")
        out["A1_vs_A0"] = {
            "angle_rel": _rel(angle0, angle1),
            "offset_rel": _rel(offset0, offset1),
            "truncation_rel": _rel(trunc0, trunc1),
            "nontruncation_rel": _rel(full0, full1),
            "pose_success_pp_L": _pp(a0["pose_L"]["success_5cm5deg"],
                                     a1["pose_L"]["success_5cm5deg"]),
            "gate_1_line_5pct": max(_rel(angle0, angle1), _rel(offset0, offset1)) >= 5.0,
            "gate_2_pose_5pp": _pp(a0["pose_L"]["success_5cm5deg"],
                                   a1["pose_L"]["success_5cm5deg"]) >= 5.0,
            "gate_3_truncation_10pct": _rel(trunc0, trunc1) >= 10.0,
            "guard_nontruncation_regression_le_5pct": _rel(full0, full1) >= -5.0,
        }
        passed = (any(out["A1_vs_A0"][k] for k in
                      ("gate_1_line_5pct", "gate_2_pose_5pp",
                       "gate_3_truncation_10pct"))
                  and out["A1_vs_A0"]["guard_nontruncation_regression_le_5pct"])
        out["A1_vs_A0"]["PASS"] = bool(passed)
    if a1 and a2:
        angle1, offset1 = line_primary(a1)
        angle2, offset2 = line_primary(a2)
        out["A2_vs_A1"] = {
            "angle_rel": _rel(angle1, angle2),
            "offset_rel": _rel(offset1, offset2),
            "corner_rel": _rel(a1["corner"]["direct_cell_median"],
                               a2["corner"]["direct_cell_median"]),
            "pose_success_pp_C": _pp(a1["pose_C"]["success_5cm5deg"],
                                     a2["pose_C"]["success_5cm5deg"]),
            "mask_iou": a2.get("mask", {}).get("iou_median"),
        }
        block = out["A2_vs_A1"]
        block["gate_primary_5pct"] = max(block["angle_rel"], block["offset_rel"],
                                         block["corner_rel"]) >= 5.0
        block["gate_pose_3pp"] = block["pose_success_pp_C"] >= 3.0
        block["PASS"] = bool(block["gate_primary_5pct"] or block["gate_pose_3pp"])
    return out


def _rel(baseline, candidate):
    """Percent improvement of candidate over baseline, positive is better."""
    if baseline in (None, 0) or candidate is None:
        return 0.0
    return round(100.0 * (baseline - candidate) / abs(baseline), 2)


def _pp(baseline, candidate):
    if baseline is None or candidate is None:
        return 0.0
    return round(100.0 * (candidate - baseline), 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", default=None, help="default: the last mark")
    parser.add_argument("--population", default=POPULATION)
    parser.add_argument("--seed", type=int, default=1)
    arguments = parser.parse_args()

    histories = {arm: load(arm, arguments.seed) for arm in MH.ARMS}
    histories = {a: h for a, h in histories.items() if h}
    if not histories:
        raise SystemExit("no screen results yet")
    step = arguments.step or sorted(next(iter(histories.values())),
                                    key=int)[-1]
    arms = {arm: history[step][arguments.population]
            for arm, history in histories.items() if step in history}
    meta = strata()

    report = [f"population {arguments.population}   step {step}   "
              f"n={arms[next(iter(arms))]['n_frames']}", ""]
    report.append(comparison(step, arms, meta))
    for arm, entry in arms.items():
        if "direct_cell" in (entry["rows"][0] if entry["rows"] else {}):
            report += ["", f"complementarity -- {arm} (diagnostic, uses GT)", ""]
            report.append(complementarity(entry, meta))
    decision = verdict(step, arms, meta)
    report += ["", "gates (pre-registered in SCREEN_LOCK.md)", "",
               json.dumps(decision, indent=1)]
    text = "\n".join(report)
    print(text)
    tag = f"step_{step}" + ("" if arguments.seed == 1 else f"_seed{arguments.seed}")
    (OUT / f"mh_report_{tag}.txt").write_text(text)
    (OUT / f"mh_verdict_{tag}.json").write_text(json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
