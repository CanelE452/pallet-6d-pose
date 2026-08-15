"""Phase E-G analysis and the pre-fixed GO/STOP gate for the corner screen.

Reads the two mechanism evaluations (epoch 0 = C0, epoch 5 = C1-base /
C1-proposal / C1-refined), computes the signed-bias, tail, PnP and gate-behaviour
metrics, applies the twelve pre-specified conditions and writes the reports.
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]


import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "data/pallet/results/paper_s2/paper_s2_corner_replacement_screen"
FIG = OUT / "figures"
MECH = ROOT / "data/pallet/results/paper_s2_mechanism_diagnostic"
NEAR, FAR = (0, 1, 2, 3), (4, 5, 6, 7)
ARMS = ("base", "proposal", "refined")


def corner_frame(table: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Long form: one row per (frame, corner)."""
    rows = []
    for _, row in table.iterrows():
        for corner in range(8):
            rows.append({
                "frame_id": row["frame_id"], "session_id": row["session_id"],
                "domain": row["domain"], "corner": corner,
                "group": "far" if corner in FAR else "near",
                "err": row[f"{arm}_err_{corner}"],
                "dx": row.get(f"{arm}_dx_{corner}", np.nan),
                "dy": row.get(f"{arm}_dy_{corner}", np.nan),
                "peak": row[f"{arm}_peak_{corner}"],
                "gate": row.get(f"gate_{corner}", np.nan),
            })
    return pd.DataFrame(rows)


def signed_bias(frame: pd.DataFrame) -> float:
    clean = frame.dropna(subset=["dx", "dy"])
    if not len(clean):
        return float("nan")
    return float(np.hypot(clean.dx.mean(), clean.dy.mean()))


def summarise(table: pd.DataFrame, arm: str, f2: set[str]) -> dict[str, float]:
    corners = corner_frame(table, arm)
    subset = corners[corners.frame_id.isin(f2)]
    far = subset[subset.group == "far"]
    near_all = corners[corners.group == "near"]
    valid = corners.dropna(subset=["err"])
    return {
        "arm": arm,
        "f2_far_median_px": float(far.err.median()),
        "f2_far_signed_bias_px": signed_bias(far),
        "f2_near_median_px": float(subset[subset.group == "near"].err.median()),
        "near_median_px": float(near_all.err.median()),
        "tail_gt20": int((valid.err > 20).sum()),
        "tail_gt50": int((valid.err > 50).sum()),
        "weak_corner_n": int((corners.peak < 0.1).sum()),
        "confident_wrong_n": int(((corners.peak >= 0.5) & (corners.err > 20)).sum()),
        "pose_success": int(table[f"{arm}_pose_success"].sum()),
        "reproj_median_px": float(pd.to_numeric(
            table[f"{arm}_reproj"], errors="coerce").median()),
        "yaw_median_deg": float(pd.to_numeric(
            table[f"{arm}_yaw_err"], errors="coerce").median()),
        "nan_err": int(corners.err.isna().sum()),
    }


def paired(epoch0: pd.DataFrame, epoch5: pd.DataFrame, arm: str,
           f2: set[str]) -> pd.DataFrame:
    before = corner_frame(epoch0, "base")
    after = corner_frame(epoch5, arm)
    merged = before.merge(after, on=["frame_id", "corner"], suffixes=("_c0", "_c1"))
    merged = merged[merged.frame_id.isin(f2) & (merged.group_c0 == "far")]
    per_frame = merged.groupby("frame_id").agg(
        err_c0=("err_c0", "median"), err_c1=("err_c1", "median")).reset_index()
    per_frame["delta"] = per_frame.err_c1 - per_frame.err_c0
    return per_frame


def gate_decision(c0: dict, c1: dict, pairs: pd.DataFrame,
                  epoch0: pd.DataFrame, epoch5: pd.DataFrame,
                  gate_median: float, base_summary: dict) -> dict:
    def drop(before: float, after: float) -> float:
        return float(1.0 - after / before) if before else 0.0

    improved = int((pairs.delta < 0).sum())
    worsened = int((pairs.delta > 0).sum())
    lost = int(((epoch0["base_pose_success"]) & (~epoch5["refined_pose_success"])).sum())
    conditions = [
        ("1 F2 far median -15%", drop(c0["f2_far_median_px"], c1["f2_far_median_px"]) >= 0.15,
         drop(c0["f2_far_median_px"], c1["f2_far_median_px"])),
        ("2 F2 far signed bias -20%",
         drop(c0["f2_far_signed_bias_px"], c1["f2_far_signed_bias_px"]) >= 0.20,
         drop(c0["f2_far_signed_bias_px"], c1["f2_far_signed_bias_px"])),
        ("3 >50px tail -20%", drop(c0["tail_gt50"], c1["tail_gt50"]) >= 0.20,
         drop(c0["tail_gt50"], c1["tail_gt50"])),
        ("4 paired improved > worsened", improved > worsened, improved - worsened),
        ("5 PnP success >= 72/87", c1["pose_success"] >= 72, c1["pose_success"]),
        ("6 reproj -10%", drop(c0["reproj_median_px"], c1["reproj_median_px"]) >= 0.10,
         drop(c0["reproj_median_px"], c1["reproj_median_px"])),
        ("7 near median <= +5%",
         c1["near_median_px"] <= c0["near_median_px"] * 1.05,
         c1["near_median_px"] / c0["near_median_px"] - 1.0),
        ("8 no new PnP failure", lost == 0, lost),
        ("9 no new >50px", c1["tail_gt50"] <= c0["tail_gt50"],
         c1["tail_gt50"] - c0["tail_gt50"]),
        ("10 no new NaN", c1["nan_err"] <= c0["nan_err"], c1["nan_err"] - c0["nan_err"]),
        ("11 gate not collapsed", 0.02 <= gate_median <= 0.98, gate_median),
        ("12 C1-base no catastrophic regression",
         base_summary["pose_success"] >= c0["pose_success"] - 3,
         base_summary["pose_success"] - c0["pose_success"]),
    ]
    return {
        "conditions": [{"name": n, "passed": bool(p), "value": float(v)}
                       for n, p, v in conditions],
        "passed": all(p for _, p, _ in conditions),
        "paired_improved": improved, "paired_worsened": worsened,
    }


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    epoch0 = pd.read_parquet(OUT / "mechanism_epoch0.parquet")
    epoch5 = pd.read_parquet(OUT / "mechanism_epoch5.parquet")
    classes = pd.read_csv(MECH / "failure_class_frames.csv")
    f2 = set(classes.loc[classes.failure_class == "F2_CONFIDENT_WRONG", "frame_id"])

    c0 = summarise(epoch0, "base", f2)
    c0["arm"] = "C0"
    rows = [c0]
    for arm in ARMS:
        entry = summarise(epoch5, arm, f2)
        entry["arm"] = f"C1-{arm}"
        rows.append(entry)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "corner_replacement_pose_metrics.csv", index=False)

    gate_columns = [f"gate_{i}" for i in range(8)]
    gates = epoch5[gate_columns].to_numpy().reshape(-1)
    gate_median = float(np.median(gates))
    pd.DataFrame({"gate": gates}).describe().to_csv(
        OUT / "corner_replacement_gate_values.csv")

    pairs = paired(epoch0, epoch5, "refined", f2)
    pairs.to_csv(OUT / "corner_replacement_f2_paired.csv", index=False)

    corners = pd.concat(
        [corner_frame(epoch0, "base").assign(arm="C0")]
        + [corner_frame(epoch5, arm).assign(arm=f"C1-{arm}") for arm in ARMS])
    corners.to_csv(OUT / "corner_replacement_corner_metrics.csv", index=False)

    decision = gate_decision(
        c0, [r for r in rows if r["arm"] == "C1-refined"][0], pairs,
        epoch0, epoch5, gate_median,
        [r for r in rows if r["arm"] == "C1-base"][0])
    decision["gate_median"] = gate_median
    (OUT / "corner_replacement_gate.json").write_text(
        json.dumps(decision, indent=1), encoding="utf-8")

    # -- figures -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(table))
    ax.bar(x - 0.2, table.f2_far_median_px, 0.4, label="F2 far median err")
    ax.bar(x + 0.2, table.f2_far_signed_bias_px, 0.4, label="F2 far signed bias")
    ax.set_xticks(x); ax.set_xticklabels(table.arm); ax.legend(); ax.grid(alpha=.3, axis="y")
    ax.set_ylabel("px"); ax.set_title("far-face error and signed bias")
    fig.tight_layout(); fig.savefig(FIG / "corner_bias_vectors.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - 0.2, table.tail_gt20, 0.4, label=">20px corners")
    ax.bar(x + 0.2, table.tail_gt50, 0.4, label=">50px corners")
    ax.set_xticks(x); ax.set_xticklabels(table.arm); ax.legend(); ax.grid(alpha=.3, axis="y")
    ax.set_title("error tail"); fig.tight_layout()
    fig.savefig(FIG / "tail_histogram.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    order = pairs.sort_values("delta")
    colours = ["seagreen" if d < 0 else "crimson" for d in order.delta]
    ax.bar(range(len(order)), order.delta, color=colours)
    ax.axhline(0, c="k", lw=1)
    ax.set_ylabel("C1-refined - C0  (px, negative is better)")
    ax.set_title(f"F2 paired far-error delta  improved {int((pairs.delta<0).sum())} / "
                 f"worsened {int((pairs.delta>0).sum())}")
    ax.grid(alpha=.3, axis="y"); fig.tight_layout()
    fig.savefig(FIG / "far_depth_paired_delta.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(gates, bins=40)
    ax.axvline(0.02, ls="--", c="r", label="collapse threshold 0.02")
    ax.set_xlabel("per-corner gate g"); ax.legend(); ax.grid(alpha=.3)
    ax.set_title(f"gate distribution (median {gate_median:.4f})")
    fig.tight_layout(); fig.savefig(FIG / "gate_distribution.png", dpi=130); plt.close(fig)

    print(table.to_string(index=False))
    print(f"\ngate median {gate_median:.5f}")
    for condition in decision["conditions"]:
        print(f"  {'PASS' if condition['passed'] else 'FAIL'}  {condition['name']:<40} "
              f"{condition['value']:.4f}")
    print(f"\nGO/STOP: {'GO' if decision['passed'] else 'STOP'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
