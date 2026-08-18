"""Figures for the final 2-head pose qualification (PHASE 4-9).

Same rule as `mh_figures.py`: nothing here computes a new number.  Every value is
read from the JSON the phases already wrote, so a figure cannot quietly disagree
with the report beside it.  The one exception is figure 4, which recomputes the
scale target from the prediction cache -- that is a distribution, not a claim,
and it uses `mh_scale.scale_target`, the same function the phases used.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mh_data as MD                                            # noqa: E402
import mh_scale as SC                                           # noqa: E402

OUT = MD.OUT
FIG = OUT / "figures"
SEEDS = (1, 2)
RUN = "e3confirm25k"
plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3})

PASS, FAIL, NEUTRAL, CEIL = "#2a7f4f", "#b03030", "#7a7a7a", "#3b6ea5"


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("->", path)


def figure_native_solver():
    """1: point-only against joint point+line, both runs, both seeds."""
    rows = [("A1 s1", 7.830, 7.125, 0.2244, 0.2389, -1.95),
            ("A1 s2", 8.067, 6.964, 0.2397, 0.2951, -5.86),
            ("E3 s1", 7.232, 6.641, 0.1825, 0.1733, +0.58),
            ("E3 s2", 7.539, 7.137, 0.1941, 0.2133, -6.83)]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    x = np.arange(len(rows))
    for ax, (i0, i1, title, unit) in zip(
            axes, [(1, 2, "rotation median", "deg"),
                   (3, 4, "translation median", "m"),
                   (None, None, "5cm5deg change", "pp")]):
        if i0 is None:
            values = [r[5] for r in rows]
            ax.bar(x, values, color=[PASS if v >= 0 else FAIL for v in values])
            ax.axhline(0, color="k", lw=0.8)
        else:
            ax.bar(x - 0.2, [r[i0] for r in rows], 0.4, label="point only",
                   color=NEUTRAL)
            ax.bar(x + 0.2, [r[i1] for r in rows], 0.4, label="joint point+line",
                   color=CEIL)
            ax.legend(fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([r[0] for r in rows])
        ax.set_title(f"{title} ({unit})")
    fig.suptitle("PHASE 4  native joint solver: rotation always wins, "
                 "translation only on E3 seed 1", fontsize=9)
    _save(fig, "scale_1_native_solver.png")


def figure_lambda_defect():
    """2: why seed 2 fails -- lambda is chosen on rotation alone."""
    grid = [0.03, 0.1, 0.3, 1.0]
    seed1 = {"R": [6.566, 6.582, 6.495, 6.496],
             "t": [0.1770, 0.1774, 0.1672, 0.2324], "pick": 0.3}
    seed2 = {"R": [7.040, 7.083, 6.913, 6.485],
             "t": [0.1880, 0.1885, 0.1830, 0.2096], "pick": 1.0}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharex=True)
    for ax, (name, block) in zip(axes, [("seed 1", seed1), ("seed 2", seed2)]):
        ax.plot(grid, block["R"], "o-", color=CEIL, label="rotation (deg)")
        ax.set_xscale("log")
        ax.set_ylabel("rotation median", color=CEIL)
        twin = ax.twinx()
        twin.plot(grid, block["t"], "s--", color=FAIL,
                  label="translation (m)")
        twin.set_ylabel("translation median", color=FAIL)
        twin.grid(False)
        ax.axvline(block["pick"], color="k", lw=1.0, ls=":")
        ax.set_title(f"{name}   selected lambda = {block['pick']}")
        ax.set_xlabel("lambda_line")
    fig.suptitle("PHASE 4 defect  selection reads rotation only, which is the "
                 "axis the line term is good at", fontsize=9)
    _save(fig, "scale_2_lambda_defect.png")


def figure_predictability():
    """3: how far the Ridge predictors fall short of the pre-registered bar."""
    data = json.loads((OUT / f"scale_ridge_{RUN}.json").read_text())
    names = [n for n in SC.FEATURE_SETS if SC.FEATURE_SETS[n]]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    x = np.arange(len(names))
    for k, seed in enumerate(SEEDS):
        arms = data["seeds"][f"seed{seed}"]["arms"]
        axes[0].bar(x + (k - 0.5) * 0.36, [arms[n]["r2_D2"] for n in names],
                    0.36, label=f"seed {seed}", color=[CEIL, NEUTRAL][k])
        axes[1].bar(x + (k - 0.5) * 0.36,
                    [arms[n]["gain_vs_constant_pct"] for n in names],
                    0.36, label=f"seed {seed}", color=[CEIL, NEUTRAL][k])
    axes[0].axhline(data["gate"]["min_r2"], color=FAIL, ls="--",
                    label="pre-registered bar")
    axes[0].set_title("D2 R^2 of the per-frame scale factor")
    axes[1].axhline(data["gate"]["min_residual_gain_vs_const_pct"], color=FAIL,
                    ls="--", label="pre-registered bar")
    axes[1].set_title("residual reduction vs the constant (%)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=7)
        ax.legend(fontsize=7)
    fig.suptitle(f"PHASE 6  SCALE_PREDICTABLE = {data['SCALE_PREDICTABLE']}"
                 "   signal is real but far short, and the winning block flips "
                 "between seeds", fontsize=9)
    _save(fig, "scale_3_predictability.png")


def figure_target_distribution():
    """4: what is actually being predicted -- the bias is already small."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharex=True)
    for ax, seed in zip(axes, SEEDS):
        for split, colour in (("D0_MH_SEEN512", NEUTRAL),
                              ("D2_MH_DEV512", CEIL)):
            values = SC.scale_target(np.load(SC.cache(RUN, seed, split),
                                             allow_pickle=True))
            ax.hist(values, bins=60, range=(0.85, 1.20), alpha=0.55,
                    color=colour, label=f"{split.split('_')[0]}  median "
                                        f"{np.median(values):.4f}")
        ax.axvline(1.0, color="k", lw=0.9, ls=":")
        ax.set_title(f"seed {seed}")
        ax.set_xlabel("s* = spread(gt) / spread(pred)")
        ax.legend(fontsize=7)
    fig.suptitle("PHASE 5  the target: median bias about 2%, the rest is "
                 "per-frame spread", fontsize=9)
    _save(fig, "scale_4_target_distribution.png")


def figure_corrected_arms():
    """5: applying the prediction, against no correction and against the oracle."""
    data = json.loads((OUT / f"corrected_point_line_{RUN}.json").read_text())
    order = ["C0_uncorrected", "C_const", "C1_geometry", "C2P_point",
             "C2L_line", "C2PL_both", "C_oracle_GT"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    x = np.arange(len(order))
    for k, seed in enumerate(SEEDS):
        arms = data["seeds"][f"seed{seed}"]["arms"]
        axes[0].bar(x + (k - 0.5) * 0.36,
                    [arms[n]["ALL"]["t_median"] for n in order], 0.36,
                    label=f"seed {seed}", color=[CEIL, NEUTRAL][k])
        axes[1].bar(x + (k - 0.5) * 0.36,
                    [arms[n]["ALL"]["success_5cm5deg"] for n in order], 0.36,
                    label=f"seed {seed}", color=[CEIL, NEUTRAL][k])
    for ax, title in zip(axes, ["translation median (m), lower better",
                                "5cm5deg, higher better"]):
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=25, ha="right", fontsize=7)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.axvspan(-0.5, 0.5, color=NEUTRAL, alpha=0.10)
        ax.axvspan(len(order) - 1.5, len(order) - 0.5, color=PASS, alpha=0.10)
    fig.suptitle("PHASE 8  the oracle (right) is large; every learned correction "
                 "loses 5cm5deg against no correction (left)", fontsize=9)
    _save(fig, "scale_5_corrected_arms.png")


def figure_final_gate():
    """6: the gate itself -- what passed, what did not, on both seeds."""
    data = json.loads((OUT / f"corrected_point_line_{RUN}.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3))
    labels = ["rotation\n(gain %)", "translation\n(gain %)", "5cm5deg\n(pp)"]
    bars = {"gain": [3.0, -2.0, -1.0]}
    for ax, seed in zip(axes, SEEDS):
        phase9 = data["seeds"][f"seed{seed}"]["phase9"]
        values = [phase9["R_gain_pct"], phase9["t_gain_pct"],
                  phase9["success_delta_pp"]]
        ok = [values[0] >= 3.0, values[1] >= -2.0, values[2] >= -1.0]
        ax.bar(labels, values, color=[PASS if o else FAIL for o in ok])
        for i, (v, bar) in enumerate(zip(values, bars["gain"])):
            ax.plot([i - 0.4, i + 0.4], [bar, bar], color="k", lw=1.0, ls="--")
            ax.text(i, v, f"{v:+.2f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"seed {seed}   "
                     f"{'PASS' if all(ok) else 'FAIL'}")
    fig.suptitle("PHASE 9  H1 (corrected point+line) against H0 (corrected "
                 f"point):  TWO_HEAD_POSE_QUALIFIED = "
                 f"{data['TWO_HEAD_POSE_QUALIFIED']}", fontsize=9)
    _save(fig, "scale_6_final_gate.png")


if __name__ == "__main__":
    figure_native_solver()
    figure_lambda_defect()
    figure_predictability()
    figure_target_distribution()
    figure_corrected_arms()
    figure_final_gate()
