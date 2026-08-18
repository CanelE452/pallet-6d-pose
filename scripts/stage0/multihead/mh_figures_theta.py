"""Figures for the theta-only solver screen (PHASE 5).

Same rule as the other figure modules: every value is read from the JSON the
phases wrote, so a figure cannot disagree with the report beside it.
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

OUT = MD.OUT
FIG = OUT / "figures"
SEEDS = (1, 2)
POPS = ("D2_MH_DEV512", "D3_MH_CONF512")
ARMS = ("T0", "T1", "T2")
NAMES = {"T0": "point only", "T1": "full line (theta,rho)", "T2": "theta only"}
COLOUR = {"T0": "#7a7a7a", "T1": "#b03030", "T2": "#2a7f4f"}
plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3})


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("->", path)


def _load():
    return {s: json.loads((OUT / f"theta_only_solver_seed{s}.json").read_text())
            for s in SEEDS}


def figure_three_arms():
    """1: the three solvers side by side, both seeds, both populations."""
    data = _load()
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6))
    labels = [f"s{s}\n{p.split('_')[0]}" for s in SEEDS for p in POPS]
    x = np.arange(len(labels))
    metrics = [("R_median", "rotation median (deg)", True),
               ("t_median", "translation median (m)", True),
               ("success_5cm5deg", "5cm5deg", False)]
    for row, subset in enumerate(["ALL", "V<8 (off-grid)"]):
        for col, (key, title, lower_better) in enumerate(metrics):
            ax = axes[row][col]
            for k, arm in enumerate(ARMS):
                values = [data[s]["populations"][p][subset][arm][key]
                          for s in SEEDS for p in POPS]
                ax.bar(x + (k - 1) * 0.27, values, 0.27, label=NAMES[arm],
                       color=COLOUR[arm])
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_title(f"{subset}  —  {title}"
                         f"{' (lower better)' if lower_better else ''}",
                         fontsize=8)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
    fig.suptitle("PHASE 5  theta-only beats full-line on rotation everywhere, "
                 "and removes full-line's translation damage on seed 2",
                 fontsize=9)
    _save(fig, "theta_1_three_arms.png")


def figure_v8_split():
    """2: where the orientation constraint pays -- truncated frames."""
    boot = json.loads((OUT / "theta_only_solver_bootstrap.json").read_text())
    subsets = ["ALL", "V=8 (in-grid)", "V<8 (off-grid)", "low-angle",
               "near/large"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    x = np.arange(len(subsets))
    for ax, metric in zip(axes, ("R", "t")):
        for k, (s, p) in enumerate([(s, p) for s in SEEDS for p in POPS]):
            block = boot["seeds"][f"seed{s}"][p]
            values, lows, highs = [], [], []
            for subset in subsets:
                entry = block.get(f"{subset}|{metric}")
                values.append(entry["effect_pct"] if entry else np.nan)
                lows.append(entry["effect_pct"] - entry["ci95"][0]
                            if entry else 0.0)
                highs.append(entry["ci95"][1] - entry["effect_pct"]
                             if entry else 0.0)
            ax.errorbar(x + (k - 1.5) * 0.16, values, yerr=[lows, highs],
                        fmt="o", ms=4, capsize=2, lw=1,
                        label=f"s{s} {p.split('_')[0]}")
        ax.axhline(0, color="k", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(subsets, rotation=18, ha="right", fontsize=7)
        ax.set_title(f"{metric} improvement of theta-only over point-only (%)")
    axes[0].legend(fontsize=7)
    fig.suptitle("PHASE 5  paired frame bootstrap, 10,000 resamples, seeds "
                 "never pooled — rotation gain is largest on V<8", fontsize=9)
    _save(fig, "theta_2_subsets.png")


def figure_pareto():
    """3: the trade the solver is actually making."""
    data = _load()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), sharex=False)
    for ax, s in zip(axes, SEEDS):
        for p, marker in zip(POPS, ("o", "s")):
            for arm in ARMS:
                e = data[s]["populations"][p]["ALL"][arm]
                ax.scatter(e["R_median"], e["t_median"], s=70, marker=marker,
                           color=COLOUR[arm],
                           label=f"{NAMES[arm]} ({p.split('_')[0]})")
            pts = [data[s]["populations"][p]["ALL"][a] for a in ARMS]
            ax.plot([q["R_median"] for q in pts], [q["t_median"] for q in pts],
                    color="k", lw=0.5, alpha=0.4)
        ax.set_xlabel("rotation median (deg)  — lower better")
        ax.set_ylabel("translation median (m)  — lower better")
        ax.set_title(f"seed {s}   lambda_theta = "
                     f"{data[s]['lambda_theta']}")
        ax.legend(fontsize=6)
    fig.suptitle("PHASE 5  R vs t Pareto: theta-only buys a large rotation gain; "
                 "seed 1's grid-edge lambda pays for it in translation",
                 fontsize=9)
    _save(fig, "theta_3_pareto.png")


def figure_d0_selection():
    """4: the D0 sweep the lambda was chosen on, and where the rule landed."""
    d0 = json.loads((OUT / "theta_only_solver_d0.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, s in zip(axes, SEEDS):
        block = d0["seeds"][f"seed{s}"]
        grid = [float(k) for k in block["candidates"]]
        R = [block["candidates"][k]["R_median"] for k in block["candidates"]]
        t = [block["candidates"][k]["t_median"] for k in block["candidates"]]
        rejected = [bool(block["candidates"][k]["rejected_for"])
                    for k in block["candidates"]]
        ax.plot(grid, R, "o-", color="#3b6ea5", label="rotation (deg)")
        ax.set_xscale("log")
        ax.set_ylabel("rotation median", color="#3b6ea5")
        twin = ax.twinx()
        twin.plot(grid, t, "s--", color="#b03030", label="translation (m)")
        twin.axhline(block["P0_point_only"]["t_median"], color="#b03030",
                     lw=0.8, ls=":")
        twin.set_ylabel("translation median", color="#b03030")
        twin.grid(False)
        for g, ok in zip(grid, rejected):
            if ok:
                ax.axvspan(g * 0.82, g * 1.22, color="#b03030", alpha=0.10)
        ax.axvline(block["selected_lambda_theta"], color="k", lw=1.2, ls=":")
        ax.set_title(f"seed {s}   selected {block['selected_lambda_theta']}"
                     f"   (shaded = rejected by safety filter)", fontsize=8)
        ax.set_xlabel("lambda_theta")
    fig.suptitle("PHASE 4  the safety filter fixed the gate, but 'smallest R "
                 "among survivors' still walks seed 1 to the grid edge",
                 fontsize=9)
    _save(fig, "theta_4_d0_selection.png")


if __name__ == "__main__":
    figure_three_arms()
    figure_v8_split()
    figure_pareto()
    figure_d0_selection()
