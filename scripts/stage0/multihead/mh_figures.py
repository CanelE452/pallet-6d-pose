"""Figures for the multihead failure diagnosis (brief PHASE 13).

Each figure answers one question a table answers badly.  Nothing here computes a
new number -- every value is read from the JSON the diagnostics already wrote, so
a figure cannot disagree with the report beside it.
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
MARKS = (6000, 12000, 18000, 25000)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3})


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("->", path)


def figure_gradient_cosine():
    """1 + 2: is the corner gradient fighting the line gradient, and where?"""
    data = json.loads((OUT / "gradient_conflict.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))

    for seed in SEEDS:
        blocks = [data[f"seed{seed}_step{m}"] for m in MARKS]
        axes[0].errorbar(MARKS, [b["cos_median"] for b in blocks],
                         yerr=[[b["cos_median"] - b["cos_p10"] for b in blocks],
                               [b["cos_p90"] - b["cos_median"] for b in blocks]],
                         marker="o", capsize=3, label=f"seed {seed}")
        axes[1].plot(MARKS, [b["norm_ratio_corner_over_line"] for b in blocks],
                     marker="o", label=f"seed {seed}")
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].axhspan(-1, -0.10, color="tab:red", alpha=0.10)
    axes[0].text(6500, -0.55, "conflict gate\ncos < -0.10", color="tab:red",
                 fontsize=8)
    axes[0].set_ylim(-0.7, 0.5)
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("cosine(g_line, g_corner)")
    axes[0].set_title("shared late-A1: gradients agree, weakly")
    axes[0].legend()

    axes[1].axhline(21.3, color="tab:gray", ls="--", lw=1)
    axes[1].text(6500, 3.0, "ratio at step 0 = 21.3", color="tab:gray", fontsize=8)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("||g_corner|| / ||g_line||")
    axes[1].set_title("corner gradient collapses by ~35,000x")
    axes[1].legend()

    layers = data["seed1_step25000"]["layers"]
    layers = [l for l in layers if l["line_grad_share"] > 0.01]
    order = np.argsort([-l["line_grad_share"] for l in layers])
    names = [layers[i]["tensor"] for i in order]
    axes[2].barh(range(len(order)),
                 [layers[i]["cos_median"] for i in order],
                 color=["tab:red" if layers[i]["cos_median"] < 0 else "tab:blue"
                        for i in order])
    axes[2].set_yticks(range(len(order)))
    axes[2].set_yticklabels([f"{n}  ({layers[i]['line_grad_share']:.0%})"
                             for n, i in zip(names, order)], fontsize=7)
    axes[2].axvline(-0.10, color="tab:red", ls="--", lw=1)
    axes[2].set_xlabel("cosine at 25,000 (seed 1)")
    axes[2].set_title("per tensor (share of line gradient)")
    _save(fig, "fig1_gradient_cosine.png")


def figure_residual_modes():
    """3 + 4: how much of the corner error is structured, and which part hurts."""
    data = json.loads((OUT / "corner_residual_modes.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))

    for seed in SEEDS:
        explained = data[f"seed{seed}"]["pca_explained"]
        axes[0].plot(range(1, len(explained) + 1), np.cumsum(explained),
                     marker="o", label=f"seed {seed}")
    axes[0].axhline(0.50, color="tab:red", ls="--", lw=1)
    axes[0].text(4.2, 0.53, "gate: top-3 >= 50%", color="tab:red", fontsize=8)
    axes[0].set_xlabel("principal component")
    axes[0].set_ylabel("cumulative explained variance")
    axes[0].set_title("16-D corner residual is low rank")
    axes[0].legend()

    keys = list(data["seed1"]["measures"])
    y = np.arange(len(keys))
    for offset, seed in zip((-0.2, 0.2), SEEDS):
        axes[1].barh(y + offset,
                     [data[f"seed{seed}"]["measures"][k]["rho_R"] for k in keys],
                     height=0.4, label=f"seed {seed}")
    axes[1].axvline(0.40, color="tab:red", ls="--", lw=1)
    axes[1].axvline(-0.40, color="tab:red", ls="--", lw=1)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(keys, fontsize=7)
    axes[1].set_xlabel("Spearman rho with rotation error")
    axes[1].set_title("which residual mode moves the pose")
    axes[1].legend(fontsize=8)

    # variance rank against pose damage -- the point is that they disagree
    for seed in SEEDS:
        block = data[f"seed{seed}"]
        axes[2].scatter([e["explained"] for e in block["pc_vs_pose"]],
                        [abs(e["rho_R"]) for e in block["pc_vs_pose"]],
                        label=f"seed {seed}")
    top = max(abs(data["seed1"]["measures"]["front_rear_shift"]["rho_R"]),
              abs(data["seed2"]["measures"]["front_rear_shift"]["rho_R"]))
    axes[2].axhline(top, color="tab:green", ls="--", lw=1)
    axes[2].text(0.02, top - 0.06, "front_rear_shift", color="tab:green",
                 fontsize=8)
    axes[2].set_xlabel("PC explained variance")
    axes[2].set_ylabel("|rho| with rotation error")
    axes[2].set_title("the loud modes are not the harmful ones")
    axes[2].legend(fontsize=8)
    _save(fig, "fig2_residual_modes.png")


def figure_pose_comparison():
    """5 + 6 + 7: point-only, CIGM and joint, split by the subsets that differ."""
    data = json.loads((OUT / "point_line_solver.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    arms = ("F0", "F1", "F2")
    labels = {"F0": "point only", "F1": "CIGM", "F2": "joint point+line"}
    colours = {"F0": "tab:blue", "F1": "tab:red", "F2": "tab:green"}

    for axis, metric, name in ((axes[0], "R_median", "rotation error (deg)"),
                               (axes[1], "t_median", "translation error (m)")):
        subsets = ("ALL", "V=8", "V<8")
        x = np.arange(len(subsets))
        for index, arm in enumerate(arms):
            values = [np.mean([data[f"seed{s}"]["subsets"][sub][arm][metric]
                               for s in SEEDS]) for sub in subsets]
            axis.bar(x + (index - 1) * 0.27, values, width=0.27,
                     label=labels[arm], color=colours[arm])
        axis.set_xticks(x)
        axis.set_xticklabels(subsets)
        axis.set_ylabel(name)
        axis.set_title(f"{name} (mean of 2 seeds)")
    axes[0].legend(fontsize=8)

    for index, arm in enumerate(arms):
        values = [np.mean([data[f"seed{s}"]["subsets"][sub][arm]["success_5cm5deg"]
                           for s in SEEDS]) for sub in ("ALL", "V=8", "V<8")]
        axes[2].bar(np.arange(3) + (index - 1) * 0.27, values, width=0.27,
                    label=labels[arm], color=colours[arm])
    axes[2].set_xticks(np.arange(3))
    axes[2].set_xticklabels(("ALL", "V=8", "V<8"))
    axes[2].set_ylabel("5cm 5deg success")
    axes[2].set_title("joint wins rotation, loses translation")
    _save(fig, "fig3_pose_comparison.png")


def figure_complementarity():
    """in-grid against off-grid, the split the whole fusion question rests on."""
    from mh_report import _corner_pairs, TIE_BAND_CELLS
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    for seed in SEEDS:
        rows = json.loads(
            (OUT / f"mh_screen_A1_CORNER_LINE_long25k_seed{seed}.json").read_text()
        )["25000"]["D2_MH_DEV512"]["rows"]
        for axis, flag, title in ((axes[0], True, "GT corner inside the grid"),
                                  (axes[1], False, "GT corner off the grid")):
            direct, cigm = _corner_pairs(rows, only=flag)
            axis.scatter(direct, cigm, s=4, alpha=0.25, label=f"seed {seed}")
            axis.set_title(f"{title}  (n={direct.size})")
    for axis in axes:
        limit = axis.get_xlim()[1]
        axis.plot([0, limit], [0, limit], color="k", lw=0.8)
        axis.set_xlabel("direct corner error (cells)")
        axis.set_ylabel("CIGM corner error (cells)")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.legend(fontsize=8)
    _save(fig, "fig4_complementarity.png")


def figure_line_uncertainty():
    """does the model know which of its lines are wrong?"""
    data = json.loads((OUT / "line_uncertainty.json").read_text())
    fig, axis = plt.subplots(figsize=(6.5, 3.2))
    width = 0.35
    roles = np.arange(12)
    for offset, seed in zip((-width / 2, width / 2), SEEDS):
        values = [r["entropy_vs_angle"]
                  for r in data[f"seed{seed}"]["per_role"]]
        axis.bar(roles + offset, values, width=width, label=f"seed {seed}")
    axis.axhline(0.35, color="tab:red", ls="--", lw=1)
    axis.text(0.1, 0.37, "gate 0.35", color="tab:red", fontsize=8)
    axis.set_xticks(roles)
    axis.set_xlabel("edge role")
    axis.set_ylabel("Spearman rho (entropy vs angle error)")
    axis.set_title("Hough entropy predicts its own line error")
    axis.legend(fontsize=8)
    _save(fig, "fig5_line_uncertainty.png")


def figure_role_sensitivity():
    data = json.loads((OUT / "role_pose_sensitivity.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.0))
    axes[0].bar([e["corner"] for e in data["corner"]],
                [e["dR_deg_median"] for e in data["corner"]], color="tab:blue")
    axes[0].set_xlabel("corner")
    axes[0].set_ylabel("dR (deg) per +-0.5 cell")
    axes[0].set_title("corner sensitivity")
    axes[1].bar([e["role"] for e in data["line_role"]],
                [e["dR_deg_median"] for e in data["line_role"]], color="tab:orange")
    axes[1].set_xlabel("edge role")
    axes[1].set_ylabel("dR (deg) per +-0.5deg / 0.25 cell")
    axes[1].set_title("line role sensitivity (through CIGM)")
    _save(fig, "fig6_role_sensitivity.png")


def figure_stopgrad():
    """E0 / E1 / E2, if the causal screen has produced anything yet."""
    arms = ("E0_CONTINUE_LINE", "E1_SHARED_CORNER_LINE", "E2_STOPGRAD_CORNER")
    available = {}
    for seed in SEEDS:
        for arm in arms:
            path = OUT / f"stopgrad_{arm}_seed{seed}.json"
            if path.exists():
                available[(arm, seed)] = json.loads(path.read_text())
    if not available:
        print("stopgrad results not present yet, skipping fig7")
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for (arm, seed), history in sorted(available.items()):
        marks = sorted((k for k in history if k.isdigit()), key=int)
        steps = [int(k) for k in marks]
        style = "-" if seed == 1 else "--"
        axes[0].plot(steps, [history[k]["D2_MH_DEV512"]["line"]["angle_median"]
                             for k in marks], style, marker="o", ms=3,
                     label=f"{arm.split('_')[0]} s{seed}")
        axes[1].plot(steps, [history[k]["D2_MH_DEV512"]["line"]["offset_median"]
                             for k in marks], style, marker="o", ms=3,
                     label=f"{arm.split('_')[0]} s{seed}")
    axes[0].set_xlabel("continuation step from A0 @18,000")
    axes[0].set_ylabel("line angle median (deg)")
    axes[1].set_xlabel("continuation step from A0 @18,000")
    axes[1].set_ylabel("line offset median (cells)")
    axes[0].set_title("stop-grad causal screen")
    axes[0].legend(fontsize=7, ncol=2)
    _save(fig, "fig7_stopgrad.png")


if __name__ == "__main__":
    for function in (figure_gradient_cosine, figure_residual_modes,
                     figure_pose_comparison, figure_complementarity,
                     figure_line_uncertainty, figure_role_sensitivity,
                     figure_stopgrad):
        try:
            function()
        except Exception as error:                    # keep the rest usable
            print(f"{function.__name__}: {type(error).__name__}: {error}")
