"""Postseal plots from saved measurements, retaining every seed and extreme."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .common import read, rows, csv_write, sha, write
from .registry import BASIC_GROUPS, VISUAL_GROUPS, SEEDS, REPRESENTATIVE_SEED

COLORS = ("#2166ac", "#1b7837", "#7b3294", "#ca7020", "#c51b7d", "#4d9221",
          "#762a83", "#a6611a", "#008b8b", "#d73027", "#636363", "#7fbc41",
          "#ce6dbd", "#8c6d31", "#637939", "#6b6ecf")
NAMES = {"N0": "N3 + SubPix (N0)", "E0": "Full statewise (E0)",
    "E0_REGENERATED": "E0 regenerated (frozen replay)",
    "POSEFIX_SUBPIX_PF_ANCHOR": "PoseFix + SubPix / PF anchor",
    "POSEFIX_SUBPIX_YOLO_ANCHOR": "PoseFix + SubPix / YOLO anchor",
    "D1_H1_MLP": "Existing full BCE (H1)", "D1_H1_PAIR_MLP": "Existing two-action MLP",
    "D1_H1_PAIR_ROLE_AFFINE": "Existing two-action affine",
    "R_CONT_F0": "Continuous regression / F0", "R_ORDER_F0": "Pareto rank / F0",
    "R_ORDER_F1_ZERO": "Rank / F1 zero control", "R_ORDER_F1": "Rank / F1 information",
    "V_ORDER_NULL": "Rank / visual NULL", "V_ORDER_RGB": "Rank / actual RGB"}


def group_name(arm):
    return arm.rsplit("_s", 1)[0] if arm.rsplit("_s", 1)[-1].isdigit() else arm


def label(arm):
    return NAMES.get(group_name(arm), arm)


def groups_and_representatives(result):
    arms = result["arms"]
    groups = ["N0", "E0", "E0_REGENERATED", "POSEFIX_SUBPIX_PF_ANCHOR", "POSEFIX_SUBPIX_YOLO_ANCHOR",
              "D1_H1_MLP", "D1_H1_PAIR_MLP", "D1_H1_PAIR_ROLE_AFFINE"]
    groups += [g for g in BASIC_GROUPS + VISUAL_GROUPS if g + "_s20261018" in arms]
    return groups, [g if g in arms else g + "_s20261018" for g in groups]


def finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate(run_dir, evaluation, output):
    run_dir, evaluation, output = map(Path, (run_dir, evaluation, output))
    output.mkdir(parents=True, exist_ok=False)
    result = read(evaluation / "EVALUATION.json")
    scored = [r for r in rows(evaluation / "SCORED_ROWS.jsonl.gz") if r["partition"] != "CAL"]
    groups, representatives = groups_and_representatives(result)
    primary = result["populations"]["PRIMARY192"]
    color = {arm: COLORS[i % len(COLORS)] for i, arm in enumerate(representatives)}
    x = np.arange(len(groups))
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), gridspec_kw={"width_ratios": [1.5, 1]})
    data = []
    for index, (group, arm) in enumerate(zip(groups, representatives)):
        available = [group + "_s" + str(seed) for seed in SEEDS if group + "_s" + str(seed) in primary] or [arm]
        values = [primary[a]["successes"] for a in available]
        axes[0].barh(index, primary[arm]["successes"], color=color[arm], alpha=.4)
        axes[0].scatter(values, index + np.linspace(-.16, .16, len(values)),
                        marker="o", facecolor="white", edgecolor=color[arm], s=35, zorder=3)
        axes[0].text(max(values) + 1, index, "/".join(map(str, values)), va="center", fontsize=8)
        for a in available:
            data.append(dict(arm=a, group=group, denominator=192, **{k: primary[a][k]
                for k in ("successes", "T_gt50cm", "T_gt100cm", "R_gt45deg", "no_pose")}))
    axes[0].set_yticks(x, [label(a) for a in representatives], fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 192)
    axes[0].set_xlabel("Joint success count / all 192 images")
    axes[0].set_title("Bar: fixed seed18; dots/counts: all three fixed seeds")
    for j, (field, title, mark) in enumerate((("T_gt50cm", "T > 50 cm", "o"),
        ("T_gt100cm", "T > 100 cm", "x"), ("R_gt45deg", "R > 45 deg", "s"))):
        axes[1].plot([primary[a][field] for a in representatives], x + (j - 1) * .13,
                     mark, label=title, alpha=.8)
    axes[1].set_yticks(x, [label(a) for a in representatives], fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Large-error count / all 192 images (representative)")
    axes[1].legend(fontsize=8)
    finish(fig, output / "success_and_large_errors.png")
    csv_write(output / "success_and_large_errors.csv", data)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Whole PRIMARY192 distributions: seed18 thick, other fixed seeds thin; all extremes retained")
    cdfdata = []
    for axis_index, (field, unit) in enumerate((("translation_cm", "cm"), ("rotation_deg", "deg"))):
        for arm in result["arms"]:
            values = sorted(r["metrics"][arm][field] for r in scored if r["metrics"][arm][field] is not None)
            cdfdata.extend(dict(arm=arm, axis=field, order=i + 1, error=value,
                               cumulative_fraction_of_full_192=(i + 1) / 192)
                           for i, value in enumerate(values))
        for arm in representatives:
            group = group_name(arm)
            variants = [a for a in result["arms"] if group_name(a) == group]
            for variant in variants:
                values = np.asarray(sorted(r["metrics"][variant][field] for r in scored
                                           if r["metrics"][variant][field] is not None))
                fraction = np.arange(1, len(values) + 1) / 192
                for row, transform in ((0, lambda a: a), (1, np.log1p)):
                    axes[row, axis_index].step(transform(values), fraction, where="post",
                        color=color[arm], lw=1.6 if variant == arm else .6,
                        alpha=.95 if variant == arm else .3, label=label(arm) if variant == arm else None)
        axes[0, axis_index].set_xlabel(f"All saved {field} ({unit}), no extreme removed")
        axes[1, axis_index].set_xlabel(f"log1p({field} in {unit}), zero retained")
        for row in range(2):
            axes[row, axis_index].set_ylabel("Cumulative fraction of all 192 images")
            axes[row, axis_index].set_ylim(0, 1.01)
            axes[row, axis_index].grid(alpha=.2)
    axes[0, 0].legend(fontsize=7, ncol=2)
    finish(fig, output / "whole_error_distributions.png")
    csv_write(output / "whole_error_distributions.csv", cdfdata)

    new_reps = [a for a in representatives if group_name(a) in BASIC_GROUPS + VISUAL_GROUPS]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    delta_data = []
    for index, arm in enumerate(new_reps):
        pair = result["paired"]["PRIMARY192"]["N0__TO__" + arm]
        axes[0].barh(index - .15, pair["counts"]["RECOVERY"], height=.27, color="#268b53")
        axes[0].barh(index + .15, -pair["counts"]["DAMAGE"], height=.27, color="#c74747")
        d = pair["both_failure_continuous"]
        for ax, field in zip(axes[1:], ("translation_cm", "rotation_deg")):
            value = d[field]["mean"]
            if value is not None:
                ax.barh(index, value, color=color[arm])
        delta_data.append(dict(arm=arm, recovery=pair["counts"]["RECOVERY"], damage=pair["counts"]["DAMAGE"],
            both_failure_n=d["denominator"], delta_T_mean_cm=d["translation_cm"]["mean"],
            delta_R_mean_deg=d["rotation_deg"]["mean"]))
    for ax in axes:
        ax.set_yticks(range(len(new_reps)), [label(a) for a in new_reps], fontsize=8)
        ax.axvline(0, color="black", lw=.7)
        ax.invert_yaxis()
    axes[0].set_xlabel("Recovery (+) / original success damage (-), vs N0")
    axes[1].set_xlabel("Mean delta T, both-failure rows vs N0 (cm)")
    axes[2].set_xlabel("Mean delta R, both-failure rows vs N0 (deg)")
    finish(fig, output / "recovery_damage_and_common_failure.png")
    csv_write(output / "recovery_damage_and_common_failure.csv", delta_data)

    failed = [r for r in scored if not r["full_bank_has_success"]]
    columns = min(3, max(1, len(new_reps)))
    fig, axes = plt.subplots(int(np.ceil(len(new_reps) / columns)), columns,
                             figsize=(15, 4 * int(np.ceil(len(new_reps) / columns))), squeeze=False)
    scatter_data = []
    for ax, arm in zip(axes.flat, new_reps):
        values = [(r["metrics"][arm]["translation_cm"] - r["metrics"]["N0"]["translation_cm"],
                   r["metrics"][arm]["rotation_deg"] - r["metrics"]["N0"]["rotation_deg"], r["id"])
                  for r in failed if all(r["metrics"][a][k] is not None
                      for a in ("N0", arm) for k in ("translation_cm", "rotation_deg"))]
        ax.scatter([v[0] for v in values], [v[1] for v in values], s=25, alpha=.7, color=color[arm])
        ax.axhline(0, color="#777", lw=.8)
        ax.axvline(0, color="#777", lw=.8)
        ax.set_title(label(arm) + " / fixed full-bank failure41", fontsize=9)
        ax.set_xlabel("Delta T vs N0 (cm); positive = worse")
        ax.set_ylabel("Delta R vs N0 (deg); positive = worse")
        scatter_data.extend(dict(arm=arm, id=fid, delta_translation_cm=t, delta_rotation_deg=r,
                                 fixed_set="FULL_FIXED_ACTION_BANK", denominator=41) for t, r, fid in values)
    for ax in list(axes.flat)[len(new_reps):]:
        ax.set_visible(False)
    finish(fig, output / "fixed_failure41_continuous_scatter.png")
    csv_write(output / "fixed_failure41_continuous_scatter.csv", scatter_data)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    source_data = []
    for panel, mode in enumerate(("ALL", "RECOVERY_VS_E0")):
        left = np.zeros(len(representatives))
        for role, col in (("N0", "#607a96"), ("E0", "#e2a448"), ("OTHER", "#4b9a78"), ("NO_POSE", "#a5a5a5")):
            counts = []
            for arm in representatives:
                selected = [r for r in scored if mode == "ALL" or
                    (not r["metrics"]["E0"]["success"] and r["metrics"][arm]["success"])]
                count = sum(r["action_sources"][arm] == role for r in selected)
                counts.append(count)
                source_data.append(dict(arm=arm, population="PRIMARY192", scope=mode,
                                        source=role, n=count, scope_n=len(selected)))
            axes[panel].barh(np.arange(len(representatives)), counts, left=left, color=col, label=role)
            left += counts
        axes[panel].set_yticks(np.arange(len(representatives)), [label(a) for a in representatives], fontsize=8)
        axes[panel].invert_yaxis()
        axes[panel].set_title("Actual returned source / " + mode)
        axes[panel].legend(fontsize=8)
    finish(fig, output / "actual_sources_and_recovery_sources.png")
    csv_write(output / "actual_sources_and_recovery_sources.csv", source_data)

    matched = [("D1_H1_MLP", "R_ORDER_F0"), ("D1_H1_MLP", "R_CONT_F0"),
               ("R_ORDER_F0", "R_CONT_F0"),
               ("R_ORDER_F1_ZERO", "R_ORDER_F1")]
    if "V_ORDER_RGB_s20261018" in primary:
        matched.append(("V_ORDER_NULL", "V_ORDER_RGB"))
    contrast_labels = {
        ("D1_H1_MLP", "R_ORDER_F0"): "H1 BCE\n→ Rank F0",
        ("D1_H1_MLP", "R_CONT_F0"): "H1 BCE\n→ Regression F0",
        ("R_ORDER_F0", "R_CONT_F0"): "Rank F0\n→ Regression F0",
        ("R_ORDER_F1_ZERO", "R_ORDER_F1"): "F1 ZERO\n→ F1",
        ("V_ORDER_NULL", "V_ORDER_RGB"): "Visual NULL\n→ RGB"}
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    matched_data = []
    for i, (before, after) in enumerate(matched):
        for seed_index, seed in enumerate(SEEDS):
            a, b = before + "_s" + str(seed), after + "_s" + str(seed)
            changes = [primary[b]["successes"] - primary[a]["successes"],
                primary[b]["translation_cm"]["mean"] - primary[a]["translation_cm"]["mean"],
                primary[b]["rotation_deg"]["mean"] - primary[a]["rotation_deg"]["mean"]]
            for ax, change in zip(axes, changes):
                ax.scatter(i + (seed_index - 1) * .12, change,
                    marker=("o", "s", "^")[seed_index], color=COLORS[i], s=45)
            matched_data.append(dict(before=a, after=b, success_count_delta=changes[0],
                translation_mean_delta_cm=changes[1], rotation_mean_delta_deg=changes[2]))
    for ax, title in zip(axes, ("Success count delta (+ better)", "Mean T delta cm (- better)", "Mean R delta deg (- better)")):
        ax.axhline(0, color="black", lw=.7)
        ax.set_xticks(range(len(matched)), [contrast_labels[a, b] for a, b in matched], fontsize=8)
        ax.set_title(title)
    axes[0].legend(handles=[Line2D([], [], color="#444444", marker=marker,
        linestyle="None", label=str(seed)) for seed, marker in zip(SEEDS, ("o", "s", "^"))],
        title="Fixed seed", fontsize=8, title_fontsize=8, loc="best")
    finish(fig, output / "matched_information_objective_controls.png")
    csv_write(output / "matched_information_objective_controls.csv", matched_data)

    histories = []
    for folder in sorted((run_dir / "training").iterdir()):
        if (folder / "RUN_RECEIPT.json").is_file():
            receipt = read(folder / "RUN_RECEIPT.json")
            histories.append((receipt, read(folder / "HISTORY.json")))
    present_groups = [g for g in BASIC_GROUPS + VISUAL_GROUPS if any(r["arm"] == g for r, _ in histories)]
    fig, axes = plt.subplots(len(present_groups), 3, figsize=(17, 3.1 * len(present_groups)), squeeze=False)
    training_data = []
    for row_index, group in enumerate(present_groups):
        for receipt, history in histories:
            if receipt["arm"] != group:
                continue
            seed = receipt["seed"]
            style = ("-", "--", ":")[SEEDS.index(seed)]
            epochs = [v["epoch"] for v in history]
            for split in ("train", "validation"):
                axes[row_index, 0].plot(epochs, [v[split]["loss"] for v in history],
                    ls=style, alpha=.8, label=str(seed) + " " + split)
            axes[row_index, 1].plot(epochs, [v["validation"]["counts"]["successes"] for v in history], ls=style, label=str(seed))
            axes[row_index, 2].plot(epochs, [v["validation"]["translation_cm"]["mean"] for v in history], ls=style, label=str(seed) + " T cm")
            axes[row_index, 2].plot(epochs, [v["validation"]["rotation_deg"]["mean"] for v in history], ls=style, alpha=.5, label=str(seed) + " R deg")
            for v in history:
                training_data.append(dict(arm=group, seed=seed, epoch=v["epoch"],
                    best_epoch=receipt["best_epoch"], train_loss=v["train"]["loss"],
                    validation_loss=v["validation"]["loss"],
                    validation_success=v["validation"]["counts"]["successes"],
                    validation_T_mean_cm=v["validation"]["translation_cm"]["mean"],
                    validation_R_mean_deg=v["validation"]["rotation_deg"]["mean"],
                    validation_T_gt50cm=v["validation"]["counts"].get("T_gt50cm", 0),
                    validation_R_gt45deg=v["validation"]["counts"].get("R_gt45deg", 0)))
        axes[row_index, 0].set_title(label(group) + " / its own objective scale")
        axes[row_index, 1].set_title("Actual validation joint success /235")
        axes[row_index, 2].set_title("Actual validation T(cm), R(deg), distinct units")
        for ax in axes[row_index]:
            ax.set_xlabel("Epoch")
            ax.grid(alpha=.2)
            ax.legend(fontsize=6, ncol=2)
    finish(fig, output / "all_fit_learning_and_pose_curves.png")
    csv_write(output / "all_fit_learning_and_pose_curves.csv", training_data)
    write(output / "FIGURES.json", dict(status="GENERATED_PIXEL_REVIEW_PENDING",
        evaluation_sha256=sha(evaluation / "EVALUATION_SEAL.json"),
        images=[dict(file=p.name, sha256=sha(p)) for p in sorted(output.glob("*.png"))],
        machine_readable_data=[p.name for p in sorted(output.glob("*.csv"))],
        all_seeds_preserved=True, no_extreme_removed=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.run_dir, args.evaluation, args.output)


if __name__ == "__main__":
    main()
