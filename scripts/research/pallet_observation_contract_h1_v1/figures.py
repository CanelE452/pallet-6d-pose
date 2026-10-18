"""Postseal reporting plots; no pose solving, fitting or real-reference inputs.

Every real-data chart retains the full PRIMARY192 denominator. The finite tau
chart uses only archived synthetic development/confirmation measurements.
"""
from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scripts.research.pallet_risk_information_v1.common import read, rows, csv_write, sha, write
from .registry import SEEDS, BASIC_GROUPS, AUGMENTATION_GROUPS
from .policy import decode_tau

COLORS = ("#2166ac", "#1b7837", "#7b3294", "#ca7020", "#c51b7d", "#4d9221",
          "#762a83", "#a6611a", "#008b8b", "#d73027", "#636363", "#7fbc41",
          "#ce6dbd", "#8c6d31")
MARKERS = ("o", "s", "^")
NAMES = {
    "N0": "N3 + SubPix (N0)", "E0": "Full statewise (E0)",
    "E0_REGENERATED": "E0 regenerated",
    "POSEFIX_SUBPIX_PF_ANCHOR": "PoseFix / PF anchor",
    "POSEFIX_SUBPIX_YOLO_ANCHOR": "PoseFix / YOLO anchor",
    "D1_H1_MLP": "Original full BCE (H1)",
    "D1_H1_PAIR_MLP": "Two-action MLP", "D1_H1_PAIR_ROLE_AFFINE": "Two-action affine",
    "R_CONT_F0": "Continuous regression (R_CONT)",
    "H1_NO_B_FEATURE": "H1 / no-B feature bundle",
    "H1_GUARDED_SWITCH": "Guard / active policy or H1 fallback",
    "H1_GUARD_CALIBRATED_TRIAL": "Guard trial (diagnostic only; not adopted)",
    "AUG_RANDOM_H1": "H1 / random additions", "AUG_STRUCTURED_H1": "H1 / structural additions",
}


def group_name(arm):
    parts = arm.rsplit("_s", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else arm


def label(arm):
    return NAMES.get(group_name(arm), arm)


def variants(group, arms):
    return [group + "_s" + str(seed) for seed in SEEDS if group + "_s" + str(seed) in arms] or [group]


def representatives(arms):
    groups = list(dict.fromkeys(group_name(a) for a in arms))
    return groups, [variants(g, arms)[0] for g in groups]


def comparator(arm):
    parts = arm.rsplit("_s", 1)
    seed = parts[1] if len(parts) == 2 and parts[1].isdigit() else str(SEEDS[0])
    return "D1_H1_MLP_s" + seed


def finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def finite(value):
    return value is not None and math.isfinite(value)


def deltas(records, before, after, only_both_failure=False):
    chosen = [r for r in records if not only_both_failure or
              (not r["metrics"][before]["success"] and not r["metrics"][after]["success"])]
    out = dict(denominator=len(chosen))
    for axis in ("translation_cm", "rotation_deg"):
        values = [r["metrics"][after][axis] - r["metrics"][before][axis] for r in chosen
                  if finite(r["metrics"][before][axis]) and finite(r["metrics"][after][axis])]
        out[axis] = dict(n=len(values), mean=float(np.mean(values)) if values else None,
                         worsened_n=sum(v > 0 for v in values), improved_n=sum(v < 0 for v in values),
                         equal_n=sum(v == 0 for v in values))
    return out


def ecdf_rows(records, arms):
    """One row per image/arm/axis, including missing output measurements."""
    output = []
    denominator = len(records)
    for arm in arms:
        for axis in ("translation_cm", "rotation_deg"):
            ordered = sorted(records, key=lambda r: (
                not finite(r["metrics"][arm][axis]),
                r["metrics"][arm][axis] if finite(r["metrics"][arm][axis]) else 0., r["id"]))
            order = 0
            for record in ordered:
                value = record["metrics"][arm][axis]
                if finite(value):order += 1
                output.append(dict(id=record["id"], arm=arm, axis=axis, error=value,
                    available=finite(value), order=order if finite(value) else None,
                    cumulative_fraction_of_full_denominator=order / denominator if finite(value) else None,
                    denominator=denominator))
    return output


def verified_postseal_inputs(run_dir, evaluation, GO):
    if not GO:
        raise PermissionError("Explicit root postseal figure GO required before reading real results")
    manifest = read(run_dir / "PREDICTION_MANIFEST.json")
    seal = read(evaluation / "EVALUATION_SEAL.json")
    if manifest["status"] != "ALL_REGISTERED_OUTPUTS_SEALED_BEFORE_REAL_SCORING" or seal["status"] != "COMPLETE":
        raise ValueError("Actual all-output and evaluation seals required")
    if manifest["real_reference_inputs"] is not False or seal["prediction_manifest_sha256"] != sha(run_dir / "PREDICTION_MANIFEST.json"):
        raise ValueError("Prediction/evaluation seal chain changed")
    for name in ("EVALUATION.json", "SCORED_ROWS.jsonl.gz"):
        if seal["artifacts"][name] != sha(evaluation / name):
            raise ValueError("Sealed reporting measurements changed: " + name)
    result = read(evaluation / "EVALUATION.json")
    records = [r for r in rows(evaluation / "SCORED_ROWS.jsonl.gz") if r["partition"] != "CAL"]
    if len(records) != 192 or len({r["id"] for r in records}) != 192 or result["arms"] != manifest["arms"]:
        raise ValueError("Whole PRIMARY192 or registered arm identity mismatch")
    if any(set(r["metrics"]) != set(result["arms"]) for r in records):
        raise ValueError("A registered result is missing")
    if any(s["denominator"] != 192 for s in result["populations"]["PRIMARY192"].values()):
        raise ValueError("A method changed the whole-image denominator")
    return result, records


def training_histories(run_dir, original_manifest):
    output = []
    for manifest_path, selected in ((original_manifest, ("D1_H1_MLP",)),
            (run_dir / "NO_B_TRAINING_MANIFEST.json", ("H1_NO_B_FEATURE",))):
        for receipt in read(manifest_path)["runs"]:
            if receipt["arm"] not in selected:continue
            path = Path(receipt.get("history", Path(receipt["checkpoint"]).parent / "HISTORY.json"))
            if sha(path) != receipt["history_sha256"]:
                raise ValueError("Immutable learning curve changed")
            output.append((receipt, read(path)))
    return output


def plot_learning(run_dir, original_manifest, output):
    histories = training_histories(run_dir, original_manifest)
    fig, axes = plt.subplots(2, 6, figsize=(25, 7.5), squeeze=False)
    data = []
    for row, group in enumerate(("D1_H1_MLP", "H1_NO_B_FEATURE")):
        for receipt, history in histories:
            if receipt["arm"] != group:continue
            seed = receipt["seed"];color = COLORS[SEEDS.index(seed)]
            epochs = [h["epoch"] for h in history]
            for split, style in (("train", "--"), ("validation", "-")):
                axes[row, 0].plot(epochs, [h[split]["loss"] for h in history], color=color,
                    ls=style, label=str(seed) + " " + split)
            for col, getter in ((1, lambda h: h["success_5cm5deg"]),
                    (2, lambda h: h["translation_cm"]["mean"]), (3, lambda h: h["rotation_deg"]["mean"]),
                    (4, lambda h: h["frame_balanced_Brier"])):
                axes[row, col].plot(epochs, [getter(h["validation"]) for h in history], color=color, label=str(seed))
            for field, style in (("T_gt50cm", "-"), ("T_gt100cm", "--"), ("R_gt45deg", ":")):
                axes[row, 5].plot(epochs, [h["validation"]["tails"][field] for h in history], color=color,
                    ls=style, label=str(seed) + " " + field)
            for h in history:
                data.append(dict(arm=group, seed=seed, epoch=h["epoch"], best_epoch=receipt["best_epoch"],
                    train_frames=h["train"]["frames"], validation_frames=h["validation"]["frames"],
                    train_loss=h["train"]["loss"], validation_loss=h["validation"]["loss"],
                    validation_success=h["validation"]["success_5cm5deg"],
                    validation_T_mean_cm=h["validation"]["translation_cm"]["mean"],
                    validation_R_mean_deg=h["validation"]["rotation_deg"]["mean"],
                    validation_Brier=h["validation"]["frame_balanced_Brier"],
                    **{"validation_" + key: value for key, value in h["validation"]["tails"].items()}))
        for col, title in enumerate(("Original frame-balanced BCE", "Validation joint success /235",
                "Validation mean T (cm)", "Validation mean R (deg)", "Validation frame Brier", "Validation large-error counts /235")):
            axes[row, col].set_title(label(group) + "\n" + title, fontsize=9)
            axes[row, col].set_xlabel("Epoch");axes[row, col].grid(alpha=.2)
            axes[row, col].legend(fontsize=6, ncol=2 if col in (0, 5) else 1)
    finish(fig, output / "learning_and_pose_curves.png")
    csv_write(output / "learning_and_pose_curves.csv", data)


def plot_tau(policy_path, output):
    policy = read(policy_path)
    if policy["schema"] != "FROZEN_H1_GUARDED_POLICY_CALIBRATION_V1" or policy["PRIMARY_or_CAL_reference_reads"] != 0:
        raise ValueError("Synthetic-only sealed policy calibration required")
    fig, axes = plt.subplots(3, 4, figsize=(21, 10), squeeze=False)
    fig.suptitle("Synthetic development thresholds only; confirmation once; no real-image threshold selection", fontsize=13)
    data = []
    for row, seed in enumerate(SEEDS):
        result = policy["seeds"][str(seed)];trials = result["all_thresholds"]
        taus = [decode_tau(t["tau"]) for t in trials]
        active, trial = decode_tau(result["active_tau"]), decode_tau(result["calibrated_trial_tau"])
        for rank, item in enumerate(trials):
            m = item["metrics"]
            data.append(dict(seed=seed, scope="SYNTHETIC_DEVELOPMENT", threshold_rank=rank,
                tau_kind=item["tau"]["kind"], tau_value=item["tau"].get("value"),
                is_active=taus[rank] == active, is_calibrated_trial=taus[rank] == trial,
                admissible=item["admissible"], denominator=m["denominator"], successes=m["successes"],
                T_mean_cm=m["translation_cm"]["mean"], R_mean_deg=m["rotation_deg"]["mean"],
                T_gt50cm=m["T_gt50cm"], T_gt100cm=m["T_gt100cm"], R_gt45deg=m["R_gt45deg"],
                changes=m["changes"], policy_status=result["status"], confirmation_passed=None))
        x = np.arange(len(trials))
        series = ((0, [t["metrics"]["successes"] for t in trials], "Joint successes"),
                  (1, [t["metrics"]["translation_cm"]["mean"] for t in trials], "Mean T (cm)"),
                  (2, [t["metrics"]["rotation_deg"]["mean"] for t in trials], "Mean R (deg)"))
        for col, values, title in series:
            axes[row, col].plot(x, values, color=COLORS[row]);axes[row, col].set_title(title, fontsize=10)
        for field, style in (("T_gt100cm", "-"), ("R_gt45deg", "--")):
            axes[row, 3].plot(x, [t["metrics"][field] for t in trials], ls=style, label=field)
        axes[row, 3].set_title("Large-error counts", fontsize=10)
        for ax in axes[row]:
            ax.axvline(taus.index(active), color="#238b45", label="Active fixed tau", lw=1.5)
            ax.axvline(taus.index(trial), color="#d95f0e", label="Trial: diagnostic, not adopted", ls="--")
            ax.set_xlabel("Fixed threshold rank; 0 = -inf (exact tau in CSV)", fontsize=8)
            ax.grid(alpha=.2)
        axes[row, 0].set_ylabel(str(seed) + "\n" + result["status"].replace("_", " "), fontsize=7)
        fmt = lambda value: "-inf" if value == -math.inf else format(value, ".7g")
        confirmation_n = result["confirmation"]["selected"]["denominator"] if result["confirmation"] is not None else 0
        axes[row, 0].text(.02, .98,
            "Active tau=" + fmt(active) + "; trial tau=" + fmt(trial) + "\n"
            + "Development N=" + str(result["development_baseline"]["denominator"])
            + "; confirmation N=" + str(confirmation_n),
            transform=axes[row, 0].transAxes, va="top", fontsize=7,
            bbox=dict(facecolor="white", edgecolor="#999", alpha=.9))
        axes[row, 3].legend(fontsize=6)
        confirmation = result["confirmation"]
        if confirmation is not None:
            for name in ("baseline", "selected"):
                m = confirmation[name]
                data.append(dict(seed=seed, scope="SYNTHETIC_CONFIRMATION_" + name.upper(),
                    threshold_rank=0 if name == "baseline" else taus.index(trial),
                    tau_kind="NEGATIVE_INFINITY" if name == "baseline" else result["calibrated_trial_tau"]["kind"],
                    tau_value=None if name == "baseline" else result["calibrated_trial_tau"].get("value"),
                    is_active=None, is_calibrated_trial=name == "selected", admissible=None,
                    denominator=m["denominator"], successes=m["successes"], T_mean_cm=m["translation_cm"]["mean"],
                    R_mean_deg=m["rotation_deg"]["mean"], T_gt50cm=m["T_gt50cm"], T_gt100cm=m["T_gt100cm"],
                    R_gt45deg=m["R_gt45deg"], changes=m["changes"], policy_status=result["status"],
                    confirmation_passed=confirmation["passed"]))
    finish(fig, output / "synthetic_tau_success_risk.png")
    csv_write(output / "synthetic_tau_success_risk.csv", data)


def generate(run_dir, evaluation, output, original_manifest, policy_path, *, postseal_GO=False):
    run_dir, evaluation, output, original_manifest, policy_path = map(Path,
        (run_dir, evaluation, output, original_manifest, policy_path))
    result, scored = verified_postseal_inputs(run_dir, evaluation, postseal_GO)
    output.mkdir(parents=True, exist_ok=False)
    groups, reps = representatives(result["arms"])
    primary = result["populations"]["PRIMARY192"]
    colors = {group: COLORS[i % len(COLORS)] for i, group in enumerate(groups)}
    fig, axes = plt.subplots(1, 2, figsize=(19, 9))
    data = []
    for index, (group, rep) in enumerate(zip(groups, reps)):
        axes[0].barh(index, primary[rep]["successes"], color=colors[group], alpha=.3)
        values = []
        for j, arm in enumerate(variants(group, result["arms"])):
            value = primary[arm];values.append(value["successes"])
            axes[0].scatter(value["successes"], index + (j - 1) * .12, marker=MARKERS[j], color=colors[group], s=35)
            for field, offset, marker in (("T_gt50cm", -.15, "o"), ("T_gt100cm", 0., "x"), ("R_gt45deg", .15, "s")):
                axes[1].scatter(value[field], index + offset, marker=marker, color=colors[group],
                    alpha=1. if arm == rep else .3, s=40 if arm == rep else 20)
            data.append(dict(arm=arm, group=group, denominator=192, diagnostic_not_adopted=group == BASIC_GROUPS[2],
                **{key: value[key] for key in ("successes", "T_gt50cm", "T_gt100cm", "R_gt45deg", "no_pose")}))
        axes[0].text(min(191, max(values) + 1), index, "/".join(map(str, values)), fontsize=8, va="center")
    for ax in axes:
        ax.set_yticks(range(len(groups)), [label(g).replace(" (diagnostic only; not adopted)", "\n(diagnostic only; not adopted)") for g in groups], fontsize=8)
        ax.invert_yaxis();ax.grid(axis="x", alpha=.15)
    axes[0].set_xlim(0, 192);axes[0].set_xlabel("Joint success / all 192 images")
    axes[0].set_title("Fixed output / seed18 bar; all three seeds when present")
    axes[0].legend(handles=[Line2D([], [], marker=m, color="black", ls="None", label=str(s))
        for s, m in zip(SEEDS, MARKERS)], title="Fixed seed", fontsize=7)
    axes[1].set_xlabel("Large-error counts / all 192 images")
    axes[1].set_title("Representative symbols opaque; other seeds faint")
    axes[1].legend(handles=[Line2D([], [], marker=m, color="black", ls="None", label=t)
        for m, t in (("o", "T > 50cm"), ("x", "T > 100cm"), ("s", "R > 45deg"))], fontsize=8)
    finish(fig, output / "success_and_large_errors.png");csv_write(output / "success_and_large_errors.csv", data)

    cdf = ecdf_rows(scored, result["arms"])
    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    fig.suptitle("Whole PRIMARY192: extremes and zero retained; missing outputs remain in denominator\n"
        "Fixed output / seed18 thick; seed19 and seed20 thin")
    for col, (axis, unit) in enumerate((("translation_cm", "cm"), ("rotation_deg", "deg"))):
        for group, rep in zip(groups, reps):
            for arm in variants(group, result["arms"]):
                selected = [r for r in cdf if r["arm"] == arm and r["axis"] == axis and r["available"]]
                values = np.array([r["error"] for r in selected]);fraction = [r["cumulative_fraction_of_full_denominator"] for r in selected]
                for row, transform in ((0, lambda x: x), (1, np.log1p)):
                    axes[row, col].step(transform(values), fraction, where="post", color=colors[group],
                        lw=1.5 if arm == rep else .6, alpha=.95 if arm == rep else .3,
                        label=label(rep) if arm == rep else None)
        for row in range(2):
            axes[row, col].set_xlabel(("All " if row == 0 else "log1p ") + axis + " (" + unit + ")")
            axes[row, col].set_ylabel("Cumulative fraction of full 192");axes[row, col].set_ylim(0, 1.01);axes[row, col].grid(alpha=.2)
    axes[0, 0].legend(fontsize=6, ncol=2)
    finish(fig, output / "whole_error_distributions.png");csv_write(output / "whole_error_distributions.csv", cdf)

    compared = [(g, variants(g, result["arms"])) for g in groups if g != "D1_H1_MLP"]
    fig, axes = plt.subplots(1, 3, figsize=(22, 9))
    data = []
    for index, (group, methods) in enumerate(compared):
        for j, arm in enumerate(methods):
            before = comparator(arm);counts = Counter()
            for r in scored:
                b, a = r["metrics"][before]["success"], r["metrics"][arm]["success"]
                counts["BOTH_SUCCESS" if b and a else "RECOVERY" if a else "DAMAGE" if b else "BOTH_FAILURE"] += 1
            d = deltas(scored, before, arm, only_both_failure=True)
            y = index + (j - 1) * .12
            axes[0].scatter(counts["RECOVERY"], y, color="#268b53", marker=MARKERS[j])
            axes[0].scatter(-counts["DAMAGE"], y, color="#c74747", marker=MARKERS[j])
            for ax, field in zip(axes[1:], ("translation_cm", "rotation_deg")):
                if d[field]["mean"] is not None:ax.scatter(d[field]["mean"], y, color=colors[group], marker=MARKERS[j])
            data.append(dict(before=before, arm=arm, denominator=192, recovery=counts["RECOVERY"], damage=counts["DAMAGE"],
                both_success_n=counts["BOTH_SUCCESS"], both_failure_n=d["denominator"],
                delta_T_n=d["translation_cm"]["n"], delta_T_mean_cm=d["translation_cm"]["mean"],
                delta_T_worsened_n=d["translation_cm"]["worsened_n"], delta_R_n=d["rotation_deg"]["n"],
                delta_R_mean_deg=d["rotation_deg"]["mean"], delta_R_worsened_n=d["rotation_deg"]["worsened_n"]))
    for ax, title in zip(axes, ("Recovery (+ green) / damage (- red) vs same-seed H1", "Both-failure mean delta T (cm), + worse", "Both-failure mean delta R (deg), + worse")):
        ax.set_yticks(range(len(compared)), [label(g) for g, _ in compared], fontsize=7)
        ax.invert_yaxis();ax.axvline(0, color="black", lw=.7);ax.set_title(title, fontsize=10)
    axes[0].legend(handles=[Line2D([], [], marker=m, color="black", ls="None", label=str(s)) for s, m in zip(SEEDS, MARKERS)], fontsize=7)
    finish(fig, output / "H1_recovery_damage_and_common_failure.png");csv_write(output / "H1_recovery_damage_and_common_failure.csv", data)

    failed = [r for r in scored if not r["full_bank_has_success"]]
    selected_groups = [g for g in ("R_CONT_F0",) + BASIC_GROUPS + AUGMENTATION_GROUPS if g in groups]
    columns = 2;fig, axes = plt.subplots(math.ceil(len(selected_groups) / columns), columns,
        figsize=(17, 5 * math.ceil(len(selected_groups) / columns)), squeeze=False)
    data = []
    for ax, group in zip(axes.flat, selected_groups):
        for j, arm in enumerate(variants(group, result["arms"])):
            before = comparator(arm);xy = []
            for record in failed:
                b, a = record["metrics"][before], record["metrics"][arm]
                comparable = all(finite(m[k]) for m in (b, a) for k in ("translation_cm", "rotation_deg"))
                t = a["translation_cm"] - b["translation_cm"] if comparable else None
                r = a["rotation_deg"] - b["rotation_deg"] if comparable else None
                if comparable:xy.append((t, r))
                data.append(dict(id=record["id"], before=before, arm=arm, fixed_set="FULL_FIXED_ACTION_BANK", denominator=len(failed),
                    comparable=comparable, delta_translation_cm=t, delta_rotation_deg=r))
            ax.scatter([v[0] for v in xy], [v[1] for v in xy], marker=MARKERS[j], alpha=.65, s=26, label=str(SEEDS[j]))
        ax.axhline(0, color="#777", lw=.7);ax.axvline(0, color="#777", lw=.7)
        ax.set_title(label(group) + "\nActual full-bank all-failure N=" + str(len(failed)), fontsize=10)
        ax.set_xlabel("Delta T vs same-seed H1 (cm), positive worse");ax.set_ylabel("Delta R vs same-seed H1 (deg), positive worse")
        ax.legend(title="Fixed seed", fontsize=7)
    for ax in list(axes.flat)[len(selected_groups):]:ax.set_visible(False)
    finish(fig, output / "fixed_bank_all_failure_continuous_scatter.png");csv_write(output / "fixed_bank_all_failure_continuous_scatter.csv", data)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9));data = []
    roles = (("N0", "#607a96"), ("E0", "#e2a448"), ("OTHER", "#4b9a78"), ("NO_POSE", "#a5a5a5"))
    for panel, scope in enumerate(("ALL", "RECOVERY_VS_H1")):
        left = np.zeros(len(groups))
        for role, color in roles:
            counts = []
            for index, (group, rep) in enumerate(zip(groups, reps)):
                for arm in variants(group, result["arms"]):
                    before = comparator(arm)
                    subset = [r for r in scored if scope == "ALL" or
                        (not r["metrics"][before]["success"] and r["metrics"][arm]["success"])]
                    count = sum(r["action_sources"][arm] == role for r in subset)
                    data.append(dict(arm=arm, before=before, scope=scope, source=role, n=count, scope_n=len(subset), whole_denominator=192))
                    if arm == rep:counts.append(count)
            axes[panel].barh(range(len(groups)), counts, left=left, color=color, label=role);left += counts
        axes[panel].set_yticks(range(len(groups)), [label(g) for g in groups], fontsize=7);axes[panel].invert_yaxis()
        axes[panel].set_title("Actual returned source: " + scope + "\nFixed seed18 chart; CSV all fixed seeds")
        axes[panel].set_xlabel("Whole images");axes[panel].legend(fontsize=8)
    finish(fig, output / "actual_sources_and_H1_recovery_sources.png");csv_write(output / "actual_sources_and_H1_recovery_sources.csv", data)

    plot_learning(run_dir, original_manifest, output)
    plot_tau(policy_path, output)
    write(output / "FIGURES.json", dict(status="GENERATED_ACTUAL_PIXEL_REVIEW_PENDING",
        evaluation_seal_sha256=sha(evaluation / "EVALUATION_SEAL.json"), prediction_manifest_sha256=sha(run_dir / "PREDICTION_MANIFEST.json"),
        original_training_manifest_sha256=sha(original_manifest), no_B_training_manifest_sha256=sha(run_dir / "NO_B_TRAINING_MANIFEST.json"),
        policy_calibration_sha256=sha(policy_path), source_sha256=sha(__file__),
        images=[dict(file=p.name, sha256=sha(p)) for p in sorted(output.glob("*.png"))],
        machine_readable_data=[dict(file=p.name, sha256=sha(p)) for p in sorted(output.glob("*.csv"))],
        whole_PRIMARY_denominator=192, all_seeds_preserved=True, no_extreme_removed=True,
        calibrated_trial_is_diagnostic_not_adopted=True, actual_pixel_review_completed=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("run-dir", "evaluation", "output", "original-training-manifest", "policy-calibration"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--postseal-GO", action="store_true")
    a = parser.parse_args()
    generate(a.run_dir, a.evaluation, a.output, a.original_training_manifest, a.policy_calibration, postseal_GO=a.postseal_GO)


if __name__ == "__main__":
    main()
