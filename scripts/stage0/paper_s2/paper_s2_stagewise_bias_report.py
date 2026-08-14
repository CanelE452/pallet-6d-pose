"""Phase I/J analysis and the pre-fixed GO/STOP gate for the stage-wise screen."""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "data/pallet/results/paper_s2_stagewise_bias_screen"
FIG = OUT / "figures"
MECH = ROOT / "data/pallet/results/paper_s2_mechanism_diagnostic"
FAR = (4, 5, 6, 7)
STAGES = (4, 5, 6)


def long_form(table: pd.DataFrame, arm: str) -> pd.DataFrame:
    rows = []
    for _, row in table.iterrows():
        for corner in range(8):
            entry = {"arm": arm, "frame_id": row["frame_id"],
                     "session_id": row["session_id"], "domain": row["domain"],
                     "corner": corner,
                     "group": "far" if corner in FAR else "near"}
            for stage in STAGES:
                for field in ("err", "peak", "mass", "wrong", "dx", "dy"):
                    entry[f"s{stage}_{field}"] = row[f"s{stage}_{field}_{corner}"]
            rows.append(entry)
    return pd.DataFrame(rows)


def sharpen_without_correction(frame: pd.DataFrame) -> pd.Series:
    """Confidence up by more than 0.10 while the error does not really fall."""
    return ((frame.s6_peak > frame.s4_peak + 0.10)
            & (frame.s6_err >= frame.s4_err - 2.0))


def summarise(frame: pd.DataFrame, poses: pd.DataFrame, f2: set[str]) -> dict:
    errors = frame.s6_err
    far = frame[frame.group == "far"]
    f2_far = frame[(frame.frame_id.isin(f2)) & (frame.group == "far")]
    clean = f2_far.dropna(subset=["s6_dx", "s6_dy"])
    valid = frame.dropna(subset=["s6_err"])
    reproj = pd.to_numeric(poses.reproj_px, errors="coerce")
    return {
        "median_px": float(errors.median()),
        "near_median_px": float(frame[frame.group == "near"].s6_err.median()),
        "far_median_px": float(far.s6_err.median()),
        "f2_far_median_px": float(f2_far.s6_err.median()),
        "f2_far_signed_bias_px": float(np.hypot(clean.s6_dx.mean(), clean.s6_dy.mean()))
        if len(clean) else np.nan,
        "p90_px": float(errors.quantile(0.90)),
        "tail_gt20": int((valid.s6_err > 20).sum()),
        "tail_gt50": int((valid.s6_err > 50).sum()),
        "tail_gt100": int((valid.s6_err > 100).sum()),
        "nan_err": int(errors.isna().sum()),
        "sharpen_no_correct": int(sharpen_without_correction(frame).sum()),
        "s6_better_than_s4": float((frame.s6_err < frame.s4_err).mean()),
        "s6_mass_up": float((frame.s6_mass > frame.s4_mass).mean()),
        "s6_wrong_down": float((frame.s6_wrong < frame.s4_wrong).mean()),
        "pose_success": int(poses.pose_success.sum()),
        "reproj_median_px": float(reproj.median()),
        "yaw_median_deg": float(pd.to_numeric(poses.yaw_err_deg,
                                              errors="coerce").median()),
    }


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    c0 = pd.read_parquet(OUT / "mechanism_C0.parquet")
    c1 = pd.read_parquet(OUT / "mechanism_C1.parquet")
    classes = pd.read_csv(MECH / "failure_class_frames.csv")
    f2 = set(classes.loc[classes.failure_class == "F2_CONFIDENT_WRONG", "frame_id"])
    f1 = set(classes.loc[classes.failure_class == "F1_NO_RESPONSE", "frame_id"])

    long0, long1 = long_form(c0, "C0"), long_form(c1, "C1")
    pd.concat([long0, long1]).to_csv(OUT / "stagewise_corner_metrics.csv", index=False)

    s0, s1 = summarise(long0, c0, f2), summarise(long1, c1, f2)
    summary = pd.DataFrame([{"arm": "C0", **s0}, {"arm": "C1", **s1}])
    summary.to_csv(OUT / "stagewise_pose_metrics.csv", index=False)

    trajectory = []
    for arm, frame in (("C0", long0), ("C1", long1)):
        for stage in STAGES:
            block = frame.dropna(subset=[f"s{stage}_err"])
            far = block[block.group == "far"]
            trajectory.append({
                "arm": arm, "stage": stage,
                "far_err_median": float(far[f"s{stage}_err"].median()),
                "near_err_median": float(
                    block[block.group == "near"][f"s{stage}_err"].median()),
                "far_peak_median": float(far[f"s{stage}_peak"].median()),
                "far_mass_median": float(far[f"s{stage}_mass"].median()),
                "far_wrong_median": float(far[f"s{stage}_wrong"].median()),
            })
    trajectory = pd.DataFrame(trajectory)
    trajectory.to_csv(OUT / "stagewise_stage_trajectory.csv", index=False)

    merged = long0.merge(long1, on=["frame_id", "corner"], suffixes=("_c0", "_c1"))
    f2_pairs = merged[merged.frame_id.isin(f2) & (merged.group_c0 == "far")]
    per_frame = f2_pairs.groupby("frame_id").agg(
        err_c0=("s6_err_c0", "median"), err_c1=("s6_err_c1", "median")).reset_index()
    per_frame["delta"] = per_frame.err_c1 - per_frame.err_c0
    per_frame.to_csv(OUT / "stagewise_f2_paired.csv", index=False)

    f1_rows = []
    for arm, frame, poses in (("C0", long0, c0), ("C1", long1, c1)):
        block = frame[frame.frame_id.isin(f1)]
        pose_block = poses[poses.frame_id.isin(f1)]
        f1_rows.append({
            "arm": arm, "frames": int(pose_block.shape[0]),
            "decoded_corners": int(block.s6_err.notna().sum()),
            "no_response_corners": int(block.s6_err.isna().sum()),
            "gt_mass_median": float(block.s6_mass.median()),
            "pose_success": int(pose_block.pose_success.sum()),
        })
    pd.DataFrame(f1_rows).to_csv(OUT / "stagewise_f1_metrics.csv", index=False)

    def drop(before, after):
        return float(1.0 - after / before) if before else 0.0

    improved = int((per_frame.delta < 0).sum())
    worsened = int((per_frame.delta > 0).sum())
    lost = int((c0.pose_success & ~c1.pose_success).sum())
    collapse = bool(np.allclose(
        c1[[f"s4_peak_{i}" for i in range(8)]].to_numpy(),
        c1[[f"s6_peak_{i}" for i in range(8)]].to_numpy()))
    conditions = [
        ("1 F2 far median -15%", drop(s0["f2_far_median_px"], s1["f2_far_median_px"]) >= 0.15,
         drop(s0["f2_far_median_px"], s1["f2_far_median_px"])),
        ("2 F2 signed bias -20%",
         drop(s0["f2_far_signed_bias_px"], s1["f2_far_signed_bias_px"]) >= 0.20,
         drop(s0["f2_far_signed_bias_px"], s1["f2_far_signed_bias_px"])),
        ("3 >50px tail -15%", drop(s0["tail_gt50"], s1["tail_gt50"]) >= 0.15,
         drop(s0["tail_gt50"], s1["tail_gt50"])),
        ("4 sharpen-no-correct -30%",
         drop(s0["sharpen_no_correct"], s1["sharpen_no_correct"]) >= 0.30,
         drop(s0["sharpen_no_correct"], s1["sharpen_no_correct"])),
        ("5 F2 improved > worsened", improved > worsened, improved - worsened),
        ("6 canonical PnP >= 72", s1["pose_success"] >= 72, s1["pose_success"]),
        ("7 reproj -10%", drop(s0["reproj_median_px"], s1["reproj_median_px"]) >= 0.10,
         drop(s0["reproj_median_px"], s1["reproj_median_px"])),
        ("8 near <= +5%", s1["near_median_px"] <= s0["near_median_px"] * 1.05,
         s1["near_median_px"] / s0["near_median_px"] - 1.0),
        ("9 no new PnP failure", lost == 0, lost),
        ("10 no new >100px", s1["tail_gt100"] <= s0["tail_gt100"],
         s1["tail_gt100"] - s0["tail_gt100"]),
        ("11 no new NaN", s1["nan_err"] <= s0["nan_err"], s1["nan_err"] - s0["nan_err"]),
        ("12 no stage collapse", not collapse, float(collapse)),
    ]
    gate = {"conditions": [{"name": n, "passed": bool(p), "value": float(v)}
                           for n, p, v in conditions],
            "passed": all(p for _, p, _ in conditions),
            "paired_improved": improved, "paired_worsened": worsened,
            "C0": s0, "C1": s1}
    (OUT / "stagewise_gate.json").write_text(json.dumps(gate, indent=1),
                                             encoding="utf-8")

    # figures
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for arm, style in (("C0", "-o"), ("C1", "--s")):
        block = trajectory[trajectory.arm == arm]
        ax[0].plot(block.stage, block.far_err_median, style, label=arm)
        ax[1].plot(block.stage, block.far_peak_median, style, label=arm)
        ax[2].plot(block.stage, block.far_wrong_median, style, label=arm)
    for axis, title in zip(ax, ("far error median (px)", "far peak median",
                                "wrong-peak median")):
        axis.set_xlabel("belief stage"); axis.set_title(title)
        axis.legend(); axis.grid(alpha=.3); axis.set_xticks([4, 5, 6])
    fig.tight_layout(); fig.savefig(FIG / "stage_trajectory_before_after.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(3); w = 0.35
    ax.bar(x - w / 2, [s0["tail_gt20"], s0["tail_gt50"], s0["tail_gt100"]], w, label="C0")
    ax.bar(x + w / 2, [s1["tail_gt20"], s1["tail_gt50"], s1["tail_gt100"]], w, label="C1")
    ax.set_xticks(x); ax.set_xticklabels([">20px", ">50px", ">100px"])
    ax.legend(); ax.grid(alpha=.3, axis="y"); ax.set_title("error tail")
    fig.tight_layout(); fig.savefig(FIG / "error_tail.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    order = per_frame.sort_values("delta")
    ax.bar(range(len(order)), order.delta,
           color=["seagreen" if d < 0 else "crimson" for d in order.delta])
    ax.axhline(0, c="k", lw=1); ax.set_ylabel("C1 - C0 (px)")
    ax.set_title(f"F2 paired far error   improved {improved} / worsened {worsened}")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(FIG / "F2_paired_delta.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(["C0", "C1"], [s0["sharpen_no_correct"], s1["sharpen_no_correct"]],
           color=["gray", "steelblue"])
    ax.set_ylabel("corners"); ax.grid(alpha=.3, axis="y")
    ax.set_title("sharpen-without-correction\n(peak +0.10 while error does not fall)")
    fig.tight_layout(); fig.savefig(FIG / "wrong_peak_suppression.png", dpi=130)
    plt.close(fig)

    print(summary.to_string(index=False))
    print()
    print(trajectory.to_string(index=False))
    print()
    for condition in gate["conditions"]:
        print(f"  {'PASS' if condition['passed'] else 'FAIL'}  "
              f"{condition['name']:<32} {condition['value']:>10.4f}")
    print(f"\nGO/STOP: {'GO' if gate['passed'] else 'STOP'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
