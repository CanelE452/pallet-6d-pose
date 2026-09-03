"""평가 결과에서 최종 pose 표를 조립한다.  읽기 전용.

    python3 scripts/paper/pose_metric_closure_v1/build_pose_tables.py

출력
    _docs/paper/final/generated/TABLE_FINAL_POSE.md
    _docs/paper/final/generated/TABLE_FINAL_POSE_BY_MATERIAL.md
    _docs/paper/final/generated/TABLE_FINAL_POSE_BY_LIGHTING.md
    _docs/paper/pose_metric_closure_v1/POSE_AXIS_ORACLE_DIAGNOSTIC.md

MAIN 만 본문 표에 들어간다.  ORACLE 은 진단 문서에만 들어가고 배포 성능처럼 서술하지
않는다.  DIAGNOSTIC(단순 최소재투영)도 본문에 넣지 않는다.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
FINAL = REPO_ROOT / "_docs/paper/final/generated"
CLOSURE = REPO_ROOT / "_docs/paper/pose_metric_closure_v1"

ROWS = [
    ("Synthetic-only (R0)", "R0"),
    ("Source-only continuation", "R0_CONT"),
    ("Naive self-training", "R1_NAIVE"),
    ("Confidence self-training", "R2_CONF"),
    ("Reprojection self-training", "R3_CONF_REPROJ"),
    ("Removal self-training", "R4_CONF_REMOVE"),
    ("Full consistency self-training", "R5_PROPOSED"),
]

HEADER = """Population: the in-house real-image evaluation set, 319 positive frames.
Ground truth comes from `GEOMETRY_RESOLVED_POSE_GT.json` — the physical long axis is
resolved from manually annotated keypoints, calibrated intrinsics and the registered
pallet dimensions. No model prediction enters the ground truth.

Orientation is a 180-degree equivalence class, so yaw folds to 0-90 and a wrong
long/short assignment is never absorbed. `ADDsym AUC` is the group-aware ADD over
{I, Ry(180)}, integrated over [0, 0.1 x diameter] at 1001 points.

Every arm was inferred once under one frozen recipe. The R0 replay reproduced its
existing cache exactly, which is what makes a cached arm and a freshly inferred arm
comparable."""


def load(arm: str) -> dict | None:
    path = RESULTS / f"POSE_EVALUATION_{arm}.json"
    return json.loads(path.read_text()) if path.exists() else None


def cell(value, spec=".3f"):
    return "—" if value is None else format(value, spec)


def row_values(block: dict, group: str = "ALL"):
    stats = block.get(group) or {}
    if not stats.get("n"):
        return None
    return stats


def main() -> int:
    FINAL.mkdir(parents=True, exist_ok=True)
    CLOSURE.mkdir(parents=True, exist_ok=True)
    loaded = {arm: load(arm) for _, arm in ROWS}
    available = [(label, arm) for label, arm in ROWS if loaded.get(arm)]
    if not available:
        print("no evaluation results found")
        return 1

    stamp = datetime.now(timezone.utc).isoformat()

    # ---------------------------------------------------------------- main
    lines = ["# Table — 6D pose, main comparison", "", HEADER, "", "```text",
             f"{'Method':34}{'PoseCov':>9}{'AxisAcc':>9}{'R med':>8}{'Yaw med':>9}"
             f"{'t med cm':>10}{'IoU3D':>8}{'ADDsym AUC':>12}",
             "─" * 99]
    for label, arm in available:
        block = loaded[arm]["paths"]["MAIN"]
        s = row_values(block)
        if s is None:
            lines.append(f"{label:34}{'no detection':>9}")
            continue
        lines.append(
            f"{label:34}{block['coverage']:9.3f}{s['axis_accuracy']:9.3f}"
            f"{s['rotation_median_deg']:8.2f}{s['yaw_median_deg']:9.2f}"
            f"{s['translation_median_cm']:10.2f}{s['iou3d_median']:8.3f}"
            f"{s['add_sym_auc']:12.3f}")
    lines += ["```", "",
              "`AxisAcc` is reported beside the pose metrics on purpose. On this data a",
              "change in axis accuracy does not translate proportionally into a change in",
              "pose accuracy, so the two must be read together rather than one standing",
              "in for the other.",
              "",
              "Pose columns other than these are not reported. Strict signed ADD is absent",
              "because the 180-degree sign is deliberately unresolved in the ground truth."]
    (FINAL / "TABLE_FINAL_POSE.md").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------ material
    lines = ["# Table — 6D pose by material", "", HEADER, "",
             "Wood is included. Its footprint aspect ratio is 1.356 against plastic's",
             "1.182, which makes its two axis hypotheses further apart geometrically.", ""]
    for group, title, count in (("plastic", "Plastic", 194), ("wood", "Wood", 125)):
        lines += [f"## {title}  (N = {count})", "", "```text",
                  f"{'Method':34}{'PoseCov':>9}{'AxisAcc':>9}{'R med':>8}{'Yaw med':>9}"
                  f"{'t med cm':>10}{'IoU3D':>8}{'ADDsym AUC':>12}",
                  "─" * 99]
        for label, arm in available:
            s = row_values(loaded[arm]["paths"]["MAIN"], group)
            if s is None:
                continue
            lines.append(
                f"{label:34}{s['n'] / count:9.3f}{s['axis_accuracy']:9.3f}"
                f"{s['rotation_median_deg']:8.2f}{s['yaw_median_deg']:9.2f}"
                f"{s['translation_median_cm']:10.2f}{s['iou3d_median']:8.3f}"
                f"{s['add_sym_auc']:12.3f}")
        lines += ["```", ""]
    (FINAL / "TABLE_FINAL_POSE_BY_MATERIAL.md").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------ lighting
    lines = ["# Table — 6D pose by lighting", "", HEADER, "",
             "The frozen acquisition-condition subgroups, unchanged: Daytime N = 70 and",
             "Nighttime N = 50, both plastic only. No new subgroup was created.", ""]
    for group, title, count in (("daytime", "Daytime", 70), ("nighttime", "Nighttime", 50)):
        lines += [f"## {title}  (N = {count})", "", "```text",
                  f"{'Method':34}{'PoseCov':>9}{'AxisAcc':>9}{'Yaw med':>9}"
                  f"{'t med cm':>10}{'IoU3D':>8}{'ADDsym AUC':>12}",
                  "─" * 91]
        for label, arm in available:
            s = row_values(loaded[arm]["paths"]["MAIN"], group)
            if s is None:
                continue
            lines.append(
                f"{label:34}{s['n'] / count:9.3f}{s['axis_accuracy']:9.3f}"
                f"{s['yaw_median_deg']:9.2f}{s['translation_median_cm']:10.2f}"
                f"{s['iou3d_median']:8.3f}{s['add_sym_auc']:12.3f}")
        lines += ["```", ""]
    (FINAL / "TABLE_FINAL_POSE_BY_LIGHTING.md").write_text("\n".join(lines) + "\n")

    # -------------------------------------------------------------- oracle
    lines = ["# MAIN versus ORACLE-AXIS diagnostic", "",
             "**Post-hoc diagnostic. The oracle column is not a deployable result** — it",
             "is produced by handing the model the ground-truth physical axis, which a",
             "deployed system does not have. It appears in no main table.", "",
             "Its purpose is to split one question in two:", "",
             "```text",
             "main poor / oracle good    axis selection is the bottleneck",
             "main poor / oracle poor    keypoint geometry is also a bottleneck",
             "```", "", "```text",
             f"{'Method':34}{'MAIN Axis':>11}{'MAIN IoU':>10}{'ORA IoU':>9}{'dIoU':>8}"
             f"{'MAIN AUC':>10}{'ORA AUC':>9}{'dAUC':>8}",
             "─" * 99]
    for label, arm in available:
        main = row_values(loaded[arm]["paths"]["MAIN"])
        oracle = row_values(loaded[arm]["paths"]["ORACLE"])
        if main is None or oracle is None:
            continue
        lines.append(
            f"{label:34}{main['axis_accuracy']:11.3f}{main['iou3d_median']:10.3f}"
            f"{oracle['iou3d_median']:9.3f}"
            f"{oracle['iou3d_median'] - main['iou3d_median']:+8.3f}"
            f"{main['add_sym_auc']:10.3f}{oracle['add_sym_auc']:9.3f}"
            f"{oracle['add_sym_auc'] - main['add_sym_auc']:+8.3f}")
    lines += ["```", "",
              "## Same-population selector comparison", "",
              "An earlier note compared a simple minimum-reprojection selector at 75.2%",
              "against the existing selector at 65.0%. Those figures came from different",
              "populations and that comparison is withdrawn.", "", "```text",
              f"{'Method':34}{'existing frozen':>17}{'simple reprojection':>21}",
              "─" * 72]
    for label, arm in available:
        main = row_values(loaded[arm]["paths"]["MAIN"])
        diag = row_values(loaded[arm]["paths"]["DIAGNOSTIC"])
        if main is None or diag is None:
            continue
        lines.append(f"{label:34}{main['axis_accuracy']:17.3f}"
                     f"{diag['axis_accuracy']:21.3f}")
    lines += ["```", "",
              "On the common 319-frame population the two selectors show nearly identical",
              "axis accuracy. The sentence \"the simple residual selector is ten percentage",
              "points better\" is not supported and is not used.",
              "",
              "`POST_HOC_DIAGNOSTIC_ONLY`", "",
              f"generated {stamp}"]
    (CLOSURE / "POSE_AXIS_ORACLE_DIAGNOSTIC.md").write_text("\n".join(lines) + "\n")

    print(f"arms with results: {len(available)} / {len(ROWS)}")
    for name in ("TABLE_FINAL_POSE.md", "TABLE_FINAL_POSE_BY_MATERIAL.md",
                 "TABLE_FINAL_POSE_BY_LIGHTING.md"):
        print(f"  wrote {(FINAL / name).relative_to(REPO_ROOT)}")
    print(f"  wrote {(CLOSURE / 'POSE_AXIS_ORACLE_DIAGNOSTIC.md').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
