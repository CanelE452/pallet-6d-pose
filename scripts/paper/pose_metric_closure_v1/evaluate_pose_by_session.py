"""이미 캐시된 예측을 세션별로 다시 집계한다.  새 추론·새 학습 없음.

    python3 scripts/paper/pose_metric_closure_v1/evaluate_pose_by_session.py

출력  POSE_EVALUATION_BY_SESSION.json
      _docs/paper/pose_metric_closure_v1/POSE_BY_SESSION.md

`run_pose_evaluation.py` 와 **같은 selector·같은 GT·같은 metric** 을 쓴다.
집계 축만 세션으로 바꾼다.  기존 `POSE_EVALUATION_<ARM>.json` 은 건드리지 않는다.

★ 자기검증: 세션별 값을 다시 합쳐 만든 ALL 이 기존 파일의 ALL 과 어긋나면
   중단한다.  코드 경로가 두 벌이 되었으니 일치를 증명해야 신뢰할 수 있다.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
DOC_DIR = REPO_ROOT / "_docs/paper/pose_metric_closure_v1"
GT_PATH = OUT_DIR / "GEOMETRY_RESOLVED_POSE_GT.json"
MANIFEST = OUT_DIR / "AXIS_REVIEW_MANIFEST.json"
PREDICTIONS = OUT_DIR / "predictions"
CONTRACT = OUT_DIR / "POSE_EVAL_OBJECT_CONTRACT.json"

ARMS = ["R0", "R0_CONT", "R1_NAIVE", "R2_CONF", "R3_CONF_REPROJ",
        "R4_CONF_REMOVE", "R5_PROPOSED"]
LABELS = {"R0": "Synthetic-only (R0)", "R0_CONT": "Source-only continuation",
          "R1_NAIVE": "Naive ST", "R2_CONF": "Confidence ST",
          "R3_CONF_REPROJ": "Reprojection ST", "R4_CONF_REMOVE": "Removal ST",
          "R5_PROPOSED": "Full consistency ST"}
CF_WIDTH, CF_DEPTH = "CF_WIDTH", "CF_DEPTH"
TOLERANCE = 1e-9


def cuboid(across, height, along):
    ha, hh, hb = across / 2.0, height / 2.0, along / 2.0
    return np.array([
        [-ha, -hh, -hb], [+ha, -hh, -hb], [+ha, +hh, -hb], [-ha, +hh, -hb],
        [-ha, -hh, +hb], [+ha, -hh, +hb], [+ha, +hh, +hb], [-ha, +hh, +hb],
    ], dtype=np.float64)


def solve(model, points, camera, usable):
    ok, rvec, tvec = cv2.solvePnP(model[usable], points[usable], camera, None,
                                  flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(model[usable], points[usable], camera, None,
                                      rvec, tvec)
    projected, _ = cv2.projectPoints(model, rvec, tvec, camera, None)
    rotation, _ = cv2.Rodrigues(rvec)
    residual = float(np.linalg.norm(
        projected.reshape(-1, 2)[usable] - points[usable], axis=1).mean())
    return rotation, tvec.reshape(-1), residual


def main() -> int:
    from pose_evaluation_paths import load_pose_object_contract, object_spec
    from symmetry_aware_pose_metrics import (cuboid_model_points, model_diameter_m,
                                             pose_auc, rotation_error_degrees,
                                             symmetry_aware_add_m,
                                             translation_components_m,
                                             yaw_error_degrees)
    from challenge.evaluation_v2.oriented_iou3d import oriented_iou_3d
    from challenge.evaluation_v2.pnp_selector import select_pnp_hypotheses

    contract = load_pose_object_contract(str(CONTRACT))
    gt = json.loads(GT_PATH.read_text())["frames"]
    frames = {f["frame_id"]: f for f in json.loads(MANIFEST.read_text())["frames_list"]}

    per_arm: dict[str, list[dict]] = {}
    for arm in ARMS:
        payload = json.loads((PREDICTIONS / f"{arm}.json").read_text())["frames"]
        predictions = {k: v for k, v in payload.items()
                       if v.get("status") == "OK" and v.get("keypoints_xy")}
        rows = []
        for frame_id, truth in gt.items():
            pred = predictions.get(frame_id)
            if not pred:
                continue
            frame = frames[frame_id]
            annotation = json.loads((REPO_ROOT / frame["annotation"]).read_text())
            raw = annotation["camera_data"]["intrinsics"]
            camera = np.array([[raw["fx"], 0.0, raw["cx"]],
                               [0.0, raw["fy"], raw["cy"]], [0.0, 0.0, 1.0]], np.float64)
            spec = object_spec(contract, frame["object_type"])
            long_m, short_m, height_m = spec["long_m"], spec["short_m"], spec["height_m"]
            models = {CF_WIDTH: cuboid(long_m, height_m, short_m),
                      CF_DEPTH: cuboid(short_m, height_m, long_m)}

            points = np.asarray(pred["keypoints_xy"], np.float64)[:8]
            usable = np.isfinite(points).all(axis=1)
            if usable.sum() < 6:
                continue

            chosen = None
            try:
                result = select_pnp_hypotheses(
                    np.asarray(pred["keypoints_xy"], np.float64), camera,
                    {"x": long_m, "y": height_m, "z": short_m}, None)
                for hypothesis in result.hypotheses:
                    if hypothesis.name == result.selected_hypothesis and hypothesis.success:
                        dims = hypothesis.camera_facing_dimensions.as_dict()
                        chosen = (CF_WIDTH if abs(float(dims["width"]) - long_m) < 1e-6
                                  else CF_DEPTH)
            except Exception:
                chosen = None

            fits = {k: solve(m, points, camera, usable) for k, m in models.items()}
            if any(v is None for v in fits.values()) or chosen is None:
                continue

            gt_R = np.asarray(truth["R_gt_representative"], np.float64)
            gt_t = np.asarray(truth["t_gt"], np.float64)
            dims = truth["physical_dimensions_m"]
            extents = (dims["across"], dims["height"], dims["along"])
            model_points = cuboid_model_points(extents)
            rotation, translation, _ = fits[chosen]
            parts = translation_components_m(translation, gt_t)
            rows.append({
                "frame_id": frame_id,
                "session_id": frame["session_id"],
                "object_type": frame["object_type"],
                "paper_domain": frame["paper_domain"],
                "axis_correct": chosen == truth["physical_long_axis"],
                "rotation_error_deg": rotation_error_degrees(rotation, gt_R),
                "yaw_error_deg": yaw_error_degrees(rotation, gt_R),
                "translation_error_cm": parts["total_m"] * 100.0,
                "iou3d": oriented_iou_3d(rotation, translation, extents,
                                         gt_R, gt_t, extents),
                "add_sym_m": symmetry_aware_add_m(model_points, rotation, translation,
                                                  gt_R, gt_t),
                "diameter_m": model_diameter_m(model_points),
            })
        per_arm[arm] = rows
        print(f"  {arm:16}{len(rows):5d} frames")

    def summarize(rows):
        if not rows:
            return {"n": 0}
        arr = lambda k: np.array([r[k] for r in rows], float)
        return {
            "n": len(rows),
            "axis_accuracy": float(np.mean([r["axis_correct"] for r in rows])),
            "rotation_median_deg": float(np.median(arr("rotation_error_deg"))),
            "yaw_median_deg": float(np.median(arr("yaw_error_deg"))),
            "translation_median_cm": float(np.median(arr("translation_error_cm"))),
            "iou3d_median": float(np.median(arr("iou3d"))),
            "add_sym_auc": pose_auc(arr("add_sym_m"),
                                    float(np.median(arr("diameter_m")))),
        }

    # ── 자기검증: 다시 만든 ALL 이 기존 파일과 같아야 한다
    mismatches = []
    for arm in ARMS:
        recomputed = summarize(per_arm[arm])
        existing = json.loads(
            (OUT_DIR / f"POSE_EVALUATION_{arm}.json").read_text())["paths"]["MAIN"]["ALL"]
        for key in ("n", "axis_accuracy", "iou3d_median", "add_sym_auc",
                    "translation_median_cm", "yaw_median_deg"):
            a, b = recomputed[key], existing[key]
            if abs(float(a) - float(b)) > TOLERANCE:
                mismatches.append(f"{arm}.{key}: {a} != {b}")
    if mismatches:
        print("\nALL 재현 실패 — 세션별 값을 신뢰할 수 없다:")
        for line in mismatches[:10]:
            print(f"  {line}")
        return 1
    print("\nALL 재현 확인 — 기존 POSE_EVALUATION_<ARM>.json 과 완전 일치")

    sessions = sorted({r["session_id"] for r in per_arm["R0"]})
    report = {
        "schema_version": "pose_evaluation_by_session_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "new_training": 0,
        "new_inference": 0,
        "prediction_source": "data/pallet/results/paper_pose_metric_closure_v1/predictions/",
        "selector": "unchanged frozen prediction-only selector (MAIN path)",
        "existing_files_modified": [],
        "all_reproduced_exactly": True,
        "sessions": sessions,
        "by_arm": {
            arm: {
                "ALL": summarize(per_arm[arm]),
                **{s: summarize([r for r in per_arm[arm] if r["session_id"] == s])
                   for s in sessions},
            } for arm in ARMS
        },
    }
    (OUT_DIR / "POSE_EVALUATION_BY_SESSION.json").write_text(
        json.dumps(report, indent=2) + "\n")

    # paired bootstrap 이 같은 프레임에서 arm 을 짝지으려면 per-frame 이 필요하다
    (OUT_DIR / "POSE_PER_FRAME_BY_ARM.json").write_text(json.dumps({
        "schema_version": "pose_per_frame_by_arm_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selector": "unchanged frozen prediction-only selector (MAIN path)",
        "all_reproduced_exactly": True,
        "per_frame": per_arm,
    }, indent=2) + "\n")

    counts = {s: report["by_arm"]["R0"][s]["n"] for s in sessions}
    lines = ["# 6D pose per evaluation session", "",
             "Same frozen selector, same ground truth, same metrics as the main pose",
             "table. Only the aggregation axis changed. No model ran again — the cached",
             "2D predictions were re-read, and the pooled numbers reproduce the existing",
             "per-arm files exactly, which is what makes this split trustworthy.", "",
             "**Session sample sizes are small (10-56).** A rank change between two arms",
             "inside one session is not evidence on its own.", ""]
    for metric, title, spec, better in (
            ("iou3d_median", "IoU3D median", "8.3f", "higher"),
            ("add_sym_auc", "ADDsym AUC", "8.3f", "higher"),
            ("translation_median_cm", "translation median [cm]", "8.2f", "lower"),
            ("axis_accuracy", "axis accuracy", "8.3f", "higher")):
        lines += [f"## {title}  ({better} is better)", "", "```text",
                  f"{'session':18}{'n':>5}" + "".join(
                      f"{LABELS[a].split('(')[0].strip()[:11]:>12}" for a in ARMS),
                  "─" * (23 + 12 * len(ARMS))]
        for session in sessions:
            row = f"{session:18}{counts[session]:5d}"
            for arm in ARMS:
                block = report["by_arm"][arm][session]
                row += (format(block[metric], spec).rjust(12) if block.get("n")
                        else "—".rjust(12))
            lines.append(row)
        row = f"{'ALL':18}{sum(counts.values()):5d}"
        for arm in ARMS:
            row += format(report["by_arm"][arm]["ALL"][metric], spec).rjust(12)
        lines += ["─" * (23 + 12 * len(ARMS)), row, "```", ""]

    DOC_DIR.mkdir(parents=True, exist_ok=True)
    (DOC_DIR / "POSE_BY_SESSION.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {(OUT_DIR / 'POSE_EVALUATION_BY_SESSION.json').relative_to(REPO_ROOT)}")
    print(f"wrote {(DOC_DIR / 'POSE_BY_SESSION.md').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
