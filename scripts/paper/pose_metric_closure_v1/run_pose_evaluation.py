"""동일 모집단에서 MAIN · DIAGNOSTIC · ORACLE 세 경로를 평가한다.

    python3 scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py \
        --pose-object-contract data/pallet/results/paper_pose_metric_closure_v1/POSE_EVAL_OBJECT_CONTRACT.json \
        --arm R0

세 경로를 절대 섞지 않는다.

    MAIN        기존 frozen prediction-only selector.  배포 가능한 성능.
    DIAGNOSTIC  단순 최소재투영 selector.  post-hoc 진단이며 main 을 대체하지 않는다.
    ORACLE      GT 축을 공급.  상한이며 배포 성능이 아니다.

GT 는 모든 선택이 끝난 뒤에만 읽는다 — 축 선택에는 어떤 경로에서도 GT 가 들어가지
않는다(ORACLE 만 예외이며 그래서 oracle 이라고 표시한다).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
GT_PATH = OUT_DIR / "GEOMETRY_RESOLVED_POSE_GT.json"
MANIFEST = OUT_DIR / "AXIS_REVIEW_MANIFEST.json"
PREDICTIONS = OUT_DIR / "predictions"
CF_WIDTH, CF_DEPTH = "CF_WIDTH", "CF_DEPTH"


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


def load_predictions(arm: str) -> dict:
    """replay runner 가 만든 raw 2D 예측.  arm 마다 같은 recipe·같은 프레임 순서."""

    path = PREDICTIONS / f"{arm}.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {frame_id: entry for frame_id, entry in payload["frames"].items()
            if entry.get("status") == "OK" and entry.get("keypoints_xy")}


def main() -> int:
    from pose_evaluation_paths import (build_argument_parser, load_pose_object_contract,
                                       object_spec)
    from symmetry_aware_pose_metrics import (cuboid_model_points, model_diameter_m,
                                             pose_auc, rotation_error_degrees,
                                             symmetry_aware_add_m,
                                             translation_components_m,
                                             yaw_error_degrees)
    from challenge.evaluation_v2.oriented_iou3d import oriented_iou_3d
    from challenge.evaluation_v2.pnp_selector import select_pnp_hypotheses

    parser = build_argument_parser(__doc__)
    parser.add_argument("--arm", default="R0")
    args = parser.parse_args()

    contract = load_pose_object_contract(args.pose_object_contract)
    gt = json.loads(GT_PATH.read_text())["frames"]
    frames = {f["frame_id"]: f for f in json.loads(MANIFEST.read_text())["frames_list"]}
    predictions = load_predictions(args.arm)
    if not predictions:
        print(f"no cached predictions for arm {args.arm}")
        return 1

    records: dict[str, list] = {"MAIN": [], "DIAGNOSTIC": [], "ORACLE": []}
    axis_hit = {"MAIN": 0, "DIAGNOSTIC": 0, "ORACLE": 0}
    covered = 0

    for frame_id, truth in gt.items():
        pred = predictions.get(frame_id)
        if not pred:
            continue
        frame = frames[frame_id]
        payload = json.loads((REPO_ROOT / frame["annotation"]).read_text())
        raw = payload["camera_data"]["intrinsics"]
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
        covered += 1

        # ---- MAIN: 기존 frozen selector.  GT 를 보지 않는다.
        chosen_main = None
        try:
            result = select_pnp_hypotheses(
                np.asarray(pred["keypoints_xy"], np.float64),
                camera, {"x": long_m, "y": height_m, "z": short_m}, None)
            for hypothesis in result.hypotheses:
                if hypothesis.name == result.selected_hypothesis and hypothesis.success:
                    dims = hypothesis.camera_facing_dimensions.as_dict()
                    chosen_main = (CF_WIDTH if abs(float(dims["width"]) - long_m) < 1e-6
                                   else CF_DEPTH)
        except Exception:
            chosen_main = None

        # ---- DIAGNOSTIC: 단순 최소재투영.  GT 를 보지 않는다.
        fits = {k: solve(m, points, camera, usable) for k, m in models.items()}
        if any(v is None for v in fits.values()):
            continue
        chosen_diag = min(fits, key=lambda k: fits[k][2])

        # ---- ORACLE: GT 축 공급.  상한이며 배포 성능이 아니다.
        chosen_oracle = truth["physical_long_axis"]

        gt_R = np.asarray(truth["R_gt_representative"], np.float64)
        gt_t = np.asarray(truth["t_gt"], np.float64)
        dims = truth["physical_dimensions_m"]
        extents = (dims["across"], dims["height"], dims["along"])
        model_points = cuboid_model_points(extents)

        for path, chosen in (("MAIN", chosen_main), ("DIAGNOSTIC", chosen_diag),
                             ("ORACLE", chosen_oracle)):
            if chosen is None:
                continue
            axis_hit[path] += (chosen == truth["physical_long_axis"])
            rotation, translation, _ = fits[chosen]
            parts = translation_components_m(translation, gt_t)
            add = symmetry_aware_add_m(model_points, rotation, translation, gt_R, gt_t)
            records[path].append({
                "frame_id": frame_id,
                "object_type": frame["object_type"],
                "paper_domain": frame["paper_domain"],
                "axis_correct": chosen == truth["physical_long_axis"],
                "rotation_error_deg": rotation_error_degrees(rotation, gt_R),
                "yaw_error_deg": yaw_error_degrees(rotation, gt_R),
                "translation_error_cm": parts["total_m"] * 100.0,
                "lateral_error_cm": parts["lateral_m"] * 100.0,
                "depth_error_cm": parts["depth_m"] * 100.0,
                "iou3d": oriented_iou_3d(rotation, translation, extents,
                                         gt_R, gt_t, extents),
                "add_sym_m": add,
                "diameter_m": model_diameter_m(model_points),
            })

    def summarize(rows):
        if not rows:
            return {"n": 0}
        arr = lambda k: np.array([r[k] for r in rows], float)
        errors = arr("add_sym_m")
        diameter = float(np.median(arr("diameter_m")))
        return {
            "n": len(rows),
            "axis_accuracy": float(np.mean([r["axis_correct"] for r in rows])),
            "rotation_median_deg": float(np.median(arr("rotation_error_deg"))),
            "yaw_median_deg": float(np.median(arr("yaw_error_deg"))),
            "translation_median_cm": float(np.median(arr("translation_error_cm"))),
            "lateral_median_cm": float(np.median(arr("lateral_error_cm"))),
            "depth_median_cm": float(np.median(arr("depth_error_cm"))),
            "iou3d_median": float(np.median(arr("iou3d"))),
            "add_sym_auc": pose_auc(errors, diameter),
        }

    report = {
        "schema_version": "pose_evaluation_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "arm": args.arm,
        "gt": str(GT_PATH.relative_to(REPO_ROOT)),
        "gt_rule_sha256": json.loads(GT_PATH.read_text())["rule_sha256"],
        "pose_object_contract_sha256": contract["contract_sha256"],
        "population": "PAPER_EVAL 319, geometry-resolved GT",
        "frames_with_usable_prediction": covered,
        "new_training": 0,
        "new_inference": "replayed once under INFERENCE_REPLAY_LOCK; R0 reproduced its cache exactly",
        "prediction_source": "data/pallet/results/paper_pose_metric_closure_v1/predictions/<ARM>.json",
        "path_semantics": {
            "MAIN": "existing frozen prediction-only selector — deployable",
            "DIAGNOSTIC": "simple minimum-reprojection selector — post-hoc, does not replace MAIN",
            "ORACLE": "GT physical axis supplied — upper bound, not deployable",
        },
        "paths": {},
    }
    for path, rows in records.items():
        block = {"ALL": summarize(rows)}
        for key, group in (("plastic", "plastic_standard_110x130x11"),
                           ("wood", "wood_small_80x59x14")):
            block[key] = summarize([r for r in rows if r["object_type"] == group])
        for domain in ("daytime", "nighttime"):
            block[domain] = summarize([r for r in rows if r["paper_domain"] == domain])
        block["coverage"] = len(rows) / len(gt) if gt else 0.0
        report["paths"][path] = block

    out = OUT_DIR / f"POSE_EVALUATION_{args.arm}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"arm {args.arm}   frames with usable prediction {covered}")
    print(f"  {'path':12}{'n':>5}{'cov':>7}{'axis':>8}{'R med':>9}{'yaw med':>9}"
          f"{'t med cm':>10}{'IoU3D':>8}{'ADD AUC':>9}")
    for path in ("MAIN", "DIAGNOSTIC", "ORACLE"):
        s = report["paths"][path]["ALL"]
        if not s.get("n"):
            print(f"  {path:12}{0:5d}")
            continue
        print(f"  {path:12}{s['n']:5d}{report['paths'][path]['coverage']:7.3f}"
              f"{s['axis_accuracy']:8.3f}{s['rotation_median_deg']:9.2f}"
              f"{s['yaw_median_deg']:9.2f}{s['translation_median_cm']:10.2f}"
              f"{s['iou3d_median']:8.3f}{s['add_sym_auc']:9.3f}")
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
