"""FAST_6D_SCREEN — S0 baseline, S1 point-only control, S2 point+bbox translation.

    python3 scripts/paper/fast6d_screen_v1/run_translation_arms.py \
        --output-dir data/pallet/results/paper_fast6d_screen_v1

S1 과 S2 는 **회전을 절대 움직이지 않는다**.  translation 3 변수만 푼다.
그래서 S2 의 이득이 있다면 bbox 때문인지, 그냥 한 번 더 최적화해서인지 S1 이 갈라준다.

목적함수가 줄었다는 이유만으로 PASS 하지 않는다 — 기각된 predicted-seed DiffPnP 가
바로 그 함정이었다.  판정은 GT 기준 IoU3D · ADDsym 으로만 한다.

모든 상수는 lock 에서 읽는다.  치수는 frozen object contract 에서 읽는다.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

REPO_ROOT = Path(__file__).resolve().parents[3]
CLOSURE = REPO_ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
sys.path.insert(0, str(REPO_ROOT / "scripts/paper/pose_metric_closure_v1"))
sys.path.insert(0, str(REPO_ROOT))


def cuboid(across, height, along):
    ha, hh, hb = across / 2, height / 2, along / 2
    return np.array([[-ha, -hh, -hb], [+ha, -hh, -hb], [+ha, +hh, -hb], [-ha, +hh, -hb],
                     [-ha, -hh, +hb], [+ha, -hh, +hb], [+ha, +hh, +hb], [-ha, +hh, +hb]],
                    dtype=np.float64)


def project(model, rotation, translation, camera):
    points = model @ rotation.T + translation
    z = points[:, 2]
    u = camera[0, 0] * points[:, 0] / z + camera[0, 2]
    v = camera[1, 1] * points[:, 1] / z + camera[1, 2]
    return np.stack([u, v], axis=1), z


def refine_translation(model, rotation, t0, camera, target_kp, valid,
                       bbox, width, height, rules):
    """회전 고정, t 3변수만.  bbox 가 None 이면 S1, 있으면 S2."""

    # lock: "huber scale 1.0 after image-diagonal normalisation" — 산문으로 적혀
    # 있어 앞의 숫자만 뽑아 쓴다.  lock 을 고치지 않는다.
    huber_scale = float(str(rules["huber_scale"]).split()[0])
    diagonal = float(np.hypot(width, height))
    n_point = int(valid.sum()) * 2
    n_bbox = 4

    def residual(t):
        projected, z = project(model, rotation, t, camera)
        if not np.isfinite(projected).all() or (z <= 0).any():
            return np.full(n_point + (n_bbox if bbox is not None else 0), 1e3)
        point_block = ((projected[valid] - target_kp[valid]) / diagonal).ravel()
        point_block = point_block / np.sqrt(n_point)
        if bbox is None:
            return point_block
        left, top = projected.min(axis=0)
        right, bottom = projected.max(axis=0)
        predicted_box = np.array([left, right, top, bottom], float)
        target_box = np.array([bbox[0], bbox[2], bbox[1], bbox[3]], float)
        box_block = ((predicted_box - target_box) / diagonal) / np.sqrt(n_bbox)
        return np.concatenate([point_block, box_block])

    try:
        # 종료 허용오차는 lock 이 정한 method parameter 가 아니라 solver 배관이다.
        # 기본값(1e-8)은 정규화된 잔차(~1e-4) 앞에서 즉시 수렴해 최적화 자체를
        # 무효로 만들었다 — 첫 실행에서 모든 arm 이 no-op 이 된 원인이다.
        outcome = least_squares(residual, np.asarray(t0, float),
                                loss="huber", f_scale=huber_scale,
                                max_nfev=rules["max_iterations"] * 4, method="trf",
                                xtol=1e-15, ftol=1e-15, gtol=1e-15)
    except Exception as error:                      # 조용히 넘어가지 않는다
        refine_translation.failures.append(repr(error))
        return np.asarray(t0, float), False
    t_new = outcome.x
    if not np.isfinite(t_new).all():
        return np.asarray(t0, float), False
    _, z = project(model, rotation, t_new, camera)
    if (z <= 0).any():
        return np.asarray(t0, float), False
    # 목적함수가 줄었을 때만 채택
    if float(np.sum(residual(t_new) ** 2)) > float(np.sum(residual(np.asarray(t0, float)) ** 2)):
        return np.asarray(t0, float), False
    return t_new, True


refine_translation.failures = []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()
    screen = out_dir / "screen"
    screen.mkdir(parents=True, exist_ok=True)

    from pose_evaluation_paths import (load_pose_object_contract, object_spec,
                                       predict_pose_without_gt)
    from symmetry_aware_pose_metrics import (cuboid_model_points, model_diameter_m,
                                             pose_auc, rotation_error_degrees,
                                             symmetry_aware_add_m,
                                             translation_components_m, yaw_error_degrees)
    from challenge.evaluation_v2.oriented_iou3d import oriented_iou_3d

    lock = json.loads((out_dir / "FAST_6D_SCREEN_LOCK.json").read_text())
    rules = lock["optimizer_rules_S1_S2"]
    contract = load_pose_object_contract(
        str(CLOSURE / "POSE_EVAL_OBJECT_CONTRACT.json"))
    gt_all = json.loads((CLOSURE / "GEOMETRY_RESOLVED_POSE_GT.json").read_text())["frames"]
    manifest = {f["frame_id"]: f for f in
                json.loads((CLOSURE / "AXIS_REVIEW_MANIFEST.json").read_text())["frames_list"]}
    predictions = json.loads((CLOSURE / "predictions/R0.json").read_text())["frames"]
    print(f"population {len(gt_all)}  predictions {len(predictions)}")

    rows = []
    for frame_id, truth in gt_all.items():
        pred = predictions.get(frame_id)
        if not pred or pred.get("status") != "OK" or not pred.get("keypoints_xy"):
            continue
        frame = manifest[frame_id]
        annotation = json.loads((REPO_ROOT / frame["annotation"]).read_text())
        raw = annotation["camera_data"]["intrinsics"]
        camera = np.array([[raw["fx"], 0, raw["cx"]], [0, raw["fy"], raw["cy"]],
                           [0, 0, 1]], np.float64)
        image = cv2.imread(str(REPO_ROOT / frame["image"]))
        if image is None:
            continue
        height, width = image.shape[:2]
        spec = object_spec(contract, frame["object_type"])
        long_m, short_m, height_m = spec["long_m"], spec["short_m"], spec["height_m"]

        keypoints = np.asarray(pred["keypoints_xy"], np.float64)
        corners = keypoints[:8]
        valid = np.isfinite(corners).all(axis=1)
        if valid.sum() < 6:
            continue

        # ── S0: frozen prediction-only selector
        try:
            outcome = predict_pose_without_gt(keypoints, camera, long_m, short_m, height_m)
        except Exception:
            continue
        selector = outcome["selector_result"]
        chosen = None
        for hypothesis in selector.hypotheses:
            if hypothesis.name == selector.selected_hypothesis and hypothesis.success:
                chosen = hypothesis
        if chosen is None:
            continue
        dims = chosen.camera_facing_dimensions.as_dict()
        across = float(dims["width"])
        along = long_m if abs(across - short_m) < 1e-6 else short_m
        extents = (across, height_m, along)
        model = cuboid(*extents)
        if chosen.rotation_camera_facing is None or chosen.translation_camera_facing is None:
            continue
        rotation = np.asarray(chosen.rotation_camera_facing, np.float64)
        t0 = np.asarray(chosen.translation_camera_facing, np.float64).reshape(3)

        gt_R = np.asarray(truth["R_gt_representative"], np.float64)
        gt_t = np.asarray(truth["t_gt"], np.float64)
        gt_dims = truth["physical_dimensions_m"]
        gt_extents = (gt_dims["across"], gt_dims["height"], gt_dims["along"])
        model_points = cuboid_model_points(gt_extents)
        diameter = model_diameter_m(model_points)

        box = pred.get("box_xyxy")
        variants = {"S0": (t0, True)}
        variants["S1"] = refine_translation(model, rotation, t0, camera, corners,
                                            valid, None, width, height, rules)
        variants["S2"] = (refine_translation(model, rotation, t0, camera, corners,
                                             valid, box, width, height, rules)
                          if box else (t0, False))

        record = {"frame_id": frame_id, "session_id": frame["session_id"]}
        for name, (t, accepted) in variants.items():
            parts = translation_components_m(t, gt_t)
            record[name] = {
                "accepted": bool(accepted),
                "rotation_deg": rotation_error_degrees(rotation, gt_R),
                "yaw_deg": yaw_error_degrees(rotation, gt_R),
                "translation_cm": parts["total_m"] * 100.0,
                # frozen evaluator 와 동일하게 양쪽 상자에 GT extents 를 쓴다
                # (run_pose_evaluation.py:165).  정의를 새로 만들지 않는다.
                "iou3d": oriented_iou_3d(rotation, t, gt_extents, gt_R, gt_t, gt_extents),
                "add_sym_m": symmetry_aware_add_m(model_points, rotation, t, gt_R, gt_t),
                "diameter_m": diameter,
                # 목적함수 쪽 숫자도 남긴다 — 이것만으로 PASS 하지 않기 위해서
                "predicted_reproj_px": float(np.median(np.linalg.norm(
                    project(model, rotation, t, camera)[0][valid] - corners[valid], axis=1))),
            }
        rows.append(record)
        if len(rows) % 100 == 0:
            print(f"  {len(rows)}", flush=True)

    def summarize(name):
        blocks = [r[name] for r in rows]
        arr = lambda k: np.array([b[k] for b in blocks], float)
        return {
            "n": len(blocks),
            "pose_coverage": len(blocks) / len(gt_all),
            "accepted_rate": float(np.mean([b["accepted"] for b in blocks])),
            "rotation_median_deg": float(np.median(arr("rotation_deg"))),
            "yaw_median_deg": float(np.median(arr("yaw_deg"))),
            "translation_median_cm": float(np.median(arr("translation_cm"))),
            "iou3d_median": float(np.median(arr("iou3d"))),
            "add_sym_auc": pose_auc(arr("add_sym_m"), float(np.median(arr("diameter_m")))),
            "predicted_reproj_median_px": float(np.median(arr("predicted_reproj_px"))),
        }

    summary = {name: summarize(name) for name in ("S0", "S1", "S2")}
    if refine_translation.failures:
        import collections
        print("optimizer 예외:", dict(collections.Counter(
            refine_translation.failures).most_common(3)))
    for name, block in summary.items():
        path = screen / {"S0": "S0_BASELINE.json", "S1": "S1_POINT_T_ONLY.json",
                         "S2": "S2_BBOX_T_ONLY.json"}[name]
        path.write_text(json.dumps({
            "schema_version": "fast6d_arm_v1",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "arm": name,
            "rotation_frozen": name != "S0",
            "variables": [] if name == "S0" else ["tx", "ty", "tz"],
            "bbox_used": name == "S2",
            "optimizer_rules": rules if name != "S0" else None,
            "new_training": 0, "depth": 0,
            "summary": block,
        }, indent=2) + "\n")
    (screen / "S5_LINE_R_BBOX_T.json").write_text(json.dumps({
        "schema_version": "fast6d_arm_v1",
        "arm": "S5",
        "status": "BLOCKED_INCOMPATIBLE_PROVENANCE",
        "reason": lock["arms"]["S5"]["why"],
        "new_line_training": 0,
    }, indent=2) + "\n")
    (screen / "PER_FRAME.json").write_text(json.dumps({"frames": rows}, indent=2) + "\n")

    print(f"\n{'arm':6}{'n':>5}{'acc':>7}{'R':>7}{'Yaw':>7}{'t cm':>8}{'IoU3D':>8}"
          f"{'ADDsym':>9}{'pred reproj':>13}")
    print("-" * 70)
    for name in ("S0", "S1", "S2"):
        b = summary[name]
        print(f"{name:6}{b['n']:5d}{b['accepted_rate']:7.3f}{b['rotation_median_deg']:7.2f}"
              f"{b['yaw_median_deg']:7.2f}{b['translation_median_cm']:8.2f}"
              f"{b['iou3d_median']:8.4f}{b['add_sym_auc']:9.4f}"
              f"{b['predicted_reproj_median_px']:13.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
