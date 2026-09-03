"""V1B C1 — YOLO 가 실제로 학습한 bbox semantics 로 translation 을 다시 맞춘다.

    python3 scripts/paper/fast6d_screen_v1b/run_bbox_corrected_arm.py \
        --output-dir data/pallet/results/paper_fast6d_screen_v1b

V1 의 S2 는 투영된 8 코너 **전부**의 min/max 를 상자로 썼다.  YOLO 가 학습한
상자는 그것이 아니다 — `build_real_ft_dataset.to_yolo_label` 은 화면 밖 코너를
v=0 으로 버리고 **화면 안 코너만**의 min/max 를 상자로 쓴다.  그래서 V1 의 S2 는
관측 불가능한 상자를 목표로 삼았다.

회전은 절대 움직이지 않는다.  translation 3 변수만 푼다.
목적함수가 줄었다는 이유로 PASS 하지 않는다 — 판정은 GT IoU3D · ADDsym 뿐이다.
"""

from __future__ import annotations

import argparse
import collections
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

MIN_INSIDE = 4
FAILURES: list[dict] = []


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


def inside_mask(projected, width, height):
    """YOLO 라벨 규약과 같은 판정 — 화면 안에 있는 코너만 상자를 만든다."""
    finite = np.isfinite(projected).all(axis=1)
    return (finite
            & (projected[:, 0] >= 0.0) & (projected[:, 0] < width)
            & (projected[:, 1] >= 0.0) & (projected[:, 1] < height))


def refine_translation(model, rotation, t0, camera, target_kp, valid,
                       bbox, width, height, rules, frame_id):
    """회전 고정, t 3변수.  bbox 가 None 이면 point-only, 있으면 point + 관측 bbox."""

    huber_scale = float(rules["huber_f_scale"])
    diagonal = float(np.hypot(width, height))
    n_point = int(valid.sum()) * 2
    n_bbox = 4
    length = n_point + (n_bbox if bbox is not None else 0)

    def residual(t):
        projected, z = project(model, rotation, t, camera)
        if not np.isfinite(projected).all() or (z <= 0).any():
            return np.full(length, 1e3)
        point_block = ((projected[valid] - target_kp[valid]) / diagonal).ravel()
        point_block = point_block / np.sqrt(n_point)
        if bbox is None:
            return point_block
        seen = inside_mask(projected, width, height)
        if int(seen.sum()) < MIN_INSIDE:
            return np.full(length, 1e3)
        observable = projected[seen]
        left, top = observable.min(axis=0)
        right, bottom = observable.max(axis=0)
        predicted_box = np.array([left, right, top, bottom], float)
        target_box = np.array([bbox[0], bbox[2], bbox[1], bbox[3]], float)
        box_block = ((predicted_box - target_box) / diagonal) / np.sqrt(n_bbox)
        return np.concatenate([point_block, box_block])

    try:
        outcome = least_squares(residual, np.asarray(t0, float),
                                loss="huber", f_scale=huber_scale,
                                max_nfev=int(rules["max_nfev"]), method="trf",
                                xtol=float(rules["xtol"]), ftol=float(rules["ftol"]),
                                gtol=float(rules["gtol"]))
    except Exception as error:
        FAILURES.append({"frame_id": frame_id, "exception_type": type(error).__name__,
                         "message": str(error)[:200],
                         "fallback_reason": "least_squares raised; t0 kept"})
        return np.asarray(t0, float), False
    t_new = outcome.x
    if not np.isfinite(t_new).all():
        FAILURES.append({"frame_id": frame_id, "exception_type": "NonFinite",
                         "message": "optimizer returned a non-finite translation",
                         "fallback_reason": "t0 kept"})
        return np.asarray(t0, float), False
    _, z = project(model, rotation, t_new, camera)
    if (z <= 0).any():
        FAILURES.append({"frame_id": frame_id, "exception_type": "NonPositiveDepth",
                         "message": "a corner fell behind the camera",
                         "fallback_reason": "t0 kept"})
        return np.asarray(t0, float), False
    if float(np.sum(residual(t_new) ** 2)) > float(np.sum(residual(np.asarray(t0, float)) ** 2)):
        return np.asarray(t0, float), False
    return t_new, True


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

    lock = json.loads((out_dir / "FAST_6D_SCREEN_V1B_LOCK.json").read_text())
    rules = lock["optimizer_rules_C1"]
    contract = load_pose_object_contract(str(CLOSURE / "POSE_EVAL_OBJECT_CONTRACT.json"))
    gt_all = json.loads((CLOSURE / "GEOMETRY_RESOLVED_POSE_GT.json").read_text())["frames"]
    manifest = {f["frame_id"]: f for f in
                json.loads((CLOSURE / "AXIS_REVIEW_MANIFEST.json").read_text())["frames_list"]}
    predictions = json.loads((CLOSURE / "predictions/R0.json").read_text())["frames"]
    print(f"population {len(gt_all)}  predictions {len(predictions)}", flush=True)

    rows = []
    inside_set_changed = 0
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

        try:
            outcome = predict_pose_without_gt(keypoints, camera, long_m, short_m, height_m)
        except Exception as error:
            FAILURES.append({"frame_id": frame_id, "exception_type": type(error).__name__,
                             "message": str(error)[:200],
                             "fallback_reason": "selector raised; frame dropped"})
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
        model = cuboid(across, height_m, along)
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

        projected0, _ = project(model, rotation, t0, camera)
        seen0 = inside_mask(projected0, width, height)
        box = pred.get("box_xyxy")
        resolved = bool(box) and int(seen0.sum()) >= MIN_INSIDE

        if resolved:
            t_c1, accepted = refine_translation(model, rotation, t0, camera, corners,
                                                valid, box, width, height, rules, frame_id)
            projected1, _ = project(model, rotation, t_c1, camera)
            if not np.array_equal(seen0, inside_mask(projected1, width, height)):
                inside_set_changed += 1
        else:
            t_c1, accepted = t0, False

        record = {"frame_id": frame_id, "session_id": frame["session_id"],
                  "object_type": frame["object_type"],
                  "paper_domain": frame.get("paper_domain", "none"),
                  "n_inside_corners": int(seen0.sum()),
                  "observable_box_resolved": resolved,
                  "unresolved_reason": None if resolved else (
                      "no YOLO box" if not box else "fewer than four in-image corners")}
        for name, (t, was_accepted) in (("C0", (t0, True)), ("C1", (t_c1, accepted))):
            parts = translation_components_m(t, gt_t)
            record[name] = {
                "accepted": bool(was_accepted),
                "rotation_deg": rotation_error_degrees(rotation, gt_R),
                "yaw_deg": yaw_error_degrees(rotation, gt_R),
                "translation_cm": parts["total_m"] * 100.0,
                "iou3d": oriented_iou_3d(rotation, t, gt_extents, gt_R, gt_t, gt_extents),
                "add_sym_m": symmetry_aware_add_m(model_points, rotation, t, gt_R, gt_t),
                "diameter_m": diameter,
                "predicted_reproj_px": float(np.median(np.linalg.norm(
                    project(model, rotation, t, camera)[0][valid] - corners[valid], axis=1))),
            }
        rows.append(record)
        if len(rows) % 100 == 0:
            print(f"  {len(rows)}", flush=True)

    def summarize(name, subset):
        blocks = [r[name] for r in subset]
        arr = lambda k: np.array([b[k] for b in blocks], float)
        return {
            "n": len(blocks),
            "pose_coverage": len(blocks) / len(gt_all),
            "accepted_rate": float(np.mean([b["accepted"] for b in blocks])) if blocks else 0.0,
            "rotation_median_deg": float(np.median(arr("rotation_deg"))),
            "yaw_median_deg": float(np.median(arr("yaw_deg"))),
            "translation_median_cm": float(np.median(arr("translation_cm"))),
            "iou3d_median": float(np.median(arr("iou3d"))),
            "add_sym_auc": pose_auc(arr("add_sym_m"), float(np.median(arr("diameter_m")))),
            "predicted_reproj_median_px": float(np.median(arr("predicted_reproj_px"))),
        }

    resolved_rows = [r for r in rows if r["observable_box_resolved"]]
    unresolved = [r["frame_id"] for r in rows if not r["observable_box_resolved"]]

    payload = {
        "schema_version": "fast6d_v1b_arm_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "population_total": len(gt_all),
        "frames_evaluated": len(rows),
        "C1_UNRESOLVED_OBSERVABLE_BOX": {
            "n": len(unresolved), "frame_ids": unresolved,
            "rule": "fewer than four in-image projected corners, or no YOLO box; "
                    "these frames are NOT counted as a C1 success"},
        "observable_box_semantics": {
            "inside_test": "finite AND 0 <= u < width AND 0 <= v < height",
            "recomputed_at_every_residual_evaluation": True,
            "rationale": "the target box was labelled from whichever corners were "
                         "in-image, so the predicted box must be built the same way "
                         "at whatever translation is being tested",
            "frames_whose_inside_set_changed_during_refinement": inside_set_changed},
        "MAIN_paired_on_resolved": {
            "n": len(resolved_rows),
            "C0": summarize("C0", resolved_rows),
            "C1": summarize("C1", resolved_rows)},
        "secondary_full_population": {
            "C0": summarize("C0", rows),
            "C1_unresolved_frames_keep_t0_and_are_flagged": summarize("C1", rows)},
        "optimizer_rules": rules,
        "exceptions": {"count": len(FAILURES),
                       "by_type": dict(collections.Counter(f["exception_type"] for f in FAILURES)),
                       "records": FAILURES[:50]},
        "new_training": 0, "depth": 0, "parameter_sweep": 0,
    }
    (screen / "C1_OBSERVABLE_BBOX.json").write_text(json.dumps(payload, indent=2) + "\n")
    (screen / "C0_BASELINE.json").write_text(json.dumps({
        "schema_version": "fast6d_v1b_arm_v1",
        "arm": "C0", "identical_to": "frozen YOLO R0 baseline",
        "full_population": summarize("C0", rows),
        "parity_target": lock["baseline_C0_L0"]["expected_parity_read_from_artifact"],
    }, indent=2) + "\n")
    (screen / "C_PER_FRAME.json").write_text(json.dumps({"frames": rows}, indent=2) + "\n")

    # 40 프레임 결정적 parity dump
    parity = sorted(rows, key=lambda r: r["frame_id"])[:40]
    (screen / "C0_PARITY_40_FRAMES.json").write_text(json.dumps({
        "note": "deterministic first 40 frame_ids, C0 only, for byte-level review",
        "frames": [{"frame_id": r["frame_id"], **r["C0"]} for r in parity]}, indent=2) + "\n")

    print(f"\nexceptions {len(FAILURES)}  unresolved {len(unresolved)}  "
          f"inside-set changed {inside_set_changed}")
    print(f"{'arm':6}{'n':>5}{'acc':>7}{'R':>7}{'Yaw':>7}{'t cm':>8}{'IoU3D':>9}"
          f"{'ADDsym':>9}{'reproj':>9}")
    print("-" * 66)
    for name, block in (("C0", payload["MAIN_paired_on_resolved"]["C0"]),
                        ("C1", payload["MAIN_paired_on_resolved"]["C1"])):
        print(f"{name:6}{block['n']:5d}{block['accepted_rate']:7.3f}"
              f"{block['rotation_median_deg']:7.2f}{block['yaw_median_deg']:7.2f}"
              f"{block['translation_median_cm']:8.2f}{block['iou3d_median']:9.4f}"
              f"{block['add_sym_auc']:9.4f}{block['predicted_reproj_median_px']:9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
