"""V1B L2/L3/L4 — YOLO 점 + SplitLate 선의 hybrid.  새 solver 를 쓰지 않는다.

    python3 scripts/paper/fast6d_screen_v1b/run_line_arms.py \
        --output-dir data/pallet/results/paper_fast6d_screen_v1b --seed 1

배선 (lock §10):

    RGB ├── YOLO R0 -> 9 keypoints -> frozen selector -> (R0, t0)
        └── SplitLate -> 12 structural lines
                             |
                     mh_fusion.rotation_only      (L2: t 고정)
                             |
                     mh_fusion.translation_refit  (L3: 코너만, line 안 들어감)

line 모델의 코너 예측은 pose point input 으로 **쓰지 않는다**.
support 는 예측된 YOLO 코너에서 만든다 — GT 로 만들면 oracle 이다.
lambda 는 theta_posealigned_d0.json 에서 읽은 historical 값 그대로다.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for sub in ("scripts/stage0", "scripts/stage0/paper_s2", "scripts/stage0/multihead",
            "scripts/stage0/line", "scripts/stage0/real_eval", "challenge",
            "scripts/annotate"):
    sys.path.insert(0, str(ROOT / sub))
sys.path.insert(0, str(ROOT / "scripts/paper/pose_metric_closure_v1"))
sys.path.insert(0, str(ROOT))

import cv2                                        # noqa: E402
import numpy as np                                # noqa: E402

import mh_cigm as CG                              # noqa: E402
import mh_diagnose as DG                          # noqa: E402
import mh_fusion as FU                            # noqa: E402
import line_feature_capacity_v2 as V2             # noqa: E402

CLOSURE = ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
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
    return np.stack([u, v], axis=1)


def support_from_grid(grid9):
    grid = np.asarray(grid9, float)[None, :, :]
    _, _, p0, p1, length = V2.gt_lines(grid, CG.EDGES)
    return V2.visible_segments(p0, p1, length)["hit"][0]


def pixels_to_grid(pixels, width, height, grid=50):
    pixels = np.asarray(pixels, float)
    return np.stack([pixels[:, 0] * grid / width, pixels[:, 1] * grid / height], 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True, choices=(1, 2))
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
    weight = float(lock["historical_lambda_read_from_file_not_hardcoded"][
        f"read_value_seed{args.seed}"])
    # lock 에 적힌 값이 정말 파일에서 온 값인지 다시 확인한다
    historical = json.loads((ROOT / "data/pallet/results/paper_s2_multihead"
                             / "theta_posealigned_d0.json").read_text())
    on_disk = float(historical["seeds"][f"seed{args.seed}"]["selected_lambda_theta"])
    if abs(on_disk - weight) > 1e-12:
        raise SystemExit(f"lambda mismatch: lock {weight} vs file {on_disk}")

    contract = load_pose_object_contract(str(CLOSURE / "POSE_EVAL_OBJECT_CONTRACT.json"))
    gt_all = json.loads((CLOSURE / "GEOMETRY_RESOLVED_POSE_GT.json").read_text())["frames"]
    manifest = {f["frame_id"]: f for f in
                json.loads((CLOSURE / "AXIS_REVIEW_MANIFEST.json").read_text())["frames_list"]}
    predictions = json.loads((CLOSURE / "predictions/R0.json").read_text())["frames"]
    line_cache = {f["frame_id"]: f for f in json.loads(
        (out_dir / "cache" / f"line_predictions_seed{args.seed}.json").read_text())["frames"]}
    gate = json.loads((out_dir / "audit" / "GEOMETRY_CONVENTION_GATE.json").read_text())
    allowed = {k for k, v in gate["object_status"].items() if v == "OK"}
    print(f"seed{args.seed}  lambda {weight}  line cache {len(line_cache)}  "
          f"objects allowed {sorted(allowed)}", flush=True)

    rows = []
    for frame_id, truth in gt_all.items():
        frame = manifest[frame_id]
        if frame["object_type"] not in allowed:
            continue
        pred = predictions.get(frame_id)
        cached = line_cache.get(frame_id)
        if not pred or pred.get("status") != "OK" or not pred.get("keypoints_xy") or not cached:
            continue
        annotation = json.loads((ROOT / frame["annotation"]).read_text())
        raw = annotation["camera_data"]["intrinsics"]
        camera = np.array([[raw["fx"], 0, raw["cx"]], [0, raw["fy"], raw["cy"]],
                           [0, 0, 1]], np.float64)
        width, height = cached["image_size"]
        spec = object_spec(contract, frame["object_type"])
        long_m, short_m, height_m = spec["long_m"], spec["short_m"], spec["height_m"]

        keypoints = np.asarray(pred["keypoints_xy"], np.float64)
        corners = keypoints[:8]
        if not np.isfinite(corners).all():
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
        if chosen is None or chosen.rotation_camera_facing is None:
            continue
        dims = chosen.camera_facing_dimensions.as_dict()
        across = float(dims["width"])
        along = long_m if abs(across - short_m) < 1e-6 else short_m
        model = cuboid(across, height_m, along)
        R0 = np.asarray(chosen.rotation_camera_facing, np.float64)
        t0 = np.asarray(chosen.translation_camera_facing, np.float64).reshape(3)

        theta = np.asarray(cached["pred_theta_canonical_rad"], float)
        rho = np.asarray(cached["pred_rho_canonical_grid"], float)
        lines = DG._line_in_pixels(theta, rho, width, height)
        support = np.asarray(cached["support_from_predicted_corners"], bool)

        poses = {"L0": (R0, t0)}
        rvec0, _ = cv2.Rodrigues(R0)
        try:
            rvec = FU.rotation_only(rvec0.reshape(3), t0, model, camera, corners,
                                    lines, CG.EDGES, support, weight)
            R_line, _ = cv2.Rodrigues(rvec)
            poses["L2"] = (R_line, t0)
            poses["L3"] = (R_line, FU.translation_refit(R_line, t0, model, camera, corners))
        except Exception as error:
            FAILURES.append({"frame_id": frame_id, "exception_type": type(error).__name__,
                             "message": str(error)[:200],
                             "fallback_reason": "rotation_only/translation_refit raised; "
                                                "L2 and L3 unavailable for this frame"})
        try:
            R_yaw, _ = FU.yaw_only(R0, t0, model, camera, corners, lines, CG.EDGES,
                                   support, weight)
            poses["L4"] = (R_yaw, FU.translation_refit(R_yaw, t0, model, camera, corners))
        except Exception as error:
            FAILURES.append({"frame_id": frame_id, "exception_type": type(error).__name__,
                             "message": str(error)[:200],
                             "fallback_reason": "yaw_only raised; L4 unavailable"})

        # GTSUP parity — 진단 전용, 어떤 paper-facing 숫자에도 들어가지 않는다
        manual = np.asarray(frame["keypoints_xy"], np.float64)[:9]
        parity_L3 = None
        if np.isfinite(manual).all():
            try:
                gt_support = support_from_grid(pixels_to_grid(manual, width, height))
                rvec_g = FU.rotation_only(rvec0.reshape(3), t0, model, camera, corners,
                                          lines, CG.EDGES, gt_support, weight)
                R_g, _ = cv2.Rodrigues(rvec_g)
                parity_L3 = (R_g, FU.translation_refit(R_g, t0, model, camera, corners))
            except Exception as error:
                FAILURES.append({"frame_id": frame_id, "exception_type": type(error).__name__,
                                 "message": str(error)[:200],
                                 "fallback_reason": "GTSUP parity only; primary unaffected"})

        gt_R = np.asarray(truth["R_gt_representative"], np.float64)
        gt_t = np.asarray(truth["t_gt"], np.float64)
        gt_dims = truth["physical_dimensions_m"]
        gt_extents = (gt_dims["across"], gt_dims["height"], gt_dims["along"])
        model_points = cuboid_model_points(gt_extents)
        diameter = model_diameter_m(model_points)

        def score(pose):
            R, t = pose
            parts = translation_components_m(t, gt_t)
            return {"rotation_deg": rotation_error_degrees(R, gt_R),
                    "yaw_deg": yaw_error_degrees(R, gt_R),
                    "translation_cm": parts["total_m"] * 100.0,
                    "iou3d": oriented_iou_3d(R, t, gt_extents, gt_R, gt_t, gt_extents),
                    "add_sym_m": symmetry_aware_add_m(model_points, R, t, gt_R, gt_t),
                    "diameter_m": diameter,
                    "predicted_reproj_px": float(np.median(np.linalg.norm(
                        project(model, R, t, camera) - corners, axis=1)))}

        record = {"frame_id": frame_id, "session_id": frame["session_id"],
                  "object_type": frame["object_type"],
                  "paper_domain": frame.get("paper_domain", "none"),
                  "n_support": int(support.sum()),
                  "rotation_moved_deg": None}
        for name, pose in poses.items():
            record[name] = score(pose)
        if "L2" in poses:
            record["rotation_moved_deg"] = rotation_error_degrees(poses["L2"][0], R0)
        if parity_L3 is not None:
            record["L3_GTSUP_PARITY_ONLY"] = score(parity_L3)
        rows.append(record)
        if len(rows) % 100 == 0:
            print(f"  {len(rows)}", flush=True)

    arms = ["L0", "L2", "L3", "L4"]

    def summarize(name, subset):
        blocks = [r[name] for r in subset if name in r]
        if not blocks:
            return {"n": 0}
        arr = lambda k: np.array([b[k] for b in blocks], float)
        return {
            "n": len(blocks),
            "pose_coverage": len(blocks) / len(gt_all),
            "rotation_median_deg": float(np.median(arr("rotation_deg"))),
            "yaw_median_deg": float(np.median(arr("yaw_deg"))),
            "translation_median_cm": float(np.median(arr("translation_cm"))),
            "iou3d_median": float(np.median(arr("iou3d"))),
            "add_sym_auc": pose_auc(arr("add_sym_m"), float(np.median(arr("diameter_m")))),
            "predicted_reproj_median_px": float(np.median(arr("predicted_reproj_px"))),
        }

    def subgroup(key, value):
        return [r for r in rows if r.get(key) == value]

    payload = {
        "schema_version": "fast6d_v1b_line_arm_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "lambda_theta": weight,
        "lambda_source": "theta_posealigned_d0.json, verified against the lock at run time",
        "line_checkpoint": lock["line_checkpoints"][f"seed{args.seed}"],
        "line_solver_constants": {"HUBER_PX": DG.HUBER_PX, "MAX_NFEV": DG.MAX_NFEV},
        "population_total": len(gt_all),
        "frames_evaluated": len(rows),
        "objects_allowed_by_geometry_gate": sorted(allowed),
        "support_median": float(np.median([r["n_support"] for r in rows])),
        "rotation_moved_median_deg": float(np.median(
            [r["rotation_moved_deg"] for r in rows if r["rotation_moved_deg"] is not None])),
        "ALL": {arm: summarize(arm, rows) for arm in arms},
        "by_material": {material: {arm: summarize(arm, subgroup("object_type", material))
                                   for arm in arms}
                        for material in sorted({r["object_type"] for r in rows})},
        "by_lighting_existing_paper_domain_field": {
            domain: {arm: summarize(arm, subgroup("paper_domain", domain)) for arm in arms}
            for domain in sorted({r["paper_domain"] for r in rows})},
        "PARITY_ONLY_not_paper_facing": {
            "L3_GTSUP": summarize("L3_GTSUP_PARITY_ONLY", rows),
            "why": "support derived from GT corners is an oracle; reported for parity "
                   "with the historical real evaluation and never used for selection"},
        "exceptions": {"count": len(FAILURES),
                       "by_type": dict(collections.Counter(f["exception_type"] for f in FAILURES)),
                       "records": FAILURES[:50]},
        "new_training": 0, "depth": 0, "parameter_sweep": 0,
    }
    (screen / f"LINE_ARMS_seed{args.seed}.json").write_text(json.dumps(payload, indent=2) + "\n")
    (screen / f"LINE_PER_FRAME_seed{args.seed}.json").write_text(
        json.dumps({"seed": args.seed, "frames": rows}, indent=2) + "\n")

    print(f"\nseed{args.seed}  n {len(rows)}  exceptions {len(FAILURES)}  "
          f"rotation moved median {payload['rotation_moved_median_deg']:.4f} deg")
    print(f"{'arm':6}{'n':>5}{'R':>8}{'Yaw':>8}{'t cm':>9}{'IoU3D':>10}{'ADDsym':>10}{'reproj':>9}")
    print("-" * 66)
    for arm in arms:
        b = payload["ALL"][arm]
        if not b.get("n"):
            print(f"{arm:6}    0   (unavailable)")
            continue
        print(f"{arm:6}{b['n']:5d}{b['rotation_median_deg']:8.3f}{b['yaw_median_deg']:8.3f}"
              f"{b['translation_median_cm']:9.3f}{b['iou3d_median']:10.4f}"
              f"{b['add_sym_auc']:10.4f}{b['predicted_reproj_median_px']:9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
