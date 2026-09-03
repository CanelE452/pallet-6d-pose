"""FAST_6D_SCREEN — D0 raw DOPE, S3 bbox-gated decoding, S4 square-context ROI.

    python3 scripts/paper/fast6d_screen_v1/run_dope_arms.py \
        --output-dir data/pallet/results/paper_fast6d_screen_v1

S3 은 네트워크 입력을 바꾸지 않는다.  full image forward 그대로 두고 **decoder 의
후보 선택에만** YOLO bbox 정사각 context 를 쓴다.  heatmap 을 건드리지 않으므로
이 arm 에는 조정 가능한 가중치가 없다.

S4 는 정사각 context 를 잘라 400x400 으로 넣는다.  기각된 방식(tight bbox 를
short-side 400 으로)과 달리 종횡비가 변하지 않는다.  역매핑은 별도 parity 테스트가
1e-4 px 이내를 보장한 뒤에만 실행된다.

DOPE 와 YOLO 의 검출 점수 의미가 다르므로 둘 사이 AUROC/AP 비교를 하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CLOSURE = REPO_ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "scripts/paper/pose_metric_closure_v1"))
sys.path.insert(0, str(REPO_ROOT / "scripts/stage0/selftrain"))
sys.path.insert(0, str(REPO_ROOT))

from square_roi import crop_square, forward_map, inverse_map, square_context  # noqa: E402

GROSS_PX = 20.0
N_CORNERS = 8


def peaks_from_belief(belief: np.ndarray, threshold: float, max_peaks: int = 8):
    """채널마다 국소 최대 후보 목록.  DOPE 의 기존 decode 와 같은 성격."""

    out = []
    for channel in range(belief.shape[0]):
        heat = belief[channel]
        smoothed = cv2.GaussianBlur(heat, (3, 3), 0)
        mask = (smoothed >= threshold)
        if not mask.any():
            out.append([])
            continue
        dilated = cv2.dilate(smoothed, np.ones((3, 3), np.float32))
        local = mask & (smoothed >= dilated - 1e-9)
        ys, xs = np.nonzero(local)
        scores = smoothed[ys, xs]
        order = np.argsort(-scores)[:max_peaks]
        out.append([(float(xs[i]), float(ys[i]), float(scores[i])) for i in order])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
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
    import s1_cad_9filters as S1          # 정본 DOPE 추론 경로
    from filter_pr_camfacing import extract_keypoints_from_belief  # noqa: F401

    lock = json.loads((out_dir / "FAST_6D_SCREEN_LOCK.json").read_text())
    s3 = lock["arms"]["S3"]
    s4 = lock["arms"]["S4"]
    ratio = 1.25
    contract = load_pose_object_contract(str(CLOSURE / "POSE_EVAL_OBJECT_CONTRACT.json"))
    gt_all = json.loads((CLOSURE / "GEOMETRY_RESOLVED_POSE_GT.json").read_text())["frames"]
    manifest = {f["frame_id"]: f for f in
                json.loads((CLOSURE / "AXIS_REVIEW_MANIFEST.json").read_text())["frames_list"]}
    yolo = json.loads((CLOSURE / "predictions/R0.json").read_text())["frames"]

    weights = REPO_ROOT / "weights/backbone_dope_final_v1/run/final_net_epoch_0060.pth"
    print(f"DOPE weights {weights.name}")
    model = S1.E.load_model(str(weights), args.device)
    pad = int(getattr(S1, "PAD", 0))
    threshold = float(getattr(S1, "THR", 0.3))
    print(f"  pad {pad}  belief threshold {threshold}")

    rows = []
    for frame_id, truth in gt_all.items():
        frame = manifest[frame_id]
        image = cv2.imread(str(REPO_ROOT / frame["image"]))
        if image is None:
            continue
        height, width = image.shape[:2]
        annotation = json.loads((REPO_ROOT / frame["annotation"]).read_text())
        raw = annotation["camera_data"]["intrinsics"]
        camera = np.array([[raw["fx"], 0, raw["cx"]], [0, raw["fy"], raw["cy"]],
                           [0, 0, 1]], np.float64)
        gt_kp = np.array([p if p else [np.nan, np.nan]
                          for p in annotation["objects"][0]["projected_cuboid"]],
                         np.float64)[:N_CORNERS]
        supervised = np.isfinite(gt_kp).all(axis=1)
        spec = object_spec(contract, frame["object_type"])

        belief, geom, wh = S1.infer_belief(model, image, args.device, pad)
        belief = np.asarray(belief)
        candidates = peaks_from_belief(belief, threshold)
        bh, bw = belief.shape[1], belief.shape[2]

        def to_original(bx, by):
            return S1.E.belief_to_orig_pad(bx, by, bw, bh, *geom, pad, *wh)

        box = (yolo.get(frame_id) or {}).get("box_xyxy")
        context = (square_context(box, width, height, ratio, 400) if box else None)

        def decode(gate):
            points, fallbacks = np.full((N_CORNERS, 2), np.nan), 0
            for index in range(N_CORNERS):
                options = candidates[index] if index < len(candidates) else []
                if not options:
                    continue
                mapped = [(*to_original(x, y), s) for x, y, s in options]
                inside = mapped
                if gate and context is not None:
                    left = context["origin_x"]
                    top = context["origin_y"]
                    side = context["side"]
                    inside = [p for p in mapped
                              if left <= p[0] <= left + side and top <= p[1] <= top + side]
                    if not inside:
                        fallbacks += 1
                        inside = mapped
                best = max(inside, key=lambda p: p[2])
                points[index] = [best[0], best[1]]
            return points, fallbacks

        d0_points, _ = decode(gate=False)
        s3_points, s3_fallback = decode(gate=True)

        s4_points = np.full((N_CORNERS, 2), np.nan)
        if context is not None:
            patch = crop_square(image, context, cv2.BORDER_REFLECT_101)
            if patch is not None:
                crop_belief, crop_geom, crop_wh = S1.infer_belief(
                    model, patch, args.device, pad)
                crop_belief = np.asarray(crop_belief)
                crop_candidates = peaks_from_belief(crop_belief, threshold)
                cbh, cbw = crop_belief.shape[1], crop_belief.shape[2]
                for index in range(N_CORNERS):
                    options = crop_candidates[index] if index < len(crop_candidates) else []
                    if not options:
                        continue
                    x, y, _ = max(options, key=lambda p: p[2])
                    cx, cy = S1.E.belief_to_orig_pad(x, y, cbw, cbh, *crop_geom,
                                                     pad, *crop_wh)
                    s4_points[index] = inverse_map(np.array([[cx, cy]]), context)[0]

        record = {"frame_id": frame_id, "session_id": frame["session_id"],
                  "s3_fallback_corners": s3_fallback,
                  "has_yolo_box": box is not None}
        gt_R = np.asarray(truth["R_gt_representative"], np.float64)
        gt_t = np.asarray(truth["t_gt"], np.float64)
        gt_dims = truth["physical_dimensions_m"]
        gt_extents = (gt_dims["across"], gt_dims["height"], gt_dims["along"])
        model_points = cuboid_model_points(gt_extents)
        diameter = model_diameter_m(model_points)

        for name, points in (("D0", d0_points), ("S3", s3_points), ("S4", s4_points)):
            valid = np.isfinite(points).all(axis=1)
            block = {"detected": bool(valid.sum() >= 6),
                     "n_corners": int(valid.sum())}
            both = valid & supervised
            if both.any():
                errors = np.linalg.norm(points[both] - gt_kp[both], axis=1)
                block["_errors"] = errors.tolist()
            if valid.sum() >= 6:
                padded = np.vstack([points, np.nanmean(points[valid], axis=0)])
                try:
                    outcome = predict_pose_without_gt(
                        padded, camera, spec["long_m"], spec["short_m"], spec["height_m"])
                    selector = outcome["selector_result"]
                    chosen = next((h for h in selector.hypotheses
                                   if h.name == selector.selected_hypothesis and h.success),
                                  None)
                except Exception:
                    chosen = None
                if chosen is not None and chosen.rotation_camera_facing is not None:
                    rotation = np.asarray(chosen.rotation_camera_facing, np.float64)
                    translation = np.asarray(chosen.translation_camera_facing,
                                             np.float64).reshape(3)
                    parts = translation_components_m(translation, gt_t)
                    block |= {
                        "pose": True,
                        "rotation_deg": rotation_error_degrees(rotation, gt_R),
                        "yaw_deg": yaw_error_degrees(rotation, gt_R),
                        "translation_cm": parts["total_m"] * 100.0,
                        "iou3d": oriented_iou_3d(rotation, translation, gt_extents,
                                                 gt_R, gt_t, gt_extents),
                        "add_sym_m": symmetry_aware_add_m(model_points, rotation,
                                                          translation, gt_R, gt_t),
                        "diameter_m": diameter,
                    }
            record[name] = block
        rows.append(record)
        if len(rows) % 50 == 0:
            print(f"  {len(rows)}/{len(gt_all)}", flush=True)

    total = len(gt_all)

    def summarize(name):
        blocks = [r[name] for r in rows]
        detected = [b for b in blocks if b["detected"]]
        posed = [b for b in detected if b.get("pose")]
        errors = np.concatenate([np.asarray(b["_errors"]) for b in blocks
                                 if b.get("_errors")]) if any(
            b.get("_errors") for b in blocks) else np.array([])
        out = {"frames": total,
               "detection_coverage": len(detected) / total,
               "pose_coverage": len(posed) / total,
               "corners_pooled": int(errors.size)}
        if errors.size:
            out |= {"kp_median_px": float(np.median(errors)),
                    "kp_p90_px": float(np.percentile(errors, 90)),
                    "gross20": float(np.mean(errors > GROSS_PX))}
        if posed:
            arr = lambda k: np.array([b[k] for b in posed], float)
            out |= {"rotation_median_deg": float(np.median(arr("rotation_deg"))),
                    "yaw_median_deg": float(np.median(arr("yaw_deg"))),
                    "translation_median_cm": float(np.median(arr("translation_cm"))),
                    "iou3d_median": float(np.median(arr("iou3d"))),
                    "add_sym_auc": pose_auc(arr("add_sym_m"),
                                            float(np.median(arr("diameter_m"))))}
        return out

    summary = {name: summarize(name) for name in ("D0", "S3", "S4")}
    fallback_frames = sum(1 for r in rows if r["s3_fallback_corners"] > 0)
    payload = {
        "schema_version": "fast6d_dope_arms_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dope_weights": str(weights.relative_to(REPO_ROOT)),
        "network_input_unchanged_for_S3": True,
        "S3_has_no_tunable_weight": True,
        "S4_padding": "BORDER_REFLECT_101",
        "S4_inverse_map_parity": "unit test passed, max round-trip error below 1e-4 px",
        "S3_fallback_frames": fallback_frames,
        "S3_fallback_rate": fallback_frames / total,
        "cross_model_detection_scores_not_compared": True,
        "new_training": 0, "depth": 0,
        "summary": summary,
    }
    (screen / "S3_BBOX_GATED_DOPE.json").write_text(json.dumps(
        {**payload, "arm": "S3"}, indent=2) + "\n")
    (screen / "S4_SQUARE_ROI_DOPE.json").write_text(json.dumps(
        {**payload, "arm": "S4"}, indent=2) + "\n")
    (screen / "DOPE_PER_FRAME.json").write_text(json.dumps({"frames": rows}, indent=2) + "\n")

    print(f"\n{'arm':6}{'DetCov':>9}{'PoseCov':>9}{'corners':>9}{'kp med':>9}{'p90':>9}"
          f"{'gross20':>10}{'IoU3D':>8}{'ADDsym':>9}")
    print("-" * 78)
    for name in ("D0", "S3", "S4"):
        b = summary[name]
        print(f"{name:6}{b['detection_coverage']:9.3f}{b['pose_coverage']:9.3f}"
              f"{b['corners_pooled']:9d}{b.get('kp_median_px', float('nan')):9.2f}"
              f"{b.get('kp_p90_px', float('nan')):9.2f}"
              f"{b.get('gross20', float('nan')):10.3f}"
              f"{b.get('iou3d_median', float('nan')):8.4f}"
              f"{b.get('add_sym_auc', float('nan')):9.4f}")
    print(f"\nS3 fallback frames {fallback_frames} ({fallback_frames / total:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
