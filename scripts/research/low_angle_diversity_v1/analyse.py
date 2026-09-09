"""A2b screen 최종 분석 — 새 추론 0, 새 학습 0.

정본 evaluator 와 **같은 selector·GT·metric 함수**를 import 한다.  집계 축만
늘린다(p90 · 2D corner · 조건층 · paired delta · 세션 median).
자기검증: 다시 만든 ALL 이 `POSE_EVALUATION_<ARM>.json` 과 어긋나면 중단한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
for sub in ("scripts/paper/pose_metric_closure_v1", "scripts/evaluation",
            "scripts/self_training_yolo", ""):
    sys.path.insert(0, str(REPO / sub) if sub else str(REPO))

CLOSURE = REPO / "data/pallet/results/paper_pose_metric_closure_v1"
OUT = REPO / "data/pallet/results/low_angle_diversity_v1"
ARMS = ["R0", "LAD_C0", "LAD_D1"]
CF_WIDTH, CF_DEPTH = "CF_WIDTH", "CF_DEPTH"
TOL = 1e-9


def cuboid(a, h, b):
    ha, hh, hb = a / 2, h / 2, b / 2
    return np.array([[-ha, -hh, -hb], [ha, -hh, -hb], [ha, hh, -hb], [-ha, hh, -hb],
                     [-ha, -hh, hb], [ha, -hh, hb], [ha, hh, hb], [-ha, hh, hb]], float)


def solve(model, pts, cam, usable):
    ok, rv, tv = cv2.solvePnP(model[usable], pts[usable], cam, None,
                              flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    rv, tv = cv2.solvePnPRefineLM(model[usable], pts[usable], cam, None, rv, tv)
    R, _ = cv2.Rodrigues(rv)
    return R, tv.reshape(-1)


def summarize(rows):
    if not rows:
        return {"n": 0}
    from symmetry_aware_pose_metrics import pose_auc
    a = lambda k: np.array([r[k] for r in rows], float)  # noqa: E731
    q = lambda k, p: float(np.percentile(a(k), p))       # noqa: E731
    return {
        "n": len(rows),
        "axis_accuracy": float(np.mean([r["axis_correct"] for r in rows])),
        "rotation_median_deg": float(np.median(a("rotation_error_deg"))),
        "rotation_p90_deg": q("rotation_error_deg", 90),
        "yaw_median_deg": float(np.median(a("yaw_error_deg"))),
        "translation_median_cm": float(np.median(a("translation_error_cm"))),
        "translation_p90_cm": q("translation_error_cm", 90),
        "depth_median_cm": float(np.median(a("depth_error_cm"))),
        "depth_p90_cm": q("depth_error_cm", 90),
        "lateral_median_cm": float(np.median(a("lateral_error_cm"))),
        "corner2d_median_px": float(np.median(a("corner2d_px"))),
        "corner2d_p90_px": q("corner2d_px", 90),
        "iou3d_median": float(np.median(a("iou3d"))),
        "add_sym_auc": pose_auc(a("add_sym_m"), float(np.median(a("diameter_m")))),
    }


def main() -> int:
    from pose_evaluation_paths import load_pose_object_contract, object_spec
    from symmetry_aware_pose_metrics import (cuboid_model_points, model_diameter_m,
                                             rotation_error_degrees, symmetry_aware_add_m,
                                             translation_components_m, yaw_error_degrees)
    from challenge.evaluation_v2.oriented_iou3d import oriented_iou_3d
    from challenge.evaluation_v2.pnp_selector import select_pnp_hypotheses

    contract = load_pose_object_contract(str(CLOSURE / "POSE_EVAL_OBJECT_CONTRACT.json"))
    gt = json.loads((CLOSURE / "GEOMETRY_RESOLVED_POSE_GT.json").read_text())["frames"]
    frames = {f["frame_id"]: f
              for f in json.loads((CLOSURE / "AXIS_REVIEW_MANIFEST.json").read_text())["frames_list"]}

    per_arm = {}
    for arm in ARMS:
        payload = json.loads((CLOSURE / "predictions" / f"{arm}.json").read_text())["frames"]
        preds = {k: v for k, v in payload.items()
                 if v.get("status") == "OK" and v.get("keypoints_xy")}
        rows = []
        for fid, truth in gt.items():
            pred = preds.get(fid)
            if not pred:
                continue
            fr = frames[fid]
            raw = json.loads((REPO / fr["annotation"]).read_text())["camera_data"]["intrinsics"]
            cam = np.array([[raw["fx"], 0, raw["cx"]], [0, raw["fy"], raw["cy"]], [0, 0, 1]], float)
            spec = object_spec(contract, fr["object_type"])
            L, S, H = spec["long_m"], spec["short_m"], spec["height_m"]
            models = {CF_WIDTH: cuboid(L, H, S), CF_DEPTH: cuboid(S, H, L)}
            pts = np.asarray(pred["keypoints_xy"], float)[:8]
            usable = np.isfinite(pts).all(axis=1)
            if usable.sum() < 6:
                continue
            chosen = None
            try:
                res = select_pnp_hypotheses(np.asarray(pred["keypoints_xy"], float), cam,
                                            {"x": L, "y": H, "z": S}, None)
                for hyp in res.hypotheses:
                    if hyp.name == res.selected_hypothesis and hyp.success:
                        d = hyp.camera_facing_dimensions.as_dict()
                        chosen = CF_WIDTH if abs(float(d["width"]) - L) < 1e-6 else CF_DEPTH
            except Exception:
                chosen = None
            fits = {k: solve(m, pts, cam, usable) for k, m in models.items()}
            if any(v is None for v in fits.values()) or chosen is None:
                continue
            gR = np.asarray(truth["R_gt_representative"], float)
            gt_t = np.asarray(truth["t_gt"], float)
            dm = truth["physical_dimensions_m"]
            ext = (dm["across"], dm["height"], dm["along"])
            mp = cuboid_model_points(ext)
            R, t = fits[chosen]
            parts = translation_components_m(t, gt_t)
            # 2D corner: GT pose 를 투영한 것과 예측 코너의 거리 (평가와 독립적인 2D 층)
            X = cuboid(*ext)
            P = X @ gR.T + gt_t
            u = np.stack([cam[0, 0] * P[:, 0] / P[:, 2] + cam[0, 2],
                          cam[1, 1] * P[:, 1] / P[:, 2] + cam[1, 2]], 1)
            rows.append({
                "frame_id": fid, "session_id": fr["session_id"],
                "object_type": fr["object_type"],
                "axis_correct": chosen == truth["physical_long_axis"],
                "rotation_error_deg": rotation_error_degrees(R, gR),
                "yaw_error_deg": yaw_error_degrees(R, gR),
                "translation_error_cm": parts["total_m"] * 100,
                "depth_error_cm": parts["depth_m"] * 100,
                "lateral_error_cm": parts["lateral_m"] * 100,
                "corner2d_px": float(np.median(np.linalg.norm(pts[usable] - u[usable], axis=1))),
                "iou3d": oriented_iou_3d(R, t, ext, gR, gt_t, ext),
                "add_sym_m": symmetry_aware_add_m(mp, R, t, gR, gt_t),
                "diameter_m": model_diameter_m(mp),
            })
        per_arm[arm] = rows
        print(f"  {arm:14s} {len(rows):4d} frames")

    # ---- 자기검증: ALL 이 정본 파일과 일치해야 한다
    bad = []
    for arm in ARMS:
        rec, ex = summarize(per_arm[arm]), json.loads(
            (CLOSURE / f"POSE_EVALUATION_{arm}.json").read_text())["paths"]["MAIN"]["ALL"]
        for k in ("n", "axis_accuracy", "translation_median_cm", "depth_median_cm",
                  "rotation_median_deg", "yaw_median_deg", "iou3d_median"):
            if abs(float(rec[k]) - float(ex[k])) > TOL:
                bad.append(f"{arm}.{k}: {rec[k]} != {ex[k]}")
    if bad:
        print("ALL 재현 실패 — 분석을 신뢰할 수 없다:", *bad[:8], sep="\n  ")
        return 1
    print("ALL 재현 확인 — 정본 POSE_EVALUATION_<ARM>.json 과 일치")

    # ---- 조건층 (정의를 새로 만들지 않는다)
    from eval_workspace import evaluation_population_views, load_frames
    from evaluate_arms import SUBGROUPS
    meta = {r["frame_id"]: r for r in evaluation_population_views(
        load_frames(REPO / "data/evaluation/pallet_eval_v1"))["PAPER_EVAL_POSITIVE"]}
    WANT = ["ALL", "Clean", "Occlusion", "Far", "Low", "Mid", "High"]

    def strata(rows):
        out = {"ALL": summarize(rows)}
        for name in WANT[1:]:
            pred = SUBGROUPS[name]
            out[name] = summarize([r for r in rows if meta.get(r["frame_id"]) and pred(meta[r["frame_id"]])])
        for lab, key in (("plastic", "plastic"), ("wood", "wood")):
            out[lab] = summarize([r for r in rows if r["object_type"].startswith(key)])
        return out

    report = {"schema_version": "a2b_screen_analysis_v1", "new_inference": 0,
              "new_training": 0, "all_reproduced_exactly": True,
              "by_arm": {a: strata(per_arm[a]) for a in ARMS}}

    # ---- paired: 세 arm 모두 예측이 있는 프레임에서만
    idx = {a: {r["frame_id"]: r for r in per_arm[a]} for a in ARMS}
    common = sorted(set(idx["LAD_C0"]) & set(idx["LAD_D1"]))
    d = {k: np.array([idx["LAD_D1"][f][k] - idx["LAD_C0"][f][k] for f in common])
         for k in ("translation_error_cm", "depth_error_cm", "rotation_error_deg", "corner2d_px")}
    report["paired_D1_minus_C0"] = {
        "n_common": len(common),
        **{k: {"median": float(np.median(v)), "mean": float(v.mean()),
               "frac_improved": float((v < 0).mean())} for k, v in d.items()}}
    sess = sorted({idx["LAD_C0"][f]["session_id"] for f in common})
    report["paired_by_session"] = {
        s: {"n": int(sum(idx["LAD_C0"][f]["session_id"] == s for f in common)),
            "delta_t_median_cm": float(np.median([idx["LAD_D1"][f]["translation_error_cm"]
                                                  - idx["LAD_C0"][f]["translation_error_cm"]
                                                  for f in common
                                                  if idx["LAD_C0"][f]["session_id"] == s])),
            "delta_depth_median_cm": float(np.median([idx["LAD_D1"][f]["depth_error_cm"]
                                                      - idx["LAD_C0"][f]["depth_error_cm"]
                                                      for f in common
                                                      if idx["LAD_C0"][f]["session_id"] == s]))}
        for s in sess}

    (OUT / "PER_FRAME.json").write_text(json.dumps(per_arm, indent=1, sort_keys=True) + "\n")
    (OUT / "SCREEN_ANALYSIS.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {(OUT / 'SCREEN_ANALYSIS.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
