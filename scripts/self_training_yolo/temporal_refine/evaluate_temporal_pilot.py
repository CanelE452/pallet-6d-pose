"""TEMPORAL PILOT — refinement 가 실제로 GT 에 가까워졌는지 평가한다.

    python3 scripts/self_training_yolo/temporal_refine/evaluate_temporal_pilot.py \
        --output-dir data/pallet/results/paper_temporal_selftrain_v1/pilot

출력  TEMPORAL_PILOT_EVALUATION.json · TEMPORAL_PILOT_REPORT.md

정답은 **여기서 처음** 열린다.  refinement 는 이미 파일로 저장돼 있고 이 파일은
그것을 바꾸지 않는다.

metric 정의는 전부 기존 것 재사용 — 2D 는 원본 좌표계 유클리드 오차, 6D 는
pose closure 의 대칭 인지 ADD·정확 oriented IoU·고정 AUC 구간.
새 정의를 만들지 않는다.  결과를 보고 임계값을 만들지 않는다.
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
WORKSPACE = REPO_ROOT / "data/evaluation/pallet_eval_v1"
sys.path.insert(0, str(REPO_ROOT / "scripts/paper/pose_metric_closure_v1"))
sys.path.insert(0, str(REPO_ROOT))

METHODS = ["RAW_TEACHER", "TEMPORAL_ONLY", "TEMPORAL_GEOMETRY"]
GROSS_PX = 20.0
N_RESAMPLES = 10000
SEED = 20260903


def cuboid(across, height, along):
    ha, hh, hb = across / 2, height / 2, along / 2
    return np.array([[-ha, -hh, -hb], [+ha, -hh, -hb], [+ha, +hh, -hb], [-ha, +hh, -hb],
                     [-ha, -hh, +hb], [+ha, -hh, +hb], [+ha, +hh, +hb], [-ha, +hh, +hb]],
                    dtype=np.float64)


def solve(model, points, camera, usable):
    ok, rvec, tvec = cv2.solvePnP(model[usable], points[usable], camera, None,
                                  flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(model[usable], points[usable], camera, None,
                                      rvec, tvec)
    rotation, _ = cv2.Rodrigues(rvec)
    projected, _ = cv2.projectPoints(model, rvec, tvec, camera, None)
    residual = float(np.median(np.linalg.norm(
        projected.reshape(-1, 2)[usable] - points[usable], axis=1)))
    return rotation, tvec.reshape(-1), residual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()

    from symmetry_aware_pose_metrics import (cuboid_model_points, model_diameter_m,
                                             pose_auc, rotation_error_degrees,
                                             symmetry_aware_add_m,
                                             translation_components_m, yaw_error_degrees)
    from challenge.evaluation_v2.oriented_iou3d import oriented_iou_3d

    lock = json.loads((out_dir.parent / "TEMPORAL_METHOD_LOCK.json").read_text())
    allowed = lock["verdict_axes"]
    dims = lock["object_geometry"]["dimensions_m"]
    long_m, short_m, height_m = max(dims["x"], dims["z"]), min(dims["x"], dims["z"]), dims["y"]

    refinement = json.loads(
        (out_dir / "TEMPORAL_REFINEMENT_PER_FRAME.json").read_text())["frames"]
    print(f"centres with refinement {len(refinement)}")

    rows = []
    for frame_id, entry in refinement.items():
        if entry["status"] != "OK":
            continue
        annotation = WORKSPACE / entry["gt_annotation_path"]        # 여기서 처음 GT
        if not annotation.exists():
            continue
        payload = json.loads(annotation.read_text())
        obj = payload["objects"][0]
        gt = np.array([p if p else [np.nan, np.nan]
                       for p in obj["projected_cuboid"]], np.float64)[:8]
        supervised = np.isfinite(gt).all(axis=1)
        if supervised.sum() < 6:
            continue
        raw_intrinsics = payload["camera_data"]["intrinsics"]
        camera = np.array([[raw_intrinsics["fx"], 0, raw_intrinsics["cx"]],
                           [0, raw_intrinsics["fy"], raw_intrinsics["cy"]],
                           [0, 0, 1]], np.float64)

        # GT 기준 자세 — 수동 좌표 + 등록 치수, 모델 예측 미사용
        gt_pose = None
        for across, along in ((long_m, short_m), (short_m, long_m)):
            model = cuboid(across, height_m, along)
            fit = solve(model, gt, camera, supervised)
            if fit and (gt_pose is None or fit[2] < gt_pose[0][2]):
                gt_pose = (fit, (across, height_m, along))
        if gt_pose is None:
            continue
        (gt_R, gt_t, _), gt_extents = gt_pose
        model_points = cuboid_model_points(gt_extents)
        diameter = model_diameter_m(model_points)

        candidates = {
            "RAW_TEACHER": np.asarray(entry["raw_teacher"], np.float64),
            "TEMPORAL_ONLY": np.asarray(entry["temporal_only"], np.float64),
            "TEMPORAL_GEOMETRY": (np.asarray(entry["temporal_geometry"], np.float64)
                                  if entry["temporal_geometry"] else None),
        }
        record = {"frame_id": frame_id, "source_recording": entry["source_recording"]}
        for name, points in candidates.items():
            if points is None:
                record[name] = None
                continue
            valid = supervised & np.isfinite(points).all(axis=1)
            if valid.sum() < 6:
                record[name] = None
                continue
            errors = np.linalg.norm(points[valid] - gt[valid], axis=1)
            block = {"n_corners": int(valid.sum()),
                     "median_px": float(np.median(errors)),
                     "p90_px": float(np.percentile(errors, 90)),
                     "gross20": float(np.mean(errors > GROSS_PX))}
            best = None
            for across, along in ((long_m, short_m), (short_m, long_m)):
                fit = solve(cuboid(across, height_m, along), points, camera, valid)
                if fit and (best is None or fit[2] < best[0][2]):
                    best = (fit, (across, height_m, along))
            if best is not None:
                (rotation, translation, _), extents = best
                parts = translation_components_m(translation, gt_t)
                block |= {
                    "pose": True,
                    "rotation_deg": rotation_error_degrees(rotation, gt_R),
                    "yaw_deg": yaw_error_degrees(rotation, gt_R),
                    "translation_cm": parts["total_m"] * 100.0,
                    "iou3d": oriented_iou_3d(rotation, translation, extents,
                                             gt_R, gt_t, gt_extents),
                    "add_sym_m": symmetry_aware_add_m(model_points, rotation, translation,
                                                      gt_R, gt_t),
                    "diameter_m": diameter,
                }
            else:
                block["pose"] = False
            record[name] = block
        rows.append(record)

    print(f"evaluated {len(rows)}")

    def summarize(name, subset):
        blocks = [r[name] for r in subset if r.get(name)]
        posed = [b for b in blocks if b.get("pose")]
        out = {"frames": len(subset), "coverage": len(blocks) / len(subset) if subset else 0,
               "n": len(blocks)}
        if blocks:
            out |= {
                "median_px": float(np.median([b["median_px"] for b in blocks])),
                "p90_px": float(np.median([b["p90_px"] for b in blocks])),
                "gross20": float(np.mean([b["gross20"] for b in blocks])),
            }
        if posed:
            arr = lambda k: np.array([b[k] for b in posed], float)
            out |= {
                "pose_coverage": len(posed) / len(subset),
                "rotation_median_deg": float(np.median(arr("rotation_deg"))),
                "yaw_median_deg": float(np.median(arr("yaw_deg"))),
                "translation_median_cm": float(np.median(arr("translation_cm"))),
                "iou3d_median": float(np.median(arr("iou3d"))),
                "add_sym_auc": pose_auc(arr("add_sym_m"), float(np.median(arr("diameter_m")))),
            }
        return out

    summary = {m: summarize(m, rows) for m in METHODS}

    # ── paired bootstrap: 공통 프레임에서만, frame 과 recording-cluster 둘 다
    rng = np.random.default_rng(SEED)

    def paired(metric_a, metric_b, key, kind):
        common = [r for r in rows if r.get(metric_a) and r.get(metric_b)
                  and (kind != "auc" or (r[metric_a].get("pose") and r[metric_b].get("pose")))]
        if kind in ("median", "mean") and key in ("iou3d",):
            common = [r for r in common
                      if r[metric_a].get("pose") and r[metric_b].get("pose")]
        if not common:
            return None
        recordings = sorted({r["source_recording"] for r in common})
        index = {rec: np.array([i for i, r in enumerate(common)
                                if r["source_recording"] == rec]) for rec in recordings}

        def statistic(subset):
            a = np.array([subset[i][metric_a][key] for i in range(len(subset))], float) \
                if False else np.array([r[metric_a][key] for r in subset], float)
            b = np.array([r[metric_b][key] for r in subset], float)
            if kind == "auc":
                d = np.array([r[metric_a]["diameter_m"] for r in subset], float)
                return (pose_auc(a, float(np.median(d))) - pose_auc(b, float(np.median(d))))
            if kind == "mean":
                return float(np.mean(a) - np.mean(b))
            return float(np.median(a) - np.median(b))

        observed = statistic(common)
        frame_draws, cluster_draws = np.empty(N_RESAMPLES), np.empty(N_RESAMPLES)
        n = len(common)
        for i in range(N_RESAMPLES):
            pick = rng.integers(0, n, n)
            frame_draws[i] = statistic([common[p] for p in pick])
            chosen = rng.integers(0, len(recordings), len(recordings))
            idx = np.concatenate([index[recordings[c]] for c in chosen])
            cluster_draws[i] = statistic([common[p] for p in idx])

        def interval(draws):
            low, high = np.percentile(draws, [2.5, 97.5])
            return {"low": float(low), "high": float(high),
                    "excludes_zero": bool(low > 0 or high < 0)}

        return {"n_common": n, "clusters": len(recordings),
                "observed_difference": observed,
                "frame_ci": interval(frame_draws),
                "cluster_ci": interval(cluster_draws)}

    contrasts = {}
    for a, b in (("TEMPORAL_ONLY", "RAW_TEACHER"),
                 ("TEMPORAL_GEOMETRY", "RAW_TEACHER"),
                 ("TEMPORAL_GEOMETRY", "TEMPORAL_ONLY")):
        contrasts[f"{a} - {b}"] = {
            "median_px": paired(a, b, "median_px", "median"),
            "p90_px": paired(a, b, "p90_px", "median"),
            "gross20": paired(a, b, "gross20", "mean"),
            "iou3d": paired(a, b, "iou3d", "median"),
            "add_sym_auc": paired(a, b, "add_sym_m", "auc"),
        }
        print(f"  {a} - {b}: n {contrasts[f'{a} - {b}']['p90_px']['n_common']}  "
              f"clusters {contrasts[f'{a} - {b}']['p90_px']['clusters']}")

    def verdict_2d(key):
        c = contrasts[f"{key} - RAW_TEACHER"]
        better = sum(1 for m in ("median_px", "p90_px", "gross20")
                     if c[m] and c[m]["observed_difference"] < 0)
        resolved = any(c[m] and c[m]["cluster_ci"]["excludes_zero"]
                       for m in ("median_px", "p90_px", "gross20"))
        if better == 3 and resolved:
            return "IMPROVED"
        if better == 0:
            return "NO_IMPROVEMENT"
        return "MIXED"

    c6 = contrasts["TEMPORAL_GEOMETRY - RAW_TEACHER"]
    up = sum(1 for m in ("iou3d", "add_sym_auc")
             if c6[m] and c6[m]["observed_difference"] > 0)
    resolved6 = any(c6[m] and c6[m]["cluster_ci"]["excludes_zero"]
                    for m in ("iou3d", "add_sym_auc"))
    verdicts = {
        "A_TEMPORAL_ONLY_2D": verdict_2d("TEMPORAL_ONLY"),
        "B_TEMPORAL_GEOMETRY_2D": verdict_2d("TEMPORAL_GEOMETRY"),
        "C_DOWNSTREAM_6D": ("IMPROVED" if up == 2 and resolved6
                            else "NO_IMPROVEMENT" if up == 0 else "MIXED"),
        "D_COVERAGE": ("ACCEPTABLE"
                       if summary["TEMPORAL_GEOMETRY"]["coverage"] >= 0.9 else "DEGRADED"),
    }
    for axis, value in verdicts.items():
        assert value in allowed[axis], f"{axis}: {value} not in the frozen vocabulary"

    report = {
        "schema_version": "temporal_pilot_evaluation_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "lock_commit": "3d6c977",
        "ground_truth_read_only_here": True,
        "student_training": 0, "new_checkpoint": 0, "depth": 0,
        "threshold_sweep": 0, "tracklet_sweep": 0, "flow_parameter_sweep": 0,
        "gross_threshold_px": GROSS_PX,
        "evaluated_centres": len(rows),
        "recordings": sorted({r["source_recording"] for r in rows}),
        "summary": summary,
        "contrasts": contrasts,
        "verdicts": verdicts,
        "by_recording": {
            rec: {m: summarize(m, [r for r in rows if r["source_recording"] == rec])
                  for m in METHODS}
            for rec in sorted({r["source_recording"] for r in rows})},
    }
    (out_dir / "TEMPORAL_PILOT_EVALUATION.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"\n{'method':22}{'cov':>7}{'med px':>9}{'p90 px':>9}{'gross20':>10}"
          f"{'IoU3D':>8}{'ADDsym':>9}{'t cm':>8}")
    print("-" * 82)
    for m in METHODS:
        s = summary[m]
        print(f"{m:22}{s['coverage']:7.3f}{s.get('median_px', float('nan')):9.2f}"
              f"{s.get('p90_px', float('nan')):9.2f}{s.get('gross20', float('nan')):10.3f}"
              f"{s.get('iou3d_median', float('nan')):8.3f}"
              f"{s.get('add_sym_auc', float('nan')):9.3f}"
              f"{s.get('translation_median_cm', float('nan')):8.2f}")
    print()
    for name, block in contrasts.items():
        print(f"{name}")
        for metric, c in block.items():
            if not c:
                continue
            print(f"    {metric:14}{c['observed_difference']:+9.3f}   "
                  f"frame [{c['frame_ci']['low']:+.3f}, {c['frame_ci']['high']:+.3f}]   "
                  f"cluster [{c['cluster_ci']['low']:+.3f}, {c['cluster_ci']['high']:+.3f}]"
                  f"{'  excludes 0' if c['cluster_ci']['excludes_zero'] else ''}")
    print()
    for axis, value in verdicts.items():
        print(f"  {axis:26}{value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
