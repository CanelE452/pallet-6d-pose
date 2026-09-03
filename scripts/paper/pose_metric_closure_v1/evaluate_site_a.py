"""SITE_A 의 held-out recording 에서 기존 arm 들을 비교한다.  새 학습 없음.

    python3 scripts/paper/pose_metric_closure_v1/evaluate_site_a.py

출력  SITE_A_ARM_EVALUATION.json
      _docs/paper/pose_metric_closure_v1/SITE_A_ARM_EVALUATION.md

모집단은 `SITE_A_EVAL_ELIGIBLE.csv` 다 — `SITE_A_EVAL_POSITIVE.csv` 100 장에서
평가 워크스페이스가 `FT_OVERLAP` 으로 표시한 12 장을 뺀 88 장.  그 12 장은
fine-tuning 데이터와 겹치므로 채점하면 train-on-test 가 된다.  이 정정은
**어떤 모델 결과도 보기 전에** 이뤄졌고 근거는 `SITE_A_EVAL_POPULATION_CORRECTION.json`.

selector · GT · metric 은 frozen pose contract 그대로.  새 학습 0, 새 checkpoint
선택 0.  A8 만 캐시가 없어 같은 frozen recipe 로 한 번 추론했다.

recording-cluster bootstrap 은 8 개가 아니라 실제 남은 recording 수로 돈다.
클러스터가 적으므로 구간이 0 을 포함해도 "성능이 같다" 가 아니라
"이 데이터로는 못 가른다" 로 읽어야 한다.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT))

P = REPO_ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
S = REPO_ROOT / "data/pallet/results/site_environment_audit_v1"
DOC = REPO_ROOT / "_docs/paper/pose_metric_closure_v1"

ARMS = ["R0", "R0_CONT", "R1_NAIVE", "R5_PROPOSED", "A8_DAY_ONLY"]
LABELS = {"R0": "R0 synthetic-only", "R0_CONT": "R0-CONT replay only",
          "R1_NAIVE": "R1 naive joint", "R5_PROPOSED": "R5 proposed joint",
          "A8_DAY_ONLY": "A8 day-only site-matched"}
CONTRASTS = [("A8_DAY_ONLY", "R5_PROPOSED"), ("A8_DAY_ONLY", "R0"),
             ("A8_DAY_ONLY", "R0_CONT")]
CF_WIDTH, CF_DEPTH = "CF_WIDTH", "CF_DEPTH"
N_RESAMPLES = 10000
SEED = 20260903


def cuboid(across, height, along):
    ha, hh, hb = across / 2.0, height / 2.0, along / 2.0
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
    return rotation, tvec.reshape(-1)


def auc(errors, diameter):
    thresholds = np.linspace(0.0, 0.1 * diameter, 1001)
    accuracy = (errors[None, :] <= thresholds[:, None]).mean(axis=1)
    return float(np.trapz(accuracy, thresholds) / (0.1 * diameter))


def main() -> int:
    from pose_evaluation_paths import load_pose_object_contract, object_spec
    from symmetry_aware_pose_metrics import (cuboid_model_points, model_diameter_m,
                                             rotation_error_degrees, symmetry_aware_add_m,
                                             translation_components_m, yaw_error_degrees)
    from challenge.evaluation_v2.oriented_iou3d import oriented_iou_3d
    from challenge.evaluation_v2.pnp_selector import select_pnp_hypotheses

    contract = load_pose_object_contract(str(P / "POSE_EVAL_OBJECT_CONTRACT.json"))
    gt = json.loads((P / "GEOMETRY_RESOLVED_POSE_GT.json").read_text())["frames"]
    frames = {f["frame_id"]: f
              for f in json.loads((P / "AXIS_REVIEW_MANIFEST.json").read_text())["frames_list"]}
    population = list(csv.DictReader((S / "SITE_A_EVAL_ELIGIBLE.csv").open()))
    recording_of = {r["frame_id"]: r["source_recording"] for r in population}
    wanted = [r["frame_id"] for r in population]
    missing_gt = [f for f in wanted if f not in gt]
    if missing_gt:
        print(f"frozen GT 에 없는 프레임 {len(missing_gt)} — 중단")
        return 1

    per_arm: dict[str, list[dict]] = {}
    for arm in ARMS:
        payload = json.loads((P / f"predictions/{arm}.json").read_text())["frames"]
        rows, detected = [], 0
        for frame_id in wanted:
            pred = payload.get(frame_id)
            if not pred or pred.get("status") != "OK" or not pred.get("keypoints_xy"):
                continue
            detected += 1
            frame = frames[frame_id]
            annotation = json.loads((REPO_ROOT / frame["annotation"]).read_text())
            raw = annotation["camera_data"]["intrinsics"]
            camera = np.array([[raw["fx"], 0.0, raw["cx"]],
                               [0.0, raw["fy"], raw["cy"]], [0.0, 0.0, 1.0]], np.float64)
            spec = object_spec(contract, frame["object_type"])
            long_m, short_m, height_m = spec["long_m"], spec["short_m"], spec["height_m"]

            # 2D keypoint 오차 — frozen metric contract 의 keypoint layer
            gt_kp = np.array([p if p else [np.nan, np.nan]
                              for p in frame["keypoints_xy"]], np.float64)
            pred_kp = np.asarray(pred["keypoints_xy"], np.float64)
            n = min(len(gt_kp), len(pred_kp))
            visible = np.isfinite(gt_kp[:n]).all(axis=1)
            kp_px = (float(np.median(np.linalg.norm(
                pred_kp[:n][visible] - gt_kp[:n][visible], axis=1)))
                if visible.any() else float("nan"))

            points = pred_kp[:8]
            usable = np.isfinite(points).all(axis=1)
            if usable.sum() < 6:
                rows.append({"frame_id": frame_id, "pose": False, "kp_px": kp_px,
                             "recording": recording_of[frame_id]})
                continue

            chosen = None
            try:
                result = select_pnp_hypotheses(pred_kp, camera,
                                               {"x": long_m, "y": height_m, "z": short_m}, None)
                for hypothesis in result.hypotheses:
                    if hypothesis.name == result.selected_hypothesis and hypothesis.success:
                        dims = hypothesis.camera_facing_dimensions.as_dict()
                        chosen = (CF_WIDTH if abs(float(dims["width"]) - long_m) < 1e-6
                                  else CF_DEPTH)
            except Exception:
                chosen = None
            models = {CF_WIDTH: cuboid(long_m, height_m, short_m),
                      CF_DEPTH: cuboid(short_m, height_m, long_m)}
            fit = solve(models[chosen], points, camera, usable) if chosen else None
            if fit is None:
                rows.append({"frame_id": frame_id, "pose": False, "kp_px": kp_px,
                             "recording": recording_of[frame_id]})
                continue

            truth = gt[frame_id]
            gt_R = np.asarray(truth["R_gt_representative"], np.float64)
            gt_t = np.asarray(truth["t_gt"], np.float64)
            dims = truth["physical_dimensions_m"]
            extents = (dims["across"], dims["height"], dims["along"])
            model_points = cuboid_model_points(extents)
            rotation, translation = fit
            parts = translation_components_m(translation, gt_t)
            rows.append({
                "frame_id": frame_id, "pose": True, "kp_px": kp_px,
                "recording": recording_of[frame_id],
                "axis_correct": chosen == truth["physical_long_axis"],
                "rotation_error_deg": rotation_error_degrees(rotation, gt_R),
                "yaw_error_deg": yaw_error_degrees(rotation, gt_R),
                "translation_error_cm": parts["total_m"] * 100.0,
                "iou3d": oriented_iou_3d(rotation, translation, extents, gt_R, gt_t, extents),
                "add_sym_m": symmetry_aware_add_m(model_points, rotation, translation,
                                                  gt_R, gt_t),
                "diameter_m": model_diameter_m(model_points),
            })
        per_arm[arm] = rows
        print(f"  {arm:16} detected {detected}/{len(wanted)}  pose "
              f"{sum(1 for r in rows if r['pose'])}")

    def summarize(rows, total):
        posed = [r for r in rows if r["pose"]]
        out = {"n_frames": total, "detection_coverage": len(rows) / total,
               "kp_median_px": float(np.nanmedian([r["kp_px"] for r in rows])) if rows else None,
               "pose_coverage": len(posed) / total, "n_pose": len(posed)}
        if posed:
            arr = lambda k: np.array([r[k] for r in posed], float)
            out |= {
                "axis_accuracy": float(np.mean([r["axis_correct"] for r in posed])),
                "rotation_median_deg": float(np.median(arr("rotation_error_deg"))),
                "yaw_median_deg": float(np.median(arr("yaw_error_deg"))),
                "translation_median_cm": float(np.median(arr("translation_error_cm"))),
                "iou3d_median": float(np.median(arr("iou3d"))),
                "add_sym_auc": auc(arr("add_sym_m"), float(np.median(arr("diameter_m")))),
            }
        return out

    total = len(wanted)
    summaries = {arm: summarize(per_arm[arm], total) for arm in ARMS}

    # ── recording-cluster paired bootstrap
    rng = np.random.default_rng(SEED)
    contrasts = []
    for arm_a, arm_b in CONTRASTS:
        a = {r["frame_id"]: r for r in per_arm[arm_a] if r["pose"]}
        b = {r["frame_id"]: r for r in per_arm[arm_b] if r["pose"]}
        shared = sorted(set(a) & set(b))
        recordings = sorted({recording_of[f] for f in shared})
        index = {rec: np.array([i for i, f in enumerate(shared)
                                if recording_of[f] == rec]) for rec in recordings}
        entry = {"arm": arm_a, "reference": arm_b, "paired_frames": len(shared),
                 "clusters": len(recordings), "metrics": {}}
        for key, kind in (("iou3d", "median"), ("add_sym_m", "auc"),
                          ("yaw_error_deg", "median"), ("translation_error_cm", "median")):
            va = np.array([a[f][key] for f in shared], float)
            vb = np.array([b[f][key] for f in shared], float)
            da = np.array([a[f]["diameter_m"] for f in shared], float)

            def stat(values, diameters):
                return (float(np.median(values)) if kind == "median"
                        else auc(values, float(np.median(diameters))))

            observed = stat(va, da) - stat(vb, da)
            draws = np.empty(N_RESAMPLES)
            for i in range(N_RESAMPLES):
                pick = rng.integers(0, len(recordings), len(recordings))
                idx = np.concatenate([index[recordings[c]] for c in pick])
                draws[i] = stat(va[idx], da[idx]) - stat(vb[idx], da[idx])
            low, high = np.percentile(draws, [2.5, 97.5])
            entry["metrics"][key] = {
                "observed_difference": observed,
                "cluster_ci_low": float(low), "cluster_ci_high": float(high),
                "excludes_zero": bool(low > 0 or high < 0),
            }
        contrasts.append(entry)
        print(f"  {arm_a} vs {arm_b}  paired {len(shared)}  clusters {len(recordings)}")

    report = {
        "schema_version": "site_a_arm_evaluation_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "population": "SITE_A_EVAL_ELIGIBLE.csv",
        "population_note": ("SITE_A_EVAL_POSITIVE.csv had 100 rows; 12 are marked "
                            "FT_OVERLAP by the evaluation workspace and were removed "
                            "before any result was read"),
        "n_frames": total,
        "new_training": 0, "new_checkpoint_selection": 0, "metric_change": 0,
        "new_inference": "A8_DAY_ONLY only, under the frozen replay recipe",
        "summaries": summaries,
        "contrasts": contrasts,
        "cluster_caveat": ("few recording clusters; an interval containing zero means "
                           "the comparison is not resolved by this data, not that the "
                           "arms are equal"),
    }
    (P / "SITE_A_ARM_EVALUATION.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = ["# SITE_A — existing arms on the site's held-out recordings", "",
             f"Population: **{total} frames** from SITE_A recordings that supply no",
             "adaptation image. Built from `SITE_A_EVAL_ELIGIBLE.csv`; the twelve",
             "`FT_OVERLAP` frames in the earlier 100-row file were removed before any",
             "model result was read, because they overlap fine-tuning data.", "",
             "No model was trained. Only A8 was inferred, under the frozen replay recipe.",
             "", "```text",
             f"{'Method':28}{'Det':>7}{'KpPx':>8}{'PoseCov':>9}{'R':>7}{'Yaw':>7}"
             f"{'tcm':>8}{'IoU3D':>8}{'ADDsym':>9}",
             "─" * 91]
    for arm in ARMS:
        s = summaries[arm]
        lines.append(
            f"{LABELS[arm]:28}{s['detection_coverage']:7.3f}{s['kp_median_px']:8.2f}"
            f"{s['pose_coverage']:9.3f}{s['rotation_median_deg']:7.2f}"
            f"{s['yaw_median_deg']:7.2f}{s['translation_median_cm']:8.2f}"
            f"{s['iou3d_median']:8.3f}{s['add_sym_auc']:9.3f}")
    lines += ["```", "", "## Pre-registered contrasts", "", "```text",
              f"{'contrast':28}{'metric':16}{'diff':>9}{'recording-cluster 95% CI':>28}",
              "─" * 81]
    for entry in contrasts:
        for key, block in entry["metrics"].items():
            ci = "[{:+.3f}, {:+.3f}]".format(block["cluster_ci_low"], block["cluster_ci_high"])
            lines.append(f"{entry['arm'] + ' - ' + entry['reference']:28}{key:16}"
                         f"{block['observed_difference']:+9.3f}{ci:>28}")
        lines.append("")
    lines += ["```", "",
              f"Clusters: {contrasts[0]['clusters']} recordings. An interval containing zero",
              "means this data does not resolve the comparison — not that the arms perform",
              "the same."]
    DOC.mkdir(parents=True, exist_ok=True)
    (DOC / "SITE_A_ARM_EVALUATION.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {(P / 'SITE_A_ARM_EVALUATION.json').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
