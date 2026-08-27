"""공통 geometry evaluator — 두 모델 계열에 **같은 계약**을 건다.

이 파일이 생긴 이유는 결함이다.  이전 `mc_score.py` 는 키포인트 신뢰도에
`>=0.5` 를 걸었는데, 그 값의 의미가 계열마다 다르다:

    yolo26n_ft   kp_conf = sigmoid visibility   median 1.000   >=0.5 통과 98.2%
    FINAL40K     kp_conf = raw belief peak      median 0.423   >=0.5 통과 45.3%

같은 임계를 걸면 FINAL40K 만 절반이 PnP 전에 탈락한다.  그래서 여기서는
**신뢰도 필터를 걸지 않는다.**  두 계열 모두 8 코너를 그대로 내고, 기하는 동일한
경로로 푼다.  모델별로 다른 것은 **native preprocessing 하나뿐**이고, 그 뒤로는
전부 같다.

계약 (브리프 §2)
    좌표      각 모델의 native preprocessing -> 원본 픽셀로 복원
    K         프레임 라벨의 intrinsics (동일)
    dims      프레임 라벨의 dimensions_m (동일)
    점        **8 코너만**.  centroid(index 8) 제외 — YOLO 는 실제 대응이지만
              FINAL40K 의 centroid 채널과 의미가 같다고 보장할 수 없다
    최소      4 점
    solver    SOLVEPNP_SQPNP -> solvePnPRefineLM
    population  presence / common-detected / unconditional 분리
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
for sub in ("scripts/stage0/real_eval", "scripts/annotate", "challenge"):
    sys.path.insert(0, os.path.join(ROOT, sub))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2                       # noqa: E402
import re_metrics as RM          # noqa: E402
import annotate_pnp as APNP      # noqa: E402
import mc_frames as MF           # noqa: E402

OUT = os.path.join(ROOT, "data/pallet/results/model_compare")
MIN_POINTS = 4


def gt_of(label):
    obj = label["objects"][0]
    dims = obj["dimensions_m"]
    pose = np.asarray(obj["pose_transform"], float)
    k = label["camera_data"]["intrinsics"]
    return {"R": pose[:3, :3], "t": pose[:3, 3],
            "extents": (dims["width"], dims["height"], dims["depth"]),
            "model": APNP.make_pallet_keypoints_3d_diagram(
                width=dims["width"], depth=dims["depth"],
                height=dims["height"])[:8],
            "gt8": np.asarray(obj["projected_cuboid"], float)[:8],
            "K": np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]],
                           [0, 0, 1.0]], float)}


def solve(points, truth):
    """SQPnP -> refineLM.  두 계열에 대해 글자 그대로 같은 코드가 돈다."""
    ok = np.isfinite(points).all(1)
    if int(ok.sum()) < MIN_POINTS:
        return None
    object_pts = truth["model"][ok].astype(np.float64)
    image_pts = points[ok].astype(np.float64)
    good, rvec, tvec = cv2.solvePnP(object_pts, image_pts, truth["K"], None,
                                    flags=cv2.SOLVEPNP_SQPNP)
    if not good:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(object_pts, image_pts, truth["K"], None,
                                      rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)


def metrics(points, truth):
    row = {"n_points": int(np.isfinite(points).all(1).sum())}
    ok = np.isfinite(points).all(1)
    if ok.any():
        d = np.linalg.norm(points[ok] - truth["gt8"][ok], axis=1)
        row["corner_med"] = float(np.median(d))
    else:
        row["corner_med"] = np.nan
    pose = solve(points, truth)
    row.update({"pnp_ok": 0, "R": np.nan, "t": np.nan, "add": np.nan,
                "adds": np.nan, "iou": np.nan, "5cm5": 0})
    if pose is not None:
        R_p, t_p = pose
        deg, met = RM.pose_error(R_p, t_p, truth["R"], truth["t"])
        row.update({"pnp_ok": 1, "R": deg, "t": met,
                    "add": RM.add(truth["model"], R_p, t_p, truth["R"], truth["t"]),
                    "adds": RM.add_s(truth["model"], R_p, t_p, truth["R"], truth["t"]),
                    "iou": RM.iou_3d(R_p, t_p, truth["extents"], truth["R"],
                                     truth["t"], truth["extents"]),
                    "5cm5": int(RM.success_5cm5deg(R_p, t_p, truth["R"], truth["t"]))})
    return row


def points_of(entry, model):
    """각 계열의 native 출력 -> 원본 픽셀 8x2.  신뢰도 필터 없음."""
    if model.startswith("FINAL40K"):
        # threshold-free argmax peaks: 두 계열 모두 '항상 8점' 이 되게 맞춘다
        return np.asarray(entry["kps_argmax"], float)[:8]
    if entry["kps"] is None:
        return np.full((8, 2), np.nan)
    return np.asarray(entry["kps"], float)[:8]


def presence_of(entry, model):
    """모델이 '물체가 있다' 고 말했는가 — 계열마다 다른 양이므로 분리 보고."""
    if model.startswith("FINAL40K"):
        return {"score": entry.get("score_4kp"),
                "criterion": "score_4kp (threshold UNSET_PENDING_REAL_DEV)",
                "declared": None}
    return {"score": entry.get("box_conf"),
            "criterion": "box conf >= 0.4 (release deployment contract)",
            "declared": entry["kps"] is not None}


def stats(rows, field):
    v = np.array([r[field] for r in rows], float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    return {"n": int(v.size), "median": round(float(np.median(v)), 4),
            "mean": round(float(v.mean()), 4),
            "p90": round(float(np.percentile(v, 90)), 4)}


def summarise(rows, denominator=None):
    total = denominator if denominator is not None else len(rows)
    if not rows:
        return None
    return {"n": len(rows), "denominator": total,
            "pnp_rate": round(sum(r["pnp_ok"] for r in rows) / max(total, 1), 4),
            "corner_px": stats(rows, "corner_med"),
            "R_deg": stats(rows, "R"), "t_m": stats(rows, "t"),
            "ADD": stats(rows, "add"), "ADD_S": stats(rows, "adds"),
            "IoU3D": stats(rows, "iou"),
            "success_5cm5deg": round(sum(r["5cm5"] for r in rows) / max(total, 1), 4)}


def main(models=("yolo26n_synth", "yolo26n_ft", "yolo26m_ft", "FINAL40K_seed1")):
    frame_rows = MF.frames()
    if len(frame_rows) != 161:
        raise RuntimeError(
            "HISTORICAL_DEV161_EVALUATOR_DISABLED_AFTER_GT_QA: "
            f"mc_geom.py requires its frozen 161-frame population, but mc_frames "
            f"now exposes the clean {len(frame_rows)}-frame population. Use "
            "challenge/evaluation_v2/paper_real_eval.py for current paper evaluation; "
            "otherwise this script would write misleading OPEN_56/DEV_105/ALL_161 keys."
        )
    truths = {}
    for key, sealed, jp, ip, label in frame_rows:
        truths[os.path.splitext(os.path.basename(jp))[0]] = (key, sealed,
                                                             gt_of(label))
    per_model, presence = {}, {}
    for name in models:
        path = os.path.join(OUT, f"kps_{name}.json")
        if not os.path.exists(path):
            continue
        dump = json.load(open(path))
        rows = {}
        for entry in dump["frames"]:
            fid = entry["fid"]
            key, sealed, truth = truths[fid]
            row = metrics(points_of(entry, name), truth)
            row.update({"set": key, "sealed": sealed, "fid": fid})
            rows[fid] = row
            presence.setdefault(name, {})[fid] = presence_of(entry, name)
        per_model[name] = rows

    # common-detected = 모든 모델이 PnP 를 푼 프레임.  같은 프레임에서만 비교.
    common = [fid for fid in truths
              if all(per_model[m][fid]["pnp_ok"] for m in per_model)]

    report = {
        "contract": {
            "points": "8 corners only (centroid excluded)",
            "min_points": MIN_POINTS,
            "solver": "SOLVEPNP_SQPNP -> solvePnPRefineLM",
            "confidence_filter": "NONE — 계열 간 신뢰도 의미가 달라 임계를 걸면 "
                                 "한쪽만 탈락한다 (yolo median 1.000 vs FINAL 0.423)",
            "K": "frame label intrinsics (identical)",
            "dims": "frame label dimensions_m (identical)",
            "per_model_difference": "native preprocessing only"},
        "fix_note": "이전 mc_score.py 는 kp conf>=0.5 를 두 계열에 똑같이 걸어 "
                    "FINAL40K 키포인트의 55%를 PnP 전에 버렸다. 그 표는 무효다.",
        "populations": {
            "unconditional": "전 프레임 분모. 못 풀면 실패로 센다",
            "common_detected": f"모든 모델이 PnP 를 푼 {len(common)} 프레임",
            "presence": "모델이 '있다' 고 말했는가 — 계열마다 다른 양이라 분리"},
        "n_common_detected": len(common),
        "models": {}}

    for name, rows in per_model.items():
        allr = list(rows.values())
        openr = [r for r in allr if not r["sealed"]]
        sealr = [r for r in allr if r["sealed"]]
        commonr = [rows[f] for f in common]
        report["models"][name] = {
            "OPEN_56": summarise(openr),
            "REAL_CHALLENGE_DEV_105": summarise(sealr),
            "ALL_161": summarise(allr),
            "COMMON_DETECTED": summarise(commonr),
            "presence_criterion": next(iter(presence[name].values()))["criterion"],
        }
        s = report["models"][name]
        print(f"  {name:16} open pnp {s['OPEN_56']['pnp_rate']:.3f} "
              f"5cm5 {s['OPEN_56']['success_5cm5deg']:.3f} | "
              f"dev105 pnp {s['REAL_CHALLENGE_DEV_105']['pnp_rate']:.3f} "
              f"5cm5 {s['REAL_CHALLENGE_DEV_105']['success_5cm5deg']:.3f} | "
              f"common R {s['COMMON_DETECTED']['R_deg']['median']}", flush=True)

    json.dump(report, open(os.path.join(OUT, "MODEL_COMPARE_GEOM.json"), "w"),
              indent=1, default=str)
    print(f"  common-detected {len(common)}/161")
    print("-> MODEL_COMPARE_GEOM.json")


if __name__ == "__main__":
    main()
