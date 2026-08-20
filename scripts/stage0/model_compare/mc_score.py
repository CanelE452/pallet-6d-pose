"""2단계 — 덤프된 키포인트를 **한 경로**로 채점한다.

네 모델은 서로 다른 환경·전처리에서 추론했지만, 여기서부터는 완전히 같다:
같은 GT, 같은 order-free Hungarian 코너 오차, 같은 3D 모델점, 같은 PnP,
같은 pose 지표.  downstream 이 갈리면 비교가 아니라 두 실험이 된다.

★ SEALED: FINAL_TEST 105 장이 여기 포함된다.  사용자가 2026-08-20 에 봉인 해제를
  명시 승인했고, 이 실행으로 소진된다.  open 56 과 sealed 105 는 **합산하지 않고
  따로** 보고한다 (합치면 어느 쪽 수치도 해석 불가).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = "/home/minjae/Documents/github/pallet-pose"
for sub in ("scripts/stage0/real_eval", "scripts/annotate", "challenge"):
    sys.path.insert(0, os.path.join(ROOT, sub))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2                       # noqa: E402
import re_metrics as RM          # noqa: E402
import annotate_pnp as APNP      # noqa: E402
import mc_frames as MF           # noqa: E402

OUT = os.path.join(ROOT, "data/pallet/results/model_compare")
MODELS = ["yolo26n_synth", "yolo26n_ft", "yolo26m_ft", "FINAL40K_seed1"]
N_DET_MIN = 6
KP_CONF_MIN = 0.5      # release 계약: PnP 에는 conf>=0.5 키포인트만


def hungarian(pred, gt):
    """order-free 코너 대응.  예측 순서 오류를 오차로 세지 않기 위한 것."""
    ok = np.isfinite(pred).all(1)
    if ok.sum() < 1:
        return None
    cost = np.linalg.norm(pred[ok][:, None, :] - gt[None, :, :], axis=2)
    r, c = linear_sum_assignment(cost)
    return cost[r, c]


def gt_of(label):
    obj = label["objects"][0]
    dims = obj["dimensions_m"]
    pose = np.asarray(obj["pose_transform"], float)
    model_pts = APNP.make_pallet_keypoints_3d_diagram(
        width=dims["width"], depth=dims["depth"], height=dims["height"])[:8]
    return {"gt8": np.asarray(obj["projected_cuboid"], float)[:8],
            "R": pose[:3, :3], "t": pose[:3, 3],
            "extents": (dims["width"], dims["height"], dims["depth"]),
            "model": model_pts,
            "K": np.array([[label["camera_data"]["intrinsics"]["fx"], 0,
                            label["camera_data"]["intrinsics"]["cx"]],
                           [0, label["camera_data"]["intrinsics"]["fy"],
                            label["camera_data"]["intrinsics"]["cy"]],
                           [0, 0, 1.0]], float)}


def score_frame(kps, kp_conf, truth, image_shape):
    pred8 = np.asarray(kps, float)[:8]
    n_det = int(np.isfinite(pred8[:, 0]).sum())
    row = {"n_det": n_det, "det": int(n_det >= N_DET_MIN)}

    dists = hungarian(pred8, truth["gt8"])
    row["corner_med"] = float(np.median(dists)) if dists is not None else np.nan
    row["corner_p90"] = float(np.percentile(dists, 90)) if dists is not None else np.nan

    kps9 = []
    for i in range(9):
        point = np.asarray(kps[i], float) if i < len(kps) else np.array([np.nan] * 2)
        good = np.isfinite(point).all()
        if good and kp_conf is not None and i < len(kp_conf) \
                and kp_conf[i] is not None and kp_conf[i] < KP_CONF_MIN:
            good = False
        kps9.append([float(point[0]), float(point[1])] if good else None)

    row.update({"pnp_ok": 0, "R": np.nan, "t": np.nan, "add": np.nan,
                "adds": np.nan, "iou": np.nan, "5cm5": 0, "reproj": np.nan})
    if sum(1 for k in kps9 if k is not None) >= N_DET_MIN:
        try:
            pose = APNP.solve_pose(kps9, truth["K"], dims=APNP.PALLET_DIMS,
                                   img_shape=image_shape)
        except Exception:
            pose = None
        if pose is not None:
            R_p = np.asarray(pose["R"], float) if "R" in pose else None
            t_p = np.asarray(pose["t"], float).reshape(3) if "t" in pose else None
            if R_p is not None and t_p is not None:
                deg, met = RM.pose_error(R_p, t_p, truth["R"], truth["t"])
                row.update({
                    "pnp_ok": 1, "R": deg, "t": met,
                    "reproj": float(pose.get("reproj_error_px", np.nan)),
                    "add": RM.add(truth["model"], R_p, t_p, truth["R"], truth["t"]),
                    "adds": RM.add_s(truth["model"], R_p, t_p, truth["R"], truth["t"]),
                    "iou": RM.iou_3d(R_p, t_p, truth["extents"],
                                     truth["R"], truth["t"], truth["extents"]),
                    "5cm5": int(RM.success_5cm5deg(R_p, t_p, truth["R"], truth["t"]))})
    return row


def stats(rows, field, total):
    values = np.array([r[field] for r in rows], float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return {"n": int(values.size), "median": round(float(np.median(values)), 4),
            "mean": round(float(values.mean()), 4),
            "p90": round(float(np.percentile(values, 90)), 4)}


def summarise(rows):
    total = len(rows)
    if total == 0:
        return None
    return {
        "n": total,
        "det_rate": round(sum(r["det"] for r in rows) / total, 4),
        "pnp_rate": round(sum(r["pnp_ok"] for r in rows) / total, 4),
        "corner_px": stats(rows, "corner_med", total),
        "R_deg": stats(rows, "R", total),
        "t_m": stats(rows, "t", total),
        "ADD": stats(rows, "add", total),
        "ADD_S": stats(rows, "adds", total),
        "IoU3D": stats(rows, "iou", total),
        "success_5cm5deg_unconditional":
            round(sum(r["5cm5"] for r in rows) / total, 4),
    }


def main():
    truths = {}
    shapes = {}
    for key, sealed, jp, ip, label in MF.frames():
        fid = os.path.splitext(os.path.basename(jp))[0]
        truths[fid] = gt_of(label)
        image = cv2.imread(ip)
        shapes[fid] = image.shape

    report = {
        "note": "네 모델을 한 downstream 으로 채점. 추론 환경만 다르고 GT·대응·PnP·"
                "지표는 동일.",
        "SEALED_CONSUMED": {
            "sets": ["eval_pallet07", "eval_pallet09", "eval_night08",
                     "eval_night09"],
            "n": 105,
            "authorised": "user, 2026-08-20",
            "consequence": "재봉인 불가. 이후 어떤 최종 주장에도 held-out 으로 "
                           "쓸 수 없다."},
        "pnp": "annotate_pnp.solve_pose, PALLET_DIMS, kp conf>=0.5 필터"
               " (release 배포 계약)",
        "models": {}}

    for name in MODELS:
        path = os.path.join(OUT, f"kps_{name}.json")
        if not os.path.exists(path):
            print(f"  {name}: 덤프 없음 -> 건너뜀"); continue
        dump = json.load(open(path))
        rows, per_set = [], {}
        for entry in dump["frames"]:
            fid = entry["fid"]
            if entry["kps"] is None:
                row = {"n_det": 0, "det": 0, "pnp_ok": 0, "R": np.nan,
                       "t": np.nan, "add": np.nan, "adds": np.nan,
                       "iou": np.nan, "5cm5": 0, "corner_med": np.nan,
                       "corner_p90": np.nan, "reproj": np.nan}
            else:
                row = score_frame(entry["kps"], entry.get("kp_conf"),
                                  truths[fid], shapes[fid])
            row.update({"set": entry["set"], "sealed": entry["sealed"],
                        "fid": fid})
            rows.append(row)
            per_set.setdefault(entry["set"], []).append(row)

        open_rows = [r for r in rows if not r["sealed"]]
        sealed_rows = [r for r in rows if r["sealed"]]
        report["models"][name] = {
            "weights": dump.get("weights"),
            "OPEN_56": summarise(open_rows),
            "SEALED_105": summarise(sealed_rows),
            "ALL_161": summarise(rows),
            "per_set": {k: summarise(v) for k, v in per_set.items()},
        }
        s = report["models"][name]
        print(f"  {name:16} open det {s['OPEN_56']['det_rate']:.3f} "
              f"5cm5 {s['OPEN_56']['success_5cm5deg_unconditional']:.3f} | "
              f"sealed det {s['SEALED_105']['det_rate']:.3f} "
              f"5cm5 {s['SEALED_105']['success_5cm5deg_unconditional']:.3f}",
              flush=True)

    json.dump(report, open(os.path.join(OUT, "MODEL_COMPARE.json"), "w"),
              indent=1, default=str)
    print("-> MODEL_COMPARE.json")


if __name__ == "__main__":
    main()
