"""Hough isolation — line 이 문제인지 point 가 문제인지 가른다.

    H1  FINAL points + FINAL theta      현재 FINAL 의 F3
    H2  FINAL points + GT    theta      line 을 완벽하게 주면 FINAL 이 살아나나
    H3  YOLO  points + FINAL theta      좋은 점 + FINAL 의 line
    H4  YOLO  points + GT    theta      상한

H1<->H2 차이 = line 품질이 주는 것.  H1<->H3 차이 = point 품질이 주는 것.
둘을 같은 프레임에서 재야 의미가 있으므로 common-detected 만 쓴다.

theta 오차는 corner 와 **독립적으로** 따로 낸다 — pose 를 거치면 두 원인이 섞인다.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
for sub in ("scripts/stage0", "scripts/stage0/multihead", "scripts/stage0/line",
            "scripts/stage0/real_eval", "scripts/annotate", "challenge"):
    sys.path.insert(0, os.path.join(ROOT, sub))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2                                # noqa: E402
import re_metrics as RM                   # noqa: E402
import annotate_pnp as APNP               # noqa: E402
import mc_frames as MF                    # noqa: E402
import mc_geom as MG                      # noqa: E402
import mh_fusion as FU                    # noqa: E402
import mh_cigm as CG                      # noqa: E402
import line_feature_capacity_v2 as V2     # noqa: E402
import mh_data as MD                      # noqa: E402
from mh_arms import DH                    # noqa: E402

OUT = os.path.join(ROOT, "data/pallet/results/model_compare")
GRID = 50


def to_grid(pixels, width, height):
    p = np.asarray(pixels, float)
    return np.stack([p[:, 0] * GRID / width, p[:, 1] * GRID / height], 1)


def gt_lines_from(grid9):
    """GT 코너에서 12 role 의 theta/rho/support — 학습 때 쓴 그 함수."""
    grid = np.asarray(grid9, float)[None, :, :]
    theta, rho, p0, p1, length = V2.gt_lines(grid, CG.EDGES)
    hit = V2.visible_segments(p0, p1, length)["hit"][0]
    import torch
    t = torch.tensor(theta, dtype=torch.float32)
    r = torch.tensor(rho, dtype=torch.float32)
    td, rc = DH.centred_from_canonical(t, r)
    theta_c = td % 180.0
    rho_c = torch.where(((td // 180.0) % 2) == 1, -rc, rc)
    can_t, can_r = DH.canonical_from_centred(theta_c, rho_c)
    return (can_t[0].numpy(), can_r[0].numpy(), hit)


def theta_error_deg(pred_t, pred_r, gt_t, gt_r):
    """정본 `DH.measure` 로 잰다 — 손으로 만든 각도차를 쓰지 않는다.

    두 가지를 직접 짜면 반드시 틀린다:
      * 단위.  `canonical_from_centred` 는 **라디안**을 낸다 (0~pi).  도로 착각하고
        `% 180` 을 걸면 아무것도 안 걸러져 오차가 5.8배 작게 나온다 — 실제로 그렇게
        틀렸었다 (0.101 rad 을 0.101 deg 로 보고).
      * wrap.  (theta, rho) 와 (theta+180, -rho) 는 같은 직선이라 theta 와 rho 를
        따로 반올림하면 이웃을 멀다고 센다.  `line_distance` 는 k in {-1,0,1} 로
        최소를 잡는다.
    """
    import torch
    ct_p, cr_p = DH.centred_from_canonical(torch.tensor(pred_t, dtype=torch.float32),
                                           torch.tensor(pred_r, dtype=torch.float32))
    ct_g, cr_g = DH.centred_from_canonical(torch.tensor(gt_t, dtype=torch.float32),
                                           torch.tensor(gt_r, dtype=torch.float32))
    angle, _ = DH.measure(ct_p, cr_p, ct_g, cr_g)
    return np.asarray(angle, float)


def main():
    weight = json.loads(open(os.path.join(
        ROOT, "data/pallet/results/paper_s2_multihead",
        "theta_posealigned_d0.json")).read())["seeds"]["seed1"]["selected_lambda_theta"]

    final = {e["fid"]: e for e in
             json.load(open(os.path.join(OUT, "kps_FINAL40K_seed1.json")))["frames"]}
    yolo = {e["fid"]: e for e in
            json.load(open(os.path.join(OUT, "kps_yolo26n_ft.json")))["frames"]}
    geom = json.load(open(os.path.join(OUT, "MODEL_COMPARE_GEOM.json")))

    rows, theta_rows = [], []
    for key, sealed, jp, ip, label in MF.frames():
        fid = os.path.splitext(os.path.basename(jp))[0]
        if yolo[fid]["kps"] is None:
            continue
        image = cv2.imread(ip)
        height, width = image.shape[:2]
        truth = MG.gt_of(label)
        gt_grid9 = to_grid(np.vstack([truth["gt8"],
                                      np.asarray(label["objects"][0]
                                                 ["projected_cuboid_centroid"],
                                                 float)]), width, height)
        gt_t, gt_r, support = gt_lines_from(gt_grid9)
        pred_t = np.asarray(final[fid]["line_theta"], float)
        pred_r = np.asarray(final[fid]["line_rho"], float)

        # --- theta 오차: corner 와 독립. support 되는 role 만.
        diff = theta_error_deg(pred_t, pred_r, gt_t, gt_r)
        theta_rows.append({"fid": fid, "set": key, "sealed": sealed,
                           "n_support": int(support.sum()),
                           "theta_med": float(np.median(diff[support]))
                           if support.any() else np.nan,
                           "theta_p90": float(np.percentile(diff[support], 90))
                           if support.any() else np.nan,
                           "per_role": diff.tolist()})

        entry = {"fid": fid, "set": key, "sealed": sealed}
        for arm, pts_src, theta_src in (("H1", "final", "pred"),
                                        ("H2", "final", "gt"),
                                        ("H3", "yolo", "pred"),
                                        ("H4", "yolo", "gt")):
            px = (np.asarray(final[fid]["kps_argmax"], float)[:8]
                  if pts_src == "final"
                  else np.asarray(yolo[fid]["kps"], float)[:8])
            if not np.isfinite(px).all():
                entry[f"{arm}_R"] = entry[f"{arm}_t"] = np.nan
                entry[f"{arm}_ok"] = 0
                continue
            grid9 = np.vstack([to_grid(px, width, height), [[0.0, 0.0]]])
            data = {"resolution": np.array([[width, height]]),
                    "model": np.array([truth["model"]]),
                    "K": np.array([truth["K"]]),
                    "pred_corner": np.array([grid9]),
                    "pred_theta": np.array([gt_t if theta_src == "gt" else pred_t]),
                    "pred_rho": np.array([gt_r if theta_src == "gt" else pred_r]),
                    "support": np.array([support])}
            arms, _, _, _ = FU.solve_arms(data, 0, weight)
            pose = arms.get("F3")
            if pose is None:
                entry[f"{arm}_R"] = entry[f"{arm}_t"] = np.nan
                entry[f"{arm}_ok"] = 0
                continue
            deg, met = RM.pose_error(pose[0], pose[1], truth["R"], truth["t"])
            entry.update({f"{arm}_R": deg, f"{arm}_t": met, f"{arm}_ok": 1,
                          f"{arm}_5cm5": int(RM.success_5cm5deg(
                              pose[0], pose[1], truth["R"], truth["t"]))})
        rows.append(entry)

    def agg(subset, arm):
        R = np.array([r.get(f"{arm}_R", np.nan) for r in subset], float)
        t = np.array([r.get(f"{arm}_t", np.nan) for r in subset], float)
        ok = np.isfinite(R)
        return {"n": int(ok.sum()),
                "R_median": round(float(np.median(R[ok])), 3) if ok.any() else None,
                "R_p90": round(float(np.percentile(R[ok], 90)), 3) if ok.any() else None,
                "t_median": round(float(np.median(t[ok])), 4) if ok.any() else None,
                "5cm5": round(float(np.mean([r.get(f"{arm}_5cm5", 0)
                                             for r in subset])), 4)}

    report = {
        "design": {"H1": "FINAL points + FINAL theta",
                   "H2": "FINAL points + GT theta",
                   "H3": "YOLO points + FINAL theta",
                   "H4": "YOLO points + GT theta",
                   "read": "H1<->H2 = line 품질 기여, H1<->H3 = point 품질 기여"},
        "solver": "mh_fusion.solve_arms F3 (동일), lambda_theta from d0 seed1",
        "n_frames": len(rows),
        "arms": {}, "theta": {}}
    for pop, subset in (("ALL", rows),
                        ("OPEN_56", [r for r in rows if not r["sealed"]]),
                        ("REAL_CHALLENGE_DEV_105",
                         [r for r in rows if r["sealed"]])):
        report["arms"][pop] = {a: agg(subset, a) for a in ("H1", "H2", "H3", "H4")}
        tsub = [t for t in theta_rows
                if (pop == "ALL" or (t["sealed"] == (pop != "OPEN_56")))]
        med = np.array([t["theta_med"] for t in tsub], float)
        med = med[np.isfinite(med)]
        report["theta"][pop] = {
            "n": int(med.size),
            "theta_error_deg_median": round(float(np.median(med)), 3),
            "theta_error_deg_p90": round(float(np.percentile(med, 90)), 3),
            "note": "support 되는 role 만. corner 와 독립 — pose 를 거치지 않았다"}

    per_role = np.array([t["per_role"] for t in theta_rows], float)
    report["theta"]["per_role_median_deg"] = [
        round(float(np.median(per_role[:, i])), 3) for i in range(per_role.shape[1])]

    json.dump(report, open(os.path.join(OUT, "HOUGH_ISOLATION.json"), "w"),
              indent=1, default=str)
    for pop in ("OPEN_56", "REAL_CHALLENGE_DEV_105"):
        print(f"  --- {pop} ---")
        for a in ("H1", "H2", "H3", "H4"):
            e = report["arms"][pop][a]
            print(f"    {a}  n={e['n']:3d}  R med {e['R_median']}  "
                  f"p90 {e['R_p90']}  t med {e['t_median']}  5cm5 {e['5cm5']}")
        t = report["theta"][pop]
        print(f"    theta 오차 median {t['theta_error_deg_median']}deg  "
              f"p90 {t['theta_error_deg_p90']}deg")
    print("-> HOUGH_ISOLATION.json")


if __name__ == "__main__":
    main()
