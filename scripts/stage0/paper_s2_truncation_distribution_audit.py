"""paper_s2_truncation_distribution_audit.py — Step 5 (H2 / Gate B).

목적: aug_trunc_v2(합성 truncation 증강)의 keypoint border-distance 분포가
real truncated frame 분포와 얼마나 다른지 **수치로** 판정한다.
("실제 truncation 을 재현한다"는 서술은 쓰지 않는다 — 분포 차이만 본다.)

비교 population
  P0 : 정상 synthetic (non-truncated 학습셋)      -> target audit parquet 재사용
  P1 : aug_trunc_v2                                -> target audit parquet 재사용
  P2 : strict filter-val N87 중 geometry-based truncated frame

좌표 parity
  P0/P1 은 loader 가 낸 belief(50x50) 좌표.
  P2 는 eval 경로와 동일한 aspect squash 로 belief 좌표에 맞춘다:
      x_b = x_640 * 50/640 ,  y_b = y_480 * 50/480
  (A.Resize(400,400) 후 stride 8 과 동일한 고정 anisotropic scale.)

truncation 정의(real): 9 keypoint 중 하나라도 원본 이미지 밖(<0 또는 >=W/H)
  이거나 cuboid bbox 가 이미지 경계에 닿는 frame.

읽기 전용. 기존 데이터/체크포인트 수정 없음. final-test 미접근.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STAGE0 = os.path.join(ROOT, "scripts", "stage0")
sys.path.insert(0, STAGE0)

OUT_DIR = os.path.join(ROOT, "data", "pallet", "results",
                       "paper_s2_target_semantics_audit")
PARQUET = os.path.join(OUT_DIR, "target_semantics_keypoints.parquet")

BELIEF = 50
SIGMA = 2.0
W_SUPPORT = int(SIGMA * 2)          # = 4, CreateBeliefMap 의 w
ORIG_W, ORIG_H = 640.0, 480.0

# aug 설계가 강제한 밴드 (utils_dataset._TRUNC_MARGIN_FRAC = 0.20)
BAND_LO, BAND_HI = 0.20, 0.80


def real_truncated_frames():
    """strict filter-val N87 -> per-keypoint belief 좌표 + truncation flag."""
    import stage25_paperbase_eval as S
    frames = S.frames_filterval()
    rows = []
    for dom, fid, jp, ip in frames:
        d = json.load(open(jp))
        cam = d.get("camera_data", {})
        w = float(cam.get("width", ORIG_W))
        h = float(cam.get("height", ORIG_H))
        o = d["objects"][0]
        pc = np.asarray(o["projected_cuboid"], dtype=np.float64)
        ctr = np.asarray(o["projected_cuboid_centroid"], dtype=np.float64)
        kp9 = np.vstack([pc, ctr.reshape(1, 2)])       # (9,2) camera-facing v4

        outside = ((kp9[:, 0] < 0) | (kp9[:, 0] >= w)
                   | (kp9[:, 1] < 0) | (kp9[:, 1] >= h))
        x0, y0 = kp9[:, 0].min(), kp9[:, 1].min()
        x1, y1 = kp9[:, 0].max(), kp9[:, 1].max()
        bbox_touch = (x0 <= 0) or (y0 <= 0) or (x1 >= w - 1) or (y1 >= h - 1)
        is_trunc = bool(outside.any() or bbox_touch)

        # eval parity: 고정 anisotropic squash -> belief grid
        xb = kp9[:, 0] * (BELIEF / w)
        yb = kp9[:, 1] * (BELIEF / h)

        for k in range(9):
            rows.append(dict(
                population="P2_real_filterval", dataset="filterval_N87",
                domain=dom, frame_id=fid, keypoint_id=k,
                x_belief=float(xb[k]), y_belief=float(yb[k]),
                dist_to_border=float(min(xb[k], yb[k],
                                         BELIEF - 1 - xb[k],
                                         BELIEF - 1 - yb[k])),
                center_inside_belief=bool(0 <= xb[k] < BELIEF
                                          and 0 <= yb[k] < BELIEF),
                full_gaussian_support_inside=bool(
                    xb[k] - W_SUPPORT >= 0 and xb[k] + W_SUPPORT < BELIEF
                    and yb[k] - W_SUPPORT >= 0 and yb[k] + W_SUPPORT < BELIEF),
                kp_outside_image=bool(outside[k]),
                frame_is_truncated=is_trunc,
                frame_bbox_touch=bool(bbox_touch),
                frame_n_kp_outside=int(outside.sum()),
                in_band=bool(BAND_LO * BELIEF <= xb[k] <= BAND_HI * BELIEF
                             and BAND_LO * BELIEF <= yb[k] <= BAND_HI * BELIEF),
            ))
    return pd.DataFrame(rows)


def synthetic_populations():
    df = pd.read_parquet(PARQUET)
    df = df[["dataset", "frame_id", "keypoint_id", "x_belief", "y_belief",
             "dist_to_border", "center_inside_belief",
             "full_gaussian_support_inside"]].copy()
    df["population"] = np.where(df.dataset == "aug_trunc_v2",
                                "P1_aug_trunc_v2", "P0_synth_nontrunc")
    df["domain"] = "synthetic"
    df["kp_outside_image"] = ~df.center_inside_belief
    df["frame_is_truncated"] = df.dataset == "aug_trunc_v2"
    df["in_band"] = ((df.x_belief >= BAND_LO * BELIEF)
                     & (df.x_belief <= BAND_HI * BELIEF)
                     & (df.y_belief >= BAND_LO * BELIEF)
                     & (df.y_belief <= BAND_HI * BELIEF))
    return df


def summarize(df, label_col="population"):
    out = []
    for pop, g in df.groupby(label_col):
        n_kp = len(g)
        out.append(dict(
            population=pop,
            n_frames=g.frame_id.nunique(), n_keypoints=n_kp,
            pct_in_band_20_80=round(100 * g.in_band.mean(), 1),
            pct_center_inside=round(100 * g.center_inside_belief.mean(), 1),
            pct_full_support=round(100 * g.full_gaussian_support_inside.mean(), 1),
            pct_in_deadzone=round(
                100 * (g.center_inside_belief
                       & ~g.full_gaussian_support_inside).mean(), 1),
            pct_outside=round(100 * (~g.center_inside_belief).mean(), 1),
            border_dist_p05=round(float(g.dist_to_border.quantile(0.05)), 2),
            border_dist_median=round(float(g.dist_to_border.median()), 2),
            border_dist_p95=round(float(g.dist_to_border.quantile(0.95)), 2),
        ))
    return pd.DataFrame(out).sort_values("population")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    real = real_truncated_frames()
    synth = synthetic_populations()

    n_fr = real.frame_id.nunique()
    trunc_fr = real[real.frame_is_truncated].frame_id.nunique()
    print(f"[real filterval] frames={n_fr}  truncated={trunc_fr}  "
          f"non-truncated={n_fr - trunc_fr}")

    # P2 는 truncated subset 만; 비교용으로 real non-truncated 도 따로 본다
    real_t = real[real.frame_is_truncated].copy()
    real_n = real[~real.frame_is_truncated].copy()
    real_n["population"] = "P2b_real_nontrunc"

    allpop = pd.concat([synth, real_t, real_n], ignore_index=True)
    summ = summarize(allpop)
    print("\n=== border-distance / band 요약 (belief 50x50, sigma=2 -> w=4) ===")
    print(summ.to_string(index=False))

    summ.to_csv(os.path.join(OUT_DIR, "truncation_distribution_summary.csv"),
                index=False)
    allpop.to_parquet(os.path.join(OUT_DIR, "truncation_populations.parquet"),
                      index=False)

    # far(4-7) 별도
    far = allpop[allpop.keypoint_id.isin([4, 5, 6, 7])]
    summ_far = summarize(far)
    summ_far.to_csv(os.path.join(OUT_DIR,
                                 "truncation_distribution_summary_far.csv"),
                    index=False)
    print("\n=== far keypoints (4-7) only ===")
    print(summ_far.to_string(index=False))

    # frame 단위 bbox-touch rate
    print("\n=== frame-level ===")
    for pop, g in allpop.groupby("population"):
        fr = g.groupby("frame_id").agg(
            any_outside=("kp_outside_image", "any"),
            all_in_band=("in_band", "all"))
        print(f"{pop:22s} frames={len(fr):6d} "
              f"any_kp_outside={100*fr.any_outside.mean():5.1f}%  "
              f"all9_in_band={100*fr.all_in_band.mean():5.1f}%")


if __name__ == "__main__":
    main()
