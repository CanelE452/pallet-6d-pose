"""paper_s2_decoder_parity_audit.py — Step 7 (H5).

목적: **동일한 ep57 heatmap** 에 training decoder 와 evaluation decoder 를 모두
적용해, weak/border keypoint 에서 좌표·missing 판정이 갈리는지 판정한다.

decoder
  D0 : training LocalSoftArgmax2D  (window 7, temp 0.1, index clamp)
       -> 경계에서 clamp 로 window index 가 **중복**됨. 그 중복 수를 직접 센다.
  D1 : border-safe LocalSoftArgmax2D (동일 window/temp, 유효 index 만 softmax)
  D2 : canonical evaluation decoder (filter_pr_camfacing.extract_keypoints_from_belief)
       gaussian sigma2 + local NMS + threshold 0.3 + 11x11 weighted centroid + 0.4395
  D3 : eval-style differentiable centroid diagnostic
       peak/NMS 선택은 detach, 실제 유효 window, raw positive weights,
       threshold/missing 은 별도 판정

평가셋: strict filter-val N87 (outside44 + night43).  final-test 미접근.
전처리는 ep57 학습과 동일한 anisotropic squash (640x480 -> 400x400),
belief(50) -> orig 는 (W/50, H/50).  paper_s2_real_eval.py 와 parity.

읽기 전용 — 가중치/데이터 수정 없음.
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]


import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "train"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))

import stage25_paperbase_eval as S            # noqa: E402
from stage25_paperbase_eval import E          # noqa: E402
from filter_pr_camfacing import extract_keypoints_from_belief  # noqa: E402
from diffpnp3d_loss import LocalSoftArgmax2D  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "pallet", "results",
                       "paper_s2_target_semantics_audit")
WEIGHTS = os.path.join(ROOT, "weights", "paper_s2_stageB", "net_epoch_0057.pth")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
BELIEF = 50
WINDOW = 7
TEMP = 0.1
THRESH = 0.3


def preprocess_squash(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r = cv2.resize(rgb, (400, 400), interpolation=cv2.INTER_LINEAR)
    t = (r.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(t.transpose(2, 0, 1)).float().unsqueeze(0)


def d0_training(belief_t):
    """D0: 학습에서 쓰인 그대로 (clamp). belief_t (1,9,50,50) -> belief px."""
    sa = LocalSoftArgmax2D(window=WINDOW, temperature=TEMP,
                           orig_size=(BELIEF, BELIEF),
                           belief_size=(BELIEF, BELIEF))
    coords, conf = sa(belief_t)          # orig_size=belief_size => belief px
    return coords[0].detach().cpu().numpy(), conf


def _peak_yx(bmap):
    idx = int(np.argmax(bmap))
    return idx // BELIEF, idx % BELIEF


def d0_duplicates(bmap):
    """D0 의 7x7 clamp window 에서 unique index 수 / 중복 수."""
    r = WINDOW // 2
    py, px = _peak_yx(bmap)
    off = np.arange(-r, r + 1)
    dy, dx = np.meshgrid(off, off, indexing="ij")
    wy = np.clip(py + dy.ravel(), 0, BELIEF - 1)
    wx = np.clip(px + dx.ravel(), 0, BELIEF - 1)
    gidx = wy * BELIEF + wx
    uniq = len(np.unique(gidx))
    return uniq, WINDOW * WINDOW - uniq, py, px


def d1_border_safe(bmap):
    """D1: 동일 window/temperature 지만 map 밖 index 는 softmax 에서 제외."""
    r = WINDOW // 2
    py, px = _peak_yx(bmap)
    ys = np.arange(py - r, py + r + 1)
    xs = np.arange(px - r, px + r + 1)
    ys = ys[(ys >= 0) & (ys < BELIEF)]
    xs = xs[(xs >= 0) & (xs < BELIEF)]
    xg, yg = np.meshgrid(xs, ys)
    vals = bmap[np.ix_(ys, xs)].astype(np.float64)
    w = np.exp((vals - vals.max()) / TEMP)
    w = w / w.sum()
    return float((w * xg).sum()), float((w * yg).sum()), int(w.size)


def d3_eval_style_centroid(bmap):
    """D3: eval 형태(11x11 raw weighted centroid) 지만 선택은 detach 로 분리.
    threshold/missing 판정은 호출측에서 별도."""
    RAN = 5
    sm = _gaussian(bmap, 2.0)
    p = 1
    pl = np.zeros_like(sm); pl[p:, :] = sm[:-p, :]
    pr = np.zeros_like(sm); pr[:-p, :] = sm[p:, :]
    pu = np.zeros_like(sm); pu[:, p:] = sm[:, :-p]
    pd_ = np.zeros_like(sm); pd_[:, :-p] = sm[:, p:]
    peaks = (sm >= pl) & (sm >= pr) & (sm >= pu) & (sm >= pd_)
    ys_, xs_ = np.nonzero(peaks)
    if len(xs_) == 0:
        py, px = _peak_yx(bmap)
    else:
        best = int(np.argmax([bmap[y, x] for y, x in zip(ys_, xs_)]))
        py, px = int(ys_[best]), int(xs_[best])
    y0, y1 = max(0, py - RAN), min(BELIEF, py + RAN + 1)
    x0, x1 = max(0, px - RAN), min(BELIEF, px + RAN + 1)
    patch = bmap[y0:y1, x0:x1].astype(np.float64)
    patch = np.clip(patch, 0, None)
    if patch.sum() <= 0:
        return float(px), float(py), 0
    ys = np.arange(y0, y1); xs = np.arange(x0, x1)
    xg, yg = np.meshgrid(xs, ys)
    return (float(np.average(xg, weights=patch)),
            float(np.average(yg, weights=patch)), int(patch.size))


def _gaussian(a, sigma):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(a, sigma=sigma)


def entropy_of(bmap):
    v = np.clip(bmap.astype(np.float64), 0, None)
    s = v.sum()
    if s <= 0:
        return float("nan")
    p = (v / s).ravel()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = E.load_model(WEIGHTS, device)
    frames = S.frames_filterval()
    print(f"[filterval] N={len(frames)}  weights=ep57  device={device}")

    rows = []
    for dom, fid, jp, ip in frames:
        img = cv2.imread(ip)
        if img is None:
            continue
        d = json.load(open(jp))
        o = d["objects"][0]
        gt8 = np.asarray(o["projected_cuboid"], float)[:8]
        gtc = np.asarray(o["projected_cuboid_centroid"], float)
        gt9 = np.vstack([gt8, gtc.reshape(1, 2)])
        H, W = img.shape[:2]
        sx, sy = W / BELIEF, H / BELIEF
        gt9_bel = np.stack([gt9[:, 0] / sx, gt9[:, 1] / sy], axis=1)
        # frame-level truncation (원본 이미지 기준)
        outside = ((gt9[:, 0] < 0) | (gt9[:, 0] >= W)
                   | (gt9[:, 1] < 0) | (gt9[:, 1] >= H))
        is_trunc = bool(outside.any())

        with torch.no_grad():
            beliefs, _ = model(preprocess_squash(img).to(device))
        bt = beliefs[-1][:, :9].float().cpu()          # (1,9,50,50)
        bel = bt[0].numpy()

        c0, conf0 = d0_training(bt)
        kps_eval = extract_keypoints_from_belief(bel, THRESH)

        for k in range(9):
            bmap = bel[k]
            peak = float(bmap.max())
            uniq, dup, py, px = d0_duplicates(bmap)
            d1x, d1y, d1n = d1_border_safe(bmap)
            d3x, d3y, d3n = d3_eval_style_centroid(bmap)
            ek = kps_eval[k]
            d2_missing = (ek is None) or (ek[0] < 0)
            d2x = float("nan") if d2_missing else float(ek[0])
            d2y = float("nan") if d2_missing else float(ek[1])

            gx, gy = float(gt9_bel[k][0]), float(gt9_bel[k][1])
            dist_border = float(min(px, py, BELIEF - 1 - px, BELIEF - 1 - py))

            def err(ax, ay):
                if not np.isfinite(ax):
                    return float("nan")
                return float(np.hypot((ax - gx) * sx, (ay - gy) * sy))

            rows.append(dict(
                domain=dom, frame_id=fid, keypoint_id=k,
                frame_is_truncated=is_trunc, kp_outside_image=bool(outside[k]),
                peak=peak, entropy=entropy_of(bmap),
                peak_x=px, peak_y=py, dist_to_border_peak=dist_border,
                d0_x=float(c0[k][0]), d0_y=float(c0[k][1]),
                d1_x=d1x, d1_y=d1y,
                d2_x=d2x, d2_y=d2y, d2_missing=bool(d2_missing),
                d3_x=d3x, d3_y=d3y,
                d0_window_unique=uniq, d0_window_duplicate=dup,
                d1_window_n=d1n, d3_window_n=d3n,
                gt_x_bel=gx, gt_y_bel=gy,
                err_d0=err(float(c0[k][0]), float(c0[k][1])),
                err_d1=err(d1x, d1y), err_d2=err(d2x, d2y), err_d3=err(d3x, d3y),
                diff_d0_d1=float(np.hypot((c0[k][0] - d1x) * sx,
                                          (c0[k][1] - d1y) * sy)),
                diff_d0_d2=(float("nan") if d2_missing else
                            float(np.hypot((c0[k][0] - d2x) * sx,
                                           (c0[k][1] - d2y) * sy))),
                diff_d1_d2=(float("nan") if d2_missing else
                            float(np.hypot((d1x - d2x) * sx,
                                           (d1y - d2y) * sy))),
                diff_d2_d3=(float("nan") if d2_missing else
                            float(np.hypot((d2x - d3x) * sx,
                                           (d2y - d3y) * sy))),
                scale_x=sx, scale_y=sy,
            ))

    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "decoder_parity.csv")
    df.to_csv(out, index=False)
    print(f"[saved] {out}  rows={len(df)}")

    def q(s):
        s = s.dropna()
        return (round(float(s.median()), 3),
                round(float(s.quantile(0.90)), 3)) if len(s) else (np.nan, np.nan)

    print("\n=== decoder 좌표 차이 (orig px, median / p90) ===")
    for c in ["diff_d0_d1", "diff_d0_d2", "diff_d1_d2", "diff_d2_d3"]:
        m, p = q(df[c])
        print(f"  {c:12s} median={m:8.3f}  p90={p:8.3f}  n={df[c].notna().sum()}")

    print("\n=== D0 clamp duplicate ===")
    print(f"  duplicate>0 인 keypoint: {(df.d0_window_duplicate > 0).sum()}"
          f" / {len(df)} ({100*(df.d0_window_duplicate>0).mean():.1f}%)")
    print(f"  duplicate 평균(전체)     : {df.d0_window_duplicate.mean():.2f} / 49")
    sub = df[df.d0_window_duplicate > 0]
    if len(sub):
        print(f"  duplicate 평균(>0 만)    : {sub.d0_window_duplicate.mean():.2f} / 49")
        print(f"  그 subset diff_d0_d1 median = "
              f"{sub.diff_d0_d1.median():.3f} px")

    print("\n=== border subset (peak가 경계 3px 이내) ===")
    bs = df[df.dist_to_border_peak <= 3]
    print(f"  n={len(bs)}")
    if len(bs):
        for c in ["diff_d0_d1", "diff_d0_d2", "err_d0", "err_d1", "err_d2"]:
            m, p = q(bs[c])
            print(f"  {c:12s} median={m:8.3f} p90={p:8.3f}")

    print("\n=== missing (D2 threshold 0.3) ===")
    print(f"  D2 missing rate = {100*df.d2_missing.mean():.1f}% "
          f"(D0/D1 은 항상 좌표를 냄 = missing 개념 없음)")
    print(df.groupby("keypoint_id").agg(
        d2_missing_pct=("d2_missing", lambda s: round(100 * s.mean(), 1)),
        peak_median=("peak", lambda s: round(float(s.median()), 3)),
        dup_pct=("d0_window_duplicate",
                 lambda s: round(100 * (s > 0).mean(), 1))).to_string())


if __name__ == "__main__":
    main()
