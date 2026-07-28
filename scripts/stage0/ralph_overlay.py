"""ralph_overlay.py — R0 vs self-train best 모델 정성 비교 (실제 eval GT 프레임).

각 도메인 몇 프레임: [R0 예측] | [best 예측] | GT(초록) 오버레이. corner 8점 + 큐보이드 엣지.
goal 달성 정성 증거.

Usage: python -u scripts/stage0/ralph_overlay.py
"""
from __future__ import annotations
import glob
import json
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))

import paper_s2_filterval_9filters as F   # noqa: E402,F401
import paper_s2_testset17_9filters as T   # noqa: E402
if os.environ.get("RALPH_THRESH"):
    T.THRESH = float(os.environ["RALPH_THRESH"])
import cv2                                # noqa: E402
import torch                              # noqa: E402

NMIN = T.N_DET_MIN
RS = "data/pallet/results/ralph_selftrain"
R0 = "weights/paper_s2_stageB/net_epoch_0057_noseg.pth"
BEST = f"{RS}/h4_s2_combined/round_02.pth"
# camera-facing 큐보이드 엣지 (0-3 앞면, 4-7 뒷면)
EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
DOMS = {
    "night": ["challenge/data/_night_eval_manual_gt"]
    + [f"challenge/data/capturenight0{i}_manual_gt" for i in range(5, 8)],
    "outside": ["challenge/data/_outside_eval_manual_gt"]
    + [f"challenge/data/capturepallet0{i}_manual_gt" for i in range(2, 10)],
}
N_PER = 3
OUT = os.path.join(ROOT, RS, "overlay_R0_vs_best.png")


def draw(img, pts8, color):
    for i in range(8):
        p = pts8[i]
        if p is None or (isinstance(p, np.ndarray) and np.isnan(p[0])):
            continue
        cv2.circle(img, (int(p[0]), int(p[1])), 5, color, -1)
    for a, b in EDGES:
        pa, pb = pts8[a], pts8[b]
        if pa is None or pb is None:
            continue
        if (isinstance(pa, np.ndarray) and np.isnan(pa[0])) or (isinstance(pb, np.ndarray) and np.isnan(pb[0])):
            continue
        cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, 2)
    return img


def pick_frames():
    out = {}
    for dom, fos in DOMS.items():
        seen = {}
        for fo in fos:
            for jf in sorted(glob.glob(os.path.join(ROOT, fo, "*.json"))):
                fid = os.path.splitext(os.path.basename(jf))[0]
                if fid in seen:
                    continue
                ip = jf[:-5] + ".png"
                if not os.path.isfile(ip):
                    continue
                try:
                    gt = np.array(json.load(open(jf))["objects"][0]["projected_cuboid"], float)[:8]
                except Exception:
                    continue
                seen[fid] = {"ip": ip, "gt": gt}
        vals = list(seen.values())
        out[dom] = vals[:: max(1, len(vals) // N_PER)][:N_PER]
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = pick_frames()
    m0 = T.E.load_model(os.path.join(ROOT, R0), device)
    mb = T.E.load_model(os.path.join(ROOT, BEST), device)
    rows = []
    for dom, fs in frames.items():
        for fr in fs:
            img = cv2.imread(fr["ip"])
            H, W = img.shape[:2]
            _, p0, _, _, _ = T.infer_squash(m0, img, device)
            _, pb, _, _, _ = T.infer_squash(mb, img, device)
            gt8 = [fr["gt"][i] for i in range(8)]
            a = draw(img.copy(), [tuple(g) for g in gt8], (0, 255, 0))
            a = draw(a, [p0[i] for i in range(8)], (0, 128, 255))
            cv2.putText(a, f"{dom} R0 (orange) vs GT (green)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2)
            b = draw(img.copy(), [tuple(g) for g in gt8], (0, 255, 0))
            b = draw(b, [pb[i] for i in range(8)], (255, 100, 0))
            cv2.putText(b, "best self-train (blue) vs GT (green)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)
            # 두 패널 가로 결합
            h = 360
            a = cv2.resize(a, (int(W * h / H), h)); b = cv2.resize(b, (int(W * h / H), h))
            rows.append(cv2.hconcat([a, np.full((h, 6, 3), 255, np.uint8), b]))
    # 각 row 너비 통일 후 세로 결합
    wmax = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 6, 0, wmax - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows]
    grid = cv2.vconcat(rows)
    cv2.imwrite(os.path.join(ROOT, OUT), grid)
    print("[save]", OUT, grid.shape)


if __name__ == "__main__":
    main()
