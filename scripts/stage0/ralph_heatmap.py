"""ralph_heatmap.py — R0 vs self-train 모델의 belief 히트맵 오버레이 (도메인별, 각각 따로).

infer_squash 의 belief(9,50,50) 를 9채널 max 로 합쳐 이미지 위 JET 오버레이.
+ 예측 8코너(파랑)·GT(초록) 점. 두 모델을 각각 따로 파일 저장.

Usage: RALPH_DOM=night python -u scripts/stage0/ralph_heatmap.py
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

RS = "data/pallet/results/ralph_selftrain"
DOM = os.environ.get("RALPH_DOM", "night")
# 도메인별 (R0, self-train 모델, GT 폴더)
CFG = {
    "night": (f"{RS}/h3_s2_night/round_02.pth",
              ["challenge/data/_night_eval_manual_gt"]
              + [f"challenge/data/capturenight0{i}_manual_gt" for i in range(5, 8)]),
    "outside": (f"{RS}/h6_s2_outside/round_02.pth",
                ["challenge/data/_outside_eval_manual_gt"]
                + [f"challenge/data/capturepallet0{i}_manual_gt" for i in range(2, 10)]),
    "noapril": (f"{RS}/h7_s2_noapril/round_02.pth",
                ["challenge/data/capture0403noapril_manual_gt"]),
}
R0 = "weights/paper_s2_stageB/net_epoch_0057_noseg.pth"
N_FRAMES = int(os.environ.get("RALPH_N", "4"))


def heat_overlay(model, img_bgr, device, gt8):
    H, W = img_bgr.shape[:2]
    belief, pred8, pred_c, peaks, _ = T.infer_squash(model, img_bgr, device)
    hm = np.max(belief[:9], axis=0)                 # (50,50) 9채널 max
    hm = np.clip(hm, 0, 1)
    hm = cv2.resize(hm, (W, H), interpolation=cv2.INTER_CUBIC)
    hmc = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    ov = cv2.addWeighted(img_bgr, 0.55, hmc, 0.45, 0)
    # GT(초록) + 예측(파랑) 점
    for g in gt8:
        cv2.circle(ov, (int(g[0]), int(g[1])), 5, (0, 255, 0), 2)
    for i in range(8):
        if not np.isnan(pred8[i, 0]):
            cv2.circle(ov, (int(pred8[i, 0]), int(pred8[i, 1])), 4, (255, 80, 0), -1)
    n_det = int(np.sum(~np.isnan(pred8[:, 0])))
    return ov, n_det, float(np.max(peaks))


def pick(gt_dirs):
    seen = {}
    for fo in gt_dirs:
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
    return vals[:: max(1, len(vals) // N_FRAMES)][:N_FRAMES]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    self_path, gt_dirs = CFG[DOM]
    frames = pick(gt_dirs)
    m0 = T.E.load_model(os.path.join(ROOT, R0), device)
    ms = T.E.load_model(os.path.join(ROOT, self_path), device)
    for tag, model in [("R0_s2", m0), (f"selftrain_{DOM}", ms)]:
        rows = []
        for fr in frames:
            img = cv2.imread(fr["ip"])
            ov, ndet, peak = heat_overlay(model, img, device, [fr["gt"][i] for i in range(8)])
            h = 340
            ov = cv2.resize(ov, (int(img.shape[1] * h / img.shape[0]), h))
            cv2.putText(ov, f"{tag}  det{ndet}/8 peak{peak:.2f}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            rows.append(ov)
        wmax = max(r.shape[1] for r in rows)
        rows = [cv2.copyMakeBorder(r, 0, 6, 0, wmax - r.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0)) for r in rows]
        out = os.path.join(ROOT, RS, f"heatmap_{DOM}_{tag}.png")
        cv2.imwrite(out, cv2.vconcat(rows))
        print("[save]", out)


if __name__ == "__main__":
    main()
