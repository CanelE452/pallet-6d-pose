"""ralph_heatmap_confwrong.py — self-train이 confident-wrong 낸 프레임 선별 → R0|self-train 히트맵.

confident-wrong severity(프레임) = self-train 모델에서 검출된 코너 중 GT서 먼(>TAU px) 코너의 peak 합.
= "틀린 위치에 진한 히트맵"이 심한 프레임. severity 상위 N개 골라 R0 vs self-train 오버레이 나란히.

Usage: RALPH_DOM=outside python -u scripts/stage0/ralph/ralph_heatmap_confwrong.py
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

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
import ralph_heatmap as RH                # noqa: E402  (heat_overlay, CFG 재사용)

DOM = os.environ.get("RALPH_DOM", "outside")
TAU = float(os.environ.get("RALPH_TAU", "20"))    # correct 기준 px
N = int(os.environ.get("RALPH_N", "4"))
R0 = "weights/paper_s2_stageB/net_epoch_0057_noseg.pth"
RS = "data/pallet/results/ralph/ralph_selftrain"


def per_kp_err(pred8, gt8):
    """same-index 2D err (px). pred None→nan."""
    e = np.full(8, np.nan)
    for i in range(8):
        if not np.isnan(pred8[i, 0]):
            e[i] = np.hypot(pred8[i, 0] - gt8[i][0], pred8[i, 1] - gt8[i][1])
    return e


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    self_path, gt_dirs = RH.CFG[DOM]
    # 모든 eval 프레임 로드
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
    frames = list(seen.values())

    ms = T.E.load_model(os.path.join(ROOT, self_path), device)
    # severity 계산 (self-train 모델)
    scored = []
    for fr in frames:
        img = cv2.imread(fr["ip"])
        _, pred8, _, peaks, _ = T.infer_squash(ms, img, device)
        err = per_kp_err(pred8, [fr["gt"][i] for i in range(8)])
        sev = 0.0
        nwrong = 0
        for i in range(8):
            if not np.isnan(err[i]) and err[i] > TAU:      # 검출됐는데 틀림
                sev += float(peaks[i])                      # 그 peak(자신감)만큼 나쁨
                nwrong += 1
        scored.append((sev, nwrong, fr))
    scored.sort(key=lambda x: -x[0])                        # severity 높은 순
    picks = scored[:N]
    print(f"[{DOM}] confident-wrong 상위 {N} (sev=틀린코너 peak합):")
    for sev, nw, fr in picks:
        print(f"   {os.path.basename(fr['ip'])[:20]}  sev={sev:.2f} nwrong={nw}")

    m0 = T.E.load_model(os.path.join(ROOT, R0), device)
    rows = []
    for sev, nw, fr in picks:
        img = cv2.imread(fr["ip"])
        gt8 = [fr["gt"][i] for i in range(8)]
        a, d0, p0 = RH.heat_overlay(m0, img.copy(), device, gt8)
        b, ds, ps = RH.heat_overlay(ms, img.copy(), device, gt8)
        cv2.putText(a, f"R0  det{d0}/8 peak{p0:.2f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(b, f"self-train  det{ds}/8 peak{ps:.2f}  conf-wrong sev={sev:.2f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        h = 340
        a = cv2.resize(a, (int(img.shape[1] * h / img.shape[0]), h))
        b = cv2.resize(b, (int(img.shape[1] * h / img.shape[0]), h))
        rows.append(cv2.hconcat([a, np.full((h, 6, 3), 255, np.uint8), b]))
    wmax = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 6, 0, wmax - r.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0)) for r in rows]
    out = os.path.join(ROOT, RS, f"heatmap_{DOM}_confwrong.png")
    cv2.imwrite(out, cv2.vconcat(rows))
    print("[save]", out, "(좌 R0 | 우 self-train, 초록=GT 파랑=예측)")


if __name__ == "__main__":
    main()
