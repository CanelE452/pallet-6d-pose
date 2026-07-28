"""s2_gt_accuracy.py — 사용자 어노 GT 로 R0 vs self-train 정확도(코너 오차 px) 확인.

셀 = 필터 통과수 아님. 각 모델을 GT 이미지에 추론 → 예측 keypoint vs GT keypoint
코너 오차(order-free Hungarian median px). 낮을수록 정확 = 좋아진 것.
검증: 프레임별 pred/GT/오차를 몇 개 출력해 '필터 아닌 정확도'임을 증명.

Usage: conda activate pallet-pose; python -u scripts/stage0/s2_gt_accuracy.py
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
import cv2                                # noqa: E402
import torch                              # noqa: E402
import matplotlib                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt           # noqa: E402

NMIN = T.N_DET_MIN
MODELS = [
    ("R0 baseline", "weights/paper_s2_stageB/net_epoch_0057.pth"),
    ("R1rf outside", "weights/paper_s2_rf_hipl/r1_outside/net_epoch_0060.pth"),
    ("R1rf night", "weights/paper_s2_rf_hipl/r1_night/net_epoch_0060.pth"),
    ("R1rf noapril", "weights/paper_s2_rf_hipl/r1_noapril/net_epoch_0060.pth"),
    ("R1rf combined", "weights/paper_s2_rf_hipl/r1_combined/net_epoch_0060.pth"),
]
DOMAINS = {
    "outside": ["challenge/data/_outside_eval_manual_gt"] + [f"challenge/data/capturepallet0{i}_manual_gt" for i in range(2, 10)],
    "night": ["challenge/data/_night_eval_manual_gt"] + [f"challenge/data/capturenight0{i}_manual_gt" for i in range(5, 8)],
    "noapril": ["challenge/data/capture0403noapril_manual_gt"],
    "cad": ["challenge/data/capturepalletcad_manual_gt"],
}
OUT = os.path.join(ROOT, "data/pallet/results/paper_s2_fullpool_full7", "gt_accuracy_hipl.png")


def load_eval():
    frames = {}
    for dom, fos in DOMAINS.items():
        seen = {}
        for fo in fos:
            for jf in glob.glob(os.path.join(ROOT, fo, "*.json")):
                fid = os.path.splitext(os.path.basename(jf))[0]
                if fid in seen:
                    continue
                ip = jf[:-5] + ".png"
                if not os.path.isfile(ip):
                    continue
                try:
                    o = json.load(open(jf))["objects"][0]
                    gt = np.array(o["projected_cuboid"], float)[:8]
                    src = o.get("gt_source", "?")
                except Exception:
                    continue
                seen[fid] = {"ip": ip, "gt": gt, "src": src}
        frames[dom] = list(seen.values())
    return frames


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = load_eval()
    print("[eval GT sizes]", {d: len(v) for d, v in frames.items()},
          " gt_source 예시:", frames["outside"][0]["src"] if frames["outside"] else "-")
    doms = list(DOMAINS.keys())

    # corner_med(px) median + det% + good%(<10px)
    cm_mat = np.full((len(MODELS), len(doms)), np.nan)
    detail = {}
    for mi, (name, wp) in enumerate(MODELS):
        model = T.E.load_model(os.path.join(ROOT, wp), device)
        for di, dom in enumerate(doms):
            cms = []
            for fr in frames[dom]:
                img = cv2.imread(fr["ip"])
                _, pr, _, _, _ = T.infer_squash(model, img, device)
                if int(np.sum(~np.isnan(pr[:, 0]))) < NMIN:
                    continue
                d, _ = T.E.hungarian(pr, fr["gt"])
                cms.append(float(np.median(d)))
            cm_mat[mi, di] = float(np.median(cms)) if cms else np.nan
            # 검증 출력: R0, outside 첫 3프레임
            if mi == 0 and dom == "outside":
                detail["verify"] = []
                for fr in frames[dom]:
                    img = cv2.imread(fr["ip"])
                    _, pr, _, _, _ = T.infer_squash(model, img, device)
                    if int(np.sum(~np.isnan(pr[:, 0]))) < NMIN:
                        continue
                    d, _ = T.E.hungarian(pr, fr["gt"])
                    detail["verify"].append((os.path.basename(fr["ip"])[:16],
                                             [round(float(pr[0, 0]), 1), round(float(pr[0, 1]), 1)],
                                             [round(float(fr["gt"][0, 0]), 1), round(float(fr["gt"][0, 1]), 1)],
                                             round(float(np.median(d)), 1)))
                    if len(detail["verify"]) >= 3:
                        break
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"  {name:<15} corner_med(px): {[f'{cm_mat[mi,j]:.1f}' for j in range(len(doms))]}")

    print("\n[검증] R0 outside 3프레임 (pred kp0 / GT kp0 / corner_med px) — 필터 아닌 pred-vs-GT:")
    for v in detail.get("verify", []):
        print(f"   {v[0]}  pred={v[1]}  GT={v[2]}  corner_med={v[3]}px")

    # ---- 그림: 도메인별 R0 vs self-train 코너오차(px), 낮을수록 좋음 ----
    x = np.arange(len(doms)); w = 0.16
    fig, ax = plt.subplots(figsize=(11, 5.5))
    cols = ["#7f7f7f", "#2ca02c", "#d62728", "#1f77b4", "#9467bd"]
    for mi, (name, _) in enumerate(MODELS):
        bars = ax.bar(x + (mi - 2) * w, cm_mat[mi], w, label=name, color=cols[mi], edgecolor="black")
        for j, b in enumerate(bars):
            if np.isfinite(cm_mat[mi, j]):
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                        f"{cm_mat[mi, j]:.1f}", ha="center", fontsize=7, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(doms, fontsize=12)
    ax.set_ylabel("corner error median (px)  — lower = better = improved", fontsize=11)
    ax.set_title("Accuracy on YOUR manual GT (pred vs GT corner error)\n"
                 f"eval GT held-out: { {d: len(v) for d, v in frames.items()} }", fontsize=12)
    ax.legend(fontsize=9, ncol=5, loc="upper center")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT, dpi=140, bbox_inches="tight")
    print("[save]", OUT)


if __name__ == "__main__":
    main()
