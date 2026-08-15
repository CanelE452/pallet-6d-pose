"""ralph_matrix.py — challenge0123 self-training 의 cross-domain matrix (phase2 형식).

행: R0 / R1(noapril PL) / R1(outside PL) / R1(night PL)  (+ 원하면 R2)
열: outside / night / noapril / cad  (eval GT held-out)
값: NN<20px per-frame % (좌 그림) + corner_med px (우 그림, 낮을수록 좋음)
self-domain cell = 빨간 박스. 과거 plot_phase2_cross_domain 형식.

Usage: RALPH_THRESH=0.1 python -u scripts/stage0/ralph/ralph_matrix.py
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
import matplotlib                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt           # noqa: E402

NMIN = T.N_DET_MIN
RS = "data/pallet/results/ralph/ralph_selftrain"
# 라운드 선택: R2(round_02) 가 self-domain accuracy 최선 경향 → R2 사용
RND = os.environ.get("RALPH_ROUND", "round_02")
# ★ s2(paper 트랙). 보유 모델: R0, night-PL(self=night), combined(best). 행/열 순서 정렬.
# (outside/noapril per-domain s2 는 미학습 → 파일 있으면 자동 포함되어 대각선 완성)
MODELS = [m for m in [
    ("R0\n(s2)", "weights/paper_s2_stageB/net_epoch_0057_noseg.pth"),
    ("outside PL", f"{RS}/h6_s2_outside/{RND}.pth"),
    ("night PL", f"{RS}/h3_s2_night/{RND}.pth"),
    ("noapril PL", f"{RS}/h7_s2_noapril/{RND}.pth"),
    ("combined PL\n(best)", f"{RS}/h4_s2_combined/{RND}.pth"),
] if os.path.isfile(os.path.join(ROOT, m[1]))]
DOMAINS = {
    "outside": ["challenge/data/01_real/eval_canonical/_outside_eval_manual_gt"]
    + [f"challenge/data/01_real/manual_gt/capturepallet0{i}_manual_gt" for i in range(2, 10)],
    "night": ["challenge/data/01_real/manual_gt/_night_eval_manual_gt"]
    + [f"challenge/data/01_real/manual_gt/capturenight0{i}_manual_gt" for i in range(5, 8)],
    "noapril": ["challenge/data/01_real/eval_canonical/capture0403noapril_manual_gt"],
    "cad": ["challenge/data/01_real/eval_canonical/capturepalletcad_manual_gt"],
}
# self-domain 셀: (row_label, col) — 대각선 정렬 (행/열 순서 동일)
SELF = {("outside PL", "outside"), ("night PL", "night"), ("noapril PL", "noapril")}
OUT = os.path.join(ROOT, RS, f"cross_domain_matrix_s2_{RND}.png")


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
                    gt = np.array(json.load(open(jf))["objects"][0]["projected_cuboid"], float)[:8]
                except Exception:
                    continue
                seen[fid] = {"ip": ip, "gt": gt}
        frames[dom] = list(seen.values())
    return frames


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = load_eval()
    doms = list(DOMAINS.keys())
    print("[eval sizes]", {d: len(v) for d, v in frames.items()}, "THRESH", T.THRESH, "ROUND", RND)
    nn = np.full((len(MODELS), len(doms)), np.nan)
    cm = np.full((len(MODELS), len(doms)), np.nan)
    for mi, (name, wp) in enumerate(MODELS):
        p = os.path.join(ROOT, wp)
        if not os.path.isfile(p):
            print("[skip missing]", wp)
            continue
        model = T.E.load_model(p, device)
        for di, dom in enumerate(doms):
            fs = frames[dom]
            ok, cms = 0, []
            for fr in fs:
                img = cv2.imread(fr["ip"])
                _, pr, _, _, _ = T.infer_squash(model, img, device)
                if int(np.sum(~np.isnan(pr[:, 0]))) < NMIN:
                    continue
                d, _ = T.E.hungarian(pr, fr["gt"])
                m = float(np.median(d))
                cms.append(m)
                if m < 20:
                    ok += 1
            nn[mi, di] = 100 * ok / len(fs) if fs else np.nan
            cm[mi, di] = float(np.median(cms)) if cms else np.nan
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"  {name.splitlines()[0]:<16} nn20 {[f'{nn[mi,j]:.0f}' for j in range(len(doms))]}"
              f"  cm {[f'{cm[mi,j]:.1f}' for j in range(len(doms))]}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, mat, title, cmap, vmax, fmt, better in [
        (axes[0], nn, f"(a) NN<20px per-frame (%) — higher better", "YlGnBu", 100, "{:.0f}", "hi"),
        (axes[1], cm, f"(b) corner error median (px) — lower better", "YlOrRd_r", 30, "{:.1f}", "lo"),
    ]:
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(doms))); ax.set_xticklabels(doms, fontsize=11)
        ax.set_yticks(range(len(MODELS))); ax.set_yticklabels([m[0] for m in MODELS], fontsize=9)
        ax.set_xlabel("Evaluation domain", fontsize=12); ax.set_title(title, fontsize=12)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center",
                            color="black", fontsize=11, fontweight="bold")
                if (MODELS[i][0], doms[j]) in SELF:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                               edgecolor="red", linewidth=2.5))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ev = {d: len(v) for d, v in frames.items()}
    fig.suptitle(f"s2 (paper) self-training cross-domain ({RND}, THRESH={T.THRESH})  —  "
                 f"red=self-domain (diagonal);  eval GT held-out {ev}", fontsize=12, y=0.98)
    # 제목-히트맵 공백 축소: 서브플롯을 위로 당김
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(OUT, dpi=140, bbox_inches="tight")
    print("[save]", OUT)


if __name__ == "__main__":
    main()
