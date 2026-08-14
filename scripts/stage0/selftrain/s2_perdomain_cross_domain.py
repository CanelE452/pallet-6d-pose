"""s2_perdomain_cross_domain.py — per-domain PL R1 들의 cross-domain transfer 매트릭스.

행: R0 / R1(outside PL) / R1(night PL) / R1(noapril PL) / R2
열: outside / night / noapril / cad (평가 도메인, 전체 GT, PL 홀드아웃)
값: NN<20px per-frame %  (corner_med<20px 프레임 비율)
과거 plot_phase2_cross_domain 형식 (self-domain 빨간 박스 + best per-domain 막대).

Usage: conda activate pallet-pose; python -u scripts/stage0/selftrain/s2_perdomain_cross_domain.py
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
import cv2                                # noqa: E402
import torch                              # noqa: E402
import matplotlib                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt           # noqa: E402

NMIN = T.N_DET_MIN
MODELS = [
    ("R0\n(baseline)", "weights/paper_s2_stageB/net_epoch_0057.pth"),
    ("R1 rf\n(outside PL,\n191)", "weights/paper_s2_rf_hipl/r1_outside/net_epoch_0060.pth"),
    ("R1 rf\n(night PL,\n39)", "weights/paper_s2_rf_hipl/r1_night/net_epoch_0060.pth"),
    ("R1 rf\n(noapril PL,\n2)", "weights/paper_s2_rf_hipl/r1_noapril/net_epoch_0060.pth"),
    ("R1 rf\n(combined,\n232)", "weights/paper_s2_rf_hipl/r1_combined/net_epoch_0060.pth"),
]
DOMAINS = {
    "outside": ["challenge/data/01_real/eval_canonical/_outside_eval_manual_gt"] + [f"challenge/data/01_real/manual_gt/capturepallet0{i}_manual_gt" for i in range(2, 10)],
    "night": ["challenge/data/01_real/manual_gt/_night_eval_manual_gt"] + [f"challenge/data/01_real/manual_gt/capturenight0{i}_manual_gt" for i in range(5, 8)],
    "noapril": ["challenge/data/01_real/eval_canonical/capture0403noapril_manual_gt"],
    "cad": ["challenge/data/01_real/eval_canonical/capturepalletcad_manual_gt"],
}
SELF_CELLS = [(1, 0), (2, 1), (3, 2)]   # R1_outside×outside, R1_night×night, R1_noapril×noapril
OUT = os.path.join(ROOT, "data/pallet/results/paper_s2_fullpool_full7", "cross_domain_hipl.png")


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
    print("[eval sizes]", {d: len(v) for d, v in frames.items()})
    doms = list(DOMAINS.keys())
    mat = np.full((len(MODELS), len(doms)), np.nan)
    for mi, (name, wp) in enumerate(MODELS):
        if not os.path.isfile(os.path.join(ROOT, wp)):
            print("[skip missing]", wp)
            continue
        model = T.E.load_model(os.path.join(ROOT, wp), device)
        for di, dom in enumerate(doms):
            fs = frames[dom]
            if not fs:
                continue
            ok = 0
            for fr in fs:
                img = cv2.imread(fr["ip"])
                _, pr, _, _, _ = T.infer_squash(model, img, device)
                if int(np.sum(~np.isnan(pr[:, 0]))) < NMIN:
                    continue
                d, _ = T.E.hungarian(pr, fr["gt"])
                if float(np.median(d)) < 20:
                    ok += 1
            mat[mi, di] = 100 * ok / len(fs)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"  {name.splitlines()[0]}: {[f'{mat[mi,j]:.0f}' for j in range(len(doms))]}")

    # ---- render (phase2_cross_domain 형식) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.35, 1]})
    im = ax1.imshow(mat, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    ax1.set_xticks(range(len(doms))); ax1.set_xticklabels(doms, fontsize=11)
    ax1.set_yticks(range(len(MODELS))); ax1.set_yticklabels([m[0] for m in MODELS], fontsize=9)
    ax1.set_xlabel("Evaluation domain", fontsize=12)
    ax1.set_title("(a) Cross-domain transfer (NN<20px per-frame, %)", fontsize=12)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax1.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center",
                         color="white" if mat[i, j] > 55 else "black", fontsize=12, fontweight="bold")
    for (r, c) in SELF_CELLS:
        ax1.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2.5))
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="NN<20px (%)")

    x = np.arange(len(doms)); w = 0.4
    r0 = mat[0]
    best = [np.nanmax(mat[1:4, j]) for j in range(len(doms))]   # R1 rows (per-domain PL)
    lbl = ["outside", "night", "noapril"]
    best_lbl = [lbl[int(np.nanargmax(mat[1:4, j]))] if np.isfinite(np.nanmax(mat[1:4, j])) else "" for j in range(len(doms))]
    ax2.bar(x - w/2, r0, w, label="R0 (baseline)", color="#bdbdbd", edgecolor="black")
    bb = ax2.bar(x + w/2, best, w, label="Best R1 (per-domain PL)", color="#1f77b4", edgecolor="black")
    for j in range(len(doms)):
        if np.isfinite(r0[j]):
            ax2.text(x[j]-w/2, r0[j]+1, f"{r0[j]:.0f}", ha="center", fontsize=10, fontweight="bold")
        if np.isfinite(best[j]):
            ax2.text(x[j]+w/2, best[j]+1, f"{best[j]:.0f}", ha="center", fontsize=10, fontweight="bold")
            ax2.text(x[j]+w/2, best[j]/2, best_lbl[j], ha="center", va="center", fontsize=8, color="white", rotation=90)
    ax2.set_xticks(x); ax2.set_xticklabels(doms, fontsize=11); ax2.set_ylim(0, 105)
    ax2.set_ylabel("NN<20px per-frame (%)", fontsize=12)
    ax2.set_title("(b) Best per-domain (Baseline vs best R1)", fontsize=12)
    ax2.legend(loc="upper right", fontsize=10); ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ev = {d: len(v) for d, v in frames.items()}
    fig.suptitle("Cross-domain transfer matrix & best per-domain (per-domain filtered PL)\n"
                 f"red box = self-domain cell;  eval GT (held-out): {ev}", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT, dpi=140, bbox_inches="tight")
    print("[save]", OUT)


if __name__ == "__main__":
    main()
