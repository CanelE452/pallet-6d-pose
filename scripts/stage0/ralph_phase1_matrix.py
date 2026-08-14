"""ralph_phase1_matrix.py — phase1 형식 cross-domain transfer 매트릭스 (s2 self-training).

(a) 매트릭스 히트맵: 행 R0 / R1(도메인PL) / R2(도메인PL), 열 eval 도메인, 값 NN<20px%.
    self-domain 셀 빨간 박스.
(b) best-per-domain: R0 vs best-R1 vs best-R2 막대.
phase1/phase2 형식 재현. 3도메인(outside/night/noapril), cad 제외(검출 붕괴).

Usage: RALPH_THRESH=0.1 python -u scripts/stage0/ralph_phase1_matrix.py
"""
from __future__ import annotations
import glob, json, os, sys
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
RS = "data/pallet/results/ralph_selftrain"
MODELS = [
    ("Baseline\n(R0)", "weights/paper_s2_stageB/net_epoch_0057_noseg.pth"),
    ("R1\n(outside PL)", f"{RS}/h6_s2_outside/round_01.pth"),
    ("R1\n(night PL)", f"{RS}/h3_s2_night/round_01.pth"),
    ("R1\n(noapril PL)", f"{RS}/h7_s2_noapril/round_01.pth"),
    ("R2\n(outside PL)", f"{RS}/h6_s2_outside/round_02.pth"),
    ("R2\n(night PL)", f"{RS}/h3_s2_night/round_02.pth"),
    ("R2\n(noapril PL)", f"{RS}/h7_s2_noapril/round_02.pth"),
]
DOMAINS = {
    "outside": ["challenge/data/01_real/eval_canonical/_outside_eval_manual_gt"] + [f"challenge/data/01_real/manual_gt/capturepallet0{i}_manual_gt" for i in range(2, 10)],
    "night": ["challenge/data/01_real/manual_gt/_night_eval_manual_gt"] + [f"challenge/data/01_real/manual_gt/capturenight0{i}_manual_gt" for i in range(5, 8)],
    "noapril": ["challenge/data/01_real/eval_canonical/capture0403noapril_manual_gt"],
}
# self-domain 셀 (row_label 접두, col)
SELF = {("R1\n(outside PL)", "outside"), ("R1\n(night PL)", "night"), ("R1\n(noapril PL)", "noapril"),
        ("R2\n(outside PL)", "outside"), ("R2\n(night PL)", "night"), ("R2\n(noapril PL)", "noapril")}
OUT = os.path.join(ROOT, RS, "phase1style_cross_domain_s2.png")


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
                    ip2 = os.path.join(ROOT, RS, "recovered_gt", dom, fid + ".png")
                    ip = ip2 if os.path.isfile(ip2) else ip
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
    print("[eval sizes]", {d: len(v) for d, v in frames.items()}, "THRESH", T.THRESH)
    mat = np.full((len(MODELS), len(doms)), np.nan)
    for mi, (name, wp) in enumerate(MODELS):
        p = os.path.join(ROOT, wp)
        if not os.path.isfile(p):
            print("[skip]", wp); continue
        model = T.E.load_model(p, device)
        for di, dom in enumerate(doms):
            fs = frames[dom]; ok = 0
            for fr in fs:
                img = cv2.imread(fr["ip"])
                _, pr, _, _, _ = T.infer_squash(model, img, device)
                if int(np.sum(~np.isnan(pr[:, 0]))) < NMIN:
                    continue
                d, _ = T.E.hungarian(pr, fr["gt"])
                if float(np.median(d)) < 20:
                    ok += 1
            mat[mi, di] = 100 * ok / len(fs) if fs else np.nan
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"  {name.splitlines()[0]:<14} {[f'{mat[mi,j]:.1f}' for j in range(len(doms))]}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.3, 1]})
    im = ax1.imshow(mat, cmap="YlGnBu", vmin=0, vmax=max(70, np.nanmax(mat)), aspect="auto")
    ax1.set_xticks(range(len(doms))); ax1.set_xticklabels(doms, fontsize=11)
    ax1.set_yticks(range(len(MODELS))); ax1.set_yticklabels([m[0] for m in MODELS], fontsize=9)
    ax1.set_xlabel("Evaluation domain", fontsize=12)
    ax1.set_title("(a) Cross-domain transfer (NN<20px per-frame, %)", fontsize=12)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax1.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center",
                         color="white" if mat[i, j] > 45 else "black", fontsize=11, fontweight="bold")
            if (MODELS[i][0], doms[j]) in SELF:
                ax1.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2.5))
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="NN<20px (%)")

    x = np.arange(len(doms)); w = 0.26
    r0 = mat[0]
    bestR1 = [np.nanmax(mat[1:4, j]) for j in range(len(doms))]
    bestR2 = [np.nanmax(mat[4:7, j]) for j in range(len(doms))]
    ax2.bar(x - w, r0, w, label="Baseline (R0)", color="#bdbdbd", edgecolor="black")
    ax2.bar(x, bestR1, w, label="Best R1", color="#1f77b4", edgecolor="black")
    ax2.bar(x + w, bestR2, w, label="Best R2", color="#ff7f0e", edgecolor="black")
    for j in range(len(doms)):
        for xx, vv in [(x[j]-w, r0[j]), (x[j], bestR1[j]), (x[j]+w, bestR2[j])]:
            if np.isfinite(vv):
                ax2.text(xx, vv + 0.6, f"{vv:.1f}", ha="center", fontsize=8, fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(doms, fontsize=11)
    ax2.set_ylabel("NN<20px per-frame (%)", fontsize=12)
    ax2.set_title("(b) Best per-domain across rounds", fontsize=12)
    ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ev = {d: len(v) for d, v in frames.items()}
    fig.suptitle("s2 self-training — Cross-domain transfer matrix & best per-domain\n"
                 f"red box = self-domain;  NN<20px per-frame %  (THRESH={T.THRESH});  eval GT {ev}", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT, dpi=140, bbox_inches="tight")
    print("[save]", OUT)


if __name__ == "__main__":
    main()
