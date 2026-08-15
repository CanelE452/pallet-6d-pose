"""s2_selftrain_eval_viz.py — R0/R1/R2 self-train 모델을 eval 표시 프레임에서 평가 + 시각화.

평가셋: split==eval 로 표시된 프레임(사용자 어노). 학습 PL(paper_s2_fullpool_r1/r2)에
        든 프레임은 누수라 제외 → R0/R1/R2 공정 비교.
지표:   corner_med(order-free Hungarian), front(0-3), rear(4-7), det%(n_det>=6), good%(<10px).
산출:   data/pallet/results/paper_s2/paper_s2_fullpool_full7/selftrain_eval/
          table.png (도메인별 R0/R1/R2 비교) + overlay_*.png (프레임별 R0/R1/R2 vs GT)

Usage: conda activate pallet-pose; python -u scripts/stage0/s2_selftrain_eval_viz.py
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
from annotate_draw import CUBOID_EDGES    # noqa: E402

NMIN = T.N_DET_MIN
MODELS = {
    "R0 (s2 diffpnp)": "weights/paper_s2_stageB/net_epoch_0057.pth",
    "R1 (self-train)": "weights/paper_s2/paper_s2_fullpool_selftrain/r1/net_epoch_0060.pth",
    "R2 (self-train)": "weights/paper_s2/paper_s2_fullpool_selftrain/r2/net_epoch_0063.pth",
}
EVAL_GT_GLOBS = [
    "challenge/data/01_real/eval_canonical/_outside_eval_manual_gt", "challenge/data/01_real/manual_gt/_night_eval_manual_gt",
    "challenge/data/01_real/eval_canonical/capture0403noapril_manual_gt", "challenge/data/01_real/eval_canonical/capturepalletcad_manual_gt",
]
PL_DIRS = ["data/pallet/training_data/paper_s2_fullpool_r1",
           "data/pallet/training_data/paper_s2_fullpool_r2"]
# ★ 구체적 prefix 먼저 (capturepalletcad 가 capturepallet 보다 앞).
DOM_OF = {"capturepalletcad": "cad", "capture0403noapril": "noapril",
          "_outside_eval": "outside", "capturepallet": "outside",
          "_night_eval": "night", "capturenight": "night"}
OUT = os.path.join(ROOT, "data/pallet/results/paper_s2/paper_s2_fullpool_full7/selftrain_eval")


def dom_of(folder):
    b = os.path.basename(folder)
    for k, v in DOM_OF.items():
        if b.startswith(k):
            return v
    return "other"


def collect_eval():
    leaked = set()
    for d in PL_DIRS:
        for p in glob.glob(os.path.join(ROOT, d, "*.json")):
            leaked.add(os.path.splitext(os.path.basename(p))[0])
    frames = []
    for g in EVAL_GT_GLOBS:
        for fo in glob.glob(os.path.join(ROOT, g)):
            dom = dom_of(fo)
            for jf in glob.glob(os.path.join(fo, "*.json")):
                d = json.load(open(jf))
                if d["objects"][0].get("split") != "eval":
                    continue
                fid = os.path.splitext(os.path.basename(jf))[0]
                if fid in leaked:            # 누수 제외
                    continue
                ip = jf[:-5] + ".png"
                if not os.path.isfile(ip):
                    continue
                gt = np.array(d["objects"][0]["projected_cuboid"], float)[:8]
                frames.append({"dom": dom, "fid": fid, "ip": ip, "gt": gt})
    return frames


def metrics(model, img, gt, device):
    _, pred8, _, _, _ = T.infer_squash(model, img, device)
    pr = pred8
    n_det = int(np.sum(~np.isnan(pr[:, 0])))
    if n_det < NMIN:
        return {"n_det": n_det, "cm": None, "front": None, "rear": None, "pred8": pr}
    d, _ = T.E.hungarian(pr, gt)
    cm = float(np.median(d))
    fr = [np.hypot(*(pr[i] - gt[i])) for i in range(4) if not np.isnan(pr[i, 0]) and gt[i, 0] >= 0]
    re = [np.hypot(*(pr[i] - gt[i])) for i in range(4, 8) if not np.isnan(pr[i, 0]) and gt[i, 0] >= 0]
    return {"n_det": n_det, "cm": cm, "front": float(np.median(fr)) if fr else None,
            "rear": float(np.median(re)) if re else None, "pred8": pr}


def main():
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = collect_eval()
    print(f"[eval] {len(frames)} eval frames (누수 제외). domains: "
          f"{ {d: sum(1 for f in frames if f['dom']==d) for d in set(f['dom'] for f in frames)} }")

    avail = {k: v for k, v in MODELS.items() if os.path.isfile(os.path.join(ROOT, v))}
    res = {k: [] for k in avail}
    for k, wp in avail.items():
        model = T.E.load_model(os.path.join(ROOT, wp), device)
        for fr in frames:
            img = cv2.imread(fr["ip"])
            m = metrics(model, img, fr["gt"], device)
            m.update(dom=fr["dom"], fid=fr["fid"])
            res[k].append(m)
        del model
        torch.cuda.empty_cache() if device == "cuda" else None

    # ---- table: 도메인별 + 전체 (모델별 corner_med / front / rear / det% / good%) ----
    doms = sorted(set(f["dom"] for f in frames))
    rows = []
    for dom in doms + ["ALL"]:
        row = [dom]
        for k in avail:
            rs = [m for m in res[k] if dom == "ALL" or m["dom"] == dom]
            det = [m for m in rs if m["cm"] is not None]
            cm = np.median([m["cm"] for m in det]) if det else float("nan")
            rr = np.median([m["rear"] for m in det if m["rear"] is not None]) if det else float("nan")
            detr = 100 * len(det) / len(rs) if rs else 0
            good = 100 * sum(1 for m in det if m["cm"] < 10) / len(det) if det else 0
            row.append(f"{cm:.1f}/{rr:.0f}/{detr:.0f}%/{good:.0f}%")
        rows.append(row)
    cols = ["domain"] + [f"{k}\ncm/rear/det/good" for k in avail]
    fig, ax = plt.subplots(figsize=(3 + 3 * len(avail), 0.5 + 0.5 * len(rows)))
    ax.axis("off")
    tab = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
    tab.auto_set_font_size(False); tab.set_fontsize(10); tab.scale(1, 1.8)
    for j in range(len(cols)):
        tab[0, j].set_facecolor("#d9d9d9"); tab[0, j].set_text_props(weight="bold")
    ax.set_title(f"Self-train R0->R1->R2 eval (corner_med / rear / det pct / good pct)  n={len(frames)}, leak-free",
                 fontsize=11, pad=10)
    plt.savefig(os.path.join(OUT, "table.png"), dpi=150, bbox_inches="tight")
    print("[save] table.png")

    # ---- round curve (과거 plot_overiteration 방식): R0->R1->R2 도메인별 곡선 ----
    mkeys = list(avail.keys())
    xr = list(range(len(mkeys)))
    xlabels = ["Baseline\n(R0)", "Round 1\n(R1)", "Round 2\n(R2)"][:len(mkeys)]
    pl_counts = {"R0": 513, "R1": 192}  # R1 학습에 쓴 PL / R2 학습에 쓴 PL
    curve_doms = [d for d in doms if d != "cad"] + ["ALL"]   # cad 는 검출~0 제외
    colors = {"outside": "#2ca02c", "noapril": "#1f77b4", "night": "#d62728",
              "cad": "#9467bd", "ALL": "#000000"}

    def series(dom, field):
        ys = []
        for k in mkeys:
            rs = [m for m in res[k] if dom == "ALL" or m["dom"] == dom]
            det = [m for m in rs if m["cm"] is not None]
            if field == "good":
                ys.append(100 * sum(1 for m in det if m["cm"] < 10) / len(det) if det else 0)
            else:
                ys.append(float(np.median([m["cm"] for m in det])) if det else np.nan)
        return ys

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for dom in curve_doms:
        ys = series(dom, "good")
        lw = 3.0 if dom == "ALL" else 2.2
        a1.plot(xr, ys, marker="o", color=colors.get(dom, "#888"),
                linewidth=lw, markersize=10, label=dom,
                linestyle="--" if dom == "ALL" else "-")
        for i, y in enumerate(ys):
            a1.annotate(f"{y:.0f}", (i, y), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=9,
                        color=colors.get(dom, "#888"), fontweight="bold")
    a1.set_xticks(xr); a1.set_xticklabels(xlabels)
    a1.set_ylabel("good%  (corner_med < 10px, 정확 프레임 비율)", fontsize=11)
    a1.set_title("(a) Self-training round-by-round accuracy\nR0->R1->R2 across domains", fontsize=12)
    a1.legend(fontsize=10); a1.grid(alpha=0.3)

    for dom in curve_doms:
        ys = series(dom, "cm")
        lw = 3.0 if dom == "ALL" else 2.2
        a2.plot(xr, ys, marker="s", color=colors.get(dom, "#888"),
                linewidth=lw, markersize=9, label=dom,
                linestyle="--" if dom == "ALL" else "-")
        for i, y in enumerate(ys):
            if np.isfinite(y):
                a2.annotate(f"{y:.1f}", (i, y), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=9,
                            color=colors.get(dom, "#888"), fontweight="bold")
    a2.set_xticks(xr); a2.set_xticklabels(xlabels)
    a2.set_ylabel("corner_med (px, 낮을수록 정확)", fontsize=11)
    a2.set_title("(b) Corner error over rounds\nPL: R1=513, R2=192 frames", fontsize=12)
    a2.legend(fontsize=10); a2.grid(alpha=0.3)
    plt.suptitle(f"Filter-PL self-training  (eval={len(frames)} leak-free frames)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "round_curve.png"), dpi=140, bbox_inches="tight")
    print("[save] round_curve.png")

    # ---- cross-domain matrix (phase2_cross_domain 형식): 행=R0/R1/R2, 열=도메인 ----
    mat_doms = [d for d in ["outside", "noapril", "cad"] if d in doms]

    def nn20(k, dom):   # per-frame NN<20px % (det 실패=미달)
        rs = [m for m in res[k] if m["dom"] == dom]
        if not rs:
            return np.nan
        return 100 * sum(1 for m in rs if m["cm"] is not None and m["cm"] < 20) / len(rs)

    mat = np.array([[nn20(k, d) for d in mat_doms] for k in mkeys])
    fig, (m1, m2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                 gridspec_kw={"width_ratios": [1.3, 1]})
    im = m1.imshow(mat, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    m1.set_xticks(range(len(mat_doms))); m1.set_xticklabels(mat_doms, fontsize=11)
    m1.set_yticks(range(len(mkeys)))
    m1.set_yticklabels([k.split()[0] for k in mkeys], fontsize=11)
    m1.set_xlabel("Evaluation domain", fontsize=12)
    m1.set_title("(a) Round x domain (NN<20px per-frame, %)", fontsize=12)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                m1.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color="white" if v > 55 else "black", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=m1, fraction=0.046, pad=0.04, label="NN<20px (%)")
    # (b) best round per domain bar
    x = np.arange(len(mat_doms)); w = 0.4
    r0 = mat[0]
    best = [np.nanmax(mat[1:, j]) if mat.shape[0] > 1 else np.nan for j in range(len(mat_doms))]
    best_lbl = [mkeys[1 + int(np.nanargmax(mat[1:, j]))].split()[0] for j in range(len(mat_doms))]
    m2.bar(x - w/2, r0, w, label="R0 (baseline)", color="#bdbdbd", edgecolor="black")
    bb = m2.bar(x + w/2, best, w, label="best self-train round", color="#1f77b4", edgecolor="black")
    for j, b in enumerate(bb):
        m2.text(b.get_x()+b.get_width()/2, b.get_height()+1, f"{best[j]:.0f}", ha="center", fontsize=10, fontweight="bold")
        m2.text(b.get_x()+b.get_width()/2, best[j]/2, best_lbl[j], ha="center", va="center", fontsize=9, color="white", rotation=90)
    for j in range(len(mat_doms)):
        m2.text(x[j]-w/2, r0[j]+1, f"{r0[j]:.0f}", ha="center", fontsize=10, fontweight="bold")
    m2.set_xticks(x); m2.set_xticklabels(mat_doms, fontsize=11); m2.set_ylim(0, 105)
    m2.set_ylabel("NN<20px per-frame (%)", fontsize=12)
    m2.set_title("(b) Best per-domain (R0 vs best round)", fontsize=12)
    m2.legend(fontsize=10); m2.grid(axis="y", alpha=0.3, linestyle="--")
    fig.suptitle("Filter-PL self-training — round x domain transfer (combined-PL R0->R1->R2)\n"
                 f"eval={len(frames)} leak-free frames", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "cross_domain_matrix.png"), dpi=140, bbox_inches="tight")
    print("[save] cross_domain_matrix.png")

    # ---- overlay montage: 도메인별 대표 프레임 1~2장, R0/R1/R2 pred vs GT ----
    def draw(img, pr, gt, title):
        vis = img.copy()
        def P(p): return None if p is None or np.isnan(p[0]) or p[0] < 0 else (int(p[0]), int(p[1]))
        gp = [P(gt[i]) for i in range(8)]; pp = [P(pr[i]) for i in range(8)]
        for a, b in CUBOID_EDGES:
            if a < 8 and b < 8:
                if gp[a] and gp[b]: cv2.line(vis, gp[a], gp[b], (255, 120, 0), 2)
                if pp[a] and pp[b]: cv2.line(vis, pp[a], pp[b], (0, 255, 0), 2)
        for i in range(8):
            if gp[i]: cv2.circle(vis, gp[i], 4, (255, 120, 0), -1)
            if pp[i] is not None: cv2.circle(vis, pp[i], 3, (0, 255, 0), -1)
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(vis, title, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    picks = []
    for dom in doms:
        cand = [f for f in frames if f["dom"] == dom][:1]
        picks += cand
    if picks:
        fig, axes = plt.subplots(len(picks), len(avail), figsize=(4 * len(avail), 3 * len(picks)))
        axes = np.atleast_2d(axes)
        for r, fr in enumerate(picks):
            img = cv2.imread(fr["ip"])
            for c, k in enumerate(avail):
                m = next(x for x in res[k] if x["fid"] == fr["fid"] and x["dom"] == fr["dom"])
                cm = f"cm={m['cm']:.1f}" if m["cm"] is not None else "no-det"
                axes[r, c].imshow(draw(img, m["pred8"], fr["gt"], f"{fr['dom']} {k.split()[0]} {cm}"))
                axes[r, c].axis("off")
        plt.suptitle("blue=GT  green=pred    R0->R1->R2", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "overlays.png"), dpi=110, bbox_inches="tight")
        print("[save] overlays.png")
    print("\n".join([" | ".join(r) for r in rows]))
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
