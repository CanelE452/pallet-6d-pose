"""paper_s2_audit_figures.py — 감사 필수 그림 생성 (Step 3/5/6/7)."""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]


import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "data", "pallet", "results",
                   "paper_s2_target_semantics_audit")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

tgt = pd.read_parquet(os.path.join(OUT, "target_semantics_keypoints.parquet"))
ci = tgt.center_inside_belief
nz = tgt.belief_target_nonzero
m1 = tgt.belief_channel_mask == 1
tgt["is_C2"] = ci & ~nz & m1

pop = pd.read_parquet(os.path.join(OUT, "truncation_populations.parquet"))
dec = pd.read_csv(os.path.join(OUT, "decoder_parity.csv"))
funnel = pd.read_csv(os.path.join(OUT, "diffpnp_funnel.csv"))


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print("[fig]", p)


# 1. target_contradiction_by_kp
g = tgt.groupby("keypoint_id").is_C2.mean().mul(100)
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.bar(g.index.astype(str), g.values, color="tab:red")
ax.set_xlabel("keypoint id (camera-facing v4: 0-3 near, 4-7 far, 8 centroid)")
ax.set_ylabel("C2 rate (%)")
ax.set_title("C2 = centre inside belief map, target all-zero, mask=1\n"
             "(ep57 training data, sigma=2 -> dead band w=4px)")
for i, v in enumerate(g.values):
    ax.text(i, v + 0.08, f"{v:.2f}", ha="center", fontsize=7)
save(fig, "target_contradiction_by_kp.png")

# 2. target_contradiction_by_dataset
g = tgt.groupby("dataset").is_C2.mean().mul(100).sort_values()
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.barh(g.index, g.values, color="tab:orange")
ax.set_xlabel("C2 rate (%)")
ax.set_title("C2 rate by dataset")
for i, v in enumerate(g.values):
    ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=8)
save(fig, "target_contradiction_by_dataset.png")

# 3. border_distance_distribution (C2 vs bin)
ins = tgt[ci].copy()
ins["bin"] = pd.cut(ins.dist_to_border, [-0.001, 1, 2, 3, 4, 6, 10, 100],
                    labels=["0-1", "1-2", "2-3", "3-4", "4-6", "6-10", "10+"])
t = ins.groupby("bin", observed=True).is_C2.mean().mul(100)
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.bar(t.index.astype(str), t.values, color="tab:blue")
ax.axvline(3.5, color="r", ls="--", label="w = int(2*sigma) = 4")
ax.set_xlabel("distance to belief-map border (belief px)")
ax.set_ylabel("C2 rate (%)")
ax.set_title("C2 is a deterministic function of border distance")
ax.legend()
save(fig, "border_distance_distribution.png")

# 4. trunc_real_vs_aug_distribution
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
order = ["P0_synth_nontrunc", "P1_aug_trunc_v2", "P2b_real_nontrunc",
         "P2_real_filterval"]
data = [pop[pop.population == p].dist_to_border.dropna().values for p in order]
ax[0].boxplot(data, labels=[p.replace("_", "\n") for p in order], showfliers=False)
ax[0].axhline(4, color="r", ls="--", label="dead band w=4")
ax[0].set_ylabel("distance to border (belief px)")
ax[0].set_title("keypoint border distance")
ax[0].legend(fontsize=7)
band = [100 * pop[pop.population == p].in_band.mean() for p in order]
outs = [100 * (~pop[pop.population == p].center_inside_belief).mean()
        for p in order]
x = np.arange(len(order))
ax[1].bar(x - 0.2, band, 0.4, label="in central 20-80% band (%)")
ax[1].bar(x + 0.2, outs, 0.4, label="outside belief map (%)", color="tab:red")
ax[1].set_xticks(x)
ax[1].set_xticklabels([p.replace("_", "\n") for p in order], fontsize=7)
ax[1].legend(fontsize=7)
ax[1].set_title("aug_trunc_v2 vs real truncated")
save(fig, "trunc_real_vs_aug_distribution.png")

# 5. diffpnp_valid_funnel
f = funnel[funnel.dataset != "val"].set_index("dataset")
fig, ax = plt.subplots(figsize=(8.5, 4))
x = np.arange(len(f))
ax.bar(x - 0.25, f.pnp_rate, 0.25, label="pnp_valid_3d %")
ax.bar(x, f.v8_rate, 0.25, label="V8 %")
ax.bar(x + 0.25, f.diffpnp_valid_rate, 0.25, label="DiffPnP valid %",
       color="tab:red")
ax.set_xticks(x)
ax.set_xticklabels(f.index, rotation=20, ha="right", fontsize=7)
ax.axhline(10, color="k", ls=":", label="10% = 사실상 미적용")
ax.set_ylabel("%")
ax.set_title("DiffPnP3D coverage funnel (ep57 index, actually consumed)")
ax.legend(fontsize=7)
save(fig, "diffpnp_valid_funnel.png")

# 6. decoder_coordinate_difference
fig, ax = plt.subplots(figsize=(7, 3.6))
cols = ["diff_d0_d1", "diff_d0_d2", "diff_d1_d2", "diff_d2_d3"]
ax.boxplot([dec[c].dropna().values for c in cols], labels=cols, showfliers=False)
ax.axhline(7.032, color="r", ls="--", label="0.4395 offset = 7.03 px")
ax.set_ylabel("coordinate difference (orig px)")
ax.set_title("decoder parity on identical ep57 heatmaps (strict N87)")
ax.legend(fontsize=7)
save(fig, "decoder_coordinate_difference.png")

# 7. decoder_missing_disagreement
g = dec.groupby("keypoint_id").d2_missing.mean().mul(100)
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.bar(g.index.astype(str), g.values, color="tab:purple")
ax.set_xlabel("keypoint id")
ax.set_ylabel("D2 missing rate (%)")
ax.set_title("eval decoder declares missing; training decoders (D0/D1) never do")
save(fig, "decoder_missing_disagreement.png")

# 8. border_duplicate_count
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.scatter(dec.dist_to_border_peak, dec.d0_window_duplicate, s=8, alpha=0.5)
ax.set_xlabel("peak distance to belief border (px)")
ax.set_ylabel("D0 clamp duplicate count (of 49)")
ax.set_title(f"D0 7x7 clamp duplication — only "
             f"{(dec.d0_window_duplicate>0).sum()}/{len(dec)} keypoints affected")
save(fig, "border_duplicate_count.png")

# 9. ep57 on real: truncated vs not
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
for i, (col, lbl) in enumerate([("peak", "belief peak"),
                                ("err_d2", "index-wise err (px)")]):
    d = [dec[~dec.frame_is_truncated][col].dropna(),
         dec[dec.frame_is_truncated][col].dropna()]
    ax[i].boxplot(d, labels=["non-trunc\n(70 fr)", "truncated\n(17 fr)"],
                  showfliers=False)
    ax[i].set_title(lbl)
fig.suptitle("ep57 on strict filter-val N87: collapse on real truncated frames",
             fontsize=10)
save(fig, "ep57_real_truncated_collapse.png")

print("done")
