"""지금까지 잰 것 한 장 — Direct-Hough 는 어디서 끊기나.

새 측정 0.  WHERE_BROKEN.json 과 학습 기록(direct_hough_long / overfit)만 읽어 그린다.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/pallet/results/hough_line_visual"
D = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation/"
     "compatibility_calibration/canonical_corner_audit/edge_mandatory_fast_search")

wb = json.load(open(OUT / "WHERE_BROKEN.json"))
long_ = json.load(open(D / "direct_hough_long.json"))
over = json.load(open(D / "direct_hough_overfit.json"))

# ---- 1) 일반화 계단
steps = [("OVERFIT32\n(32 frames memorised)", over["history"]["3000"]["OVERFIT32"]["angle_median"], "#2a9d3f"),
         ("own dev\n(train set deleted)", long_["history"]["8515"]["D2_LINE_DEV512"]["angle_median"], "#7cb342")]
order = ["mixed_v8_train", "aug_trunc_v2", "REAL_clean", "aug_squash_v2",
         "v4_split_base", "paper_4pallet_mask_v1", "aug_scale_v2"]
for k in order:
    v = wb["populations"].get(k)
    if v:
        col = "#1e6fd9" if k == "REAL_clean" else "#d94f2a"
        steps.append((k.replace("_", "\n"), v["angle_median"], col))

fig = plt.figure(figsize=(17, 9.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.28)

ax = fig.add_subplot(gs[0, :])
names = [s[0] for s in steps]; vals = [s[1] for s in steps]; cols = [s[2] for s in steps]
b = ax.bar(range(len(vals)), vals, color=cols, width=.62)
for i, v in enumerate(vals):
    ax.text(i, v + 1.2, f"{v:.2f}", ha="center", fontsize=10, weight="bold")
ax.axhline(45, ls="--", c="#888", lw=1.2)
ax.text(len(vals) - .4, 46.5, "45 deg = uniform angle (chance)", fontsize=9, ha="right", color="#555")
ax.axhline(1.0, ls=":", c="#2a9d3f", lw=1.4)
ax.text(0.05, 2.4, "1.0 deg = pre-registered gate", fontsize=9, color="#2a9d3f")
ax.axvspan(-0.5, 1.5, color="#2a9d3f", alpha=.07)
ax.axvspan(1.5, len(vals) - 0.5, color="#d94f2a", alpha=.06)
ax.text(0.5, 40, "inside training distribution", ha="center", fontsize=11, color="#2a9d3f", weight="bold")
ax.text((len(vals) + 1) / 2, 40, "outside it - this is where it breaks", ha="center",
        fontsize=11, color="#d94f2a", weight="bold")
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=8.5)
ax.set_ylabel("line angle error, median (deg)")
ax.set_title("One checkpoint, seven populations: the capacity is there, the generalisation is not\n"
             "green = inside training distribution (recorded)   red = other synthetic   blue = real", fontsize=12)
ax.set_ylim(0, 56)

# ---- 2) 오라클 (배선 검증)
ax = fig.add_subplot(gs[1, 0])
pops = [k for k in order if k in wb["populations"]]
ora = [wb["populations"][k]["oracle_angle_median"] for k in pops]
ax.barh(range(len(pops)), ora, color="#5a5a5a")
ax.axvline(0.5, ls="--", c="#2a9d3f", lw=1.5)
ax.text(0.52, len(pops) - 1.2, "lattice ceiling 0.5 deg", color="#2a9d3f", fontsize=9)
ax.set_yticks(range(len(pops)))
ax.set_yticklabels([p.replace("_", " ") for p in pops], fontsize=8)
ax.set_xlabel("oracle angle median (deg)")
ax.set_xlim(0, 0.75)
ax.set_title("wiring check: nearest-lattice oracle\nall <= 0.25 deg, so the measurement is sound", fontsize=10.5)

# ---- 3) >5도 비율
ax = fig.add_subplot(gs[1, 1])
frac = [wb["populations"][k]["frac_angle_gt5"] * 100 for k in pops]
ax.barh(range(len(pops)), frac,
        color=["#1e6fd9" if p == "REAL_clean" else "#d94f2a" for p in pops])
ax.set_yticks(range(len(pops))); ax.set_yticklabels([])
ax.set_xlabel("share of roles with angle error > 5 deg (%)")
ax.set_xlim(0, 100)
ax.set_title("almost every role is badly wrong", fontsize=10.5)
for i, v in enumerate(frac):
    ax.text(v + 1.5, i, f"{v:.0f}%", va="center", fontsize=8.5)

# ---- 4) role 별
ax = fig.add_subplot(gs[1, 2])
for k in pops:
    pr = wb["populations"][k].get("per_role_angle_median", {})
    if not pr:
        continue
    xs = sorted(int(r) for r in pr)
    ax.plot(xs, [pr[str(r)] for r in xs], marker="o", ms=3.5, lw=1.2,
            color="#1e6fd9" if k == "REAL_clean" else "#d94f2a",
            alpha=.95 if k == "REAL_clean" else .45,
            label=k if k in ("REAL_clean", "mixed_v8_train") else None)
ax.axhline(45, ls="--", c="#888", lw=1)
ax.set_xlabel("cuboid edge role (0-11)"); ax.set_ylabel("angle median (deg)")
ax.set_title("not one bad role - every role collapses", fontsize=10.5)
ax.legend(fontsize=8); ax.grid(alpha=.25)

fig.suptitle("Direct-Hough: where it breaks   (no new training, no new measurement)",
             fontsize=13.5)
p = OUT / "hough_generalisation.png"
fig.savefig(p, dpi=125, bbox_inches="tight")
print("wrote", p.relative_to(ROOT))
