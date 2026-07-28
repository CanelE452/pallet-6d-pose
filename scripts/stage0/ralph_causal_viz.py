"""ralph_causal_viz.py — R0 vs synthetic-only 대조군 vs PL 모델 정직 비교.

eval_causal.json 재사용(재추론 없음). nn20 를 도메인별 막대로.
핵심 메시지: corner_med 개선은 synthetic anchor(대조군) 몫, nn20 개선은 PL 몫.

Usage: python -u scripts/stage0/ralph_causal_viz.py
"""
from __future__ import annotations
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
RS = os.path.join(ROOT, "data/pallet/results/ralph_selftrain")
JSON = os.environ.get("VIZ_JSON", "eval_causal.json")
TAG = os.environ.get("VIZ_TAG", "challenge0123")
R0K = os.environ.get("VIZ_R0", "R0")
CTRLK = os.environ.get("VIZ_CTRL", "CTRL_synOnly")
PLK = os.environ.get("VIZ_PL", "PL_night")
OUTNAME = os.environ.get("VIZ_OUT", "causal_split.png")
J = json.load(open(os.path.join(RS, JSON)))
res = J["results"]
DOMS = ["outside", "night", "noapril", "cad"]
ROWS = [(f"R0 ({TAG})", R0K, "#7f7f7f"),
        ("CTRL synthetic-only", CTRLK, "#ff7f0e"),
        (f"{PLK} (self-train)", PLK, "#1f77b4")]


def main():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(len(DOMS)); w = 0.26
    for a, key, ylabel, fmt in [
        (a1, "nn20_pct", "NN<20px per-frame (%)  — higher better", "{:.0f}"),
        (a2, "corner_med", "corner error median (px)  — lower better", "{:.1f}"),
    ]:
        for k, (lbl, rk, col) in enumerate(ROWS):
            vals = [res[rk]["domains"][d][key] or 0 for d in DOMS]
            b = a.bar(x + (k - 1) * w, vals, w, label=lbl, color=col, edgecolor="black")
            for j, bar in enumerate(b):
                a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                       fmt.format(vals[j]), ha="center", fontsize=8, fontweight="bold")
        a.set_xticks(x); a.set_xticklabels(DOMS, fontsize=12)
        a.set_ylabel(ylabel, fontsize=11); a.legend(fontsize=9)
        a.grid(axis="y", alpha=0.3)
    ov = {rk: res[rk]["overall"]["nn20_pct"] for _, rk, _ in ROWS}
    a1.set_title(f"(a) NN20: PL beats control (self-train effect)\nOVERALL nn20  "
                 f"R0={ov[R0K]}  CTRL={ov[CTRLK]}  PL={ov[PLK]}", fontsize=11)
    a2.set_title("(b) corner_med: control already gets it (synthetic anchor, NOT PL)", fontsize=11)
    fig.suptitle(f"Causal split — self-training(PL) vs synthetic-only control ({TAG}, R2, THRESH0.1)\n"
                 "eval GT held-out; night nn20 gain is PL-driven (control does not reproduce it)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    out = os.path.join(RS, OUTNAME)
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print("[save]", out)


if __name__ == "__main__":
    main()
