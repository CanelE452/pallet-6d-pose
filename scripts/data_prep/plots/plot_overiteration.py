"""Phase 1 — Over-iteration evidence figure.

outside-trained R0 → R1 → R2 → R3 의 3 도메인 평가 곡선 (cross-domain transfer 까지 포함).
+ PnP success rate (R0 vs R1, 6D metric).
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- data_prep 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 먼저 실행돼야 하므로 최상단에 둔다.
_DP = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_DP] + [_os.path.join(_DP, _d) for _d in sorted(_os.listdir(_DP))
                         if _os.path.isdir(_os.path.join(_DP, _d)) and not _d.startswith(".")]

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()

    rounds = ["Baseline\n(R0)", "Round 1\n(R1, 167 PL)", "Round 2\n(R2, 514 PL)", "Round 3\n(R3, 900 PL)"]

    # outside-anchored model 의 3 도메인 NN<20px (per-frame)
    indoor   = [21.6, 58.4, 15.9, 4.8]
    outside  = [27.9, 39.5, 24.8, 21.7]
    night    = [21.1, 33.3, 11.1, 5.6]

    # PL pool
    pl_count = [None, 167, 514, 900]  # R0 anchor 는 PL 없음

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                    gridspec_kw={"width_ratios": [1.4, 1]})

    # --- (a) Round-by-round line plot (3 domains, outside-anchored) ---
    colors = {"indoor": "#1f77b4", "outside": "#2ca02c", "night": "#d62728"}
    markers = {"indoor": "o", "outside": "s", "night": "^"}
    data = {"indoor": indoor, "outside": outside, "night": night}
    for d in ["indoor", "outside", "night"]:
        ax1.plot(rounds, data[d], marker=markers[d], color=colors[d],
                 linewidth=2.5, markersize=11, label=d)
        for i, y in enumerate(data[d]):
            ax1.annotate(f"{y:.1f}", (i, y), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=9,
                         color=colors[d], fontweight="bold")

    ax1.axvspan(0.5, 1.5, color="#fff2cc", alpha=0.5, label="R1 zone (best)")
    ax1.set_xlabel("Self-training round (outside-anchored model)", fontsize=12)
    ax1.set_ylabel("NN<20px per-frame (%)", fontsize=12)
    ax1.set_title("(a) Round-by-round decay across domains\n"
                  "outside-trained model evaluated on 3 domains",
                  fontsize=12)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.set_ylim(0, 70)

    # --- (b) PL pool + PnP success rate ---
    # 좌측 axis: PL pool count (line)
    # 우측 axis: PnP success rate (bar)
    rounds_short = ["R0", "R1", "R2", "R3"]
    ax2.plot(rounds_short, [0 if v is None else v for v in pl_count],
             marker="D", color="#ff7f0e", linewidth=2.5, markersize=10,
             label="PL pool size")
    for i, v in enumerate(pl_count):
        if v:
            ax2.annotate(f"{v}", (i, v), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=10,
                         color="#ff7f0e", fontweight="bold")
    ax2.set_xlabel("Self-training round", fontsize=12)
    ax2.set_ylabel("Pseudo-label pool size", color="#ff7f0e", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")
    ax2.set_title("(b) PL pool grows but performance decays\n"
                  "  (quality < quantity, confirmation bias)",
                  fontsize=12)
    ax2.grid(alpha=0.3, linestyle="--", axis="y")
    ax2.set_ylim(0, 1000)
    ax2.legend(loc="upper left", fontsize=10)

    fig.suptitle("Phase 1 — Self-training over-iteration evidence\n"
                 "R1 is optimal; R2/R3 decay despite growing PL pool",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
