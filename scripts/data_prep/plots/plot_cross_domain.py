"""Phase 1 cross-domain transfer heatmap + best-per-domain bar chart."""
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

    # Cross-domain transfer matrix (NN<20px per-frame, %)
    models = [
        "Baseline\n(R0)",
        "R1\n(indoor PL)",
        "R1\n(outside PL)",
        "R1\n(night PL)",
        "R2\n(indoor PL)",
        "R2\n(outside PL)",
        "R2\n(night PL)",
    ]
    domains = ["indoor", "outside", "night"]
    # rows = models, cols = eval domains
    data = np.array([
        [21.6, 27.9, 21.1],   # R0
        [60.5, 31.8, 32.2],   # R1_indoor (F5)
        [58.4, 39.5, 33.3],   # R1_outside_loo
        [43.9, 22.5, 26.7],   # R1_night_loo
        [30.5, 11.6, 21.1],   # R2_indoor
        [15.9, 24.8, 11.1],   # R2_outside
        [33.4, 20.2, 26.7],   # R2_night
    ])

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.3, 1]}
    )

    # --- (a) Cross-domain heatmap ---
    im = ax1.imshow(data, cmap="YlGnBu", vmin=0, vmax=70, aspect="auto")
    ax1.set_xticks(np.arange(len(domains)))
    ax1.set_yticks(np.arange(len(models)))
    ax1.set_xticklabels(domains, fontsize=11)
    ax1.set_yticklabels(models, fontsize=10)
    ax1.set_xlabel("Evaluation domain", fontsize=12)
    ax1.set_title("(a) Cross-domain transfer (NN<20px per-frame, %)", fontsize=12)

    # annotate
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            color = "white" if val > 35 else "black"
            ax1.text(j, i, f"{val:.1f}", ha="center", va="center",
                     color=color, fontsize=10, fontweight="bold")

    # mark self-domain (diagonal) with red border
    # R0 → all 3 are "baseline" not self-domain
    # R1_indoor → col 0 (indoor)
    # R1_outside → col 1 (outside)
    # R1_night → col 2 (night)
    # R2_indoor → col 0
    # R2_outside → col 1
    # R2_night → col 2
    self_cells = [(1, 0), (2, 1), (3, 2), (4, 0), (5, 1), (6, 2)]
    for (r, c) in self_cells:
        ax1.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                     fill=False, edgecolor="red", linewidth=2))

    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="NN<20px (%)")

    # --- (b) Best per-domain bar ---
    width = 0.27
    x = np.arange(3)

    r0 = [21.6, 27.9, 21.1]
    best_r1 = [60.5, 39.5, 33.3]
    best_r1_label = ["indoor", "outside", "outside"]
    best_r2 = [33.4, 24.8, 26.7]
    best_r2_label = ["night PL", "outside PL", "night PL"]

    b1 = ax2.bar(x - width, r0, width, label="Baseline (R0)", color="#bdbdbd", edgecolor="black", linewidth=0.5)
    b2 = ax2.bar(x, best_r1, width, label="Best R1 (1 round ST)", color="#1f77b4", edgecolor="black", linewidth=0.5)
    b3 = ax2.bar(x + width, best_r2, width, label="Best R2 (2 rounds ST)", color="#ff7f0e", edgecolor="black", linewidth=0.5)

    for bars, labels in [(b1, [None]*3), (b2, best_r1_label), (b3, best_r2_label)]:
        for b, lbl in zip(bars, labels):
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width()/2, h + 1.2, f"{h:.1f}",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
            if lbl:
                ax2.text(b.get_x() + b.get_width()/2, h/2, lbl,
                         ha="center", va="center", fontsize=7, color="white",
                         rotation=90)

    ax2.set_xticks(x)
    ax2.set_xticklabels(domains, fontsize=11)
    ax2.set_ylabel("NN<20px per-frame (%)", fontsize=12)
    ax2.set_title("(b) Best per-domain across rounds", fontsize=12)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.set_ylim(0, 75)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle("Phase 1 — Cross-domain transfer matrix & best per-domain summary\n"
                 "(red box: self-domain cell; off-diagonal R1 cells show domain transfer)",
                 fontsize=12, y=1.01)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
