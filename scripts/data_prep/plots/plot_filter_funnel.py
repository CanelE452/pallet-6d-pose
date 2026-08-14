"""Filter funnel — 추출 단계별 PL count 시각화."""
import os as _os, sys as _sys

# --- data_prep 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 먼저 실행돼야 하므로 최상단에 둔다.
_DP = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_DP] + [_os.path.join(_DP, _d) for _d in sorted(_os.listdir(_DP))
                         if _os.path.isdir(_os.path.join(_DP, _d)) and not _d.startswith(".")]

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


def main():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis("off")

    # 3 domain columns
    domains = [
        ("indoor", "#1f77b4", 1.5, [188, 20, 19]),
        ("outside", "#2ca02c", 5.0, [9894, 1432, 48]),
        ("night",   "#d62728", 8.5, [9134, 1506, 41]),
    ]
    stages = [
        ("Raw unlabeled\n(captured)", 9.2, 1.0),
        ("After [A][B][C] filter\n(≥5 kp + RANSAC c≥6 + LOO)", 6.4, 0.9),
        ("After strict v2\n(+spread + area + depth\n+ AR + FB + HL)", 3.5, 0.8),
    ]

    # 단계 라벨 (왼쪽)
    for i, (stage, y, _) in enumerate(stages):
        ax.text(0.1, y, stage, fontsize=10, fontweight="bold", va="center",
                ha="left", color="black")

    # 도메인별 box + 화살표
    for dname, color, x, counts in domains:
        # 도메인 제목
        ax.text(x, 9.85, dname, ha="center", fontsize=12, fontweight="bold", color=color)

        # 단계 box
        for i, (_, y, h) in enumerate(stages):
            v = counts[i]
            # box width scales with log count
            w = 0.4 + 0.7 * np.log10(v + 1) / np.log10(10000)
            xc, yc = x, y
            box = FancyBboxPatch((xc - w, yc - h/2), 2*w, h,
                                 boxstyle="round,pad=0.05",
                                 facecolor=color, edgecolor="black", alpha=0.55)
            ax.add_patch(box)
            ax.text(xc, yc, f"{v:,}", ha="center", va="center", fontsize=13,
                    fontweight="bold", color="white")

        # 화살표 + 통과율
        for i in range(len(stages) - 1):
            y_top = stages[i][1] - stages[i][2]/2 - 0.05
            y_bot = stages[i+1][1] + stages[i+1][2]/2 + 0.05
            ax.annotate("", xy=(x, y_bot), xytext=(x, y_top),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2))
            rate = counts[i+1] / counts[i] * 100
            ax.text(x + 0.55, (y_top + y_bot)/2,
                    f"{rate:.1f}%", color=color, fontsize=10, fontweight="bold",
                    va="center")

    # title
    fig.suptitle("Phase 2 — Filter funnel: how many PL survive each stage?\n"
                 "Strict v2 filter (6 criteria) keeps only top-quality pseudo-labels",
                 fontsize=13, y=0.98)

    plt.tight_layout()
    os.makedirs("_docs/figures", exist_ok=True)
    out = "_docs/figures/phase2_filter_funnel.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
