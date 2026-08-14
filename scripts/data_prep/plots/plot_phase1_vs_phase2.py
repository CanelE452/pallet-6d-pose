"""Phase 1 (Baseline base) vs Phase 2 (Proposed base) — 종합 비교."""
import os as _os, sys as _sys

# --- data_prep 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 먼저 실행돼야 하므로 최상단에 둔다.
_DP = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_DP] + [_os.path.join(_DP, _d) for _d in sorted(_os.listdir(_DP))
                         if _os.path.isdir(_os.path.join(_DP, _d)) and not _d.startswith(".")]

import os
import matplotlib.pyplot as plt
import numpy as np


def main():
    domains = ["indoor", "outside", "night"]

    # Phase 1: v8_A_coord base
    p1_r0 = [21.6, 27.9, 21.1]
    p1_r1 = [60.5, 39.5, 33.3]   # best per domain
    # Phase 2: challenge0123 base (strong base)
    p2_r0 = [63.6, 52.7, 58.9]
    p2_r1 = [77.0, 54.3, 60.0]   # best per domain (self-domain trained)

    x = np.arange(3); w = 0.20
    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - 1.5*w, p1_r0, w, label="Phase 1 — Baseline R0 (synthetic only)",
                color="#d9d9d9", edgecolor="black")
    b2 = ax.bar(x - 0.5*w, p1_r1, w, label="Phase 1 — best R1 (1 round ST)",
                color="#9ecae1", edgecolor="black")
    b3 = ax.bar(x + 0.5*w, p2_r0, w, label="Phase 2 — Baseline R0 (camera-facing)",
                color="#bdbdbd", edgecolor="black")
    b4 = ax.bar(x + 1.5*w, p2_r1, w, label="Phase 2 — best R1 (strict PL + 1 round ST)",
                color="#1f77b4", edgecolor="black")

    for bars in [b1, b2, b3, b4]:
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 1.0, f"{h:.1f}",
                    ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(domains, fontsize=12)
    ax.set_ylabel("NN<20px per-frame (%)", fontsize=12)
    ax.set_title("Phase 1 vs Phase 2 — base upgrade contribution vs Self-training contribution",
                 fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(0, 90)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # annotate gains
    for j in range(3):
        gain_base = p2_r0[j] - p1_r0[j]
        ax.annotate(f"+{gain_base:.1f}",
                    xy=(j + 0.5*w, p2_r0[j]),
                    xytext=(j + 0.5*w, p2_r0[j] - 5),
                    ha="center", fontsize=9, color="darkgreen", fontweight="bold")

    plt.tight_layout()
    os.makedirs("_docs/figures", exist_ok=True)
    out = "_docs/figures/phase1_vs_phase2.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
