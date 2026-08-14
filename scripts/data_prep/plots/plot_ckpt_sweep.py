"""Phase 2 — R1 strict 학습 곡선 (ep65~80 sweep). over-training vs under-training 시각화."""
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
    eps = [60, 65, 70, 75, 80]  # 60 = R0 baseline

    # outside R1 strict (48 PL) on outside eval
    outside_y = [52.7, 53.5, 54.3, 51.2, 51.9]
    # indoor R1 strict (19 PL) on indoor eval
    indoor_y = [63.6, 70.0, 72.7, 73.6, 77.0]
    # night R1 strict (41 PL) on night eval
    night_y = [58.9, 58.9, 60.0, 46.7, 54.4]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # (a) indoor curve
    ax1.plot(eps, indoor_y, marker="s", color="#1f77b4",
             linewidth=2.4, markersize=10, label="R1 indoor-trained (19 PL)")
    for e, y in zip(eps, indoor_y):
        ax1.annotate(f"{y:.1f}", (e, y), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=10, fontweight="bold")
    ax1.axhline(63.6, color="gray", linestyle="--", linewidth=1.5, alpha=0.7,
                label="Baseline R0 (63.6%)")
    ax1.axvline(80, color="red", linestyle=":", linewidth=1.5, alpha=0.6,
                label="best ckpt (ep80)")
    ax1.set_xlabel("epoch", fontsize=12)
    ax1.set_ylabel("indoor NN<20px per-frame (%)", fontsize=11)
    ax1.set_title("(a) indoor — still improving at ep80", fontsize=12)
    ax1.legend(fontsize=10, loc="lower right")
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.set_ylim(60, 82)

    # (b) outside curve
    ax2.plot(eps, outside_y, marker="o", color="#2ca02c",
             linewidth=2.4, markersize=10, label="R1 outside-trained (48 PL)")
    for e, y in zip(eps, outside_y):
        ax2.annotate(f"{y:.1f}", (e, y), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=10, fontweight="bold")
    ax2.axhline(52.7, color="gray", linestyle="--", linewidth=1.5, alpha=0.7,
                label="Baseline R0 (52.7%)")
    ax2.axvline(70, color="red", linestyle=":", linewidth=1.5, alpha=0.6,
                label="best ckpt (ep70)")
    ax2.set_xlabel("epoch", fontsize=12)
    ax2.set_ylabel("outside NN<20px per-frame (%)", fontsize=11)
    ax2.set_title("(b) outside — over-training at ep75+", fontsize=12)
    ax2.legend(fontsize=10, loc="lower right")
    ax2.grid(alpha=0.3, linestyle="--")
    ax2.set_ylim(48, 60)

    # (c) night curve
    ax3.plot(eps, night_y, marker="^", color="#d62728",
             linewidth=2.4, markersize=10, label="R1 night-trained (41 PL)")
    for e, y in zip(eps, night_y):
        ax3.annotate(f"{y:.1f}", (e, y), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=10, fontweight="bold")
    ax3.axhline(58.9, color="gray", linestyle="--", linewidth=1.5, alpha=0.7,
                label="Baseline R0 (58.9%)")
    ax3.axvline(70, color="red", linestyle=":", linewidth=1.5, alpha=0.6,
                label="best ckpt (ep70)")
    ax3.set_xlabel("epoch", fontsize=12)
    ax3.set_ylabel("night NN<20px per-frame (%)", fontsize=11)
    ax3.set_title("(c) night — peak at ep70, sharp drop ep75", fontsize=12)
    ax3.legend(fontsize=10, loc="lower left")
    ax3.grid(alpha=0.3, linestyle="--")
    ax3.set_ylim(44, 64)

    fig.suptitle("Phase 2 — R1 training curves (ckpt sweep on each self-domain)\n"
                 "ep selection: indoor (more is better) ≠ outside/night (early stop optimal)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    os.makedirs("_docs/figures", exist_ok=True)
    out = "_docs/figures/phase2_ckpt_sweep.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
