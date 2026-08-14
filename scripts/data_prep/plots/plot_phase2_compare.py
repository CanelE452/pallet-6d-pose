"""Phase 2 — base 모델 비교: v8_A_coord vs challenge0123 (camera-facing)."""
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

    domains = ["indoor", "outside", "night"]
    nn20_v8   = [21.6, 27.9, 21.1]
    nn20_cf   = [63.6, 52.7, 58.9]
    pnp_v8    = [44.3, 31.0, 17.8]
    pnp_cf    = [31.1, 65.9, 60.0]
    reproj_v8 = [317.6, 133.7, 108.1]
    reproj_cf = [223.2,  72.6,   0.0]   # night reproj 미수집

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(3); w = 0.35

    # (a) NN<20px
    label_base = "Baseline (synthetic only)"
    label_prop = "Proposed (synthetic + camera-facing GT)"
    ax = axes[0]
    b1 = ax.bar(x - w/2, nn20_v8, w, label=label_base, color="#bdbdbd", edgecolor="black")
    b2 = ax.bar(x + w/2, nn20_cf, w, label=label_prop, color="#1f77b4", edgecolor="black")
    for bars in [b1, b2]:
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x()+b.get_width()/2, h+1.5, f"{h:.1f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(domains)
    ax.set_ylabel("NN<20px per-frame (%)", fontsize=11)
    ax.set_title("(a) Keypoint NN<20px", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 75); ax.grid(axis="y", alpha=0.3, linestyle="--")

    # (b) PnP success
    ax = axes[1]
    b1 = ax.bar(x - w/2, pnp_v8, w, label=label_base, color="#bdbdbd", edgecolor="black")
    b2 = ax.bar(x + w/2, pnp_cf, w, label=label_prop, color="#2ca02c", edgecolor="black")
    for bars in [b1, b2]:
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x()+b.get_width()/2, h+1.5, f"{h:.1f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(domains)
    ax.set_ylabel("PnP success rate (%)", fontsize=11)
    ax.set_title("(b) 6D PnP success rate", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 75); ax.grid(axis="y", alpha=0.3, linestyle="--")

    # (c) Reproj error
    ax = axes[2]
    b1 = ax.bar(x - w/2, reproj_v8, w, label=label_base, color="#bdbdbd", edgecolor="black")
    b2 = ax.bar(x + w/2, reproj_cf, w, label=label_prop, color="#d62728", edgecolor="black")
    for bars, vals in [(b1, reproj_v8), (b2, reproj_cf)]:
        for b, v in zip(bars, vals):
            h = b.get_height()
            label = f"{v:.0f}" if v > 0 else "N/A"
            ax.text(b.get_x()+b.get_width()/2, h+5, label, ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(domains)
    ax.set_ylabel("Reproj error mean (px)", fontsize=11)
    ax.set_title("(c) Reprojection error (PnP fit)", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle("Base model upgrade — Baseline vs Proposed (camera-facing keypoint convention)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
