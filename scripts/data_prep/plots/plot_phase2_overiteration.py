"""Phase 2 over-iteration figure — R0 → R1 → R2 outside-anchored 3 도메인 평가."""
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
    rounds = ["Baseline\n(R0)", "Round 1\n(R1, 48 PL)", "Round 2\n(R2, 137 PL)"]
    indoor   = [63.6, 67.0, 60.9]
    outside  = [52.7, 54.3, 46.5]
    night    = [58.9, 48.9, 38.9]
    pl_count = [None, 48, 137]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                    gridspec_kw={"width_ratios": [1.4, 1]})

    colors = {"indoor": "#1f77b4", "outside": "#2ca02c", "night": "#d62728"}
    markers = {"indoor": "o", "outside": "s", "night": "^"}
    data = {"indoor": indoor, "outside": outside, "night": night}
    for d in ["indoor", "outside", "night"]:
        ax1.plot(rounds, data[d], marker=markers[d], color=colors[d],
                 linewidth=2.5, markersize=12, label=d)
        for i, y in enumerate(data[d]):
            ax1.annotate(f"{y:.1f}", (i, y), textcoords="offset points",
                         xytext=(0, 11), ha="center", fontsize=10,
                         color=colors[d], fontweight="bold")
    ax1.axvspan(0.5, 1.5, color="#fff2cc", alpha=0.5, label="R1 zone (best)")
    ax1.set_xlabel("Self-training round (outside-anchored model)", fontsize=12)
    ax1.set_ylabel("NN<20px per-frame (%)", fontsize=12)
    ax1.set_title("(a) Round-by-round across domains\n"
                  "Phase 2 (strong base + strict PL): same over-iteration pattern",
                  fontsize=12)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.set_ylim(35, 75)

    rounds_short = ["R0", "R1", "R2"]
    ax2.plot(rounds_short, [0 if v is None else v for v in pl_count],
             marker="D", color="#ff7f0e", linewidth=2.5, markersize=12,
             label="strict PL pool size")
    for i, v in enumerate(pl_count):
        if v:
            ax2.annotate(f"{v}", (i, v), textcoords="offset points",
                         xytext=(0, 11), ha="center", fontsize=11,
                         color="#ff7f0e", fontweight="bold")
    ax2.set_xlabel("Self-training round", fontsize=12)
    ax2.set_ylabel("Pseudo-label pool size", color="#ff7f0e", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")
    ax2.set_title("(b) PL pool grows R1 48 → R2 137 (2.8×)\n"
                  "but performance decays in (a)", fontsize=12)
    ax2.grid(alpha=0.3, linestyle="--", axis="y")
    ax2.set_ylim(0, 180)
    ax2.legend(loc="upper left", fontsize=10)

    fig.suptitle("Phase 2 — Self-training over-iteration (Proposed base + strict PL)\n"
                 "R1 is optimal; R2 decays despite growing strict PL pool — same pattern as Phase 1",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    os.makedirs("_docs/figures", exist_ok=True)
    out = "_docs/figures/phase2_overiteration.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
