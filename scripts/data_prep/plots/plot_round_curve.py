"""Phase 1 결과 figure — R0/R1/R2 라운드별 성능 곡선 + PL 수 변화.

results JSON 파일을 읽어 plot. JSON 형식:
{
  "indoor":  {"R0": {"nn20": 18.9, "pl": 0},   "R1": {"nn20": 60.5, "pl": 2},   "R2": {"nn20": ..., "pl": ...}},
  "outside": {"R0": ..., "R1": ..., "R2": ...},
  "night":   {"R0": ..., "R1": ..., "R2": ...}
}

사용:
    python scripts/data_prep/plots/plot_round_curve.py \\
        --results _docs/experiments/self_training/phase1_results.json \\
        --output _docs/figures/phase1_round_curve.png
"""
import os as _os, sys as _sys

# --- data_prep 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 먼저 실행돼야 하므로 최상단에 둔다.
_DP = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_DP] + [_os.path.join(_DP, _d) for _d in sorted(_os.listdir(_DP))
                         if _os.path.isdir(_os.path.join(_DP, _d)) and not _d.startswith(".")]

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--metric", default="nn20", choices=["nn20", "nn10", "nn5"],
                   help="metric to plot")
    args = p.parse_args()

    with open(args.results, encoding="utf-8") as f:
        results = json.load(f)

    rounds = ["Baseline\n(R0)", "Round 1\n(R1)", "Round 2\n(R2)", "Round 3\n(R3)"]
    domains = ["indoor", "outside", "night"]
    colors = {"indoor": "#1f77b4", "outside": "#2ca02c", "night": "#d62728"}
    markers = {"indoor": "o", "outside": "s", "night": "^"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- (1) NN<20px 곡선 ---
    key_map = {"Baseline\n(R0)": "R0", "Round 1\n(R1)": "R1",
               "Round 2\n(R2)": "R2", "Round 3\n(R3)": "R3"}
    for domain in domains:
        if domain not in results:
            continue
        ys = []
        for r in rounds:
            v = results[domain].get(key_map[r], {}).get(args.metric)
            ys.append(v if v is not None else np.nan)
        ax1.plot(rounds, ys, marker=markers[domain], color=colors[domain],
                 linewidth=2.2, markersize=10, label=domain)
        for i, y in enumerate(ys):
            if not np.isnan(y):
                ax1.annotate(f"{y:.1f}", (i, y),
                             textcoords="offset points", xytext=(0, 9),
                             ha="center", fontsize=9, color=colors[domain])

    ax1.set_xlabel("Self-training round", fontsize=12)
    ylabel = {"nn20": "NN matching <20px (%)",
              "nn10": "NN matching <10px (%)",
              "nn5":  "NN matching <5px (%)"}[args.metric]
    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.set_title("(a) Round-by-round performance", fontsize=12)
    ax1.legend(loc="lower right", fontsize=11)
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.set_ylim(0, 100)

    # --- (2) PL 수 변화 (R0 의 PL pool size) ---
    width = 0.25
    x = np.arange(len(domains))
    for i, r in enumerate(rounds):
        ys = []
        for domain in domains:
            v = results.get(domain, {}).get(key_map[r], {}).get("pl", 0)
            ys.append(v if v is not None else 0)
        colors = ["#aec7e8", "#7fcdb4", "#ff9896", "#c5b0d5"]
        bars = ax2.bar(x + (i - 1.5) * width, ys, width, label=r.replace("\n", " "),
                       color=colors[i % len(colors)], edgecolor="black", linewidth=0.5)
        for j, b in enumerate(bars):
            h = b.get_height()
            if h > 0:
                ax2.annotate(f"{int(h)}", (b.get_x() + b.get_width() / 2, h),
                             textcoords="offset points", xytext=(0, 3),
                             ha="center", fontsize=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(domains)
    ax2.set_ylabel("Pseudo-labels passed (count)", fontsize=12)
    ax2.set_title("(b) PL pool size per round", fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    # --- Suptitle ---
    fig.suptitle("Phase 1 — Self-training iteration across domains",
                 fontsize=14, y=1.02)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
