"""Phase 2 — cross-domain transfer matrix + best per-domain (phase1 형식)."""
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
    p.add_argument("--include_night", action="store_true",
                   help="include R1 night PL row (needs night R1 result)")
    p.add_argument("--night_results", nargs="+", type=float, default=None,
                   help="night R1: indoor outside night")
    args = p.parse_args()

    models = [
        "Baseline\n(R0)",
        "R1\n(indoor PL,\n19 PL)",
        "R1\n(outside PL,\n48 PL)",
    ]
    self_cells = [(1, 0), (2, 1)]

    data = [
        [63.6, 52.7, 58.9],   # R0 (challenge0123)
        [77.0, 48.1, 50.0],   # R1 indoor strict ep80
        [67.0, 54.3, 48.9],   # R1 outside strict ep70 best
    ]

    if args.include_night and args.night_results:
        models.append("R1\n(night PL,\n41 PL)")
        data.append(args.night_results)
        self_cells.append((3, 2))

    # Phase 2 R2 (outside-anchored over-iteration)
    models.append("R2\n(outside PL,\n137 PL)")
    data.append([60.9, 46.5, 38.9])
    self_cells.append((len(models) - 1, 1))

    data = np.array(data)
    domains = ["indoor", "outside", "night"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.3, 1]}
    )

    im = ax1.imshow(data, cmap="YlGnBu", vmin=0, vmax=80, aspect="auto")
    ax1.set_xticks(np.arange(len(domains)))
    ax1.set_yticks(np.arange(len(models)))
    ax1.set_xticklabels(domains, fontsize=11)
    ax1.set_yticklabels(models, fontsize=10)
    ax1.set_xlabel("Evaluation domain", fontsize=12)
    ax1.set_title("(a) Cross-domain transfer (NN<20px per-frame, %)", fontsize=12)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            color = "white" if val > 50 else "black"
            ax1.text(j, i, f"{val:.1f}", ha="center", va="center",
                     color=color, fontsize=11, fontweight="bold")

    for (r, c) in self_cells:
        ax1.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                     fill=False, edgecolor="red", linewidth=2))

    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="NN<20px (%)")

    # (b) Best per-domain bar
    width = 0.40
    x = np.arange(3)
    r0 = data[0]
    # best per domain across R1 rows
    best_r1 = [float(data[1:, j].max()) for j in range(3)]
    best_r1_label = []
    label_models = [m.split("\n")[1].strip("(").strip(",").strip() for m in models[1:]]
    for j in range(3):
        idx = int(np.argmax(data[1:, j]))
        best_r1_label.append(label_models[idx])

    b1 = ax2.bar(x - width/2, r0, width, label="Baseline (R0)",
                 color="#bdbdbd", edgecolor="black")
    b2 = ax2.bar(x + width/2, best_r1, width, label="Best R1 (strict PL)",
                 color="#1f77b4", edgecolor="black")
    for bars, labels in [(b1, [None]*3), (b2, best_r1_label)]:
        for b, lbl in zip(bars, labels):
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width()/2, h + 1, f"{h:.1f}",
                     ha="center", fontsize=10, fontweight="bold")
            if lbl:
                ax2.text(b.get_x() + b.get_width()/2, h/2, lbl,
                         ha="center", va="center", fontsize=8,
                         color="white", rotation=90)
    ax2.set_xticks(x)
    ax2.set_xticklabels(domains, fontsize=11)
    ax2.set_ylabel("NN<20px per-frame (%)", fontsize=12)
    ax2.set_title("(b) Best per-domain (Baseline vs best R1)", fontsize=12)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.set_ylim(0, 85)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle("Phase 2 — cross-domain transfer matrix & best per-domain summary\n"
                 "(red box: self-domain cell; strong base + strict-filtered PL)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
