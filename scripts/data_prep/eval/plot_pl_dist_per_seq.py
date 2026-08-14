"""PL pool 의 시퀀스 분포 시각화 — 학습 데이터 다양성 분석."""
import argparse
import json
import os
import collections
import matplotlib.pyplot as plt
import numpy as np


def count_seq(log_path, tag):
    with open(log_path) as f:
        log = json.load(f)
    cnt = collections.Counter()
    for item in log:
        p = item['src_image'].replace(os.sep, '/')
        parts = p.split('/')
        for token in ("outside", "night"):
            if token in parts:
                i = parts.index(token)
                if i + 1 < len(parts):
                    cnt[parts[i + 1]] += 1
                break
    return dict(sorted(cnt.items()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()

    sources = [
        ("data/pallet/results/_pl_experiments/pl_outside_r0_loo/_accepted_log.json",      "outside R0 (Baseline)",        "#bdbdbd"),
        ("data/pallet/results/_pl_experiments/pl_outside_R0_cf_loo/_accepted_log.json",   "outside R0 (Proposed)",        "#1f77b4"),
        ("data/pallet/results/_pl_experiments/pl_night_r0_loo/_accepted_log.json",        "night R0 (Baseline)",          "#d9d9d9"),
    ]

    fig, ax = plt.subplots(figsize=(13, 6))
    all_seqs = set()
    data = []
    for log_path, label, color in sources:
        if not os.path.exists(log_path):
            continue
        d = count_seq(log_path, label)
        data.append((label, color, d))
        all_seqs.update(d.keys())

    seqs = sorted(all_seqs)
    x = np.arange(len(seqs))
    w = 0.27
    for i, (label, color, d) in enumerate(data):
        ys = [d.get(s, 0) for s in seqs]
        bars = ax.bar(x + (i - 1) * w, ys, w, label=f"{label} (total {sum(ys)})",
                      color=color, edgecolor="black", linewidth=0.5)
        for b, y in zip(bars, ys):
            if y > 0:
                ax.text(b.get_x() + b.get_width() / 2, y + 5, f"{int(y)}",
                        ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(seqs, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("PL count per sequence", fontsize=12)
    ax.set_title("[A][B][C] filter pass — PL distribution per sequence\n"
                 "Baseline (synthetic-only) vs Proposed (camera-facing) base models",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
