"""PL pool 시각화 — outside/night 의 [A][B][C] 필터 통과 frame 의 keypoint overlay."""
import argparse
import glob
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


CUBOID_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def draw(ax, img_path, json_path, title):
    img = cv2.imread(img_path)
    with open(json_path) as f:
        ndds = json.load(f)
    kps = ndds["objects"][0]["projected_cuboid"]
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for (a, b) in CUBOID_EDGES:
        pa, pb = kps[a], kps[b]
        if pa[0] < 0 or pb[0] < 0:
            continue
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]],
                color="yellow", linewidth=1.8, alpha=0.85)
    for i in range(8):
        x, y = kps[i]
        if x < 0:
            continue
        ax.scatter(x, y, c="cyan", s=50, edgecolor="black", linewidth=0.6, zorder=5)
        ax.text(x + 5, y - 5, str(i), color="cyan", fontsize=9, fontweight="bold")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--n", type=int, default=4)
    args = p.parse_args()

    sources = [
        ("data/pallet/results/_pl_experiments/pl_outside_r0_loo", "outside R0 PL pool (167 통과)"),
        ("data/pallet/results/_pl_experiments/pl_night_r0_loo",   "night R0 PL pool (105 통과)"),
    ]

    n = args.n
    fig, axes = plt.subplots(len(sources), n, figsize=(3.5 * n, 3 * len(sources)))
    if len(sources) == 1:
        axes = np.array([axes])

    for row, (src_dir, title) in enumerate(sources):
        jsons = sorted(glob.glob(os.path.join(src_dir, "*.json")))
        # skip _summary.json, _accepted_log.json
        jsons = [j for j in jsons if not os.path.basename(j).startswith("_")]
        # 균등 간격 N 개
        if len(jsons) >= n:
            step = max(1, len(jsons) // n)
            picks = [jsons[i * step] for i in range(n)]
        else:
            picks = jsons[:n]

        for col, jp in enumerate(picks):
            base = os.path.splitext(os.path.basename(jp))[0]
            ip = os.path.join(src_dir, base + ".png")
            if not os.path.exists(ip):
                continue
            ttl = f"{title}\n{base}" if col == 0 else base
            draw(axes[row][col], ip, jp, ttl)

    fig.suptitle("[A][B][C] 필터 통과 PL 시각화 (output/pl_*_r0_loo/)\n"
                 "cyan = NDDS projected_cuboid (PL dump), yellow = cuboid wireframe",
                 fontsize=12, y=1.0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=130, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
