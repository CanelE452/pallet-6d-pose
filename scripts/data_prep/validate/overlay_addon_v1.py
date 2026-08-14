"""Visual eye-check overlays for addon_v1: mask_rle + projected_cuboid on RGB.
Picks frames spanning V=8 / V<8 / small / large so reviewer can confirm both
integrity (mask aligns, corners on the pallet) and the intended augmentation.
Outputs individual overlays + a montage to data/pallet/results/addon_v1_audit/.
"""
import json, os, math
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/minjae/Documents/github/pallet-pose"
ADDON = os.path.join(ROOT, "challenge/data/02_synthetic/training/addon_v1")
OUT = os.path.join(ROOT, "data/pallet/results/addon_v1_audit")
os.makedirs(OUT, exist_ok=True)

# camera-facing 0123 v4 edges (cuboid wireframe): front 0-1-2-3, back 4-5-6-7, verticals
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
         (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
COL = ["#ff3b3b", "#ffb13b", "#fff03b", "#7bff3b", "#3bffd1", "#3b7bff",
       "#b13bff", "#ff3bd1", "#000000"]


def rle_decode(rle):
    h, w = rle["size"]; counts = rle["counts"]
    flat = np.zeros(h*w, dtype=np.uint8); idx = 0; val = 0
    for c in counts:
        flat[idx:idx+c] = val; idx += c; val ^= 1
    return flat.reshape((w, h)).T


def pick():
    import glob
    js = sorted(glob.glob(os.path.join(ADDON, "[0-9]*.json")))
    meta = []
    for jp in js[::7]:  # sparse scan
        d = json.load(open(jp)); o = d["objects"][0]
        pc = np.array(o["projected_cuboid"])
        xs = pc[:, 0]; ys = pc[:, 1]
        diag = math.hypot(xs.max()-xs.min(), ys.max()-ys.min())
        meta.append((jp, o.get("num_corners_in_frame"), diag))
    v8 = [m for m in meta if m[1] == 8]
    vlt = [m for m in meta if m[1] is not None and m[1] < 8]
    v8.sort(key=lambda m: m[2])
    vlt.sort(key=lambda m: m[2])
    picks = []
    if v8:
        picks += [v8[0], v8[len(v8)//2], v8[-1]]      # small / mid / large clean
    if vlt:
        picks += [vlt[len(vlt)//2], vlt[-1]]           # truncated cases
    return [p[0] for p in picks[:5]]


def main():
    sel = pick()
    n = len(sel)
    fig, axes = plt.subplots(n, 3, figsize=(12, n*2.6), dpi=170)
    if n == 1:
        axes = axes[None, :]
    for i, jp in enumerate(sel):
        d = json.load(open(jp)); o = d["objects"][0]
        stem = jp[:-5]
        rgb = np.array(Image.open(stem + ".png").convert("RGB"))
        m = rle_decode(o["mask_rle"])
        pc = np.array(o["projected_cuboid"])
        ctr = o["projected_cuboid_centroid"]
        V = o.get("num_corners_in_frame"); area = o.get("mask_area_px")
        # col0 RGB
        axes[i, 0].imshow(rgb); axes[i, 0].set_title(f"{os.path.basename(stem)} RGB", fontsize=8)
        # col1 mask over rgb
        axes[i, 1].imshow(rgb)
        axes[i, 1].imshow(np.ma.masked_where(m == 0, m), cmap="autumn", alpha=0.55)
        axes[i, 1].set_title(f"mask_rle  area={area}", fontsize=8)
        # col2 cuboid over rgb
        axes[i, 2].imshow(rgb)
        for a, b in EDGES:
            axes[i, 2].plot([pc[a, 0], pc[b, 0]], [pc[a, 1], pc[b, 1]], "-", color="#00e5ff", lw=1.2)
        for k in range(8):
            axes[i, 2].scatter([pc[k, 0]], [pc[k, 1]], c=COL[k], s=26, edgecolors="k", linewidths=0.4, zorder=5)
            axes[i, 2].text(pc[k, 0]+3, pc[k, 1]-3, str(k), fontsize=7, color="white",
                            bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.6))
        axes[i, 2].scatter([ctr[0]], [ctr[1]], c="white", marker="x", s=40, zorder=6)
        axes[i, 2].set_title(f"cuboid 0123 v4  V={V}", fontsize=8)
        for j in range(3):
            axes[i, j].axis("off")
        # individual full-size overlay (cuboid)
        figi, axi = plt.subplots(figsize=(6.4, 4.8), dpi=150)
        axi.imshow(rgb)
        for a, b in EDGES:
            axi.plot([pc[a, 0], pc[b, 0]], [pc[a, 1], pc[b, 1]], "-", color="#00e5ff", lw=1.4)
        for k in range(8):
            axi.scatter([pc[k, 0]], [pc[k, 1]], c=COL[k], s=40, edgecolors="k", linewidths=0.5, zorder=5)
            axi.text(pc[k, 0]+4, pc[k, 1]-4, str(k), fontsize=9, color="white",
                     bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.6))
        axi.imshow(np.ma.masked_where(m == 0, m), cmap="autumn", alpha=0.35)
        axi.set_title(f"{os.path.basename(stem)} V={V} area={area}")
        axi.axis("off")
        figi.tight_layout()
        figi.savefig(os.path.join(OUT, f"overlay_{os.path.basename(stem)}.png"))
        plt.close(figi)
    fig.suptitle("addon_v1 eye-check: RGB | mask_rle | cuboid 0123 v4 + centroid", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUT, "montage_overlay_check.png"))
    plt.close(fig)
    print("saved overlays:", [f"overlay_{os.path.basename(s[:-5])}.png" for s in sel])
    print("montage: montage_overlay_check.png")


if __name__ == "__main__":
    main()
