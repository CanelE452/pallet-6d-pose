"""visualize_pl_pnp.py — passed-PL visual QA (green 0~8 dots + SQPnP cuboid).

Reads self-training PL (output/pl_paper_r1_{domain}) and overlays, for visual
inspection of whether the diag / diag^ratio filter let through PL whose full
9 keypoints + corners are actually correct.

Style (user-specified):
  * predicted 9 keypoints  -> GREEN filled dot + number 0~8 (annotate style)
  * PnP cuboid wireframe    -> 9kp -> SQPnP (dims 1.1/1.3/0.11) -> project cuboid
                               (visualization-only PnP; dims assumed; the paper
                                self-training filter itself stays PnP-FREE.)

Output: output/pl_paper_r1_{domain}/_overlays_pnp/{id}.jpg  + contact_sheet.jpg
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "challenge", "scripts"))

sys.path[:0] = [os.path.join(ROOT, "challenge", "scripts", _s)
                for _s in ("annotate", "infer", "live")]
from annotate_pnp import make_pallet_keypoints_3d  # noqa: E402

VIZ_DIMS = (1.1, 1.3, 0.11)  # width, depth, height (m) — visualization-only
CUBOID_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # near face (thick)
    (4, 5), (5, 6), (6, 7), (7, 4),   # far face
    (0, 4), (1, 5), (2, 6), (3, 7),   # verticals
]
GREEN = (0, 255, 0)


def load_pl(json_path):
    """Return (kps_9x2 with NaN for invalid, valid_mask_9, K_3x3, meta)."""
    d = json.load(open(json_path))
    cd = d["camera_data"]
    intr = cd["intrinsics"]
    K = np.array([[intr["fx"], 0, intr["cx"]],
                  [0, intr["fy"], intr["cy"]],
                  [0, 0, 1]], dtype=np.float64)
    o = d["objects"][0]
    pts = list(o["projected_cuboid"]) + [o["projected_cuboid_centroid"]]
    kps = np.full((9, 2), np.nan, dtype=np.float64)
    valid = np.zeros(9, dtype=bool)
    for i, p in enumerate(pts):
        if p is None:
            continue
        u, v = float(p[0]), float(p[1])
        if u <= -50 or v <= -50:   # [-100,-100] off-image sentinel
            continue
        kps[i] = (u, v)
        valid[i] = True
    return kps, valid, K, o.get("pl_meta", {})


def solve_pnp_viz(kps, valid, K, dims=VIZ_DIMS):
    """SQPnP from valid keypoints. Return (ok, rvec, tvec, med_reproj)."""
    kp3d = make_pallet_keypoints_3d(*dims)
    obj, img = [], []
    for i in range(9):
        if valid[i]:
            obj.append(kp3d[i]); img.append(kps[i])
    if len(obj) < 6:
        return False, None, None, None
    obj = np.asarray(obj, np.float64).reshape(-1, 1, 3)
    img = np.asarray(img, np.float64).reshape(-1, 1, 2)
    dist = np.zeros((5, 1))
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return False, None, None, None
    try:
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    except cv2.error:
        pass
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    med = float(np.median(np.linalg.norm(
        proj.reshape(-1, 2) - img.reshape(-1, 2), axis=1)))
    return True, rvec, tvec, med


def draw(img, kps, valid, K, meta):
    vis = img.copy()
    # 1) PnP cuboid wireframe (visualization-only)
    ok, rvec, tvec, med = solve_pnp_viz(kps, valid, K)
    if ok:
        cub3d = make_pallet_keypoints_3d(*VIZ_DIMS)[:8]
        proj, _ = cv2.projectPoints(cub3d, rvec, tvec, K, np.zeros((5, 1)))
        proj = proj.reshape(-1, 2)
        for k, (a, b) in enumerate(CUBOID_EDGES):
            col = (0, 200, 255) if k < 4 else (200, 160, 0)  # near amber / far blue
            thick = 2 if k < 4 else 1
            cv2.line(vis, tuple(np.int32(proj[a])), tuple(np.int32(proj[b])),
                     col, thick, cv2.LINE_AA)
    # 2) green numbered keypoints 0~8 (on top)
    for i in range(9):
        if not valid[i]:
            continue
        p = (int(round(kps[i, 0])), int(round(kps[i, 1])))
        cv2.circle(vis, p, 4, GREEN, -1)
        cv2.circle(vis, p, 6, (0, 0, 0), 1)
        cv2.putText(vis, str(i), (p[0] + 5, p[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2, cv2.LINE_AA)
    # 3) header: filter scores + min_conf + PnP med reproj
    fs = meta.get("filter_scores", {})
    fs_txt = " ".join(f"{k}={v}" for k, v in fs.items())
    hdr = f"conf>={meta.get('min_conf','?')} {fs_txt}"
    if ok:
        hdr += f"  pnp_med={med:.1f}px"
    else:
        hdr += "  pnp=FAIL"
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(vis, hdr, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    return vis, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True,
                    choices=["indoor", "outside", "night"])
    ap.add_argument("--n", type=int, default=9, help="frames to render")
    ap.add_argument("--all", action="store_true", help="render every PL")
    args = ap.parse_args()

    pl_dir = os.path.join(ROOT, "output", f"pl_paper_r1_{args.domain}")
    out_dir = os.path.join(pl_dir, "_overlays_pnp")
    os.makedirs(out_dir, exist_ok=True)

    jsons = sorted(p for p in glob.glob(os.path.join(pl_dir, "*.json"))
                   if not os.path.basename(p).startswith("_"))
    if not args.all:
        # even sampling across the set so the contact sheet is representative
        if len(jsons) > args.n:
            idx = np.linspace(0, len(jsons) - 1, args.n).astype(int)
            jsons = [jsons[i] for i in idx]

    tiles, n_ok = [], 0
    for jp in jsons:
        base = os.path.splitext(os.path.basename(jp))[0]
        png = os.path.join(pl_dir, base + ".png")
        img = cv2.imread(png)
        if img is None:
            continue
        kps, valid, K, meta = load_pl(jp)
        vis, ok = draw(img, kps, valid, K, meta)
        n_ok += int(ok)
        cv2.imwrite(os.path.join(out_dir, base + ".jpg"), vis)
        tiles.append(cv2.resize(vis, (480, 360)))

    # contact sheet (3 cols)
    if tiles:
        cols = 3
        rows = (len(tiles) + cols - 1) // cols
        while len(tiles) < rows * cols:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols])
                          for r in range(rows)])
        sheet = os.path.join(out_dir, "contact_sheet.jpg")
        cv2.imwrite(sheet, grid)
        print(f"[{args.domain}] rendered {len(jsons)} (pnp_ok={n_ok}) -> {out_dir}")
        print(f"  contact sheet: {sheet}")


if __name__ == "__main__":
    main()
