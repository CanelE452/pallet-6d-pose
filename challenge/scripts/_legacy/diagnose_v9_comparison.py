"""diagnose_v9_comparison.py — night03 + pallet08 v8 vs v9 Before/After 시각화.

v9 fix: degenerate threshold = image_area * 1.5% → image_area * 0.5%.
Night03 와 같이 small/far pallet 케이스에서 정답 candidate 가 살아남도록.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from annotate_pnp import solve_pose, PALLET_DIMS

REPO = r"C:\Users\minjae\Documents\github\FoundationPose"
OUT  = os.path.join(REPO, "data/pallet/results/annotate_v9_oblique")
os.makedirs(OUT, exist_ok=True)


def draw_overlay(img, proj_all, clicks, label, color=(0, 255, 0)):
    vis = img.copy()
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for a, b in edges:
        pa, pb = proj_all[a], proj_all[b]
        if pa[0] == -1.0 and pa[1] == -1.0: continue
        if pb[0] == -1.0 and pb[1] == -1.0: continue
        cv2.line(vis, (int(round(pa[0])), int(round(pa[1]))),
                 (int(round(pb[0])), int(round(pb[1]))), color, 2, cv2.LINE_AA)
    for i in range(8):
        p = proj_all[i]
        if p[0] == -1.0 and p[1] == -1.0: continue
        cv2.circle(vis, (int(round(p[0])), int(round(p[1]))), 4, color, -1)
        cv2.putText(vis, str(i), (int(p[0])+5, int(p[1])-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
    click_cols = [(0,0,255),(0,165,255),(0,255,255),(0,255,0),
                  (255,255,0),(255,0,0),(255,0,255),(255,255,255)]
    for i, c in enumerate(clicks[:8]):
        if c is None: continue
        cv2.circle(vis, (int(c[0]), int(c[1])), 7, click_cols[i], 2)
    cv2.putText(vis, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255,255,255), 1, cv2.LINE_AA)
    return vis


def run_case(name, img_path, K, clicks, out_dir):
    img = cv2.imread(img_path)
    # v9 (current code)
    pose_v9 = solve_pose(clicks, K, img_shape=img.shape)
    vis_v9 = draw_overlay(img, pose_v9["projected_all"], clicks,
                          f"v9 (fixed): reproj={pose_v9['reproj_error_px']:.2f}px "
                          f"tilt={pose_v9['_v8_tilt']:.2f} tz={pose_v9['t'][2]:.2f}m")

    # v8-only simulation: temporarily increase min_bbox_area to 1.5%
    # Approach: monkey-patch solve_pose_single via env
    # Simpler: load v8 results from saved json
    # Here we just save v9 output
    out_path = os.path.join(out_dir, f"v9_{name}.png")
    cv2.imwrite(out_path, vis_v9)
    print(f"{name}: v9 reproj={pose_v9['reproj_error_px']:.2f}px tilt={pose_v9['_v8_tilt']:.2f} -> {out_path}")
    return vis_v9


def main():
    # === night03 ===
    K = np.loadtxt(os.path.join(REPO, "data/night/capturenight03/cam_K.txt"))
    img_path = os.path.join(REPO, "data/night/capturenight03/rgb/1779448848688752640.png")
    clicks = [None]*9
    clicks[0] = [353.0, 267.0]
    clicks[1] = [472.0, 266.0]
    clicks[2] = [476.0, 281.0]
    clicks[3] = [349.0, 284.0]
    clicks[4] = [401.0, 263.0]
    clicks[5] = [492.0, 260.0]
    run_case("night03_1779448848688752640", img_path, K, clicks, OUT)

    # === pallet08 (regression check, saved JSON) ===
    p08 = os.path.join(REPO, "challenge/data/capturepallet08_manual_gt/1778653498432396288.json")
    if os.path.exists(p08):
        with open(p08) as f: d = json.load(f)
        obj = d["objects"][0]
        K_dict = d["camera_data"]["intrinsics"]
        K2 = np.array([[K_dict['fx'],0,K_dict['cx']],[0,K_dict['fy'],K_dict['cy']],[0,0,1]])
        img2 = os.path.join(REPO, "data/outside/capturepallet08/rgb/1778653498432396288.png")
        c2 = list(obj["manual_kps"])
        while len(c2) < 9: c2.append(None)
        run_case("pallet08_1778653498432396288_oblique", img2, K2, c2, OUT)

    # === pallet03 normal frame (regression check) ===
    p03 = os.path.join(REPO, "challenge/data/capturepallet03_manual_gt/1778651569891693056.json")
    if os.path.exists(p03):
        with open(p03) as f: d = json.load(f)
        obj = d["objects"][0]
        K_dict = d["camera_data"]["intrinsics"]
        K3 = np.array([[K_dict['fx'],0,K_dict['cx']],[0,K_dict['fy'],K_dict['cy']],[0,0,1]])
        img3 = os.path.join(REPO, "data/outside/capturepallet03/rgb/1778651569891693056.png")
        c3 = list(obj["manual_kps"])
        while len(c3) < 9: c3.append(None)
        run_case("pallet03_1778651569891693056_normal", img3, K3, c3, OUT)


if __name__ == "__main__":
    main()
