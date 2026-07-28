#!/usr/bin/env python3
"""저앙각(<8deg)+rear 관측 프레임 오버레이 20장. 화면밖 코너 clamp 없이 그림.
convention camera_dynamic_0123_v4: 0-3 front(cyan), 4-7 rear(magenta), 8 centroid(yellow).
depth edges 0-4,1-5,2-6,3-7 (green). front face 0-1-2-3, rear 4-5-6-7."""
import os, json, glob
import numpy as np
import cv2

DATA = "/home/minjae/Documents/github/pallet-pose/challenge/data/training/truncation_addon_v1"
OUT = "/home/minjae/Documents/github/pallet-pose/data/pallet/eval_results/stage16_audit/overlays"
os.makedirs(OUT, exist_ok=True)

recs = json.load(open("/home/minjae/Documents/github/pallet-pose/data/pallet/eval_results/stage16_audit/_recs.json"))
# low elev + at least some rear observability; spread across elev band, pick diverse V_geom
low = [r for r in recs if r["elev"] < 8]
low.sort(key=lambda r: r["elev"])
# sample 20 evenly across the sorted low band
idx = np.linspace(0, len(low) - 1, 20).astype(int)
sel = [low[i] for i in idx]

FRONT_E = [(0, 1), (1, 2), (2, 3), (3, 0)]
REAR_E = [(4, 5), (5, 6), (6, 7), (7, 4)]
DEPTH_E = [(0, 4), (1, 5), (2, 6), (3, 7)]


def pt(p):
    return (int(round(p[0])), int(round(p[1])))


for k, r in enumerate(sel):
    fid = r["fid"]
    img = cv2.imread(os.path.join(DATA, f"{fid}.png"))
    d = json.load(open(os.path.join(DATA, f"{fid}.json")))
    o = d["objects"][0]
    pc = np.array(o["projected_cuboid"], float)  # 8, unclamped
    ctr = o.get("projected_cuboid_centroid")
    kif = o["keypoint_in_frame"]
    H, W = img.shape[:2]
    # pad canvas so off-screen corners visible
    pad = 200
    canvas = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    pc2 = pc + pad
    for a, b in DEPTH_E:
        cv2.line(canvas, pt(pc2[a]), pt(pc2[b]), (0, 200, 0), 2)
    for a, b in FRONT_E:
        cv2.line(canvas, pt(pc2[a]), pt(pc2[b]), (255, 255, 0), 2)
    for a, b in REAR_E:
        cv2.line(canvas, pt(pc2[a]), pt(pc2[b]), (255, 0, 255), 2)
    for i in range(8):
        col = (255, 255, 0) if i < 4 else (255, 0, 255)
        onscr = (0 <= pc[i][0] < W) and (0 <= pc[i][1] < H)
        cv2.circle(canvas, pt(pc2[i]), 6, col, -1 if onscr else 2)
        cv2.putText(canvas, str(i), pt(pc2[i] + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
    if ctr:
        cv2.circle(canvas, pt(np.array(ctr) + pad), 5, (0, 255, 255), -1)
    # frame border (original image extent)
    cv2.rectangle(canvas, (pad, pad), (pad + W, pad + H), (100, 100, 255), 1)
    txt = f"{fid} elev={r['elev']:.1f} Vin={r['num_in_frame']} rear_inf={r['rear_in_front']} frsep={r['frsep']:.0f}"
    cv2.putText(canvas, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(os.path.join(OUT, f"{k:02d}_{fid}_e{r['elev']:.0f}.jpg"), canvas)

print(f"[done] {len(sel)} overlays -> {OUT}")
