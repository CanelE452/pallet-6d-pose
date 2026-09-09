"""이 프레임에서 'Direct-Hough' 가 실제로 무엇을 보는지 / 원본 픽셀에는 무엇이 있는지.

★ 전제 정정: 저장소의 `direct_hough_*` 계열은 **원본 이미지를 보지 않는다**.
   점수는 role @ hypothesis_embedding.T 이고 embedding 은 (theta, rho) 만의
   해석적 함수다 (HOUGH_IMPLEMENTATION_AUDIT.md: LINE_FEATURE_AGGREGATION = NO).
   그래서 "원본 위에 그 Hough 를 그린다" 는 성립하지 않는다.
   대신 (a) 원본에 실제로 있는 line 증거 (Canny + 고전 Hough) 와
       (b) 그 계열이 실제로 쓰는 (theta, rho) 가설 공간 을 나란히 그려 간극을 보인다.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
C = REPO / "data/pallet/results/paper_pose_metric_closure_v1"
OUT = REPO / "data/pallet/results/hough_line_visual"
FID = "eval_pallet07__1778652166837872128"

EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


def cuboid(a, h, b):
    ha, hh, hb = a/2, h/2, b/2
    return np.array([[-ha,-hh,-hb],[ha,-hh,-hb],[ha,hh,-hb],[-ha,hh,-hb],
                     [-ha,-hh,hb],[ha,-hh,hb],[ha,hh,hb],[-ha,hh,hb]], float)


def main():
    man = {f["frame_id"]: f for f in json.load(open(C/"AXIS_REVIEW_MANIFEST.json"))["frames_list"]}
    fr = man[FID]
    img = cv2.imread(str(REPO/fr["image"]))
    H, W = img.shape[:2]
    gt = json.load(open(C/"GEOMETRY_RESOLVED_POSE_GT.json"))["frames"][FID]
    ann = json.load(open(REPO/fr["annotation"]))["camera_data"]["intrinsics"]
    K = np.array([[ann["fx"],0,ann["cx"]],[0,ann["fy"],ann["cy"]],[0,0,1]], float)
    pred = json.load(open(C/"predictions/R0.json"))["frames"][FID]
    P = np.asarray(pred["keypoints_xy"], float)[:8]

    dm = gt["physical_dimensions_m"]
    X = cuboid(dm["across"], dm["height"], dm["along"])
    R = np.asarray(gt["R_gt_representative"], float); t = np.asarray(gt["t_gt"], float)
    Q = X @ R.T + t
    G = np.stack([K[0,0]*Q[:,0]/Q[:,2]+K[0,2], K[1,1]*Q[:,1]/Q[:,2]+K[1,2]], 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 1.4)
    canny = cv2.Canny(blur, 60, 160)
    segs = cv2.HoughLinesP(canny, 1, np.pi/360, threshold=60,
                           minLineLength=int(0.10*max(H,W)), maxLineGap=12)
    segs = segs[:,0,:] if segs is not None else np.zeros((0,4), int)

    def theta_rho(p, q):
        d = q - p; n = np.array([-d[1], d[0]], float)
        n /= max(np.linalg.norm(n), 1e-9)
        return float(np.arctan2(n[1], n[0])), float(n @ p)

    seg_tr = np.array([theta_rho(s[:2].astype(float), s[2:].astype(float)) for s in segs]) \
             if len(segs) else np.zeros((0,2))
    gt_tr = np.array([theta_rho(G[i], G[j]) for i, j in EDGES])
    pr_tr = np.array([theta_rho(P[i], P[j]) for i, j in EDGES])

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    e = np.linalg.norm(P-G, axis=1)
    YAW270 = [4,0,3,7,5,1,2,6]
    e270 = np.linalg.norm(P - G[YAW270], axis=1)

    fig, ax = plt.subplots(2, 2, figsize=(16, 12))

    a = ax[0,0]; a.imshow(rgb)
    for i,j in EDGES: a.plot(*zip(G[i],G[j]), color="#00d000", lw=2.0)
    for i,j in EDGES: a.plot(*zip(P[i],P[j]), color="#1e90ff", lw=1.8)
    a.scatter(*G.T, c="#00d000", s=26, zorder=3); a.scatter(*P.T, c="yellow", s=26, zorder=3)
    for k in range(8):
        a.annotate("", xy=G[k], xytext=P[k],
                   arrowprops=dict(arrowstyle="->", color="red", lw=1.0, alpha=.75))
    a.set_title(f"(1) NOT a localisation failure — a 270 deg LABEL rotation\n"
                f"indexed error median {np.median(e):.1f} px   ->   "
                f"after the yaw-270 relabel {np.median(e270):.1f} px (max {e270.max():.1f})\n"
                f"red arrows = index i of prediction vs index i of GT", fontsize=11)

    a = ax[0,1]; a.imshow(canny, cmap="gray")
    a.set_title(f"(2) what the RAW pixels actually contain\nCanny edges  "
                f"({int((canny>0).sum()):,} edge pixels)", fontsize=11)

    a = ax[1,0]; a.imshow(rgb, alpha=0.55)
    for s in segs: a.plot([s[0],s[2]],[s[1],s[3]], color="#ff8c00", lw=1.4)
    for i,j in EDGES: a.plot(*zip(G[i],G[j]), color="#00d000", lw=2.2)
    a.set_title(f"(3) classical Hough on the raw image\n"
                f"{len(segs)} segments (orange) vs GT cuboid edges (green)\n"
                f"the line evidence is ALREADY in the right place — so a line cannot fix (1)",
                fontsize=11)

    a = ax[1,1]
    if len(seg_tr):
        a.scatter(np.degrees(seg_tr[:,0]), seg_tr[:,1], s=14, c="#ff8c00",
                  label=f"raw-image Hough segments ({len(seg_tr)})")
    a.scatter(np.degrees(gt_tr[:,0]), gt_tr[:,1], s=90, marker="*", c="#00a000",
              edgecolor="k", linewidth=.4, label="GT cuboid edges (12)", zorder=3)
    a.scatter(np.degrees(pr_tr[:,0]), pr_tr[:,1], s=60, marker="X", c="#1e90ff",
              edgecolor="k", linewidth=.4, label="R0 predicted edges (12)", zorder=3)
    a.set_xlabel("theta (deg)"); a.set_ylabel("rho (px)")
    a.set_title("(4) the (theta, rho) plane — where direct_hough_* actually works\n"
                "scoring is role @ f(theta, rho); the pixels of (2)/(3) are never read.\n"
                "note GT stars and predicted crosses nearly coincide: an undirected line is\n"
                "invariant to the very 270 deg relabel that broke this frame", fontsize=10)
    a.legend(fontsize=9, loc="best"); a.grid(alpha=.3)

    for r in ax[0]: r.set_xticks([]); r.set_yticks([])
    ax[1,0].set_xticks([]); ax[1,0].set_yticks([])
    fig.suptitle(f"{FID}   ·   Direct-Hough: what it is assumed to do vs what it does",
                 fontsize=13)
    fig.tight_layout(rect=[0,0,1,0.97])
    p1 = OUT/"hough_on_raw_image.png"
    fig.savefig(p1, dpi=130); plt.close(fig)

    rep = {"frame_id": FID, "image": str(Path(fr["image"])),
           "image_size_hw": [H, W],
           "canny_edge_pixels": int((canny>0).sum()),
           "hough_segments": int(len(segs)),
           "corner_error_px": {"median": float(np.median(e)), "max": float(e.max()),
                               "n_gt_20px": int((e>20).sum())},
           "after_yaw270_relabel_px": {"median": float(np.median(e270)),
                                       "max": float(e270.max())},
           "failure_class": "90/270-degree index permutation, not corner localisation",
           "box_conf": pred["box_conf"],
           "note": ("repo direct_hough_* does not read the raster; scoring is "
                    "role @ f(theta,rho). See HOUGH_IMPLEMENTATION_AUDIT.md"),
           "figure": str(p1.relative_to(REPO))}
    (OUT/"HOUGH_VISUAL.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
