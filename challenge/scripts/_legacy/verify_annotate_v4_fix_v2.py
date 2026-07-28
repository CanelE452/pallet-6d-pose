"""verify_annotate_v4_fix_v2.py — annotate.py v4 fix v2 검증 시각화.

capturepallet03/1778651569891693056.png frame 에서:
  - Case A: 사용자 0~3 만 클릭 (큰 face)
  - Case B: 사용자 0~7 모두 클릭

각 케이스에 대해 solve_pose 결과의 wireframe + manual_kps overlay 저장.
"""
import os, sys, json
import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from annotate_pnp import solve_pose, make_pallet_keypoints_3d

REPO = os.path.dirname(os.path.dirname(_HERE))
SRC_JSON = os.path.join(REPO, "challenge/data/capturepallet03_manual_gt/1778651569891693056.json")
SRC_IMG = os.path.join(REPO, "challenge/data/capturepallet03_manual_gt/1778651569891693056.png")
OUT = os.path.join(REPO, "data/pallet/results/annotate_v4_fix_v2")
os.makedirs(OUT, exist_ok=True)

with open(SRC_JSON) as f:
    d = json.load(f)
cam = d["camera_data"]["intrinsics"]
K = np.array([[cam["fx"],0,cam["cx"]],[0,cam["fy"],cam["cy"]],[0,0,1]], dtype=np.float64)
manual = d["objects"][0]["manual_kps"]
kps_full = manual[:9]
img0 = cv2.imread(SRC_IMG)

CUBOID_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # FRONT face — 굵게
    (4, 5), (5, 6), (6, 7), (7, 4),   # REAR
    (0, 4), (1, 5), (2, 6), (3, 7),   # 수직
]
KP_COLORS = [
    (0,0,255), (0,255,255), (0,128,255), (0,255,0),
    (255,128,0), (255,0,0), (255,0,128), (128,0,255), (255,255,255),
]

def draw_case(label, kps, out_name):
    img = img0.copy()
    res = solve_pose(kps, K)
    if res is None:
        cv2.putText(img, "PnP FAILED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    else:
        proj = res["projected_all"]
        pts = [(int(p[0]), int(p[1])) if p[0] >= 0 else None for p in proj[:8]]
        for k, (a, b) in enumerate(CUBOID_EDGES):
            if pts[a] and pts[b]:
                col = (0, 220, 0) if k < 4 else (0, 160, 0)
                thick = 3 if k < 4 else 1
                cv2.line(img, pts[a], pts[b], col, thick, cv2.LINE_AA)
        # projected_all 의 0~3 위치를 큰 X 로 표시 (post-perm)
        for i in range(4):
            if pts[i]:
                cv2.drawMarker(img, pts[i], (0, 255, 0), cv2.MARKER_TILTED_CROSS, 18, 2)
                cv2.putText(img, f"P{i}", (pts[i][0]+6, pts[i][1]-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        info = (f"dims={res['dims']} reproj={res['reproj_error_px']:.1f}px "
                f"perm={res['v4_perm']} warn={res['v4_warning']}")
        cv2.putText(img, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)

    # 사용자 클릭 점 (manual_kps) — 색깔 점 + 번호
    for i, p in enumerate(kps[:9]):
        if p is None: continue
        c = (int(p[0]), int(p[1]))
        cv2.circle(img, c, 6, KP_COLORS[i], -1)
        cv2.circle(img, c, 8, (0,0,0), 1)
        cv2.putText(img, str(i), (c[0]+6, c[1]+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, KP_COLORS[i], 2)

    cv2.putText(img, label, (10, img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)
    fp = os.path.join(OUT, out_name)
    cv2.imwrite(fp, img)
    print(f"[saved] {fp}")
    if res:
        print(f"  perm={res['v4_perm']} warn={res['v4_warning']} reproj={res['reproj_error_px']:.2f}")

# Case A: 0~3 만
draw_case("Case A: clicked 0-3 only (big face)",
          list(kps_full[:4]) + [None]*5, "caseA_only_0_3.png")

# Case B: 0~7 모두
draw_case("Case B: clicked 0-7 (user manual_kps)",
          list(kps_full[:8]) + [None], "caseB_all_0_7.png")

# Case C: 0~8 (centroid 포함)
draw_case("Case C: clicked all 9",
          kps_full, "caseC_all_9.png")

print(f"\n[done] outputs: {OUT}")
