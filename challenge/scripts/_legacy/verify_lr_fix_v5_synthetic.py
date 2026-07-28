"""Synthetic verification for fix v5.

Take a frame where the user clicked LR-reversed (capturepallet07 lrc=3).
Swap user clicks back to v4 convention -> now user clicks are NORMAL but
the underlying pallet is the same. Solver should now have to choose between:
  - LR-correct R, t (matches swapped clicks)
  - LR-flip R, t (matches original visual, but inconsistent with new clicks)

fix v4 would pick by reproj alone -> still picks LR-flip if its reproj is low.
fix v5 adds lr_viol weight -> forces LR-correct.
"""
import os, sys, json
import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from annotate_pnp import PALLET_DIMS, solve_pose
from diagnose_lr_v4 import enumerate_candidates

OUT = r"C:\Users\minjae\Documents\github\FoundationPose\data\pallet\results\annotate_v4_fix_v5"


def draw(img, proj, title=None, footer=None, color_title=(0, 255, 0)):
    out = img.copy()
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    for (a, b) in edges:
        pa = proj[a]; pb = proj[b]
        if pa[0] < 0 or pb[0] < 0: continue
        if a < 4 and b < 4:
            c = (0, 255, 0); t = 3   # front face green
        elif a >= 4 and b >= 4:
            c = (255, 0, 0); t = 2   # rear blue
        else:
            c = (255, 255, 0); t = 2
        cv2.line(out, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), c, t)
    for i in range(8):
        u, v = proj[i]
        if u < 0: continue
        col = (0, 0, 255) if i < 4 else (255, 0, 0)
        cv2.circle(out, (int(u), int(v)), 5, col, -1)
        cv2.putText(out, str(i), (int(u)+5, int(v)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    if title:
        cv2.putText(out, title, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    color_title, 2, cv2.LINE_AA)
    if footer:
        cv2.putText(out, footer, (5, out.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color_title, 1, cv2.LINE_AA)
    return out


def main():
    # Pick a frame with lr_click_viol=3 (user clicked LR-reversed)
    p = r"C:\Users\minjae\Documents\github\FoundationPose\challenge\data\capturepallet07_manual_gt\1778652174598774528.json"
    with open(p) as f: d = json.load(f)
    intr = d["camera_data"]["intrinsics"]
    K = np.array([[intr["fx"],0,intr["cx"]],[0,intr["fy"],intr["cy"]],[0,0,1]])
    mkp_orig = d["objects"][0]["manual_kps"]
    kps_orig = [tuple(x) if x else None for x in mkp_orig]

    # Swap pairs (0<->1, 2<->3, 4<->5, 6<->7) -> simulate user clicking v4-correctly
    kps_swap = list(kps_orig)
    for (a, b) in [(0,1),(2,3),(4,5),(6,7)]:
        kps_swap[a], kps_swap[b] = kps_swap[b], kps_swap[a]

    img_path = p.replace(".json", ".png")
    img = cv2.imread(img_path)

    # === CASE A : user clicks v4-correct (swapped). Solver fix v4 (no lr) vs fix v5 ===
    cands = enumerate_candidates(kps_swap, K, PALLET_DIMS)
    cands += enumerate_candidates(
        kps_swap, K, (PALLET_DIMS[1], PALLET_DIMS[0], PALLET_DIMS[2]))

    # fix v4 score: NO lr terms
    def s_v4(c):
        return c["err"] + 1000*c["n_v"] + 10000*c["nf"] + 5000*c["av"] + 50000*c["gv"]
    best_v4 = sorted(cands, key=s_v4)[0]

    # fix v5: actual solve_pose
    pose_v5 = solve_pose(kps_swap, K, PALLET_DIMS)

    print("Frame:", os.path.basename(p))
    print(f"Swapped clicks: lr_click_viol = {cands[0]['lr_click_v']}  (should be 0)")
    print(f"#cands lr=0: {sum(1 for c in cands if c['lr']==0)}")
    print(f"#cands lr=1: {sum(1 for c in cands if c['lr']==1)}")
    print()
    print(f"fix v4 best: err={best_v4['err']:.2f}  lr={best_v4['lr']}  "
          f"L-x={best_v4['left_x']:+.3f}  R-x={best_v4['right_x']:+.3f}")
    print(f"fix v5 best: err={pose_v5['reproj_error_px']:.2f}  "
          f"lr={pose_v5['_v5_lr_viol']}  "
          f"L-R={pose_v5['_v5_left_right_cam_x']}")
    print(f"             warning={pose_v5['v4_warning']}")

    # Overlay clicks
    img_with_clicks = img.copy()
    for i, q in enumerate(kps_swap):
        if q is None: continue
        cv2.circle(img_with_clicks, (int(q[0]), int(q[1])), 6, (0, 255, 255), 2)
        cv2.putText(img_with_clicks, str(i), (int(q[0])+6, int(q[1])-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)

    out_v4 = draw(img_with_clicks, best_v4["proj_all"],
                  title="CASE A.1: fix v4 (NO lr invariant) - WRONG LR-flip",
                  color_title=(0, 0, 255),
                  footer=f"L-x={best_v4['left_x']:+.3f}  R-x={best_v4['right_x']:+.3f}  "
                         f"lr_viol={best_v4['lr']}  err={best_v4['err']:.2f}")
    out_v5 = draw(img_with_clicks, pose_v5["projected_all"],
                  title="CASE A.2: fix v5 (with lr invariant) - LR-correct",
                  color_title=(0, 255, 0),
                  footer=f"L-x={pose_v5['_v5_left_right_cam_x'][0]:+.3f}  "
                         f"R-x={pose_v5['_v5_left_right_cam_x'][1]:+.3f}  "
                         f"lr_viol={pose_v5['_v5_lr_viol']}  "
                         f"err={pose_v5['reproj_error_px']:.2f}")
    side = np.hstack([out_v4, out_v5])
    p_out = os.path.join(OUT, "v5_case_a_normal.png")
    cv2.imwrite(p_out, side)
    print(f"Saved CASE A: {p_out}")

    # === CASE B : user clicks LR-reversed (original). v5 warning ===
    pose_B = solve_pose(kps_orig, K, PALLET_DIMS)
    img_B = img.copy()
    for i, q in enumerate(kps_orig):
        if q is None: continue
        cv2.circle(img_B, (int(q[0]), int(q[1])), 6, (0, 255, 255), 2)
        cv2.putText(img_B, str(i), (int(q[0])+6, int(q[1])-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
    out_B = draw(img_B, pose_B["projected_all"],
                 title="CASE B: user clicks LR-reversed -> v5 WARNING",
                 color_title=(0, 165, 255),
                 footer=f"lr_click={pose_B['_v5_lr_click_viol']}  "
                        f"lr_viol={pose_B['_v5_lr_viol']}  "
                        f"warning={pose_B['v4_warning']}  "
                        f"err={pose_B['reproj_error_px']:.2f}")
    p_outB = os.path.join(OUT, "v5_case_b_lr_reversed.png")
    cv2.imwrite(p_outB, out_B)
    print(f"Saved CASE B: {p_outB}  warning={pose_B['v4_warning']}")

    # === Diagnostic bar chart: cam-X distribution of all candidates ===
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    cs = sorted(cands, key=lambda c: c["err"])[:30]
    xs = list(range(len(cs)))
    Lx = [max(min(c["left_x"], 5), -5) for c in cs]    # clip ridiculous values
    Rx = [max(min(c["right_x"], 5), -5) for c in cs]
    ax.bar([x-0.2 for x in xs], Lx, width=0.4, label="LEFT {0,3,4,7} cam-X", color="tab:blue")
    ax.bar([x+0.2 for x in xs], Rx, width=0.4, label="RIGHT {1,2,5,6} cam-X", color="tab:red")
    for i, c in enumerate(cs):
        if c["lr"] == 1:
            ax.axvspan(i-0.4, i+0.4, alpha=0.15, color="red")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Candidate (sorted by reproj err)")
    ax.set_ylabel("cam-frame X (m, clipped to +/-5)")
    ax.set_title(f"Frame {os.path.basename(p)}: cam-X per candidate\n"
                 "Red span = LR-flip (lr_viol=1). Top reproj candidates are mostly LR-flip\n"
                 "fix v5 lr_viol weight forces LR-correct selection.")
    ax.legend()
    plt.tight_layout()
    p_diag = os.path.join(OUT, "v5_lr_invariant_diagnostic.png")
    plt.savefig(p_diag, dpi=110); plt.close()
    print(f"Saved diagnostic: {p_diag}")


if __name__ == "__main__":
    main()
