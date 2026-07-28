"""Final verification artifacts for fix v5.

Saves three images:
  v5_case_a_normal.png       — capturepallet03 saved frame (clicks v4-OK).
                                Both fix v4 and fix v5 produce same correct pose.
  v5_case_b_lr_reversed.png  — capturepallet07 frame (clicks LR-reversed).
                                fix v5 issues a warning. Solver follows clicks.
  v5_lr_invariant_diagnostic.png — bar chart of cam-X distribution across
                                candidates for a frame with many LR-flip
                                candidates (capturepallet07 reversed-click frame).
                                Shows what lr_viol catches.
"""
import os, sys, json
import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from annotate_pnp import PALLET_DIMS, solve_pose
from diagnose_lr_v4 import enumerate_candidates

OUT = r"C:\Users\minjae\Documents\github\FoundationPose\data\pallet\results\annotate_v4_fix_v5"
os.makedirs(OUT, exist_ok=True)


def draw(img, proj, title=None, footer=None, color_title=(0, 255, 0),
         clicks=None):
    out = img.copy()
    if clicks is not None:
        for i, q in enumerate(clicks):
            if q is None or i >= 9: continue
            cv2.circle(out, (int(q[0]), int(q[1])), 6, (0, 255, 255), 2)
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    for (a, b) in edges:
        pa = proj[a]; pb = proj[b]
        if pa[0] < 0 or pb[0] < 0: continue
        if a < 4 and b < 4:
            c = (0, 255, 0); t = 3
        elif a >= 4 and b >= 4:
            c = (255, 0, 0); t = 2
        else:
            c = (200, 200, 0); t = 2
        cv2.line(out, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), c, t)
    for i in range(8):
        u, v = proj[i]
        if u < 0: continue
        col = (0, 0, 255) if i < 4 else (255, 0, 0)
        cv2.circle(out, (int(u), int(v)), 5, col, -1)
        cv2.putText(out, str(i), (int(u)+5, int(v)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    if title:
        cv2.putText(out, title, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color_title, 2, cv2.LINE_AA)
    if footer:
        cv2.putText(out, footer, (5, out.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, color_title, 1, cv2.LINE_AA)
    return out


def main():
    # ---------- CASE A : v4 normal clicks (capturepallet03) ----------
    pA = r"C:\Users\minjae\Documents\github\FoundationPose\challenge\data\capturepallet03_manual_gt\1778651569891693056.json"
    with open(pA) as f: d = json.load(f)
    intr = d["camera_data"]["intrinsics"]
    K_A = np.array([[intr["fx"],0,intr["cx"]],[0,intr["fy"],intr["cy"]],[0,0,1]])
    kps_A = [tuple(x) if x else None for x in d["objects"][0]["manual_kps"]]
    pose_A = solve_pose(kps_A, K_A, PALLET_DIMS)
    img_A = cv2.imread(pA.replace(".json", ".png"))

    print("=" * 70)
    print("CASE A : capturepallet03 — user clicks v4-correct")
    print("=" * 70)
    print(f"  err={pose_A['reproj_error_px']:.2f}  "
          f"lr_viol={pose_A['_v5_lr_viol']}  "
          f"lr_click_viol={pose_A['_v5_lr_click_viol']}  "
          f"warning={pose_A['v4_warning']}")
    print(f"  cam-X L,R = {pose_A['_v5_left_right_cam_x']}")
    out_A = draw(
        img_A, pose_A["projected_all"], clicks=kps_A,
        title="CASE A : v4-correct clicks (capturepallet03)",
        color_title=(0, 255, 0),
        footer=(f"lr_viol={pose_A['_v5_lr_viol']}  "
                f"lr_click={pose_A['_v5_lr_click_viol']}  "
                f"warning={pose_A['v4_warning']}  "
                f"err={pose_A['reproj_error_px']:.2f}"))
    pA_out = os.path.join(OUT, "v5_case_a_normal.png")
    cv2.imwrite(pA_out, out_A)
    print(f"  Saved: {pA_out}")

    # ---------- CASE B : user clicks LR-reversed (capturepallet07 lrc=3) ----------
    pB = r"C:\Users\minjae\Documents\github\FoundationPose\challenge\data\capturepallet07_manual_gt\1778652174598774528.json"
    with open(pB) as f: d = json.load(f)
    intr = d["camera_data"]["intrinsics"]
    K_B = np.array([[intr["fx"],0,intr["cx"]],[0,intr["fy"],intr["cy"]],[0,0,1]])
    kps_B = [tuple(x) if x else None for x in d["objects"][0]["manual_kps"]]
    pose_B = solve_pose(kps_B, K_B, PALLET_DIMS)
    img_B = cv2.imread(pB.replace(".json", ".png"))

    print()
    print("=" * 70)
    print("CASE B : capturepallet07 — user clicks LR-REVERSED (lrc=3)")
    print("=" * 70)
    print(f"  err={pose_B['reproj_error_px']:.2f}  "
          f"lr_viol={pose_B['_v5_lr_viol']}  "
          f"lr_click_viol={pose_B['_v5_lr_click_viol']}  "
          f"warning={pose_B['v4_warning']}")
    out_B = draw(
        img_B, pose_B["projected_all"], clicks=kps_B,
        title="CASE B : LR-reversed clicks (capturepallet07)",
        color_title=(0, 165, 255),
        footer=(f"lr_click={pose_B['_v5_lr_click_viol']}  "
                f"lr_viol={pose_B['_v5_lr_viol']}  "
                f"WARNING={pose_B['v4_warning']}  "
                f"err={pose_B['reproj_error_px']:.2f}"))
    pB_out = os.path.join(OUT, "v5_case_b_lr_reversed.png")
    cv2.imwrite(pB_out, out_B)
    print(f"  Saved: {pB_out}")

    # ---------- DIAGNOSTIC : capturepallet07 frame, bar chart of cam-X ----------
    cands = enumerate_candidates(kps_B, K_B, PALLET_DIMS)
    cands += enumerate_candidates(
        kps_B, K_B, (PALLET_DIMS[1], PALLET_DIMS[0], PALLET_DIMS[2]))
    nl0 = sum(1 for c in cands if c["lr"] == 0)
    nl1 = sum(1 for c in cands if c["lr"] == 1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    cs = sorted(cands, key=lambda c: c["err"])[:30]
    xs = list(range(len(cs)))
    Lx = [max(min(c["left_x"], 3), -3) for c in cs]
    Rx = [max(min(c["right_x"], 3), -3) for c in cs]
    ax.bar([x-0.2 for x in xs], Lx, width=0.4,
           label="LEFT {0,3,4,7} mean cam-X", color="tab:blue")
    ax.bar([x+0.2 for x in xs], Rx, width=0.4,
           label="RIGHT {1,2,5,6} mean cam-X", color="tab:red")
    for i, c in enumerate(cs):
        if c["lr"] == 1:
            ax.axvspan(i-0.4, i+0.4, alpha=0.13, color="red")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Top 30 PnP candidates (sorted by reproj err)")
    ax.set_ylabel("cam-frame X (m, clipped +/-3)")
    ax.set_title(
        f"capturepallet07/1778652174598774528 — total {len(cands)} cands "
        f"(LR-correct={nl0}  LR-flip={nl1})\n"
        f"Red span = LR-flip (lr_viol=1).  Most low-err candidates fail LR invariant.\n"
        f"fix v5 lr_viol weight (+50000) forces LR-correct unless lr_click_viol>=2."
    )
    ax.legend()
    plt.tight_layout()
    p_diag = os.path.join(OUT, "v5_lr_invariant_diagnostic.png")
    plt.savefig(p_diag, dpi=110); plt.close()
    print()
    print(f"Saved diagnostic: {p_diag}")
    print(f"  total candidates: {len(cands)}, LR-flip: {nl1}, LR-correct: {nl0}")

    # ---------- Cross-check : full sweep summary ----------
    import glob
    root = r"C:\Users\minjae\Documents\github\FoundationPose\challenge\data"
    paths = sorted(glob.glob(os.path.join(root, "*manual_gt", "*.json")))
    stats = {"v4OK_solverOK": 0, "v4OK_solverLRflip": 0,
             "v4Rev_solverFollowsClicks": 0,
             "warning_count": 0, "total": 0}
    for p in paths:
        with open(p) as f: d = json.load(f)
        intr = d["camera_data"]["intrinsics"]
        K = np.array([[intr["fx"],0,intr["cx"]],[0,intr["fy"],intr["cy"]],[0,0,1]])
        mkp = d["objects"][0].get("manual_kps")
        if not mkp: continue
        kps = [tuple(x) if x else None for x in mkp]
        pose = solve_pose(kps, K, PALLET_DIMS)
        if pose is None: continue
        stats["total"] += 1
        lrc = pose["_v5_lr_click_viol"]
        lrv = pose["_v5_lr_viol"]
        if pose["v4_warning"]: stats["warning_count"] += 1
        if lrc < 2:    # user clicked v4-correct
            if lrv == 0:
                stats["v4OK_solverOK"] += 1
            else:
                stats["v4OK_solverLRflip"] += 1
        else:          # user clicked LR-reversed
            if lrv == 1:   # solver follows clicks
                stats["v4Rev_solverFollowsClicks"] += 1

    print()
    print("=" * 70)
    print("Full sweep with fix v5 applied")
    print("=" * 70)
    print(f"Total frames                      : {stats['total']}")
    print(f"User clicks v4-OK,  solver LR-OK  : {stats['v4OK_solverOK']}  (no bug)")
    print(f"User clicks v4-OK,  solver LR-bad : {stats['v4OK_solverLRflip']}  "
          f"(should be 0 - fix v5 working)")
    print(f"User clicks reversed, solver follows: {stats['v4Rev_solverFollowsClicks']}  "
          f"(expected - fix v5 disables lr_viol when lr_click_viol>=2)")
    print(f"v4_warning fired (any frame)      : {stats['warning_count']}")


if __name__ == "__main__":
    main()
