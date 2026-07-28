"""Verify fix v5: compare fix v4 best vs fix v5 best on all manual_gt frames.

Saves overlay diff for the most dramatic case + summary stats."""
import os, sys, json, glob
import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from annotate_pnp import (
    PALLET_DIMS, solve_pose, make_pallet_keypoints_3d, project_3d,
)

OUT = r"C:\Users\minjae\Documents\github\FoundationPose\data\pallet\results\annotate_v4_fix_v5"
os.makedirs(OUT, exist_ok=True)


def draw(img, proj, color_face_front=(0, 255, 0), color_face_rear=(255, 0, 0),
         color_v=(255, 255, 0), title=None, footer=None, color_title=(0, 255, 0)):
    out = img.copy()
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    for (a, b) in edges:
        pa = proj[a]; pb = proj[b]
        if pa[0] < 0 or pb[0] < 0: continue
        if a < 4 and b < 4:
            c = color_face_front; t = 3
        elif a >= 4 and b >= 4:
            c = color_face_rear; t = 2
        else:
            c = color_v; t = 2
        cv2.line(out, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), c, t)
    for i in range(8):
        u, v = proj[i]
        if u < 0: continue
        col = (0, 0, 255) if i < 4 else (255, 0, 0)
        cv2.circle(out, (int(u), int(v)), 4, col, -1)
        cv2.putText(out, str(i), (int(u)+5, int(v)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    if title:
        cv2.putText(out, title, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color_title, 2, cv2.LINE_AA)
    if footer:
        cv2.putText(out, footer, (5, out.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color_title, 1, cv2.LINE_AA)
    return out


def main():
    # We can't easily call the OLD fix v4 (already overwritten) — so we re-derive
    # v4 result by computing a separate score without lr terms.
    from annotate_pnp import _solve_pose_single  # patched
    # patched _solve_pose_single returns best under v5 score. For v4 comparison,
    # rerun candidate enumeration without lr in score.
    sys.path.insert(0, _HERE)
    from diagnose_lr_v4 import enumerate_candidates

    root = r"C:\Users\minjae\Documents\github\FoundationPose\challenge\data"
    paths = sorted(glob.glob(os.path.join(root, "*manual_gt", "*.json")))

    stats = {"total": 0, "v4_lr_flip": 0, "v5_lr_flip": 0,
             "fixed_by_v5": 0, "v5_warning_fired": 0}
    dramatic = None
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        intr = d["camera_data"]["intrinsics"]
        K = np.array([[intr["fx"],0,intr["cx"]],[0,intr["fy"],intr["cy"]],[0,0,1]])
        mkp = d["objects"][0].get("manual_kps")
        if not mkp: continue
        kps_2d = [tuple(x) if x else None for x in mkp]

        cands = enumerate_candidates(kps_2d, K, PALLET_DIMS)
        cands += enumerate_candidates(
            kps_2d, K, (PALLET_DIMS[1], PALLET_DIMS[0], PALLET_DIMS[2]))
        if not cands: continue

        best_v4 = sorted(cands, key=lambda c: c["score_v4"])[0]
        # v5 fallback aware: if all lr=1 in this dim, fall back
        all_lr = all(c["lr"] == 1 for c in cands)
        if all_lr:
            best_v5 = sorted(cands, key=lambda c: c["err"])[0]
        else:
            best_v5 = sorted(cands, key=lambda c: c["score_v5"])[0]

        # Only count frames where user clicks are v4-correct (lr_click_viol < 2).
        # Frames with reversed clicks are user-error, not solver bug.
        user_correct = best_v4["lr_click_v"] < 2
        if not user_correct:
            continue

        stats["total"] += 1
        if best_v4["lr"] == 1: stats["v4_lr_flip"] += 1
        if best_v5["lr"] == 1: stats["v5_lr_flip"] += 1
        if best_v4["lr"] == 1 and best_v5["lr"] == 0:
            stats["fixed_by_v5"] += 1
            # Pick frame where v5 also has reasonable err (not degenerate)
            if best_v5["err"] < 30 and (dramatic is None
                                         or best_v4["err"] < dramatic.get("err4", 1e9)):
                dramatic = {"path": p, "K": K, "kps_2d": kps_2d,
                            "v4": best_v4, "v5": best_v5,
                            "err4": best_v4["err"]}

        # Also verify via actual solve_pose API:
        pose = solve_pose(kps_2d, K, PALLET_DIMS)
        if pose and pose.get("v4_warning"):
            stats["v5_warning_fired"] += 1

    print("=" * 60)
    print("FIX v5 verification summary")
    print("=" * 60)
    print(f"Total frames     : {stats['total']}")
    print(f"v4 best LR-flip  : {stats['v4_lr_flip']}  (these were buggy)")
    print(f"v5 best LR-flip  : {stats['v5_lr_flip']}  (after fix)")
    print(f"Fixed by v5      : {stats['fixed_by_v5']}")
    print(f"solve_pose warning fired (any frame): {stats['v5_warning_fired']}")
    print()

    if dramatic:
        print(f"Dramatic example: {os.path.basename(dramatic['path'])}")
        v4 = dramatic["v4"]; v5 = dramatic["v5"]
        print(f"  v4: err={v4['err']:.2f}  lr={v4['lr']}  L-x={v4['left_x']:+.3f}  R-x={v4['right_x']:+.3f}")
        print(f"  v5: err={v5['err']:.2f}  lr={v5['lr']}  L-x={v5['left_x']:+.3f}  R-x={v5['right_x']:+.3f}")
        img_path = dramatic["path"].replace(".json", ".png")
        img = cv2.imread(img_path)
        # overlay clicks
        for i, p in enumerate(dramatic["kps_2d"]):
            if p is None: continue
            cv2.circle(img, (int(p[0]), int(p[1])), 5, (0, 255, 255), 1)
        side_by_side = np.hstack([
            draw(img, v4["proj_all"],
                 title="fix v4 (LR-mirror BUG)", color_title=(0, 0, 255),
                 footer=f"L-x={v4['left_x']:+.3f}  R-x={v4['right_x']:+.3f}  err={v4['err']:.2f}"),
            draw(img, v5["proj_all"],
                 title="fix v5 (LR-correct)", color_title=(0, 255, 0),
                 footer=f"L-x={v5['left_x']:+.3f}  R-x={v5['right_x']:+.3f}  err={v5['err']:.2f}"),
        ])
        p_out = os.path.join(OUT, "v5_case_a_normal.png")
        cv2.imwrite(p_out, side_by_side)
        print(f"  Saved: {p_out}")

    # Case B: LR-reversed clicks on the same dramatic frame (or capturepallet03)
    p_caseB_src = (dramatic["path"] if dramatic else
                   r"C:\Users\minjae\Documents\github\FoundationPose\challenge\data\capturepallet03_manual_gt\1778651569891693056.json")
    with open(p_caseB_src) as f:
        d = json.load(f)
    intr = d["camera_data"]["intrinsics"]
    K = np.array([[intr["fx"],0,intr["cx"]],[0,intr["fy"],intr["cy"]],[0,0,1]])
    mkp = d["objects"][0]["manual_kps"]
    kps_B = [tuple(x) if x else None for x in mkp]
    for (a, b) in [(0,1),(2,3),(4,5),(6,7)]:
        kps_B[a], kps_B[b] = kps_B[b], kps_B[a]
    pose_B = solve_pose(kps_B, K, PALLET_DIMS)
    img_B = cv2.imread(p_caseB_src.replace(".json", ".png"))
    for i, p in enumerate(kps_B):
        if p is None: continue
        cv2.circle(img_B, (int(p[0]), int(p[1])), 6, (0, 255, 255), 2)
        cv2.putText(img_B, str(i), (int(p[0])+6, int(p[1])-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
    title_B = "CASE B (LR-reversed clicks) — v5 WARNING"
    foot_B = (f"lr_click={pose_B.get('_v5_lr_click_viol', '?')}  "
              f"lr_viol={pose_B.get('_v5_lr_viol', '?')}  "
              f"warning={pose_B.get('v4_warning', '?')}")
    out_B = draw(img_B, pose_B["projected_all"], title=title_B, footer=foot_B,
                 color_title=(0, 165, 255))
    p_outB = os.path.join(OUT, "v5_case_b_lr_reversed.png")
    cv2.imwrite(p_outB, out_B)
    print(f"  Case B saved: {p_outB}  (warning={pose_B.get('v4_warning')})")


if __name__ == "__main__":
    main()
