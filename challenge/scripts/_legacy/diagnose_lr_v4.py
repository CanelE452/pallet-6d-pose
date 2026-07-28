"""LR-flip diagnosis for fix v4 best pose.

manual_kps + K 로 _solve_pose_single 의 candidates 를 모두 열거하여
각 후보의 LR cam-frame X 분포 + score 를 출력. fix v4 best (LR-flip) 와
fix v5 가 선택해야 하는 LR-correct 해를 동시 비교.
"""
import os, sys, json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from annotate_pnp import (
    PALLET_DIMS,
    make_pallet_keypoints_3d,
    _CUBE_FLIPS_DEG, _rot_axis_angle, _V4_VERTICAL_FACES,
    _refine_with_init, _polyarea4, _reproj_err, project_3d,
)


def enumerate_candidates(kps_2d, K, dims):
    kp3d = make_pallet_keypoints_3d(*dims)
    valid_idx = [i for i in range(min(9, len(kps_2d))) if kps_2d[i] is not None]
    obj = np.array([kp3d[i] for i in valid_idx], dtype=np.float64)
    img = np.array([kps_2d[i] for i in valid_idx], dtype=np.float64)
    img_w_est = int(round(2.0 * K[0, 2]))
    img_h_est = int(round(2.0 * K[1, 2]))

    inits = []
    for flag in (cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_SQPNP):
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=flag)
            if ok and tvec[2, 0] > 0:
                R, _ = cv2.Rodrigues(rvec)
                inits.append((R, tvec.flatten()))
        except cv2.error:
            pass
    try:
        ok_n, rvl, tvl, _ = cv2.solvePnPGeneric(
            obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
        if ok_n:
            for rv, tv in zip(rvl, tvl):
                if tv[2, 0] > 0:
                    R_ippe, _ = cv2.Rodrigues(rv)
                    inits.append((R_ippe, tv.flatten()))
    except cv2.error:
        pass

    cx_K, cy_K, fx_K = K[0, 2], K[1, 2], K[0, 0]
    mean_u = np.mean([kps_2d[i][0] for i in valid_idx])
    mean_v = np.mean([kps_2d[i][1] for i in valid_idx])
    img_w = max(kps_2d[i][0] for i in valid_idx) - min(kps_2d[i][0] for i in valid_idx)
    z_guess = max(0.5, fx_K * dims[0] / max(img_w, 50.0))
    t_manual = np.array([(mean_u - cx_K) * z_guess / fx_K,
                         (mean_v - cy_K) * z_guess / fx_K, z_guess])
    Rx180 = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
    inits.append((Rx180.copy(), t_manual.copy()))
    inits.append((np.eye(3), t_manual.copy()))

    flips = []
    for axd in _CUBE_FLIPS_DEG:
        rx = _rot_axis_angle((1,0,0), axd[0])
        ry = _rot_axis_angle((0,1,0), axd[1])
        rz = _rot_axis_angle((0,0,1), axd[2])
        flips.append(rz @ ry @ rx)

    cands = []
    for R0, t0 in inits:
        for F in flips:
            res = _refine_with_init(obj, img, K, R0 @ F, t0)
            if res is None: continue
            R, t = res
            if t[2] <= 0: continue
            pts_cam = (R @ kp3d.T).T + t
            if (pts_cam[:,2] <= 0).any(): continue
            err, proj_all = _reproj_err(kp3d, R, t, K, valid_idx, kps_2d)

            # invariants
            z_front = float(pts_cam[:4,2].mean()); z_rear = float(pts_cam[4:8,2].mean())
            nf_v = 1 if z_front >= z_rear else 0

            proj8 = np.array(proj_all[:8]).copy()
            proj8[:,0] = np.clip(proj8[:,0], 0.0, float(img_w_est-1))
            proj8[:,1] = np.clip(proj8[:,1], 0.0, float(img_h_est-1))
            areas = {n: _polyarea4(proj8[idx]) for n, idx in _V4_VERTICAL_FACES.items()}
            av_v = 1 if areas["FRONT"] < max(a for n,a in areas.items() if n != "FRONT") - 1.0 else 0

            top_v_proj = np.mean([proj_all[i][1] for i in (0,1,4,5)])
            bot_v_proj = np.mean([proj_all[i][1] for i in (2,3,6,7)])
            gv_v = 1 if top_v_proj >= bot_v_proj else 0

            # LR invariant (NEW)
            left_idx = [0,3,4,7]; right_idx = [1,2,5,6]
            left_x = float(pts_cam[left_idx, 0].mean())
            right_x = float(pts_cam[right_idx, 0].mean())
            lr_v = 1 if left_x >= right_x else 0

            # click u-ordering (LR-specific) — candidate-independent but cached per cand for score
            lr_click_v = 0
            for (a,b) in [(0,1),(3,2),(4,5),(7,6)]:
                if (a < len(kps_2d) and b < len(kps_2d)
                    and kps_2d[a] is not None and kps_2d[b] is not None):
                    if kps_2d[a][0] >= kps_2d[b][0] - 1.0:
                        lr_click_v += 1   # user clicked LEFT to the right of RIGHT

            # n_viol (full)
            n_v = 0
            for ii in range(len(valid_idx)):
                for jj in range(ii+1, len(valid_idx)):
                    a, b = valid_idx[ii], valid_idx[jj]
                    dv_c = kps_2d[a][1] - kps_2d[b][1]
                    dv_p = proj_all[a][1] - proj_all[b][1]
                    du_c = kps_2d[a][0] - kps_2d[b][0]
                    du_p = proj_all[a][0] - proj_all[b][0]
                    if abs(dv_c) > 10 and (dv_c > 0) != (dv_p > 0): n_v += 1
                    if abs(du_c) > 10 and (du_c > 0) != (du_p > 0): n_v += 1

            score_v4 = err + 1000*n_v + 10000*nf_v + 5000*av_v + 50000*gv_v
            score_v5 = score_v4 + 50000*lr_v + 20000*lr_click_v

            cands.append({
                "R": R, "t": t, "err": err,
                "nf": nf_v, "av": av_v, "gv": gv_v, "lr": lr_v,
                "lr_click_v": lr_click_v,
                "n_v": n_v,
                "left_x": left_x, "right_x": right_x,
                "left_x_minus_right_x": left_x - right_x,
                "score_v4": score_v4, "score_v5": score_v5,
                "proj_all": proj_all,
            })
    return cands


def main():
    j_path = r"C:\Users\minjae\Documents\github\FoundationPose\challenge\data\capturepallet03_manual_gt\1778651569891693056.json"
    img_path = r"C:\Users\minjae\Documents\github\FoundationPose\challenge\data\capturepallet03_manual_gt\1778651569891693056.png"
    with open(j_path) as f:
        d = json.load(f)
    intr = d["camera_data"]["intrinsics"]
    K = np.array([[intr["fx"],0,intr["cx"]],
                  [0,intr["fy"],intr["cy"]],
                  [0,0,1.0]])
    manual_kps = d["objects"][0]["manual_kps"]
    kps_2d = [tuple(p) if p else None for p in manual_kps]

    # Case A: normal v4 clicks (as saved)
    cands_A = enumerate_candidates(kps_2d, K, PALLET_DIMS)
    cands_A += enumerate_candidates(
        kps_2d, K, (PALLET_DIMS[1], PALLET_DIMS[0], PALLET_DIMS[2]))

    best_v4 = sorted(cands_A, key=lambda c: c["score_v4"])[0]
    best_v5 = sorted(cands_A, key=lambda c: c["score_v5"])[0]

    print("=" * 70)
    print("CASE A (normal v4 clicks)")
    print("=" * 70)
    print(f"#candidates: {len(cands_A)}")
    print(f"#LR-correct (lr_viol=0): {sum(1 for c in cands_A if c['lr']==0)}")
    print(f"#LR-flip    (lr_viol=1): {sum(1 for c in cands_A if c['lr']==1)}")
    print()
    print("BEST under fix v4 score:")
    print(f"  err={best_v4['err']:.2f}  nf={best_v4['nf']} av={best_v4['av']} "
          f"gv={best_v4['gv']} lr={best_v4['lr']} n_v={best_v4['n_v']}  "
          f"L-x={best_v4['left_x']:.3f} R-x={best_v4['right_x']:.3f}  "
          f"L-R={best_v4['left_x_minus_right_x']:.3f}  score={best_v4['score_v4']:.1f}")
    print("BEST under fix v5 score (+lr_viol +lr_click_v):")
    print(f"  err={best_v5['err']:.2f}  nf={best_v5['nf']} av={best_v5['av']} "
          f"gv={best_v5['gv']} lr={best_v5['lr']} n_v={best_v5['n_v']}  "
          f"L-x={best_v5['left_x']:.3f} R-x={best_v5['right_x']:.3f}  "
          f"L-R={best_v5['left_x_minus_right_x']:.3f}  score={best_v5['score_v5']:.1f}")
    print()
    print("Top 10 by err only:")
    for c in sorted(cands_A, key=lambda c: c["err"])[:10]:
        print(f"  err={c['err']:6.2f}  lr={c['lr']} gv={c['gv']} av={c['av']} nf={c['nf']} "
              f"n_v={c['n_v']:2d}  L-R={c['left_x_minus_right_x']:+.3f} "
              f"sv4={c['score_v4']:.1f}")
    print("Top 10 by score_v4:")
    for c in sorted(cands_A, key=lambda c: c["score_v4"])[:10]:
        print(f"  err={c['err']:6.2f}  lr={c['lr']} gv={c['gv']} av={c['av']} nf={c['nf']} "
              f"n_v={c['n_v']:2d}  L-R={c['left_x_minus_right_x']:+.3f} "
              f"sv4={c['score_v4']:.1f}")

    # Save overlays
    out_dir = r"C:\Users\minjae\Documents\github\FoundationPose\data\pallet\results\annotate_v4_fix_v5"
    os.makedirs(out_dir, exist_ok=True)

    def draw(img, proj, color_face_front=(0,255,0), color_face_rear=(255,0,0), color_v=(255,255,0)):
        out = img.copy()
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                 (0,4),(1,5),(2,6),(3,7)]
        for (a,b) in edges:
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
            col = (0,0,255) if i < 4 else (255,0,0)
            cv2.circle(out, (int(u), int(v)), 4, col, -1)
            cv2.putText(out, str(i), (int(u)+5, int(v)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
        return out

    img = cv2.imread(img_path)
    # v4 best (LR-flip wireframe expected)
    out_v4 = draw(img, best_v4["proj_all"])
    cv2.putText(out_v4, "fix v4 best (saved behavior)", (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
    cv2.putText(out_v4,
                f"L-x={best_v4['left_x']:+.2f}  R-x={best_v4['right_x']:+.2f}  "
                f"lr_viol={best_v4['lr']}",
                (5, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
    p1 = os.path.join(out_dir, "v5_case_a_v4best.png")
    cv2.imwrite(p1, out_v4)

    # v5 best (LR-correct wireframe expected)
    out_v5 = draw(img, best_v5["proj_all"])
    cv2.putText(out_v5, "fix v5 best (LR invariant added)", (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(out_v5,
                f"L-x={best_v5['left_x']:+.2f}  R-x={best_v5['right_x']:+.2f}  "
                f"lr_viol={best_v5['lr']}",
                (5, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    p2 = os.path.join(out_dir, "v5_case_a_normal.png")
    cv2.imwrite(p2, out_v5)

    # LR diagnostic bar chart
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    cand_sorted = sorted(cands_A, key=lambda c: c["err"])[:20]
    xs = list(range(len(cand_sorted)))
    Lx = [c["left_x"] for c in cand_sorted]
    Rx = [c["right_x"] for c in cand_sorted]
    ax.bar([x-0.2 for x in xs], Lx, width=0.4, label="LEFT {0,3,4,7} cam-X", color="tab:blue")
    ax.bar([x+0.2 for x in xs], Rx, width=0.4, label="RIGHT {1,2,5,6} cam-X", color="tab:red")
    for i, c in enumerate(cand_sorted):
        if c["lr"] == 1:
            ax.axvspan(i-0.4, i+0.4, alpha=0.15, color="red")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Candidate (sorted by reproj err)")
    ax.set_ylabel("cam-frame X (meters)")
    ax.set_title("LR cam-X per candidate - red span = LR-flip (lr_viol=1)\n"
                 "Correct: LEFT.x < RIGHT.x.  Note many low-err candidates fail LR.")
    ax.legend()
    plt.tight_layout()
    p3 = os.path.join(out_dir, "v5_lr_invariant_diagnostic.png")
    plt.savefig(p3, dpi=110); plt.close()

    print()
    print("Saved:")
    print(f"  {p1}")
    print(f"  {p2}")
    print(f"  {p3}")

    # Case B: LR-reversed user clicks (swap 0<->1, 2<->3, 4<->5, 6<->7)
    kps_2d_B = list(kps_2d)
    for (a, b) in [(0,1),(2,3),(4,5),(6,7)]:
        kps_2d_B[a], kps_2d_B[b] = kps_2d_B[b], kps_2d_B[a]
    cands_B = enumerate_candidates(kps_2d_B, K, PALLET_DIMS)
    cands_B += enumerate_candidates(
        kps_2d_B, K, (PALLET_DIMS[1], PALLET_DIMS[0], PALLET_DIMS[2]))
    best_v5_B = sorted(cands_B, key=lambda c: c["score_v5"])[0]
    print()
    print("=" * 70)
    print("CASE B (LR-reversed clicks - user violates v4 convention)")
    print("=" * 70)
    print(f"BEST under fix v5: lr_click_v={best_v5_B['lr_click_v']}  "
          f"lr_viol={best_v5_B['lr']}  err={best_v5_B['err']:.2f}  "
          f"score_v5={best_v5_B['score_v5']:.1f}")
    img_b = img.copy()
    # show user click points (B order) + best v5 wireframe
    for i, p in enumerate(kps_2d_B[:8]):
        cv2.circle(img_b, (int(p[0]), int(p[1])), 6, (0,255,255), 2)
        cv2.putText(img_b, str(i), (int(p[0])+6, int(p[1])-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
    out_b = draw(img_b, best_v5_B["proj_all"])
    cv2.putText(out_b, f"CASE B (LR reversed user)  lr_click_v={best_v5_B['lr_click_v']}",
                (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2, cv2.LINE_AA)
    cv2.putText(out_b, "WARNING: v5 should flag this as LR-reversed",
                (5, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1, cv2.LINE_AA)
    p4 = os.path.join(out_dir, "v5_case_b_lr_reversed.png")
    cv2.imwrite(p4, out_b)
    print(f"  {p4}")


if __name__ == "__main__":
    main()
