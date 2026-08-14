"""_diag_funnel_flip.py — DIAGNOSTIC ONLY (does not modify production code).

목적: internet_pallet 이미지에서 PnP 큐보이드가 깔때기(frustum)/뒤집힘으로 나오는
원인 확정. solve_pose 후보 풀을 재열거하여 각 후보의 (deck-normal 방향, tb/fr/lr/tilt,
reproj) 를 덤프하고, 선택된 후보 vs "정상(deck가 카메라 향함)" 후보의 reproj 격차 측정.

- pred8/pred_c/K 는 internet_pallet_infer 와 동일 파이프라인(reflect-pad150 + squash-parity).
- 후보 열거는 annotate_pnp._solve_pose_single 의 로직을 그대로 재현(노출된 helper 사용).
- deck(top face {0,1,4,5}) normal = object -Y = R@(0,-1,0). n_cam[2]<0 => deck가 카메라 향함(정상).

Usage: conda activate pallet-pose; python scripts/stage0/diag/_diag_funnel_flip.py
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))
sys.path.insert(0, os.path.join(ROOT, "challenge", "scripts"))

sys.path[:0] = [os.path.join(ROOT, "challenge", "scripts", _s)
                for _s in ("annotate", "infer", "live")]
import paper_s2_testset17_9filters as T   # noqa: E402
import internet_pallet_infer as IPI       # noqa: E402
import annotate_pnp as APNP               # noqa: E402
import cv2                                # noqa: E402
import torch                              # noqa: E402

M = T.M
DEV = "cuda" if torch.cuda.is_available() else "cpu"

TARGETS = {
    "09": "중고파렛트(18kg) [1100*1100*150mm] .png",
    "08": "수출용파렛트(1000×1000×120)4.jpg",
    "01": "1000×1000×120 11.jpg",
    "03": "[1100×1000×135] 목재 파렛트.jpg",
}


def deck_normal_cam(R):
    """deck(top face) normal in camera frame = R @ (0,-1,0). z<0 => faces camera."""
    return R @ np.array([0.0, -1.0, 0.0])


def enumerate_candidates(kps_2d, K, dims, img_shape):
    """annotate_pnp._solve_pose_single 후보 열거 재현. 모든 후보 dict 리스트 반환."""
    kp3d = APNP.make_pallet_keypoints_3d(*dims)
    valid_idx = [i for i in range(min(9, len(kps_2d))) if kps_2d[i] is not None]
    obj = np.array([kp3d[i] for i in valid_idx], dtype=np.float64)
    img = np.array([kps_2d[i] for i in valid_idx], dtype=np.float64)
    img_area = float(img_shape[0] * img_shape[1])
    min_bbox_area = 0.005 * img_area

    inits = []
    for _name, face in APNP._CUBOID_FACES:
        inits.extend(APNP._seed_from_ippe_face(kps_2d, K, kp3d, list(face)))
    for flag in (cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_SQPNP):
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=flag)
            if ok and tvec[2, 0] > 0:
                R, _ = cv2.Rodrigues(rvec)
                inits.append((R, tvec.flatten()))
        except cv2.error:
            pass
    try:
        ok_n, rl, tl, _ = cv2.solvePnPGeneric(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
        if ok_n:
            for rv, tv in zip(rl, tl):
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
    for a in APNP._CUBE_FLIPS_DEG:
        rx = APNP._rot_axis_angle((1, 0, 0), a[0])
        ry = APNP._rot_axis_angle((0, 1, 0), a[1])
        rz = APNP._rot_axis_angle((0, 0, 1), a[2])
        flips.append(rz @ ry @ rx)

    click_pts = np.array([kps_2d[i] for i in valid_idx])
    click_span = max(click_pts[:, 0].ptp(), click_pts[:, 1].ptp(), 50.0)
    z_far_limit = 50.0 * fx_K * max(dims) / click_span

    cands = []
    for R0, t0 in inits:
        for F in flips:
            res = APNP._refine_with_init(obj, img, K, R0 @ F, t0)
            if res is None:
                continue
            R, t = res
            if t[2] <= 0 or t[2] > z_far_limit:
                continue
            pts_cam = (R @ kp3d.T).T + t
            if (pts_cam[:, 2] <= 0).any():
                continue
            lrv, tbv, frv, proj_all, _ = APNP._eval_pair_invariants(R, t, K, kp3d)
            proj_8 = np.array(proj_all[:8])
            if proj_8[:, 0].ptp() * proj_8[:, 1].ptp() < min_bbox_area:
                continue
            err = APNP._reproj_err_dict(proj_all, valid_idx, kps_2d, weights=None)
            cands.append({
                "err": err, "lr": lrv, "tb": tbv, "fr": frv,
                "viol": lrv + tbv + frv, "R": R, "t": t,
                "tilt": APNP._eval_v8_tilt(R),
                "n_cam_deck": deck_normal_cam(R),
            })
    return cands


def summarize(tag, name):
    fp = os.path.join(IPI.SRC, name)
    img = cv2.imread(fp)
    dims, hdef = IPI.parse_dims_m(name)
    APNP.PALLET_DIMS = dims
    belief, geom, wh = M.infer_belief(IPI.load_or_model, img, DEV, IPI.PAD)
    pred8, pred_c, peaks, ratios = M.belief_to_pred(belief, geom, wh, IPI.PAD, IPI.THRESH)
    n_det = int((~np.isnan(pred8[:, 0])).sum())
    K = IPI.K_from_hfov(img.shape[1], img.shape[0], 60.0)  # 진단은 옛 고정 60° 기준
    print(f"\n{'='*78}\n[{tag}] {name}")
    print(f"  dims(WxDxH m)={dims}  n_det={n_det}/8  img={img.shape[1]}x{img.shape[0]}")
    print(f"  K: fx={K[0,0]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f} (HFOV={IPI.HFOV_DEG})")
    det = [i for i in range(8) if not np.isnan(pred8[i, 0])]
    print(f"  detected corners: {det}  centroid={'yes' if pred_c is not None else 'no'}")
    if n_det < M.N_DET_MIN:
        print("  -> under-det, PnP skip (no cuboid)")
        return

    kps9 = [None if np.isnan(pred8[i, 0]) else [float(pred8[i, 0]), float(pred8[i, 1])]
            for i in range(8)]
    kps9.append(list(pred_c) if pred_c is not None else None)

    # official selection
    pose = APNP.solve_pose(kps9, K, dims=dims, img_shape=img.shape)
    Rsel, tsel = pose["R"], pose["t"]
    nsel = deck_normal_cam(Rsel)
    print("\n  --- OFFICIAL solve_pose selection ---")
    print(f"    reproj={pose['reproj_error_px']:.2f}px  strict_passed={pose.get('_v6_strict_passed')}")
    print(f"    tb_viol={pose['_v6_tb_viol']} fr_viol={pose['_v6_fr_viol']} lr_viol={pose['_v6_lr_viol']} "
          f"viol_sum={pose['_v6_viol_sum']}")
    print(f"    tilt=|R[1,1]|={pose['_v8_tilt']:.3f}  v4_warning={pose['v4_warning']}")
    print(f"    n_cand={pose['_v6_n_candidates']} n_strict_ok={pose['_v6_n_strict_ok']}")
    print(f"    deck_normal_cam={nsel.round(3)}  n_cam.z={nsel[2]:+.3f} "
          f"-> deck {'FACES camera(normal)' if nsel[2] < 0 else 'FACES AWAY = FLIPPED'}")
    print(f"    t(cam)={tsel.round(3)}  R[2,1]={Rsel[2,1]:+.3f}")

    # full pool (single-dim, matching selected dims — solve_pose tries dims_a & dims_b;
    # replicate both for completeness)
    for dtag, dd in [("dims_a=W,D", dims), ("dims_b=D,W", (dims[1], dims[0], dims[2]))]:
        cands = enumerate_candidates(kps9, K, dd, img.shape)
        if not cands:
            print(f"\n  --- pool [{dtag}]: EMPTY")
            continue
        facing = [c for c in cands if c["n_cam_deck"][2] < 0]   # deck faces camera
        flipped = [c for c in cands if c["n_cam_deck"][2] >= 0]
        strict = [c for c in cands if c["viol"] == 0]
        best_overall = min(cands, key=lambda c: c["err"])
        print(f"\n  --- pool [{dtag}]: {len(cands)} cand, "
              f"{len(facing)} deck-facing / {len(flipped)} flipped, {len(strict)} strict(viol=0)")
        print(f"      best-reproj-overall: err={best_overall['err']:.2f} "
              f"viol={best_overall['viol']} tilt={best_overall['tilt']:.2f} "
              f"deckz={best_overall['n_cam_deck'][2]:+.2f} "
              f"{'FACING' if best_overall['n_cam_deck'][2]<0 else 'FLIPPED'}")
        if facing:
            bf = min(facing, key=lambda c: c["err"])
            print(f"      best deck-FACING : err={bf['err']:.2f} viol={bf['viol']} "
                  f"(lr{bf['lr']}/tb{bf['tb']}/fr{bf['fr']}) tilt={bf['tilt']:.2f} "
                  f"deckz={bf['n_cam_deck'][2]:+.2f}")
        if flipped:
            bfl = min(flipped, key=lambda c: c["err"])
            print(f"      best deck-FLIPPED: err={bfl['err']:.2f} viol={bfl['viol']} "
                  f"(lr{bfl['lr']}/tb{bfl['tb']}/fr{bfl['fr']}) tilt={bfl['tilt']:.2f} "
                  f"deckz={bfl['n_cam_deck'][2]:+.2f}")
        if facing and flipped:
            gap = min(flipped, key=lambda c: c["err"])["err"] - min(facing, key=lambda c: c["err"])["err"]
            print(f"      reproj gap (best-flipped - best-facing) = {gap:+.2f}px")
        # top-5 by reproj
        print("      top-6 by reproj:")
        for c in sorted(cands, key=lambda c: c["err"])[:6]:
            fc = "FACING " if c["n_cam_deck"][2] < 0 else "FLIPPED"
            print(f"        err={c['err']:6.2f} viol={c['viol']} "
                  f"(lr{c['lr']}/tb{c['tb']}/fr{c['fr']}) tilt={c['tilt']:.2f} "
                  f"deckz={c['n_cam_deck'][2]:+.2f} {fc}")


def main():
    IPI.load_or_model = T.E.load_model(T.WEIGHTS, DEV)
    for tag, name in TARGETS.items():
        summarize(tag, name)


if __name__ == "__main__":
    main()
