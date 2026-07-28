"""oracle_pnp_sweep.py — §1 STAGE8 offset probe: GT-2D-corner noise -> pose 안정성.

목적: "keypoint corner 오차(현재 heatmap ~9px)를 줄이는 게 pose 정확도에 실용 가치가
있나"를 학습 없이 정량화. GT 3D 8 corner + GT 2D 8 corner + K 로 reference pose 를
PnP(solve_pose, order-free ITERATIVE) 로 잡고, 2D corner 에 Gaussian noise 를 주입한 뒤
재-PnP 한 pose 가 reference 대비 얼마나 흔들리는지 측정.

조건:
  A: 8 corner 전체에 동일 σ Gaussian noise.
  B: back corner(GT 3D 의 far 4점)에만 noise — flat/edge-on depth 붕괴가 back 에
     집중된다는 가설(diag-filter-not-reliable memory) 검증.
  C: 현재 heatmap 모델의 *실제* residual(predicted_k - GT_k)을 noise 로 — systematic
     (flip/flatten) 분포 반영. extract_keypoints_from_belief 로 예측, GT 와 hungarian
     매칭 후 corner 별 residual 벡터를 reference 2D 에 더해 재-PnP.

좌표/dims: real(manual) = dimensions_m, synthetic = PALLET_DIMS (val 엔 dims 없음).
PnP = challenge/scripts/annotate_pnp.solve_pose (auto-dim, ITERATIVE refine, order-free).

출력: σ×조건별 rotation delta(deg), translation delta(m, %), GT corner 재투영 reproj(px),
PnP failure rate, positive-depth failure rate. 공백정렬 표 + json.
  data/pallet/eval_results/stage8_offset/oracle_pnp/

★ 학습 없음. heatmap ckpt 는 조건 C residual 측정에만 사용.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "challenge", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))

import torch  # noqa: E402
from annotate_pnp import (  # noqa: E402
    solve_pose, make_pallet_keypoints_3d, project_3d, PALLET_DIMS,
)
from eval_pvnet_heads import (  # noqa: E402
    collect_manual, collect_syn, load_gt8, preprocess, belief_to_orig,
    load_pvnet_model,
)
from filter_pr_camfacing import extract_keypoints_from_belief  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "pallet", "eval_results",
                       "stage8_offset", "oracle_pnp")
SIGMAS = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0]
N_SAMPLE = 20
BACK_IDX = [4, 5, 6, 7]   # annotate_pnp make_pallet_keypoints_3d far corners


def _K_from_json(jp):
    d = json.load(open(jp))
    intr = d["camera_data"]["intrinsics"]
    return np.array([[intr["fx"], 0, intr["cx"]],
                     [0, intr["fy"], intr["cy"]], [0, 0, 1]], float)


def _dims_from_json(jp, dom):
    """manual = dimensions_m, synthetic = PALLET_DIMS (val 엔 dims 없음)."""
    d = json.load(open(jp))
    o = d["objects"][0]
    dm = o.get("dimensions_m")
    if dm is not None:
        return (dm["width"], dm["depth"], dm["height"])
    return PALLET_DIMS


def _rot_delta_deg(R1, R2):
    """두 회전 사이 geodesic angle (deg)."""
    Rrel = R1.T @ R2
    c = (np.trace(Rrel) - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def _solve_ref(gt8, K, dims):
    """GT 2D 8 corner(noise 없음) -> reference pose. kps_2d = 9칸(centroid None)."""
    kps = [list(map(float, gt8[i])) for i in range(8)] + [None]
    pose = solve_pose(kps, K, dims=dims, img_shape=(480, 640, 3))
    return pose


def _solve_noisy(gt8, K, dims, noise8):
    """gt8 + noise8 (8,2) -> pose. centroid 미사용."""
    pert = gt8 + noise8
    kps = [list(map(float, pert[i])) for i in range(8)] + [None]
    return solve_pose(kps, K, dims=dims, img_shape=(480, 640, 3))


def _reproj_gt(pose, gt8, dims, K):
    """pose 로 GT 3D 8 corner 재투영 -> GT 2D 8 corner 와 mean px err.
    order-free: hungarian (pose corner index 와 gt8 index convention 차이 흡수)."""
    kp3d = make_pallet_keypoints_3d(*dims)
    proj = np.array(project_3d(kp3d, pose["R"], pose["t"], K))[:8]
    valid = np.all(proj > -1.0, axis=1)
    if valid.sum() < 6:
        return float("inf")
    P = proj[valid]
    cost = np.linalg.norm(P[:, None, :] - gt8[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(cost)
    return float(cost[ri, ci].mean())


# ── 조건 C: 현재 heatmap 모델 실제 residual (predicted_k - GT_k) ─────────────
def _heatmap_residual(model, device, ip, gt8, threshold=0.3):
    """이미지 추론 -> 예측 8 corner(원본px) -> GT 8 corner 와 hungarian 매칭 ->
    매칭된 corner 별 residual 벡터(pred-gt) 반환. 미검출 corner 는 NaN.
    반환: (8,2) residual (gt index 정렬), n_matched.
    """
    img = cv2.imread(ip)
    if img is None:
        return np.full((8, 2), np.nan), 0
    tensor, nw, nh, sc = preprocess(img)
    tensor = tensor.to(device)
    with torch.no_grad():
        out = model(tensor)
    beliefs = out[0]
    belief = beliefs[-1][0].cpu().numpy()
    bh, bw = belief.shape[1], belief.shape[2]
    kps_bel = extract_keypoints_from_belief(belief, threshold)
    pred8 = np.full((8, 2), np.nan)
    for i in range(8):
        k = kps_bel[i]
        if k[0] < 0:
            continue
        ox, oy = belief_to_orig(k[0], k[1], bw, bh, nw, nh, sc)
        pred8[i] = (ox, oy)
    valid = ~np.isnan(pred8[:, 0])
    if valid.sum() < 6:
        return np.full((8, 2), np.nan), int(valid.sum())
    P = pred8[valid]
    pidx = np.nonzero(valid)[0]
    cost = np.linalg.norm(P[:, None, :] - gt8[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(cost)
    resid = np.full((8, 2), np.nan)
    for m, gi in zip(ri, ci):
        resid[gi] = P[m] - gt8[gi]   # residual at GT corner gi
    return resid, int(valid.sum())


def run(frames, model, device, seed=0):
    rng = np.random.default_rng(seed)
    # cond -> sigma -> list of dict(rot,trans,trans_pct,reproj) + counters
    records = {c: {s: [] for s in SIGMAS} for c in ("A", "B", "C")}
    fail = {c: {s: {"pnp": 0, "depth": 0, "tries": 0} for s in SIGMAS}
            for c in ("A", "B", "C")}
    n_used = 0
    for dom, fid, jp, ip in frames:
        gt8 = load_gt8(jp)
        K = _K_from_json(jp)
        dims = _dims_from_json(jp, dom)
        ref = _solve_ref(gt8, K, dims)
        if ref is None:
            continue
        t_ref = ref["t"]
        dist_ref = float(np.linalg.norm(t_ref))
        # 조건 C residual (모델 있을 때만)
        residC = None
        if model is not None:
            residC, _ = _heatmap_residual(model, device, ip, gt8)
        n_used += 1

        for cond in ("A", "B", "C"):
            if cond == "C" and (residC is None or np.isnan(residC).all()):
                continue
            for s in SIGMAS:
                reps = 1 if (s == 0.0 and cond != "C") else N_SAMPLE
                for _ in range(reps):
                    if cond == "C":
                        # residual scaling: σ=0 -> 0, else residual 방향 그대로
                        # 사용(실제 systematic). σ 는 추가 isotropic noise 강도.
                        base = np.where(np.isnan(residC), 0.0, residC)
                        extra = rng.normal(0, s, size=(8, 2)) if s > 0 else 0.0
                        noise8 = base + extra
                    else:
                        noise8 = rng.normal(0, s, size=(8, 2))
                        if cond == "B":
                            mask = np.zeros((8, 1))
                            mask[BACK_IDX] = 1.0
                            noise8 = noise8 * mask
                    fail[cond][s]["tries"] += 1
                    pose = _solve_noisy(gt8, K, dims, noise8)
                    if pose is None:
                        fail[cond][s]["pnp"] += 1
                        continue
                    if pose["t"][2] <= 0:
                        fail[cond][s]["depth"] += 1
                        continue
                    rot = _rot_delta_deg(ref["R"], pose["R"])
                    tr = float(np.linalg.norm(pose["t"] - t_ref))
                    reproj = _reproj_gt(pose, gt8, dims, K)
                    records[cond][s].append({
                        "rot": rot, "trans": tr,
                        "trans_pct": 100.0 * tr / max(dist_ref, 1e-6),
                        "reproj": reproj,
                    })
    return records, fail, n_used


def _agg(lst, key):
    v = [r[key] for r in lst if np.isfinite(r[key])]
    if not v:
        return None, None
    return float(np.median(v)), float(np.percentile(v, 95))


def fmt(records, fail, title):
    lines = [f"# {title}"]
    hdr = (f"{'cond':<5}{'sigma':>7}{'N':>6}{'rot_med':>9}{'rot_p95':>9}"
           f"{'tr_med':>9}{'tr_p95':>9}{'tr%med':>8}{'rep_med':>9}{'rep_p95':>9}"
           f"{'pnpF%':>7}{'depF%':>7}")
    lines.append(hdr)
    lines.append("─" * len(hdr))
    for cond in ("A", "B", "C"):
        for s in SIGMAS:
            lst = records[cond][s]
            f = fail[cond][s]
            rm, rp = _agg(lst, "rot")
            tm, tp = _agg(lst, "trans")
            tpm, _ = _agg(lst, "trans_pct")
            em, ep = _agg(lst, "reproj")
            tries = max(f["tries"], 1)
            pnpf = 100.0 * f["pnp"] / tries
            depf = 100.0 * f["depth"] / tries

            def g(x, d=2):
                return "-" if x is None else f"{x:.{d}f}"
            lines.append(
                f"{cond:<5}{s:>7.1f}{len(lst):>6}"
                f"{g(rm):>9}{g(rp):>9}{g(tm,3):>9}{g(tp,3):>9}{g(tpm,1):>8}"
                f"{g(em,1):>9}{g(ep,1):>9}{pnpf:>7.1f}{depf:>7.1f}")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(
        ROOT, "weights", "challenge0123", "final_net_epoch_0060.pth"))
    ap.add_argument("--n_syn", type=int, default=200)
    ap.add_argument("--n_manual", type=int, default=0, help="0=전체(36)")
    ap.add_argument("--no_model", action="store_true",
                    help="조건 C 생략(heatmap 추론 안 함)")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = None
    if not args.no_model:
        model, nv, ns = load_pvnet_model(args.weights, device)
        print(f"[model] {args.weights} numVec={nv} numSeg={ns}")

    manual = collect_manual(args.n_manual)
    syn = collect_syn(args.n_syn)
    print(f"[frames] manual={len(manual)} syn={len(syn)}")

    out_all = {}
    for name, frames in (("manual", manual), ("synthetic", syn)):
        if not frames:
            continue
        rec, fl, nu = run(frames, model, device)
        title = (f"STAGE8 Oracle PnP noise sweep — {name} "
                 f"(frames_used={nu}, n_sample={N_SAMPLE})")
        table = fmt(rec, fl, title)
        print("\n" + table)
        out_all[name] = {
            "frames_used": nu,
            "n_sample": N_SAMPLE,
            "sigmas": SIGMAS,
            "table": {
                cond: {str(s): {
                    "N": len(rec[cond][s]),
                    "rot_med": _agg(rec[cond][s], "rot")[0],
                    "rot_p95": _agg(rec[cond][s], "rot")[1],
                    "trans_med": _agg(rec[cond][s], "trans")[0],
                    "trans_p95": _agg(rec[cond][s], "trans")[1],
                    "trans_pct_med": _agg(rec[cond][s], "trans_pct")[0],
                    "reproj_med": _agg(rec[cond][s], "reproj")[0],
                    "reproj_p95": _agg(rec[cond][s], "reproj")[1],
                    "pnp_fail": fl[cond][s]["pnp"],
                    "depth_fail": fl[cond][s]["depth"],
                    "tries": fl[cond][s]["tries"],
                } for s in SIGMAS} for cond in ("A", "B", "C")},
        }
        with open(os.path.join(OUT_DIR, f"oracle_{name}.txt"), "w") as f:
            f.write(table + "\n")
    json.dump(out_all, open(os.path.join(OUT_DIR, "oracle_pnp.json"), "w"),
              indent=2)
    print(f"\n[save] {OUT_DIR}/oracle_pnp.json")


if __name__ == "__main__":
    main()
