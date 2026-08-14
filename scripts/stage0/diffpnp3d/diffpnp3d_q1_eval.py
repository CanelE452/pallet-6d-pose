#!/usr/bin/env python
"""diffpnp3d_q1_eval.py — PAPER_S2 Phase 3 (Q1) synthetic-val eval for one weights.

Evaluates a DOPE heatmap checkpoint on the fixed q1_split val (500) with the
SAME preprocessing the aspect_resize training used (ANISOTROPIC squash 640x480 ->
400x400, ImageNet norm) and inverts belief(50)->orig with the fixed anisotropic
scale (x*12.8, y*9.6). This is the train/infer-parity path for these models
(the shared eval_frame's aspect-preserving preprocess would be OFF-distribution).

Metrics (verified primitives reused):
  - order-free Hungarian corner + front/back split (eval_pvnet_heads.split_metrics)
  - belief keypoint decode (filter_pr_camfacing.extract_keypoints_from_belief)
  - honest8 = solved-pose 8corner(PER-FRAME dims) vs GT projected_cuboid, order-free
    Hungarian. Pose via annotate_pnp._solve_pose_single with per-frame (W,D,H) and
    (D,W,H) swap (order-free), strict-pass-then-reproj selection (mirrors solve_pose).
  - rear_med(=back)/front_med/corner_med/worst2/det%/pnp%/good%/gross%.
Breakdowns: overall / V8 / V<8 / low-angle(elev<10) / near(elev in [10,15)).

Usage:
  python scripts/stage0/diffpnp3d/diffpnp3d_q1_eval.py --weights <path.pth> --tag lam0.005
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path.insert(0, os.path.join(ROOT, "challenge", "scripts"))

sys.path[:0] = [os.path.join(ROOT, "challenge", "scripts", _s)
                for _s in ("annotate", "infer", "live")]
from models import DopeNetwork  # noqa: E402
from filter_pr_camfacing import extract_keypoints_from_belief  # noqa: E402
from eval_pvnet_heads import split_metrics  # noqa: E402
from stage18_elevation_threshold import elev_from_pose  # noqa: E402
import annotate_pnp as APNP  # noqa: E402

VAL_LIST = os.path.join(ROOT, "data/pallet/results/paper_s2_scratch_diffpnp/q1_split/val_list.json")
OUT_DIR = os.path.join(ROOT, "data/pallet/eval_results/paper_s2_scratch_diffpnp")

GOOD_PX, GROSS_PX, N_DET_MIN = 10.0, 20.0, 6
FRONT, BACK = [0, 1, 2, 3], [4, 5, 6, 7]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SCALE_X, SCALE_Y = 640.0 / 50.0, 480.0 / 50.0   # belief(50) -> orig, fixed aniso
THRESH = 0.3


def load_model(wp, device):
    state = torch.load(wp, map_location=device)
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    m = DopeNetwork(numVec=0, numSeg=0)
    m.load_state_dict(state, strict=False)
    return m.to(device).eval()


def preprocess_squash(img_bgr):
    """640x480 -> 400x400 anisotropic squash + ImageNet norm (train parity)."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r = cv2.resize(rgb, (400, 400), interpolation=cv2.INTER_LINEAR)
    t = (r.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(t.transpose(2, 0, 1)).float().unsqueeze(0)


def K_from_json(d):
    it = d["camera_data"]["intrinsics"]
    return np.array([[it["fx"], 0, it["cx"]], [0, it["fy"], it["cy"]],
                    [0, 0, 1]], float)


def hungarian(pred, gt):
    valid = ~np.isnan(pred[:, 0])
    if valid.sum() < N_DET_MIN:
        return None
    from scipy.optimize import linear_sum_assignment
    P = pred[valid]
    cost = np.linalg.norm(P[:, None, :] - gt[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(cost)
    return cost[ri, ci]


def solve_perframe(kps9, K, dims_whd, img_shape):
    """order-free per-frame-dims pose. Try (W,D,H) & (D,W,H); strict-then-reproj."""
    W, D, H = dims_whd
    cands = []
    for dm in ((W, D, H), (D, W, H)):
        try:
            p = APNP._solve_pose_single(kps9, K, dm, None, img_shape)
        except Exception:
            p = None
        if p is not None:
            cands.append(p)
    if not cands:
        return None, None
    strict = [p for p in cands if p.get("_v6_strict_passed", False)]
    best = (min(strict, key=lambda p: p["reproj_error_px"]) if strict
            else min(cands, key=lambda p: (p.get("_v6_viol_sum", 0),
                                           p["reproj_error_px"])))
    proj_all = APNP.project_3d(APNP.make_pallet_keypoints_3d(*best["dims"]),
                               best["R"], best["t"], K)
    return best, np.array(proj_all, float)


def eval_frame(model, entry, device):
    jp, ip = entry["json"], entry["png"]
    img = cv2.imread(ip)
    if img is None:
        return None
    d = json.load(open(jp))
    o = d["objects"][0]
    gt8 = np.array(o["projected_cuboid"], float)[:8]
    H, Wd = img.shape[:2]
    K = K_from_json(d)
    inb = ((gt8[:, 0] >= 0) & (gt8[:, 0] < Wd)
           & (gt8[:, 1] >= 0) & (gt8[:, 1] < H))
    v_geom = int(inb.sum())

    with torch.no_grad():
        beliefs, _ = model(preprocess_squash(img).to(device))
    belief = beliefs[-1][0].cpu().numpy()
    kps_bel = extract_keypoints_from_belief(belief, THRESH)

    pred8 = np.full((8, 2), np.nan)
    for i, k in enumerate(kps_bel[:8]):
        if k[0] < 0:
            continue
        pred8[i] = [k[0] * SCALE_X, k[1] * SCALE_Y]
    pred_c = None
    if kps_bel[8][0] >= 0:
        pred_c = [kps_bel[8][0] * SCALE_X, kps_bel[8][1] * SCALE_Y]

    m = split_metrics(pred8, gt8)
    dists = hungarian(pred8, gt8)
    worst2 = (float(np.mean(np.sort(dists)[-2:]))
              if dists is not None and len(dists) >= 2 else np.inf)

    # PnP honest8 with per-frame dims
    kps9 = [None if np.isnan(pred8[i, 0]) else
            [float(pred8[i, 0]), float(pred8[i, 1])] for i in range(8)]
    kps9.append(pred_c)
    nvalid = sum(1 for k in kps9 if k is not None)
    pnp_ok, honest8 = 0, np.inf
    dims = entry["entry"]["dims"]
    if nvalid >= N_DET_MIN:
        best, proj_all = solve_perframe(kps9, K, (dims["W"], dims["D"], dims["H"]),
                                        img.shape)
        if best is not None:
            pnp_ok = 1
            pa = proj_all[:8].copy()
            bad = (pa[:, 0] == -1.0) & (pa[:, 1] == -1.0)
            pa[bad] = np.nan
            hd = hungarian(pa, gt8)
            if hd is not None:
                honest8 = float(np.mean(hd))

    elev = elev_from_pose(o.get("pose_transform"))
    return {"fid": entry["fid"], "v_geom": v_geom, "n_det": m["n_det"],
            "det": 1 if m["n_det"] >= N_DET_MIN else 0,
            "corner": m["overall"], "front": m["front"], "back": m["back"],
            "worst2": worst2, "pnp_ok": pnp_ok, "pnp_honest8": honest8,
            "elev": elev, "pred8": pred8.tolist(), "gt8": gt8.tolist(),
            "pred_c": pred_c, "ip": ip}


def med(a):
    a = [x for x in a if x is not None and np.isfinite(x)]
    return round(float(np.median(a)), 1) if a else None


def summarize(rows):
    n = len(rows)
    if n == 0:
        return {"n": 0}
    good = gross = ncorner = 0
    for r in rows:
        pr, gt = np.array(r["pred8"], float), np.array(r["gt8"], float)
        d = hungarian(pr, gt)
        if d is None:
            continue
        for x in d:
            ncorner += 1
            good += int(x < GOOD_PX)
            gross += int(x > GROSS_PX)
    honest = [r["pnp_honest8"] for r in rows
              if r["pnp_ok"] and np.isfinite(r["pnp_honest8"])]
    return {
        "n": n,
        "det_pct": round(100 * np.mean([r["det"] for r in rows]), 1),
        "front_med": med([r["front"] for r in rows]),
        "rear_med": med([r["back"] for r in rows]),
        "corner_med": med([r["corner"] for r in rows]),
        "worst2_med": med([r["worst2"] for r in rows]),
        "pnp_pct": round(100 * np.mean([r["pnp_ok"] for r in rows]), 1),
        "honest8_med": med(honest),
        "good_pct": round(100 * good / ncorner, 1) if ncorner else None,
        "gross_pct": round(100 * gross / ncorner, 1) if ncorner else None,
        "ncorner": ncorner,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)

    val = json.load(open(VAL_LIST))
    model = load_model(args.weights, device)
    rows = []
    for e in val:
        r = eval_frame(model, e, device)
        if r is not None:
            rows.append(r)

    v8 = [r for r in rows if r["v_geom"] == 8]
    vlt = [r for r in rows if r["v_geom"] < 8]
    low = [r for r in rows if r["elev"] is not None and r["elev"] < 10]
    near = [r for r in rows if r["elev"] is not None and 10 <= r["elev"] < 15]
    out = {"weights": args.weights, "tag": args.tag, "n": len(rows),
           "overall": summarize(rows), "V8": summarize(v8), "Vlt8": summarize(vlt),
           "elev_lt10": summarize(low), "elev_10_15": summarize(near),
           "rows": [{k: r[k] for k in ("fid", "v_geom", "n_det", "corner",
                                       "front", "back", "worst2", "pnp_ok",
                                       "pnp_honest8", "elev")} for r in rows]}
    fp = os.path.join(OUT_DIR, f"q1_val_{args.tag}.json")
    with open(fp, "w") as f:
        json.dump(out, f, indent=2, default=lambda x: None
                  if isinstance(x, float) and not np.isfinite(x) else x)
    o = out["overall"]
    print(f"[{args.tag}] n={o['n']} det%={o['det_pct']} front={o['front_med']} "
          f"rear={o['rear_med']} corner={o['corner_med']} honest8={o['honest8_med']} "
          f"pnp%={o['pnp_pct']} good%={o['good_pct']} gross%={o['gross_pct']}")
    print(f"[save] {fp}")


if __name__ == "__main__":
    main()
