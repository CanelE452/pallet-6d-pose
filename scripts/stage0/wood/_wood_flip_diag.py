"""_wood_flip_diag.py — diagnose the flip_infer_squash coordinate/offset in wood filter.

For each wood GT frame:
  - normal decode -> pred8 (orig px)
  - flip decode RAW -> predF8 in FLIPPED-image px (before un-flip / swap)
Match physical corner: flipped index i == physical corner swap[i] (FLIP_PAIRS).
For a physical corner c present in both:
    x_n = pred8[c, 0]                       (normal, orig px)
    x_f = predF8[i, 0]  where swap[i]=c     (flip raw, flipped-image px)
If model were perfectly flip-equivariant AND un-flip axis correct:
    x_n + x_f == (W-1)    -> residual r = x_n + x_f - (W-1) ~ 0
    y_n == y_f            -> ry ~ 0
r != 0 constant => systematic horiz bias (2*beta). We report r stats overall,
and separately for GT-good vs GT-bad, plus f3 under un-flip variants:
    v_curr  : x_o = W - x_f          (current code)
    v_axis  : x_o = (W-1) - x_f      (principled pixel-flip axis)
    v_debias: x_o = (W-1) - x_f + r_med  (subtract global residual = calibration)
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import glob
import importlib.util
import json
import os
import sys

import cv2
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(ROOT, "challenge", "scripts"))
import torch  # noqa: E402


def _load(n, p):
    spec = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


wf = _load("wf", os.path.join(ROOT, "scripts", "stage0", "wood", "wood_infer_filter.py"))
s1 = wf.s1
from eval_pvnet_heads import split_metrics  # noqa: E402
from annotate_wood import K_for_resolution  # noqa: E402

FLIP_PAIRS = s1.FLIP_PAIRS
GT_DIRS = {
    "pallet_20260618_183705":
        os.path.join(ROOT, "challenge", "data", "wood_pallet_20260618_183705_manual_gt"),
    "pallet_20260618_184309":
        os.path.join(ROOT, "challenge", "data", "wood_pallet_20260618_184309_manual_gt"),
}
FRAME_ROOT = os.path.join(ROOT, "data", "pallet", "raw_data", "wood", "selected")
N_DET_MIN = wf.N_DET_MIN
GOOD_PX = 10.0


def swap_arr():
    sw = list(range(9))
    for a, b in FLIP_PAIRS:
        sw[a], sw[b] = b, a
    return sw


def flip_score_variant(pred8, pred_c, predF8, predF_c, W, delta):
    """un-flip: x_o = (W-1) - x_f + delta ; then swap; compare to normal."""
    sw = swap_arr()
    kpf = [None] * 9
    for i in range(8):
        if not np.isnan(predF8[i, 0]):
            kpf[i] = ((W - 1) - predF8[i, 0] + delta, predF8[i, 1])
    if predF_c is not None:
        kpf[8] = ((W - 1) - predF_c[0] + delta, predF_c[1])
    kpB = [kpf[sw[i]] for i in range(9)]
    return s1.flip_score(pred8, pred_c, kpB)


from filter_pr_camfacing import extract_keypoints_from_belief  # noqa: E402


def raw_belief_flip(model, img_bgr, device):
    """flip image, run model, return raw belief (9,50,50) of the FLIPPED image."""
    flip = cv2.flip(img_bgr, 1)
    with torch.no_grad():
        beliefs, _ = model(wf.preprocess_squash(flip).to(device))
    return beliefs[-1][0].cpu().numpy()


def flip_infer_beliefmirror(model, img_bgr, device):
    """PRINCIPLED: mirror belief in grid-space + channel-swap, re-decode (same OFFSET)."""
    H, W = img_bgr.shape[:2]
    sx, sy = W / 50.0, H / 50.0
    bel_f = raw_belief_flip(model, img_bgr, device)      # (9,50,50) flipped
    bel_u = bel_f[:, :, ::-1].copy()                     # mirror x in grid space
    # channel swap (left<->right corners) so channel i = physical corner i
    sw = swap_arr()
    bel_u = bel_u[[sw[i] for i in range(9)]]
    kps = extract_keypoints_from_belief(bel_u, wf.THRESH)
    kpB = [None] * 9
    for i in range(9):
        if kps[i][0] >= 0:
            kpB[i] = (kps[i][0] * sx, kps[i][1] * sy)
    return kpB


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = s1.E.load_model(wf.WEIGHTS, device)
    sw = swap_arr()

    frames = []
    for ses, gtdir in GT_DIRS.items():
        for jp in sorted(glob.glob(os.path.join(gtdir, "*.json"))):
            fid = os.path.splitext(os.path.basename(jp))[0]
            ip = os.path.join(FRAME_ROOT, ses, fid + ".jpg")
            if os.path.exists(ip):
                frames.append((ses, fid, jp, ip))

    all_rx, all_ry = [], []
    good_rx, bad_rx = [], []
    rows = []  # (fid, cmed, good, f3_curr, f3_axis)
    per_frame_r = []
    for ses, fid, jp, ip in frames:
        img = cv2.imread(ip)
        H, W = img.shape[:2]
        gt8 = np.array(json.load(open(jp))["objects"][0]["projected_cuboid"], float)[:8]
        pred8, pred_c, _, _ = wf.decode_squash(model, img, device)
        if (~np.isnan(pred8[:, 0])).sum() < N_DET_MIN:
            continue
        predF8, predF_c, _, _ = wf.decode_squash(model, cv2.flip(img, 1), device)
        m = split_metrics(pred8, gt8)
        cmed = m["overall"] if np.isfinite(m["overall"]) else None
        good = cmed is not None and cmed < GOOD_PX

        # per-corner residual r = x_n + x_f - (W-1) for physical corner c
        frame_rx = []
        for i in range(8):
            c = sw[i]  # physical corner that flipped-index i represents
            if np.isnan(predF8[i, 0]) or np.isnan(pred8[c, 0]):
                continue
            rx = pred8[c, 0] + predF8[i, 0] - (W - 1)
            ry = pred8[c, 1] - predF8[i, 1]
            all_rx.append(rx)
            all_ry.append(ry)
            frame_rx.append(rx)
            (good_rx if good else bad_rx).append(rx)
        if frame_rx:
            per_frame_r.append((fid, good, float(np.median(frame_rx))))

        f3_curr = flip_score_variant(pred8, pred_c, predF8, predF_c, W, +1.0)  # W - x_f
        f3_axis = flip_score_variant(pred8, pred_c, predF8, predF_c, W, 0.0)   # (W-1)-x_f
        rows.append((f"{ses[-6:]}/{fid}", cmed, good, f3_curr, f3_axis))

    rx = np.array(all_rx)
    ry = np.array(all_ry)
    r_med = float(np.median(rx))
    print("=" * 70)
    print(f"per-corner residual r = x_n + x_f - (W-1)   (N corners = {len(rx)})")
    print(f"  median={r_med:.2f}px  mean={rx.mean():.2f}  std={rx.std():.2f}  "
          f"IQR=[{np.percentile(rx,25):.1f},{np.percentile(rx,75):.1f}]")
    print(f"  y-residual (x_n vs flip y): median={np.median(ry):.2f} std={ry.std():.2f}")
    print(f"  GT-good corners: median r={np.median(good_rx):.2f} std={np.std(good_rx):.2f} "
          f"(n={len(good_rx)})")
    print(f"  GT-bad  corners: median r={np.median(bad_rx):.2f} std={np.std(bad_rx):.2f} "
          f"(n={len(bad_rx)})")
    print(f"  -> sx=W/50={W/50:.2f}px ; r_med/sx={r_med/(W/50):.3f} cells ; "
          f"implied model horiz bias beta=r_med/2={r_med/2:.2f}px")
    print("=" * 70)

    # f3 under variants (add de-bias = subtract r_med i.e. delta = r_med with (W-1) axis)
    print("\nf3 under un-flip variants (good frames only):")
    print(f"{'frame':<22}{'cmed':>6}{'f3_curr(W-x)':>14}{'f3_axis(W-1-x)':>16}"
          f"{'f3_debias':>11}")
    for fr, cmed, good, f3c, f3a in rows:
        if not good:
            continue
        # de-bias needs recompute; approximate: shift axis by -r_med (since residual +r
        # means x_o too large; subtract r_med). Recompute properly:
        print(f"{fr:<22}{cmed:>6.1f}{_f(f3c):>14}{_f(f3a):>16}", end="")
        print(f"{'(see below)':>11}")

    # proper de-bias f3 for ALL frames: x_o = (W-1) - x_f + r_med  (delta=+r_med)
    print(f"\nf3 de-biased (x_o=(W-1)-x_f + r_med, r_med={r_med:.1f}) split by GT label:")
    rows2 = []
    for ses, fid, jp, ip in frames:
        img = cv2.imread(ip)
        H, W = img.shape[:2]
        gt8 = np.array(json.load(open(jp))["objects"][0]["projected_cuboid"], float)[:8]
        pred8, pred_c, _, _ = wf.decode_squash(model, img, device)
        if (~np.isnan(pred8[:, 0])).sum() < N_DET_MIN:
            continue
        predF8, predF_c, _, _ = wf.decode_squash(model, cv2.flip(img, 1), device)
        m = split_metrics(pred8, gt8)
        cmed = m["overall"] if np.isfinite(m["overall"]) else None
        good = cmed is not None and cmed < GOOD_PX
        f3c = flip_score_variant(pred8, pred_c, predF8, predF_c, W, +1.0)
        f3d = flip_score_variant(pred8, pred_c, predF8, predF_c, W, r_med)
        if f3c is None:
            continue
        rows2.append((f"{ses[-6:]}/{fid}", cmed, good, f3c, f3d))
    g_curr = [r[3] for r in rows2 if r[2]]
    g_deb = [r[4] for r in rows2 if r[2]]
    b_curr = [r[3] for r in rows2 if not r[2]]
    b_deb = [r[4] for r in rows2 if not r[2]]
    print(f"  GOOD (n={len(g_curr)}): f3_curr med={np.median(g_curr):.1f}  "
          f"f3_debias med={np.median(g_deb):.1f}  range_debias=[{min(g_deb):.1f},{max(g_deb):.1f}]")
    print(f"  BAD  (n={len(b_curr)}): f3_curr med={np.median(b_curr):.1f}  "
          f"f3_debias med={np.median(b_deb):.1f}  range_debias=[{min(b_deb):.1f},{max(b_deb):.1f}]")
    # ── PRINCIPLED belief-mirror (no fitted constant) ────────────────────
    print("\n" + "=" * 70)
    print("PRINCIPLED belief-mirror un-flip (no fitted constant):")
    bm_rows = []
    bm_rx = []
    for ses, fid, jp, ip in frames:
        img = cv2.imread(ip)
        H, W = img.shape[:2]
        gt8 = np.array(json.load(open(jp))["objects"][0]["projected_cuboid"], float)[:8]
        pred8, pred_c, _, _ = wf.decode_squash(model, img, device)
        if (~np.isnan(pred8[:, 0])).sum() < N_DET_MIN:
            continue
        m = split_metrics(pred8, gt8)
        cmed = m["overall"] if np.isfinite(m["overall"]) else None
        good = cmed is not None and cmed < GOOD_PX
        kpB = flip_infer_beliefmirror(model, img, device)
        f3bm = s1.flip_score(pred8, pred_c, kpB)
        # residual per corner under belief-mirror
        for i in range(8):
            if kpB[i] is not None and not np.isnan(pred8[i, 0]):
                bm_rx.append(pred8[i, 0] - kpB[i][0])
        if f3bm is not None:
            bm_rows.append((f"{ses[-6:]}/{fid}", cmed, good, f3bm))
    bm_rx = np.array(bm_rx)
    g_bm = [r[3] for r in bm_rows if r[2]]
    b_bm = [r[3] for r in bm_rows if not r[2]]
    print(f"  x-residual (pred_n - beliefmirror): median={np.median(bm_rx):.2f} "
          f"std={bm_rx.std():.2f}  (should be ~0 if principled)")
    print(f"  GOOD (n={len(g_bm)}): f3_beliefmirror med={np.median(g_bm):.1f} "
          f"range=[{min(g_bm):.1f},{max(g_bm):.1f}]")
    print(f"  BAD  (n={len(b_bm)}): f3_beliefmirror med={np.median(b_bm):.1f} "
          f"range=[{min(b_bm):.1f},{max(b_bm):.1f}]")

    print("\n  per-frame (sorted by cmed): curr vs debias vs belief-mirror")
    bmmap = {r[0]: r[3] for r in bm_rows}
    for fr, cmed, good, f3c, f3d in sorted(rows2, key=lambda x: (x[1] or 1e9)):
        cm = f"{cmed:.1f}" if cmed is not None else "n/a"
        bm = bmmap.get(fr)
        bms = f"{bm:.1f}" if bm is not None else " n/a"
        print(f"    {fr:<22} cmed={cm:>6}  GTgood={'Y' if good else 'n'}  "
              f"curr={f3c:>6.1f}  debias={f3d:>6.1f}  beliefmirror={bms:>6}")


def _f(v):
    return "  n/a " if v is None else f"{v:.1f}"


if __name__ == "__main__":
    main()
