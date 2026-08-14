"""paper_s2_testset17_overlay.py — RENDER-ONLY overlays for the designated
hand-annotated test set (cad11 + noapril6 = 17 frames).

Model: paper_s2 Stage B best (net_epoch_0057.pth), squash eval-parity
(640x480->400x400, belief(50)->orig x(W/50, H/50)) — reuses eval_frame_squash
verbatim from paper_s2_real_eval.

PnP dims = (W, D, H) = (1.1, 1.3, 0.12) m  [USER-SPECIFIED, height 0.12 NOT 0.11].
solve_pose reads module-global APNP.PALLET_DIMS internally (ignores its `dims` arg),
so we override APNP.PALLET_DIMS = (1.1,1.3,0.12) before eval. order-free W/D swap
(110-front vs 130-front) is preserved by solve_pose. corner_med is order-free
Hungarian => dims-independent; only honest8/pose change vs P5's 0.11.

Every frame is rendered, incl. under-detected (n_det<6): raw belief peaks are still
drawn and caption flagged UNDER-DET.

Overlay per frame (original image):
  GT (green)    : hand-anno projected_cuboid 8 corners + centroid + cuboid edges
  raw kp (blue) : Stage B belief-peak prediction (+ centroid)
  PnP (red)     : Stage B PnP-reprojected cuboid (1.1x1.3x0.12) — only if n_det>=6

Usage: conda activate pallet-pose; python scripts/stage0/paper_s2/paper_s2_testset17_overlay.py
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

# challenge/scripts 도 계열 폴더로 나뉘었다. 예전엔 먼저 로드된 다른 모듈이
# 넣어 주는 데 기대고 있었는데(아래 import annotate_pnp), 그건 로드 순서 의존이다.
_CS = _os.path.join(_os.path.dirname(_os.path.dirname(_S0)), "challenge", "scripts")
_sys.path[:0] = [_CS] + [_os.path.join(_CS, _d) for _d in ("annotate", "infer", "live")]

import os

import numpy as np

# paper_s2_real_eval sets sys.path (stage0, eval, DOPE/common, challenge/scripts)
# and imports the harness; reuse its eval_frame_squash + ROOT + device path.
from paper_s2_real_eval import ROOT, E, eval_frame_squash  # noqa: E402
import annotate_pnp as APNP  # noqa: E402

import cv2  # noqa: E402
import torch  # noqa: E402

# ── USER-SPECIFIED dims (height 0.12, NOT 0.11) ──────────────────────────────
APNP.PALLET_DIMS = (1.1, 1.3, 0.12)

MANIFEST = os.path.join(
    ROOT, "data/pallet/eval_results/stage22_myannot_eval/testset_full8_manifest.txt")
WEIGHTS = os.path.join(ROOT, "weights/paper_s2_stageB/net_epoch_0057.pth")
OUT_DIR = os.path.join(
    ROOT, "data/pallet/eval_results/paper_s2_scratch_diffpnp/testset17_overlays")

EDG = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
       (0, 4), (1, 5), (2, 6), (3, 7)]
GREEN = (60, 200, 60)   # GT
BLUE = (255, 90, 0)     # raw belief-peak pred
RED = (0, 0, 220)       # PnP reproj cuboid
N_DET_MIN = 6


def read_manifest(path):
    rows = []
    for ln in open(path):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        dom, fid, jp, ip = ln.split()
        rows.append((dom, fid, os.path.join(ROOT, jp), os.path.join(ROOT, ip)))
    return rows


def _cuboid(img, pts, color, thick=1, dot=4):
    pts = np.asarray(pts, float)
    for a, b in EDG:
        if (a >= len(pts) or b >= len(pts)
                or np.isnan(pts[a, 0]) or np.isnan(pts[b, 0])):
            continue
        cv2.line(img, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)),
                 color, thick)
    for i in range(min(8, len(pts))):
        if not np.isnan(pts[i, 0]):
            cv2.circle(img, (int(pts[i, 0]), int(pts[i, 1])), dot, color, -1)


def _caption(img, lines):
    h = 17 * len(lines) + 6
    cv2.rectangle(img, (0, 0), (img.shape[1], h), (0, 0, 0), -1)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (4, 16 + 17 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (255, 255, 255), 1)


def draw(row, dom):
    img = cv2.imread(row["ip"])
    if img is None:
        return None, None
    gt8 = np.array(row["gt8"], float)
    gtc = np.array(row["gtc"], float) if row.get("gtc") is not None else None
    pr8 = np.array(row["pred8"], float)
    proj = row["proj_all"]
    det = row["n_det"] >= N_DET_MIN

    # GT green (8 corner + edges + centroid)
    _cuboid(img, gt8, GREEN, 1, 3)
    if gtc is not None and gtc.size >= 2:
        cv2.circle(img, (int(gtc[0]), int(gtc[1])), 4, GREEN, 1)
    # PnP reproj cuboid red (only when detected & pnp_ok)
    if det and proj is not None:
        _cuboid(img, np.array(proj, float), RED, 1, 3)
    # raw belief-peak blue (+ centroid)
    for i in range(8):
        if not np.isnan(pr8[i, 0]):
            cv2.circle(img, (int(pr8[i, 0]), int(pr8[i, 1])), 4, BLUE, -1)
    if row.get("pred_c") is not None:
        pc = row["pred_c"]
        cv2.circle(img, (int(pc[0]), int(pc[1])), 4, BLUE, 1)

    cm = row["corner"] if np.isfinite(row["corner"]) else -1
    h8 = row["pnp_honest8"] if np.isfinite(row["pnp_honest8"]) else -1
    if det:
        l0 = f"{dom} {row['fid']}"
        l1 = (f"DET n_det={row['n_det']} corner_med={cm:.1f}px "
              f"honest8={h8:.1f}px (dims 1.1x1.3x0.12)")
    else:
        l0 = f"{dom} {row['fid']}"
        l1 = f"UNDER-DET (n_det={row['n_det']}) [raw peaks shown, no PnP]"
    _caption(img, [l0, l1,
                   "GT green | raw pred blue | PnP-reproj red"])
    tag = "det" if det else "underdet"
    return img, tag


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = read_manifest(MANIFEST)
    print(f"[manifest] {len(frames)} frames; dims=(W,D,H)={APNP.PALLET_DIMS}")

    mdl = E.load_model(WEIGHTS, device)
    saved, n_det, corners, honest8s = [], 0, [], []
    dom_stats = {}
    for dom, fid, jp, ip in frames:
        row = eval_frame_squash(mdl, jp, ip, device)
        if row is None:
            print(f"[skip] {dom} {fid}: image/json load failed ({ip})")
            continue
        img, tag = draw(row, dom)
        if img is None:
            print(f"[skip] {dom} {fid}: image load failed for draw")
            continue
        fn = f"{dom}_{tag}_{fid}.jpg"
        fp = os.path.join(OUT_DIR, fn)
        cv2.imwrite(fp, img)
        saved.append(fn)
        st = dom_stats.setdefault(dom, {"det": 0, "under": 0})
        if row["n_det"] >= N_DET_MIN:
            n_det += 1
            st["det"] += 1
            if np.isfinite(row["corner"]):
                corners.append(row["corner"])
            if np.isfinite(row["pnp_honest8"]):
                honest8s.append(row["pnp_honest8"])
        else:
            st["under"] += 1
        c = row["corner"] if np.isfinite(row["corner"]) else float("nan")
        h = row["pnp_honest8"] if np.isfinite(row["pnp_honest8"]) else float("nan")
        print(f"  {fn:<52} n_det={row['n_det']} corner={c:.1f} honest8={h:.1f}")

    def q(a):
        a = np.array(a, float)
        return (f"n={len(a)} med={np.median(a):.1f} "
                f"min={a.min():.1f} max={a.max():.1f}") if len(a) else "n=0"

    print("\n" + "=" * 68)
    print(f"[SAVED] {len(saved)} overlays -> {OUT_DIR}")
    print(f"[DET]   {n_det}/{len(frames)} detected (n_det>=6); "
          f"{len(frames) - n_det} under-det")
    for dom in sorted(dom_stats):
        print(f"   {dom}: det={dom_stats[dom]['det']} "
              f"under={dom_stats[dom]['under']}")
    print(f"[corner_med px] (order-free, dims-indep)  {q(corners)}")
    print(f"[honest8 px]    (dims 1.1x1.3x0.12)       {q(honest8s)}")


if __name__ == "__main__":
    main()
