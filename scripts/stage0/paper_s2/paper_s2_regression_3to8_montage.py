"""paper_s2_regression_3to8_montage.py — 4-panel FAILURE MONTAGE (NO retrain).

Goal (paper/report figure): show that the PAPER_S2 3-8deg regression is ALREADY
present in the RAW belief-peak heatmap keypoints (pre-PnP), not a PnP/DiffPnP
post-processing artefact. Same original frame, 4 panels side-by-side:

  P1  paper_s1        RAW keypoint (blue)  + GT (green)
  P2  paper_s2_stageB RAW keypoint (blue)  + GT (green)
  P3  paper_s2_stageB PnP-reproj cuboid (red) + GT (green)
  P4  GT projected_cuboid (green) alone

Case types (each 2-3 frames if available), tagged in filename + panel caption:
  (a) raw_worse   : s2 raw_corner > s1 raw_corner  (regression born in raw heatmap)
  (b) rear_gain_front_cost : s2 rear < s1 rear (rear improved) but s2 front/corner
       worse (front-sacrifice / rear-redistribution)
  (c) honest8_worse : s2 pnp_corner(honest8) > s1 pnp_corner

EVAL-PARITY (non-negotiable, reused verbatim from paper_s2_real_eval):
  paper_s1        -> aspect no-pad (E.eval_frame, PAD_ASPECT=0)
  paper_s2_stageB -> squash 640x480->400x400, belief(50)->orig x(W/50,H/50)
Only the RENDER differs from the diag script (which draws 2-panel triplets).

Usage: conda activate pallet-pose; python scripts/stage0/paper_s2/paper_s2_regression_3to8_montage.py
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import os

import numpy as np

import paper_s2_regression_3to8_diag as D  # sets sys.path, imports harness
from paper_s2_regression_3to8_diag import (
    frame_split_errors, BASELINE, CHALLENGER, _clean,
)
from paper_s2_real_eval import S, E, ROOT, THRESH, PAD_ASPECT, eval_frame_squash

import cv2  # noqa: E402
import torch  # noqa: E402

OUT_DIR = D.OUT_DIR
MONT_DIR = os.path.join(OUT_DIR, "regression_3to8_montage")

EDG = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
       (0, 4), (1, 5), (2, 6), (3, 7)]
GREEN = (60, 200, 60)
BLUE = (255, 90, 0)   # BGR -> blue-ish
RED = (0, 0, 220)

# per-case selection margins (px), directional (diag: 3-8 raw d=+2.5 median)
RAW_MARGIN = 2.0     # (a) s2 raw worse than s1 by >= this
REAR_GAIN = 2.0      # (b) s2 rear better (lower) than s1 by >= this
FRONT_COST = 1.0     # (b) s2 front worse (higher) than s1 by >= this
H8_MARGIN = 3.0      # (c) s2 honest8 worse than s1 by >= this
N_PER_CASE = 3


def _draw_cuboid(img, pts, color, thick=1, dot=4):
    pts = np.asarray(pts, float)
    for a, b in EDG:
        if np.isnan(pts[a, 0]) or np.isnan(pts[b, 0]):
            continue
        cv2.line(img, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)),
                 color, thick)
    for i in range(8):
        if not np.isnan(pts[i, 0]):
            cv2.circle(img, (int(pts[i, 0]), int(pts[i, 1])), dot, color, -1)


def _caption(img, lines):
    h = 16 * len(lines) + 6
    cv2.rectangle(img, (0, 0), (img.shape[1], h), (0, 0, 0), -1)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (4, 15 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1)


def panel(ip, title, gt8, kp=None, kp_color=BLUE, proj=None, cap2=""):
    img = cv2.imread(ip)
    if img is None:
        return None
    _draw_cuboid(img, gt8, GREEN, 1, 3)          # GT always green
    if proj is not None:
        _draw_cuboid(img, proj, RED, 1, 3)        # PnP reproj red cuboid
    if kp is not None:
        kp = np.asarray(kp, float)
        for i in range(8):
            if not np.isnan(kp[i, 0]):
                cv2.circle(img, (int(kp[i, 0]), int(kp[i, 1])), 4, kp_color, -1)
    _caption(img, [title, cap2])
    return img


def make_montage(rb_row, rc_row, rb_split, rc_split, elev, case_tag, fid):
    """4 panels for one frame. rb=s1 row, rc=stageB row (harness dicts)."""
    ip = rc_row["ip"]
    gt8 = np.array(rc_row["gt8"], float)
    s1_raw = np.array(rb_row["pred8"], float)
    s2_raw = np.array(rc_row["pred8"], float)
    s2_proj = None if rc_row["proj_all"] is None else _clean(rc_row["proj_all"])

    p1 = panel(ip, "P1 s1 RAW (blue) +GT(green)", gt8, kp=s1_raw, kp_color=BLUE,
               cap2=f"raw_cnr={rb_split['raw_corner']:.0f} rear={rb_split['rear']:.0f}")
    p2 = panel(ip, "P2 s2B RAW (blue) +GT(green)", gt8, kp=s2_raw, kp_color=BLUE,
               cap2=f"raw_cnr={rc_split['raw_corner']:.0f} rear={rc_split['rear']:.0f}")
    p3 = panel(ip, "P3 s2B PnP-reproj (red) +GT", gt8, proj=s2_proj,
               cap2=f"honest8={rc_split['pnp_corner']:.0f} rear={rc_split['rear']:.0f}")
    p4 = panel(ip, "P4 GT cuboid (green)", gt8,
               cap2=f"el={elev:.0f} V={rc_row['v_geom']} {case_tag}")
    panels = [p1, p2, p3, p4]
    if any(p is None for p in panels):
        return None
    h = max(p.shape[0] for p in panels)
    gap = 4
    W = sum(p.shape[1] for p in panels) + gap * 3
    canvas = np.zeros((h, W, 3), np.uint8)
    x = 0
    for p in panels:
        canvas[:p.shape[0], x:x + p.shape[1]] = p
        x += p.shape[1] + gap
    d_raw = rc_split["raw_corner"] - rb_split["raw_corner"]
    d_h8 = rc_split["pnp_corner"] - rb_split["pnp_corner"]
    fn = (f"{case_tag}_el{elev:.0f}_draw{d_raw:+.0f}_dh8{d_h8:+.0f}_{fid}.jpg")
    path = os.path.join(MONT_DIR, fn)
    cv2.imwrite(path, canvas)
    return path


def run():
    os.makedirs(MONT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fl = S.frames_filterval()
    elev = {fid: S.elev_of(jp) for _, fid, jp, _ in fl}

    rows = {BASELINE: {}, CHALLENGER: {}}
    for mname, wrel, mode in [
        (BASELINE, "weights/paper_s1/paper_s1_maskaux/net_epoch_0065.pth", "aspect"),
        (CHALLENGER, "weights/paper_s2_stageB/net_epoch_0057.pth", "squash"),
    ]:
        mdl = E.load_model(os.path.join(ROOT, wrel), device)
        for dom, fid, jp, ip in fl:
            if mode == "squash":
                r = eval_frame_squash(mdl, jp, ip, device)
            else:
                r = E.eval_frame(mdl, jp, ip, device, THRESH, PAD_ASPECT)
            if r is not None:
                rows[mname][r["fid"]] = r
        del mdl
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"[done] {mname} ({mode}) -> {len(rows[mname])} frames")

    # 3-8deg shared frames with both detected + s2 pnp_ok (panel P3 needs proj)
    cand = []
    for fid, rc in rows[CHALLENGER].items():
        e = elev.get(fid)
        if e is None or not (3 <= e < 8):
            continue
        rb = rows[BASELINE].get(fid)
        if rb is None:
            continue
        sb = frame_split_errors(rb)
        sc = frame_split_errors(rc)
        if not (rc["pnp_ok"] and np.isfinite(sc["pnp_corner"])):
            continue  # P3 (s2 PnP reproj) must exist
        if not (np.isfinite(sb["raw_corner"]) and np.isfinite(sc["raw_corner"])):
            continue  # need both raw for P1/P2 comparison
        cand.append((fid, e, rb, rc, sb, sc))
    print(f"[cand] 3-8deg shared (both raw + s2 pnp_ok): {len(cand)}")

    # case (a) raw_worse: s2 raw - s1 raw, desc
    a_sorted = sorted(cand, key=lambda x: x[5]["raw_corner"] - x[4]["raw_corner"],
                      reverse=True)
    a_pick = [c for c in a_sorted
              if c[5]["raw_corner"] - c[4]["raw_corner"] >= RAW_MARGIN][:N_PER_CASE]

    # case (b) rear_gain_front_cost: rear improved AND front (or honest8) regressed
    def b_key(c):
        _, _, _, _, sb, sc = c
        rear_gain = sb["rear"] - sc["rear"]        # positive = s2 rear better
        front_cost = sc["front"] - sb["front"]     # positive = s2 front worse
        return rear_gain + front_cost              # rank by combined effect
    b_cand = [c for c in cand
              if np.isfinite(c[4]["rear"]) and np.isfinite(c[5]["rear"])
              and np.isfinite(c[4]["front"]) and np.isfinite(c[5]["front"])
              and (c[4]["rear"] - c[5]["rear"]) >= REAR_GAIN
              and (c[5]["front"] - c[4]["front"]) >= FRONT_COST]
    b_pick = sorted(b_cand, key=b_key, reverse=True)[:N_PER_CASE]

    # case (c) honest8_worse: s2 pnp_corner - s1 pnp_corner, desc
    c_cand = [c for c in cand
              if np.isfinite(c[4]["pnp_corner"]) and np.isfinite(c[5]["pnp_corner"])]
    c_sorted = sorted(c_cand, key=lambda x: x[5]["pnp_corner"] - x[4]["pnp_corner"],
                      reverse=True)
    c_pick = [c for c in c_sorted
              if c[5]["pnp_corner"] - c[4]["pnp_corner"] >= H8_MARGIN][:N_PER_CASE]

    saved = {"a_raw_worse": [], "b_rear_gain_front_cost": [], "c_honest8_worse": []}
    for tag, picks in [("a_raw_worse", a_pick),
                       ("b_rear_gain_front_cost", b_pick),
                       ("c_honest8_worse", c_pick)]:
        for fid, e, rb, rc, sb, sc in picks:
            p = make_montage(rb, rc, sb, sc, e, tag, fid)
            if p:
                saved[tag].append(os.path.basename(p))

    print("\n[MONTAGE SAVED] ->", MONT_DIR)
    for tag, files in saved.items():
        print(f"  {tag}: {len(files)}")
        for f in files:
            print("    " + f)
    return saved


if __name__ == "__main__":
    run()
