#!/usr/bin/env python
"""diffpnp3d_val_select.py — PAPER_S2 Phase 4 synthetic-val COMPOSITE best selector.

Given a weights dir containing per-epoch checkpoints (net_epoch_XXXX.pth), evaluate
every candidate on the synthetic validation set with the SAME train/infer-parity
path as Q1 (anisotropic squash 640x480->400, belief(50)->orig x12.8/x9.6, per-frame
dims order-free PnP honest8) and pick the COMPOSITE best epoch.

Reuses diffpnp3d_q1_eval.{load_model,eval_frame,summarize} verbatim (parity locked).

Composite rule (PLAN P4):
  - primary  : rear_med (down) + honest8_med (down)  [rank-sum, scale-robust]
  - guard    : front_med not worse / det% not collapse / good% not collapse /
               gross% not increase  (relative to the best-over-candidates value)
  - tie-break: corner_med (down)
If no candidate passes the guards, fall back to all candidates (guards logged).

val set:
  --val full  -> all 1500 frames from pnp_valid_3d_index/val.json (recommended)
  --val q1    -> the fixed q1_split/val_list.json (500)

Outputs (under data/pallet/results/paper_s2_scratch_diffpnp/):
  <tag>_val_select.md    (audit table)
  <tag>_val_select.json  (all candidate summaries + chosen)
  <weights_dir>/<tag>_best.pth   (symlink -> chosen checkpoint)
  writes chosen abs path to stdout line "BEST_CKPT=<path>" (driver parses this)

Usage:
  python scripts/stage0/diffpnp3d_val_select.py \
      --weights_dir weights/paper_s2_stageA --tag stageA --val full
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))
import diffpnp3d_q1_eval as Q1E  # noqa: E402

IDX_DIR = os.path.join(ROOT, "data/pallet/results/paper_s2_scratch_diffpnp/pnp_valid_3d_index")
DATA = os.path.join(ROOT, "data/pallet/training_data")
RES_DIR = os.path.join(ROOT, "data/pallet/results/paper_s2_scratch_diffpnp")

# guard tolerances (px / pct): a candidate is guarded-OK if it is not worse than
# the best-over-candidates value by more than these margins.
FRONT_TOL = 1.5   # px
DET_TOL = 5.0     # pct
GOOD_TOL = 8.0    # pct
GROSS_TOL = 5.0   # pct


def build_val_full():
    vidx = json.load(open(os.path.join(IDX_DIR, "val.json")))
    out = []
    for rel in sorted(vidx.keys()):
        stem = os.path.basename(rel)[:-5]
        jp = os.path.join(DATA, rel)
        ip = jp.replace(".json", ".png")
        if os.path.exists(jp) and os.path.exists(ip):
            out.append({"fid": stem, "json": jp, "png": ip, "entry": vidx[rel]})
    return out


def build_val_q1():
    return json.load(open(os.path.join(
        RES_DIR, "q1_split/val_list.json")))


def list_candidates(wdir):
    """net_epoch_XXXX.pth candidates (exclude final_*), sorted by epoch."""
    cands = []
    for p in glob.glob(os.path.join(wdir, "net_*_*.pth")):
        b = os.path.basename(p)
        if b.startswith("final_"):
            continue
        try:
            ep = int(os.path.splitext(b)[0].split("_")[-1])
        except Exception:
            continue
        cands.append((ep, p))
    cands.sort()
    return cands


def eval_ckpt(wp, val, device):
    model = Q1E.load_model(wp, device)
    rows = []
    for e in val:
        r = Q1E.eval_frame(model, e, device)
        if r is not None:
            rows.append(r)
    del model
    torch.cuda.empty_cache()
    v8 = [r for r in rows if r["v_geom"] == 8]
    return Q1E.summarize(rows), Q1E.summarize(v8), len(rows)


def _num(x, default):
    return x if (x is not None and np.isfinite(x)) else default


def choose(cands):
    """cands: list of dict with overall metrics. Returns (best_idx, guard_pass_flags)."""
    front_best = min(_num(c["front"], 1e9) for c in cands)
    det_best = max(_num(c["det"], -1e9) for c in cands)
    good_best = max(_num(c["good"], -1e9) for c in cands)
    gross_best = min(_num(c["gross"], 1e9) for c in cands)
    guard = []
    for c in cands:
        ok = (_num(c["front"], 1e9) <= front_best + FRONT_TOL
              and _num(c["det"], -1e9) >= det_best - DET_TOL
              and _num(c["good"], -1e9) >= good_best - GOOD_TOL
              and _num(c["gross"], 1e9) <= gross_best + GROSS_TOL
              and c["rear"] is not None)
        guard.append(ok)
    pool = [i for i, ok in enumerate(guard) if ok]
    if not pool:
        pool = [i for i, c in enumerate(cands) if c["rear"] is not None]
    if not pool:
        return None, guard
    # rank-sum on rear + honest8 (lower better), tie-break corner
    def rank(key):
        vals = [( _num(cands[i][key], 1e9), i) for i in pool]
        vals.sort()
        r = {}
        for pos, (_, i) in enumerate(vals):
            r[i] = pos
        return r
    rr, rh = rank("rear"), rank("honest8")
    best = min(pool, key=lambda i: (rr[i] + rh[i], _num(cands[i]["corner"], 1e9)))
    return best, guard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights_dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--val", choices=["full", "q1"], default="full")
    args = ap.parse_args()
    wdir = args.weights_dir if os.path.isabs(args.weights_dir) else os.path.join(ROOT, args.weights_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    val = build_val_full() if args.val == "full" else build_val_q1()
    n_v8 = sum(1 for v in val if v["entry"].get("V8"))
    print(f"[val-select] tag={args.tag} val={args.val} n={len(val)} (V8={n_v8})")

    cands_paths = list_candidates(wdir)
    if not cands_paths:
        print(f"[val-select] ERROR: no net_*.pth candidates in {wdir}")
        sys.exit(2)
    print(f"[val-select] {len(cands_paths)} candidates: "
          f"epochs {[e for e, _ in cands_paths]}")

    cands = []
    for ep, p in cands_paths:
        ov, v8, nrows = eval_ckpt(p, val, device)
        c = {"epoch": ep, "path": p, "n": nrows,
             "rear": ov["rear_med"], "honest8": ov["honest8_med"],
             "front": ov["front_med"], "corner": ov["corner_med"],
             "det": ov["det_pct"], "good": ov["good_pct"],
             "gross": ov["gross_pct"], "pnp": ov["pnp_pct"],
             "v8_rear": v8["rear_med"], "v8_corner": v8["corner_med"],
             "overall": ov, "v8": v8}
        cands.append(c)
        print(f"  ep{ep:04d} rear={c['rear']} honest8={c['honest8']} "
              f"front={c['front']} corner={c['corner']} det={c['det']} "
              f"good={c['good']} gross={c['gross']}")

    best_idx, guard = choose(cands)
    for i, c in enumerate(cands):
        c["guard_ok"] = bool(guard[i])
    best = cands[best_idx] if best_idx is not None else None

    # ---- write audit md/json ----
    os.makedirs(RES_DIR, exist_ok=True)
    md = [f"# {args.tag} synthetic-val composite selection",
          f"", f"val={args.val} (n={len(val)}, V8={n_v8})  weights_dir={wdir}",
          f"guard tol: front+{FRONT_TOL}px det-{DET_TOL} good-{GOOD_TOL} gross+{GROSS_TOL}",
          f"primary=rank(rear)+rank(honest8) among guard-pass, tie=corner", "",
          "```",
          "ep    rear  honest8 front corner  det   good  gross  guard  best",
          "-----------------------------------------------------------------"]
    for i, c in enumerate(cands):
        star = "  <==" if i == best_idx else ""
        md.append(f"{c['epoch']:<5d} {str(c['rear']):>5} {str(c['honest8']):>7} "
                  f"{str(c['front']):>5} {str(c['corner']):>6} {str(c['det']):>5} "
                  f"{str(c['good']):>5} {str(c['gross']):>6}  "
                  f"{'Y' if c['guard_ok'] else '.':>4}{star}")
    md.append("```")
    if best is not None:
        md += ["", f"**BEST = epoch {best['epoch']}** -> `{best['path']}`",
               f"rear={best['rear']} honest8={best['honest8']} front={best['front']} "
               f"corner={best['corner']} det={best['det']} good={best['good']} "
               f"gross={best['gross']}"]
    mdp = os.path.join(RES_DIR, f"{args.tag}_val_select.md")
    with open(mdp, "w") as f:
        f.write("\n".join(md) + "\n")
    jp = os.path.join(RES_DIR, f"{args.tag}_val_select.json")
    with open(jp, "w") as f:
        json.dump({"tag": args.tag, "val": args.val, "n": len(val),
                   "best_epoch": best["epoch"] if best else None,
                   "best_path": best["path"] if best else None,
                   "candidates": cands}, f, indent=2,
                  default=lambda x: None if isinstance(x, float)
                  and not np.isfinite(x) else x)

    if best is None:
        print("[val-select] ERROR: no valid candidate chosen")
        sys.exit(3)

    link = os.path.join(wdir, f"{args.tag}_best.pth")
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(os.path.abspath(best["path"]), link)
    print(f"[val-select] audit -> {mdp}")
    print(f"[val-select] symlink -> {link}")
    print(f"BEST_CKPT={os.path.abspath(best['path'])}")


if __name__ == "__main__":
    main()
