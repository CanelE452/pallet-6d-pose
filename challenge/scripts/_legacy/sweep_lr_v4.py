"""Sweep all manual_gt frames to find ones where PnP candidates contain
LR-flip (lr_viol=1) solutions, and check whether fix v4 score picks them."""
import os, sys, json, glob
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from diagnose_lr_v4 import enumerate_candidates
from annotate_pnp import PALLET_DIMS


def main():
    root = r"C:\Users\minjae\Documents\github\FoundationPose\challenge\data"
    paths = sorted(glob.glob(os.path.join(root, "*manual_gt", "*.json")))
    print(f"{len(paths)} manual_gt files")
    print(f"{'file':70s}  cand  lr=0  lr=1  v4best_lr  v4best_err  v5diff")
    rows = []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        intr = d["camera_data"]["intrinsics"]
        K = np.array([[intr["fx"],0,intr["cx"]],[0,intr["fy"],intr["cy"]],[0,0,1]])
        objs = d.get("objects", [])
        if not objs: continue
        mkp = objs[0].get("manual_kps")
        if not mkp: continue
        kps_2d = [tuple(x) if x else None for x in mkp]
        try:
            cands = enumerate_candidates(kps_2d, K, PALLET_DIMS)
            cands += enumerate_candidates(
                kps_2d, K, (PALLET_DIMS[1], PALLET_DIMS[0], PALLET_DIMS[2]))
        except Exception as e:
            print(f"  ERR {p}: {e}")
            continue
        if not cands:
            continue
        nl0 = sum(1 for c in cands if c["lr"] == 0)
        nl1 = sum(1 for c in cands if c["lr"] == 1)
        best4 = sorted(cands, key=lambda c: c["score_v4"])[0]
        best5 = sorted(cands, key=lambda c: c["score_v5"])[0]
        diff = "SAME" if (best4["score_v4"] == best5["score_v4"]
                          and best4["lr"] == best5["lr"]) else "DIFF"
        rel = os.path.relpath(p, root)
        lrc = cands[0]["lr_click_v"]
        print(f"{rel:70s}  {len(cands):4d}  {nl0:4d}  {nl1:4d}  "
              f"{best4['lr']:9d}  {best4['err']:10.2f}  {diff}  lrc={lrc}")
        rows.append((rel, nl0, nl1, best4["lr"], best4["err"], diff, lrc))
    # filter interesting cases
    print()
    print("Frames where fix v4 picks LR-flip (lr=1) but fix v5 disagrees:")
    for r in rows:
        if r[3] == 1 or r[5] == "DIFF":
            print(f"  {r[0]}")

if __name__ == "__main__":
    main()
