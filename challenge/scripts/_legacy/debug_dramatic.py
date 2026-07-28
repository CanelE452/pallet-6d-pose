import os, sys, json
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from diagnose_lr_v4 import enumerate_candidates
from annotate_pnp import PALLET_DIMS

p = r"C:\Users\minjae\Documents\github\FoundationPose\challenge\data\capturepallet07_manual_gt\1778652174598774528.json"
with open(p) as f: d = json.load(f)
intr = d["camera_data"]["intrinsics"]
K = np.array([[intr["fx"],0,intr["cx"]],[0,intr["fy"],intr["cy"]],[0,0,1]])
mkp = d["objects"][0]["manual_kps"]
kps_2d = [tuple(x) if x else None for x in mkp]
cands = enumerate_candidates(kps_2d, K, PALLET_DIMS)
cands += enumerate_candidates(
    kps_2d, K, (PALLET_DIMS[1], PALLET_DIMS[0], PALLET_DIMS[2]))
print(f"#cands: {len(cands)}")
print(f"#lr=0: {sum(1 for c in cands if c['lr']==0)}")
print(f"#lr=1: {sum(1 for c in cands if c['lr']==1)}")
print()
print("All LR-correct candidates:")
for c in sorted(cands, key=lambda c: c["err"]):
    if c["lr"] == 0:
        print(f"  err={c['err']:.2f}  nf={c['nf']} av={c['av']} gv={c['gv']} "
              f"lrc={c['lr_click_v']} n_v={c['n_v']:2d}  "
              f"L-x={c['left_x']:+12.3f} R-x={c['right_x']:+12.3f}  "
              f"t={c['t']}")
print()
print("Top 5 LR-flip:")
for c in sorted([c for c in cands if c["lr"]==1], key=lambda c: c["err"])[:5]:
    print(f"  err={c['err']:.2f}  nf={c['nf']} av={c['av']} gv={c['gv']} "
          f"lrc={c['lr_click_v']} n_v={c['n_v']:2d}  "
          f"L-x={c['left_x']:+8.3f} R-x={c['right_x']:+8.3f}  "
          f"t={c['t']}")
