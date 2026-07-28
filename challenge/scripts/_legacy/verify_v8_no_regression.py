"""verify_v8_no_regression.py — v8 patch 가 기존 saved frames 의 PnP 결과를 망가뜨리지 않는지 검사.

각 saved JSON 의 manual_kps 를 다시 solve_pose 에 넣고:
  - v6_strict_passed 여부 동일한지
  - reproj_error 차이 < 1.0 px 인지
  - tilt 값 출력 (저장된 frame 들의 tilt 분포 확인 — 정상 frame 의 tilt threshold 검증)
"""
import os, sys, json, glob
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from annotate_pnp import solve_pose, V8_TILT_SOFT_THR, V8_TILT_HARD_THR

REPO = r"C:\Users\minjae\Documents\github\FoundationPose"
GT_DIRS = [
    "challenge/data/capturepallet07_manual_gt",
    "challenge/data/capturepallet08_manual_gt",
    "challenge/data/capturepallet09_manual_gt",
    "challenge/data/capturepalletcad_manual_gt",
]

tilts = []
regressions = []
for d_rel in GT_DIRS:
    d = os.path.join(REPO, d_rel)
    if not os.path.isdir(d): continue
    jsons = sorted(glob.glob(os.path.join(d, "*.json")))
    for j in jsons:
        with open(j) as f:
            data = json.load(f)
        obj0 = data["objects"][0]
        manual = obj0.get("manual_kps")
        if not manual: continue
        clicks = [list(p) if p is not None else None for p in manual]
        while len(clicks) < 9:
            clicks.append(None)
        ci = data["camera_data"]["intrinsics"]
        K = np.array([[ci["fx"], 0, ci["cx"]],
                       [0, ci["fy"], ci["cy"]],
                       [0, 0, 1]], dtype=np.float64)
        pose = solve_pose(clicks, K, img_shape=(480, 640, 3))
        if pose is None:
            print(f"[NONE] {os.path.basename(j)}")
            regressions.append((j, "solve_pose returned None"))
            continue
        new_err = pose["reproj_error_px"]
        old_err = obj0.get("reproj_error_px", -1)
        tilt = pose.get("_v8_tilt", -1)
        strict = pose.get("_v6_strict_passed", False)
        flag = ""
        if tilt < V8_TILT_HARD_THR: flag = " [HARD]"
        elif tilt < V8_TILT_SOFT_THR: flag = " [SOFT]"
        diff = abs(new_err - old_err)
        regression = diff > 1.0
        marker = " *REGRESSION*" if regression else ""
        print(f"{os.path.basename(d_rel)}/{os.path.basename(j)[:25]:>25}  "
              f"err: old={old_err:6.2f} new={new_err:6.2f} diff={diff:5.2f}  "
              f"tilt={tilt:.3f} strict={strict}{flag}{marker}")
        tilts.append(tilt)
        if regression:
            regressions.append((j, f"err diff {diff:.2f}"))

print(f"\n=== Summary ===")
print(f"  N frames processed: {len(tilts)}")
print(f"  Tilt stats: min={min(tilts):.3f} max={max(tilts):.3f} "
      f"median={sorted(tilts)[len(tilts)//2]:.3f}")
n_below_soft = sum(1 for t in tilts if t < V8_TILT_SOFT_THR)
n_below_hard = sum(1 for t in tilts if t < V8_TILT_HARD_THR)
print(f"  Below SOFT thr ({V8_TILT_SOFT_THR}): {n_below_soft} / {len(tilts)} "
      f"({100*n_below_soft/max(1,len(tilts)):.0f}%)")
print(f"  Below HARD thr ({V8_TILT_HARD_THR}): {n_below_hard} / {len(tilts)}")
print(f"  Regressions: {len(regressions)}")
for j, msg in regressions:
    print(f"    {j}: {msg}")
