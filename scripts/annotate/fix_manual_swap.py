"""Manual annotation 자동 교정 — swap + dim 조합 brute-force.

원인: 라벨러가 diagram convention 으로 클릭했어도 좌우 / 앞뒤 mapping 이
어긋난 frame 이 많아서 reproj 가 큼 (capturepallet07: 24/25 가 >10px).

이 스크립트는:
  • 4 swap (identity, LR, FB, both) × 2 dim ((1.1,1.3,0.11), (1.3,1.1,0.11)) = 8 조합
  • 각 frame 에서 PnP 풀어 reproj 측정 → 최소 채택
  • JSON 의 manual_kps / pose_transform / projected_cuboid / dimensions_m / reproj_error_px 업데이트
  • 원본은 .bak 백업

사용:
  python challenge/scripts/annotate/fix_manual_swap.py \\
      --dir challenge/data/01_real/manual_gt/capturepallet07_manual_gt
"""
import os as _os, sys as _sys

# --- challenge/scripts 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_CS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_CS] + [_os.path.join(_CS, _d) for _d in sorted(_os.listdir(_CS))
                         if _os.path.isdir(_os.path.join(_CS, _d)) and not _d.startswith(".")]

import argparse
import glob
import json
import os
import shutil
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# 2026-08-15 재편: 이 파일이 challenge/scripts/<계열>/ 로 한 단계 내려갔다.
# dirname 을 하나 더 감싸야 저장소 루트다 (계열 -> scripts -> challenge -> repo).
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, "challenge", "scripts"))

sys.path[:0] = [os.path.join(REPO, "challenge", "scripts", _s)
                for _s in ("annotate", "infer", "live")]
from annotate import make_pallet_keypoints_3d_diagram

# task.yaml 의 camera intrinsics (RealSense D435i)
K_DEFAULT = np.array([[614.18, 0,    329.28],
                      [0,    614.31, 234.53],
                      [0,      0,      1   ]], dtype=np.float64)

# 8 corner index permutation (centroid=8 은 항상 그대로)
SWAPS = {
    "identity": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "LR":       [1, 0, 3, 2, 5, 4, 7, 6, 8],   # 좌우 반전 (front + rear face 각각)
    "FB":       [4, 5, 6, 7, 0, 1, 2, 3, 8],   # 앞뒤 반전
    "LR+FB":    [5, 4, 7, 6, 1, 0, 3, 2, 8],   # 둘 다
}

# (width, depth, height) — 110면 정면 / 130면 정면
DIMS_CANDIDATES = [
    (1.1, 1.3, 0.11),   # 110 정면
    (1.3, 1.1, 0.11),   # 130 정면
]


def pnp_and_reproj(manual_kps_swapped, kp3d, K):
    """주어진 swap+dim 으로 PnP → reproject → 평균 reproj 반환. 실패면 None."""
    valid = [i for i in range(min(9, len(manual_kps_swapped)))
             if manual_kps_swapped[i] is not None
             and manual_kps_swapped[i][0] >= 0]
    if len(valid) < 4:
        return None
    obj = np.array([kp3d[i] for i in valid], dtype=np.float64)
    img2d = np.array([manual_kps_swapped[i] for i in valid], dtype=np.float64)
    flag = cv2.SOLVEPNP_ITERATIVE if len(valid) >= 6 else cv2.SOLVEPNP_EPNP
    ok, rvec, tvec = cv2.solvePnP(obj, img2d, K, None, flags=flag)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.flatten()
    pts_cam = (R @ kp3d.T).T + t
    projected = []
    for p in pts_cam:
        if p[2] <= 0:
            projected.append([-1.0, -1.0])
        else:
            u = K[0, 0] * p[0] / p[2] + K[0, 2]
            v = K[1, 1] * p[1] / p[2] + K[1, 2]
            projected.append([float(u), float(v)])
    errs = [float(np.hypot(projected[i][0] - manual_kps_swapped[i][0],
                           projected[i][1] - manual_kps_swapped[i][1]))
            for i in valid]
    reproj = float(np.mean(errs))
    return {"reproj": reproj, "R": R, "t": t, "projected": projected}


def best_combo(manual_kps, K):
    """Brute-force 모든 swap × dim 조합. 최소 reproj 반환."""
    best = None
    for swap_name, perm in SWAPS.items():
        kps_sw = [manual_kps[perm[i]] if perm[i] < len(manual_kps) else None
                  for i in range(9)]
        for dims in DIMS_CANDIDATES:
            kp3d = make_pallet_keypoints_3d_diagram(*dims)
            res = pnp_and_reproj(kps_sw, kp3d, K)
            if res is None:
                continue
            res.update({"swap": swap_name, "dims": dims, "kps_swapped": kps_sw})
            if best is None or res["reproj"] < best["reproj"]:
                best = res
    return best


def fix_one(json_path, K, dry_run=False):
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    obj = d["objects"][0]
    manual = obj.get("manual_kps") or []
    if not manual:
        return None, "no_manual"

    orig_reproj = obj.get("reproj_error_px")
    res = best_combo(manual, K)
    if res is None:
        return None, "all_failed"

    if not dry_run:
        # backup once
        bak = json_path + ".bak"
        if not os.path.exists(bak):
            shutil.copy2(json_path, bak)
        # update fields
        obj["manual_kps"] = res["kps_swapped"]
        obj["projected_cuboid"] = res["projected"][:8]
        obj["projected_cuboid_centroid"] = (res["projected"][8]
                                             if res["projected"][8][0] >= 0
                                             else [-1, -1])
        T = np.eye(4); T[:3, :3] = res["R"]; T[:3, 3] = res["t"]
        obj["pose_transform"] = T.tolist()
        W, D, H = res["dims"]
        obj["dimensions_m"] = {"width": W, "depth": D, "height": H}
        obj["reproj_error_px"] = res["reproj"]
        obj["fix_swap"] = res["swap"]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)

    return (orig_reproj, res["reproj"], res["swap"], res["dims"]), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--dry_run", action="store_true",
                    help="실제 파일 수정 X, 통계만 출력")
    args = ap.parse_args()

    folder = args.dir if os.path.isabs(args.dir) else os.path.join(REPO, args.dir)
    paths = sorted(glob.glob(os.path.join(folder, "*.json")))
    paths = [p for p in paths if not p.endswith(".bak")]
    print(f"[Fix] {len(paths)} JSON in {folder}  dry_run={args.dry_run}")
    print(f"      Try {len(SWAPS)} swaps x {len(DIMS_CANDIDATES)} dims = {len(SWAPS)*len(DIMS_CANDIDATES)} combos / frame\n")

    rows = []
    swap_count = {k: 0 for k in SWAPS}
    dim_count = {d: 0 for d in DIMS_CANDIDATES}

    for p in paths:
        result, err = fix_one(p, K_DEFAULT, dry_run=args.dry_run)
        name = os.path.basename(p)
        if err:
            print(f"  {name}: SKIP ({err})")
            continue
        orig, new, sw, dims = result
        rows.append((orig, new))
        swap_count[sw] += 1
        dim_count[dims] += 1
        flag = '★' if (orig is not None and new < orig - 2) else ''
        print(f"  {name}: {orig:>6.2f} → {new:>6.2f} px  ({sw}, W={dims[0]})  {flag}"
              if orig is not None else
              f"  {name}: {new:.2f} px  ({sw}, W={dims[0]})")

    if rows:
        import statistics
        old = [r[0] for r in rows if r[0] is not None]
        new = [r[1] for r in rows]
        print(f"\n[Summary]")
        print(f"  before: median={statistics.median(old):.2f}  mean={statistics.mean(old):.2f}  bad(>10)={sum(1 for v in old if v>10)}/{len(old)}")
        print(f"  after:  median={statistics.median(new):.2f}  mean={statistics.mean(new):.2f}  bad(>10)={sum(1 for v in new if v>10)}/{len(new)}")
        print(f"\n  swap chosen:")
        for k, v in swap_count.items():
            print(f"    {k:>10}: {v}")
        print(f"  dims chosen:")
        for k, v in dim_count.items():
            print(f"    {k}: {v}")
    print(f"\n  (백업: 각 JSON 옆에 .bak 으로 저장됨, dry_run={args.dry_run})")


if __name__ == "__main__":
    main()
