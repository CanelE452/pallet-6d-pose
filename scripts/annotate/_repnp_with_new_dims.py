"""DEPRECATED_FOR_PAPER_GT — legacy fixed-dims re-PnP diagnostic.

기본은 dry-run이다. 같은 keypoint index에 새 dimensions를 넣는 이 방식은 signed
canonical axis를 확정하지 못하므로 paper GT에 사용할 수 없다. 새 v2 migration을 쓴다.
"""
import os as _os, sys as _sys

# --- challenge/scripts 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_CS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_CS] + [_os.path.join(_CS, _d) for _d in sorted(_os.listdir(_CS))
                         if _os.path.isdir(_os.path.join(_CS, _d)) and not _d.startswith(".")]

import glob
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# scripts/annotate -> scripts -> repo.
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "challenge", "scripts"))
sys.path[:0] = [os.path.join(REPO, "challenge", "scripts", _s)
                for _s in ("annotate", "infer", "live")]
# Legacy camera-facing diagram convention generator 재사용.
from annotate_pnp import make_pallet_keypoints_3d_diagram as make_pallet_keypoints_3d
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
from challenge.data_paths import get as _dp  # 경로 단일 출처

NEW_DIMS = (1.1, 1.3, 0.11)


def repnp_one(p, allow_legacy_mutation=False):
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    if d.get("schema_version") == "real_pallet_gt_v2":
        return False, "paper_gt_v2_forbidden"
    o = d["objects"][0]
    kps = o.get("manual_kps") or []
    valid = [i for i in range(min(9, len(kps))) if kps[i] is not None]
    if len(valid) < 4:
        return False, "n_kp<4"

    kp3d = make_pallet_keypoints_3d(*NEW_DIMS)
    Ki = d["camera_data"]["intrinsics"]
    K = np.array([[Ki["fx"], 0, Ki["cx"]], [0, Ki["fy"], Ki["cy"]], [0, 0, 1]], dtype=np.float64)
    obj = np.array([kp3d[i] for i in valid], dtype=np.float64)
    img2d = np.array([kps[i] for i in valid], dtype=np.float64)
    flag = cv2.SOLVEPNP_ITERATIVE if len(valid) >= 6 else cv2.SOLVEPNP_EPNP
    ok, rvec, tvec = cv2.solvePnP(obj, img2d, K, None, flags=flag)
    if not ok:
        return False, "pnp_fail"
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.flatten()
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t

    pts_cam = (R @ kp3d.T).T + t
    projected = []
    for pt in pts_cam:
        if pt[2] <= 0:
            projected.append([-1.0, -1.0])
        else:
            u = K[0, 0] * pt[0] / pt[2] + K[0, 2]
            v = K[1, 1] * pt[1] / pt[2] + K[1, 2]
            projected.append([float(u), float(v)])
    errs = []
    for i in valid:
        du = projected[i][0] - kps[i][0]
        dv = projected[i][1] - kps[i][1]
        errs.append(float(np.hypot(du, dv)))
    reproj = float(np.mean(errs))

    if allow_legacy_mutation:
        o["pose_transform"] = T.tolist()
        o["projected_cuboid"] = projected[:8]
        o["projected_cuboid_centroid"] = (projected[8]
                                            if projected[8][0] >= 0 else [-1, -1])
        o["dimensions_m"] = {
            "width": NEW_DIMS[0], "height": NEW_DIMS[2], "depth": NEW_DIMS[1],
        }
        o["reproj_error_px"] = reproj
        if "visibility" not in o:
            o["visibility"] = 1

        with open(p, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
    return True, reproj


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=_dp("manual.pallet07"))
    ap.add_argument("--dry-run", action="store_true",
                    help="호환 옵션; 기본 동작이 이미 dry-run")
    ap.add_argument("--allow-legacy-mutation", action="store_true",
                    help="위험: legacy JSON만 실제 수정. 기본은 dry-run")
    args = ap.parse_args()
    mutate = bool(args.allow_legacy_mutation and not args.dry_run)
    d = args.dir if os.path.isabs(args.dir) else os.path.join(REPO, args.dir)
    paths = sorted(glob.glob(os.path.join(d, "*.json")))
    print("[DEPRECATED_FOR_PAPER_GT] fixed-index re-PnP는 canonical pose를 "
          "정의하지 않습니다. GT v2 migration을 사용하세요.")
    print(f"[Repnp] {len(paths)} JSON files in {d}  "
          f"dry_run={not mutate}")
    print(f"        NEW_DIMS = {NEW_DIMS}")
    ok = 0
    for p in paths:
        ret, msg = repnp_one(p, allow_legacy_mutation=mutate)
        name = os.path.basename(p)
        if ret:
            ok += 1
            print(f"  {name}: reproj {msg:.2f}px")
        else:
            print(f"  {name}: SKIP ({msg})")
    action = "updated" if mutate else "would update"
    print(f"\n[Done] {ok}/{len(paths)} {action}")
