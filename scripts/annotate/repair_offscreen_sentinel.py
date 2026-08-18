"""화면 밖(u<0)이라 [-1,-1] 로 잘못 저장된 코너를 재투영으로 복구한다.

annotate_io.make_annotation 이 `proj[i][0] >= 0` 으로 sentinel 을 판정하는 바람에,
화면 왼쪽으로 잘린 정상 코너(z>0)가 "안 보임"([-1,-1])으로 기록됐다. 코드는
2026-08-15 에 고쳤지만 이미 저장된 GT 는 그대로다. 100px reflect padding 계약은
"화면 밖 코너도 유효하다" 는 전제이므로, 이 점들은 v=0 이 아니라 실좌표여야 한다.

복구값은 JSON 안의 pose_transform + dimensions_m + intrinsics 로 재투영해서 얻는다
(추정이 아니라 그 프레임이 스스로 들고 있는 값이다). 카메라 뒤(z<=0)인 점은
진짜 sentinel 이므로 그대로 둔다.

원본은 .presentinel 로 남긴다 — 되돌릴 수 있게.

사용:
  python scripts/annotate/repair_offscreen_sentinel.py --dirs <폴더...> [--apply]
  (--apply 없으면 무엇이 바뀌는지만 보고한다)
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from annotate_pnp import make_pallet_keypoints_3d   # noqa: E402


def reproject(obj, cam):
    """이 프레임의 pose/dims/K 로 9 keypoint 를 다시 투영. (proj(9,2), z(9,)) 또는 None."""
    try:
        T = np.asarray(obj["pose_transform"], dtype=np.float64)
        d = obj["dimensions_m"]
        it = cam["intrinsics"]
    except (KeyError, TypeError):
        return None
    if T.shape != (4, 4):
        return None
    R, t = T[:3, :3], T[:3, 3]
    # make_pallet_keypoints_3d(width, depth, height); 저장 키는 width/height/depth
    kp3 = make_pallet_keypoints_3d(d["width"], d["depth"], d["height"])
    cam_pts = (R @ kp3.T).T + t
    z = cam_pts[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = it["fx"] * cam_pts[:, 0] / z + it["cx"]
        v = it["fy"] * cam_pts[:, 1] / z + it["cy"]
    return np.stack([u, v], axis=1), z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    tot_f = tot_c = tot_skip = 0
    for d in args.dirs:
        n_f = n_c = n_skip = 0
        for j in sorted(glob.glob(os.path.join(d, "*.json"))):
            try:
                data = json.load(open(j, encoding="utf-8"))
                obj = data["objects"][0]
            except Exception:
                continue
            cub = obj.get("projected_cuboid") or []
            bad = [i for i, c in enumerate(cub[:8])
                   if list(map(float, c)) == [-1.0, -1.0]]
            if not bad:
                continue
            got = reproject(obj, data.get("camera_data", {}))
            if got is None:
                n_skip += len(bad)
                continue
            proj, z = got
            fixed = 0
            for i in bad:
                if z[i] > 1e-6 and np.isfinite(proj[i]).all():
                    cub[i] = [float(proj[i][0]), float(proj[i][1])]
                    fixed += 1
            if not fixed:
                n_skip += len(bad)
                continue
            n_f += 1
            n_c += fixed
            if args.apply:
                bak = j + ".presentinel"
                if not os.path.exists(bak):
                    shutil.copy2(j, bak)
                obj["projected_cuboid"] = cub
                obj["sentinel_repaired"] = fixed
                tmp = j + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, j)
        print(f"{os.path.basename(d):<40} 프레임 {n_f:>4}  복구코너 {n_c:>4}"
              f"{'  (카메라뒤/정보부족 ' + str(n_skip) + ')' if n_skip else ''}")
        tot_f += n_f
        tot_c += n_c
        tot_skip += n_skip
    print("-" * 74)
    print(f"{'합계':<40} 프레임 {tot_f:>4}  복구코너 {tot_c:>4}  건너뜀 {tot_skip}")
    print("APPLIED (원본은 *.presentinel)" if args.apply
          else "DRY-RUN — --apply 로 실제 적용")


if __name__ == "__main__":
    main()
