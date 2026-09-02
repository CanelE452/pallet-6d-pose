#!/usr/bin/env python3
"""4방향 진입 팔레트 GT 의 "어느 면이 앞면(0~3)인가" 를 하나의 규칙으로 통일한다.

45° 근처에서는 어느 면을 앞으로 볼지가 사람마다 갈린다.  실측하면 한 사람은 왼쪽을
90%, 다른 사람은 오른쪽을 64% 골랐다.  같은 형상에 서로 다른 keypoint 정답이 생기면

* keypoint 학습이 두 정답 사이에서 흐려지고,
* PnP 의 원점(전면 중심)이 인접 면으로 옮겨가 x, z 가 어긋난다
  (정사각 1.10 m 기준 두 면 중심 사이 0.55·√2 ≈ 0.78 m).

정사각 팔레트는 90° 회전이 등가라 **정보를 하나도 잃지 않고** 규칙을 통일할 수 있다.
yaw 가 ``[0, 90)`` 에 오도록 90° 씩 되돌리고, keypoint 를 그만큼 재배열한다.

기하는 이렇게 유도했다(``make_pallet_keypoints_3d_diagram`` 의 local frame
X=right / Y=down / Z=near→far 기준).  Y축 둘레 90° 회전을 가하면 코너가

    new[j] = old[[1, 5, 6, 2, 0, 4, 7, 3][j]]

로 재배열된다.  이 값은 기존 synthetic 라벨의 ``perm_v4`` 와 일치한다 — 독립 유도가
기존 데이터와 맞는지 교차검증한 것이다.

카메라 좌표의 물리 점은 그대로여야 하므로 회전은 ``R_new = R_old @ Ry(-90°)`` 로 따라간다.
원점이 cuboid 중심이라 translation 은 불변이다.

기본은 dry-run 이다.  ``--apply`` 를 줘야 실제로 쓴다.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# Y축(아래) 둘레 90° 회전이 만드는 코너 재배열.  index 8(centroid)은 불변.
ROT90_PERMUTATION = (1, 5, 6, 2, 0, 4, 7, 3, 8)
QUARTER = math.pi / 2.0


def _rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def yaw_of(rotation: np.ndarray) -> float:
    """배포 규약과 같은 식: atan2(R[0,2], R[2,2])."""
    return float(math.atan2(rotation[0, 2], rotation[2, 2]))


def _permute(sequence, times: int):
    """리스트를 90° 회전 ``times`` 번만큼 재배열한다."""
    out = list(sequence)
    for _ in range(times % 4):
        out = [out[i] for i in ROT90_PERMUTATION[:len(out)]]
    return out


def canonicalise(document: dict) -> tuple[dict, int, float, float]:
    """문서 하나를 정규화하고 ``(문서, 회전횟수, 이전 yaw, 이후 yaw)`` 를 돌려준다."""
    obj = document["objects"][0]
    transform = np.asarray(obj["pose_transform"], dtype=np.float64)
    rotation = transform[:3, :3]
    before = yaw_of(rotation)

    # yaw 를 [0, 90) 로 보내는 데 필요한 90° 되돌림 횟수
    turns = int(math.floor(before / QUARTER)) % 4
    if turns == 0:
        return document, 0, before, before

    new_rotation = rotation @ _rotation_y(-turns * QUARTER)
    new_transform = transform.copy()
    new_transform[:3, :3] = new_rotation
    obj["pose_transform"] = new_transform.tolist()

    # 코너 순서를 따라가야 하는 필드들.  9개(코너 8 + centroid) 또는 8개 길이만 다룬다.
    for field in ("projected_cuboid", "cuboid", "manual_kps", "extrapolated_mask",
                  "visibility", "keypoints_3d_world"):
        value = obj.get(field)
        if isinstance(value, list) and len(value) in (8, 9):
            obj[field] = _permute(value, turns)

    return document, turns, before, yaw_of(new_rotation)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", help="GT 폴더 또는 JSON 파일")
    ap.add_argument("--apply", action="store_true", help="실제로 파일을 쓴다 (기본은 dry-run)")
    args = ap.parse_args(argv)

    files: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        files.extend(sorted(path.glob("*.json")) if path.is_dir() else [path])
    if not files:
        ap.error("JSON 을 찾지 못했다")

    turn_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    changed = 0
    failures: list[str] = []
    for path in files:
        try:
            with open(path, encoding="utf-8") as handle:
                document = json.load(handle)
            before_points = None
            obj = document["objects"][0]
            if isinstance(obj.get("projected_cuboid"), list):
                before_points = sorted(map(tuple, obj["projected_cuboid"]))

            document, turns, before, after = canonicalise(document)
            turn_counts[turns] += 1
            if turns == 0:
                continue

            # 등가 변환이므로 2D 점의 '집합' 은 변하지 않아야 한다.  순서만 바뀐다.
            if before_points is not None:
                after_points = sorted(map(tuple, document["objects"][0]["projected_cuboid"]))
                if before_points != after_points:
                    failures.append(f"{path.name}: 2D 점 집합이 바뀌었다")
                    continue
            if not (0.0 <= after < QUARTER + 1e-9):
                failures.append(f"{path.name}: yaw 가 [0,90) 밖 ({math.degrees(after):.2f}°)")
                continue

            changed += 1
            if args.apply:
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(document, handle, indent=2)
        except (OSError, ValueError, KeyError, IndexError) as exc:
            failures.append(f"{path.name}: {exc}")

    mode = "적용" if args.apply else "DRY-RUN (쓰지 않음)"
    print(f"[{mode}] 대상 {len(files)}개")
    print(f"  회전 없음 {turn_counts[0]}  90° {turn_counts[1]}  "
          f"180° {turn_counts[2]}  270° {turn_counts[3]}")
    print(f"  정규화 대상 {changed}개")
    if failures:
        print(f"  ⚠️ 실패 {len(failures)}건")
        for line in failures[:8]:
            print(f"     {line}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
