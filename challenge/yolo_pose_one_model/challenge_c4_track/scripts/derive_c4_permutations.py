#!/usr/bin/env python3
"""정사각 팔레트의 C4 (yaw 90도) 대칭 permutation 을 3D 기하에서 유도한다.

손으로 적지 않는다.  `plastic_standard_110x110x15` 의 실제 cuboid vertex 를 만들고,
yaw 회전을 적용한 뒤 회전된 vertex 가 원래 어느 index 와 같아지는지 **좌표 매칭으로**
찾는다.  그 다음 요구된 성질을 전수 검증한다.

좌표 규약은 downstream(`newauto` `depth_cam/calib/geometry.py`)과 같다::

    x 오른쪽 · y 아래 · z 카메라에서 멀어지는 방향
    0~3 전면(좌상, 우상, 우하, 좌하) · 4~7 후면(동일 순서) · 8 centroid

학습 라벨 규약(camera-facing 0123: 0~3 앞면, {0,1,4,5} 위 / {2,3,6,7} 아래)과
일치하는지도 여기서 확인한다 — 위는 y<0, 아래는 y>0 이다.

yaw 회전축은 **높이 축 = y** 다.  정사각(x==z)일 때만 90도 회전이 같은 물체를 준다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1]

# plastic_standard_110x110x15 — 실측 1.10(x) x 0.15(y) x 1.10(z)
DIMS = {"x_m": 1.10, "y_m": 0.15, "z_m": 1.10}
TOL = 1e-9

# cuboid 12 edge (index 쌍).  앞면 4 + 뒷면 4 + 연결 4.
EDGES = frozenset(map(frozenset, [
    (0, 1), (1, 2), (2, 3), (3, 0),          # 전면
    (4, 5), (5, 6), (6, 7), (7, 4),          # 후면
    (0, 4), (1, 5), (2, 6), (3, 7),          # 전-후 연결
]))
TOP, BOTTOM = frozenset({0, 1, 4, 5}), frozenset({2, 3, 6, 7})


def vertices() -> np.ndarray:
    """9 x 3.  index 8 은 centroid(원점)."""
    hx, hy, hz = DIMS["x_m"] / 2, DIMS["y_m"] / 2, DIMS["z_m"] / 2
    # y 는 아래가 + 이므로 '위' 는 -hy 다.
    front = [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz)]
    back = [(x, y, +hz) for x, y, _ in front]
    return np.array(front + back + [(0.0, 0.0, 0.0)], dtype=np.float64)


def rot_y(deg: float) -> np.ndarray:
    """높이축(y) 둘레 회전.  x 오른쪽 / y 아래 / z 전방 규약의 표준 행렬."""
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def derive(deg: float, V: np.ndarray) -> tuple[int, ...]:
    """``new[j] = old[perm[j]]`` 형태의 permutation 을 좌표 매칭으로 찾는다.

    회전 후 j 번 자리에 오는 물리 점이 원래 몇 번이었는지를 구한다.  즉 회전된
    vertex 집합에서 원래 vertex j 와 같은 좌표를 갖는 index 를 찾는다.
    """
    Vr = V @ rot_y(deg).T
    perm = []
    for j in range(len(V)):
        d = np.linalg.norm(Vr - V[j], axis=1)
        hit = np.flatnonzero(d < 1e-9)
        if len(hit) != 1:
            raise ValueError(f"{deg}도: index {j} 의 대응이 유일하지 않다 (후보 {hit})")
        perm.append(int(hit[0]))
    return tuple(perm)


def compose(a: tuple, b: tuple) -> tuple:
    """new[j] = old[p[j]] 규약에서 a 를 적용한 뒤 b 를 적용한 것."""
    return tuple(a[b[j]] for j in range(len(b)))


def check(perms: dict[int, tuple]) -> list[str]:
    """요구된 성질을 전수 검증한다.  실패 목록을 돌려준다."""
    fails = []
    ident = tuple(range(9))
    for deg, p in perms.items():
        tag = f"{deg}도"
        if sorted(p[:8]) != list(range(8)):
            fails.append(f"{tag}: 0~7 bijection 아님 {p[:8]}")
        if p[8] != 8:
            fails.append(f"{tag}: centroid 가 움직임 (perm[8]={p[8]})")
        # 위/아래 보존 — y 좌표 부호가 회전으로 바뀌면 안 된다
        if {p[i] for i in TOP} != TOP or {p[i] for i in BOTTOM} != BOTTOM:
            fails.append(f"{tag}: top/bottom 관계 깨짐")
        # 12 edge adjacency 보존
        moved = {frozenset({p[a], p[b]}) for a, b in map(tuple, EDGES)}
        if moved != EDGES:
            fails.append(f"{tag}: cuboid edge 보존 안 됨")
    # 군 성질
    p90 = perms[90]
    if compose(p90, p90) != perms[180]:
        fails.append("90도 두 번 != 180도")
    if compose(perms[180], p90) != perms[270]:
        fails.append("180+90 != 270도")
    if compose(compose(p90, p90), compose(p90, p90)) != ident:
        fails.append("90도 네 번 != identity")
    for deg, p in perms.items():
        inv = tuple(int(np.argsort(p)[j]) for j in range(9))
        if inv != perms[(360 - deg) % 360]:
            fails.append(f"{deg}도의 역이 {(360-deg)%360}도와 다르다")
    return fails


def main() -> int:
    V = vertices()
    print("cuboid vertex (x 오른쪽 / y 아래 / z 전방, 원점 = 중심)")
    for i, v in enumerate(V):
        where = "centroid" if i == 8 else (
            ("전면" if v[2] < 0 else "후면") + ("·위" if v[1] < 0 else "·아래"))
        print(f"  {i}  ({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f})  {where}")

    # 학습 라벨 규약과 맞는지 — 위 = {0,1,4,5}
    top_from_geom = {i for i in range(8) if V[i][1] < 0}
    assert top_from_geom == set(TOP), f"위 집합 불일치: {top_from_geom}"
    print(f"\n규약 확인: 위 {sorted(top_from_geom)} · 아래 {sorted(set(range(8))-top_from_geom)}"
          "  → camera-facing 0123 과 일치")

    if abs(DIMS["x_m"] - DIMS["z_m"]) > TOL:
        print("\n[FAIL] x != z — 90도는 이 물체의 대칭이 아니다")
        return 1
    print(f"정사각 확인: x={DIMS['x_m']} == z={DIMS['z_m']}  → C4 적용 가능")

    perms = {0: tuple(range(9))}
    for deg in (90, 180, 270):
        perms[deg] = derive(deg, V)

    print("\n유도된 permutation  (new[j] = old[perm[j]])")
    for deg in (0, 90, 180, 270):
        print(f"  {deg:3d}도  {list(perms[deg])}")

    fails = check(perms)
    print("\n검증")
    for name in ("bijection", "centroid 고정", "top/bottom 보존", "12-edge 보존",
                 "90x2=180", "180+90=270", "90x4=identity", "inverse 일치"):
        bad = [f for f in fails if name.split()[0] in f or name[:4] in f]
        print(f"  {name:16} {'FAIL' if bad else 'PASS'}")
    if fails:
        print("\n실패 상세:")
        for f in fails:
            print("  -", f)
        return 1

    # 기존 파일과 대조 — 참고만 하고, 우리 유도가 정답 근거다.
    legacy = OUT.parents[0] / "pallet_yolo_loss/PERMUTATION_GROUP.json"
    legacy_note = "파일 없음"
    if legacy.is_file():
        legacy_note = f"존재: {legacy}"
    canon = (1, 5, 6, 2, 0, 4, 7, 3, 8)   # canonicalize_fourfold_yaw.py 의 값
    match = {deg: (tuple(p) == canon) for deg, p in perms.items()}
    print(f"\n기존 canonicalize_fourfold_yaw.ROT90_PERMUTATION {list(canon)}")
    print(f"  일치하는 회전각: {[d for d, m in match.items() if m] or '없음'}")

    (OUT / "C4_PERMUTATIONS.json").write_text(json.dumps({
        "object_type": "plastic_standard_110x110x15",
        "dims_m": DIMS,
        "coordinate_convention": "x right, y down, z away from camera; origin at cuboid center",
        "index_semantics": {
            "0-3": "front face (top-left, top-right, bottom-right, bottom-left)",
            "4-7": "back face, same order",
            "8": "centroid",
            "top": sorted(TOP), "bottom": sorted(BOTTOM),
        },
        "rotation_axis": "y (height)",
        "permutation_semantics": "new[j] = old[perm[j]]",
        "permutations": {str(d): list(perms[d]) for d in (0, 90, 180, 270)},
        "verified": {
            "bijection_0_7": True, "centroid_fixed": True,
            "top_bottom_preserved": True, "cuboid_12_edges_preserved": True,
            "closure_and_inverse": True, "four_rotations_identity": True,
        },
        "derivation": "3D vertex coordinates rotated by Ry(deg), matched back to "
                      "original indices by exact coordinate distance (<1e-9)",
        "legacy_reference": legacy_note,
        "matches_canonicalize_fourfold_yaw_rot90": [d for d, m in match.items() if m],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {OUT / 'C4_PERMUTATIONS.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
