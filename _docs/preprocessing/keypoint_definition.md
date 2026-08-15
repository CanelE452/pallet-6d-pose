# Keypoint Definition — camera-facing 0123

> 현행 convention (2026-05-22 사용자 결정, v4). 폐기된 object-frame Y=UP 정의는
> `archive/keypoint_definition.md` 참조. memory `camera-facing-0123-convention`.

## 9 keypoint

팔레트 cuboid 8 코너 + centroid = 9 keypoint. DOPE belief map 9채널.

```
인덱스   위치
──────────────────────────────────────────────
0~3     카메라에 가까운 큰 앞면 (FRONT, image polygon area 최대)
          0 = 좌상, 1 = 우상, 2 = 우하, 3 = 좌하
4~7     뒷면 (REAR), 앞면과 대응: 0-4, 1-5, 2-6, 3-7
8       centroid (3D 중심의 투영)
```

- **TOP (위) = {0, 1, 4, 5}**, **BOTTOM (아래) = {2, 3, 6, 7}**
- 앞↔뒤 대응 edge (depth): 0-4, 1-5, 2-6, 3-7
- 앞면 위 edge: 0-1, 뒷면 위 edge: 4-5
- 좌측 depth edge: 0-4, 우측: 1-5

## camera-facing 의 의미

매 frame 카메라에서 본 앞면(가장 큰 면)이 항상 0-1-2-3 으로 라벨된다. 물체
고정(object-frame)이 아니라 **시점 기준**. 따라서 직사각형의 2D 기하 관계(앞면
4점이 한 사각형, 좌우 대칭 등)가 image 상에서 일관되게 성립 → 2D 기하 필터 가능.

## 변환

- 학습 데이터: `challenge/scripts/annotate/convert_to_camera_facing_v4.py` (`compute_perm_v4`).
  origin frame 3D 좌표 기준 top/bot split → vertical pairing → image polygon area
  최대 face = FRONT → image x 로 LR. (cam-frame 부호 의존 X)
- 적용 확인: `data/pallet/training_data/mixed_v8_train` 의 `.json`(변환) ≠ `.json.orig`(원본).

## 2D 기하 필터에 쓰는 관계 (정확한 인덱스/불변량은 3d-expert 확정 예정)

- 위/아래 순서: {0,1,4,5} 가 {2,3,6,7} 보다 image y 위쪽
- 변 비율: 앞면 위변(0-1) ≈ 뒷면 위변(4-5), 좌 depth(0-4) ≈ 우 depth(1-5) — perspective 보정 필요
- 공간 대각선(0-6, 2-4 등) 교점 ≈ centroid(8) — projective invariant
