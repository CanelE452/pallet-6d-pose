# 3DoF CONTRACT — direct pallet pose track

이 파일이 `x / z / yaw` 규약의 **유일한 출처**다. 다른 문서는 여기를 가리킨다.
확인하지 못한 항목은 `[추정]` 으로 채우지 않고 **미확인** 으로 남긴다.

---

## 1. 좌표계 — deployment 가 실제로 쓰는 것

배포 소비자는 lifter FSM 하나뿐이고, 그 값은
`25y_automatic_lifter-master/depth_cam/calib/geometry.py` 에서 나온다 `[확인]`.

```
frame   camera_optical (OpenCV):  +X right, +Y down, +Z forward
origin  RealSense color optical center
unit    metre
```
근거: rec 녹화 `*_meta.json` 의 `"frame"` / `"origin"` / `"units"` 필드,
그리고 `geometry.py:_box_object_points()` docstring
("x 오른쪽, y 아래, z 카메라에서 멀어지는 방향").

### x, z
```python
# geometry.py  solve_pose_from_keypoints() 끝
center = tvec.reshape(3)      # (x, y, z)  전면 중심의 카메라 좌표
x = center[0]                 # lateral offset  [m]  (오른쪽 +)
z = center[2]                 # forward distance [m]  (멀어질수록 +)
```
`tvec` 이 곧 소비되는 값이라 별도 변환이 없다 `[확인]`.

### yaw
```python
# geometry.py  _angles_from_R()
n   = R @ [0, 0, 1]                    # 모델 +z (전면 법선)
yaw = degrees(atan2(n[0], n[2]))
```
`n = R[:, 2] = (R[0,2], R[1,2], R[2,2])` 이므로 이는 곧
```
yaw = atan2(R[0,2], R[2,2])
```
와 같다 `[확인]`. **zero 정의**: 팔레트 전면이 카메라를 정면으로 향할 때 0.
**positive 방향**: 법선이 화면 오른쪽(+x)으로 기울 때 +.

> 이 식을 새로 만들지 않았다. 배포 코드에서 읽어 그대로 채택했다.

### 물체 원점
```
전면(포크 진입면)의 중심
```
근거: `_box_object_points()` docstring — "원점은 전면(포크 진입면) 중심 —
tvec 이 그대로 전면 중심 3D 좌표가 되어 기존 offset/distance 해석이 바뀌지 않는다".
**팔레트 중심이 아니다.** 3D 키포인트를 만들 때 이 원점을 반드시 맞춘다.

---

## 2. 물체 기하 — 현장 팔레트

```
object_type   plastic_standard_110x110x15
x (전면 가로)  1.10 m
y (높이)       0.15 m
z (깊이)       1.10 m
```
출처: `challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json` (사용자 실측) `[확인]`.
논문 정본 `challenge/real_gt_v2/OBJECT_GEOMETRY_REGISTRY.json` 의
`plastic_standard_110x130x11` 은 **다른 물체**다. 섞지 않는다.

> ⚠️ 배포 코드 `depth_cam/calib/config.py` 는 아직
> `PALLET_FACE_W=1.000 / PALLET_FACE_H=0.150 / PALLET_DEPTH=1.200` 을 쓴다.
> 실측 대비 x −9.1% / z +9.1% 다. baseline 을 이 트랙과 나란히 평가할 때
> 이 불일치를 반드시 명시한다.

---

## 3. 대칭 — 이 트랙에서 가장 중요한 계약

현장 팔레트는 **4방향 포크 진입**이라 90° 회전이 등가다 (사용자 확인) `[확인]`.

```
equivalent yaw rotations : 0°, 90°, 180°, 270°
yaw period               : 90°  (π/2)
canonical yaw range      : [0, π/2)
```

기존 `challenge/real_gt_v2/SYMMETRY_CONTRACT.json` 은 `[0, 180]` 이고
"a 90-degree yaw is not equivalent **because the canonical X and Z extents differ**"
라고 적혀 있다. 그 근거는 X≠Z 인 110×130 에만 성립한다.
정사각 팔레트는 X==Z 라 근거가 무효다. **기존 계약 파일은 건드리지 않는다** —
그건 논문 평가용이고 SHA 가 산출물 ~50곳에 박혀 있다.

### yaw 인코딩 — scalar 도 (sinψ,cosψ) 도 쓰지 않는다
```
encode :  (sin 4ψ, cos 4ψ)
decode :  ψ = atan2(s, c) / 4        → [0, π/2) 로 wrap
```
**이유**: 등가 회전이 있는데 `(sinψ, cosψ)` 를 쓰면 같은 이미지에 대해
ψ 와 ψ+90° 가 서로 다른(90° 등가면 서로 직교하는) 타깃을 준다.
학습 신호가 상쇄되어 yaw head 가 수렴하지 않는다.
4배각을 쓰면 네 등가 회전이 **정확히 같은 타깃**으로 접힌다.

부수 효과: 어느 면을 앞면으로 라벨했든 타깃이 같아지므로,
사람 어노테이션의 앞면 선택 모호성이 자동 해소된다.

### yaw 오차
등가를 반영한 circular 거리로만 계산한다.
```python
d = (a - b) % (pi/2)
err = min(d, pi/2 - d)          # 0 ≤ err ≤ π/4
```
`179° vs -179°` 를 358° 로 세는 실수도, 4-fold 등가를 무시하는 실수도 막는다.

---

## 4. x / z 정규화
raw metre 를 같은 스케일 loss 에 그대로 넣지 않는다.
정규화 통계는 **config / checkpoint / export metadata 세 곳에 함께** 저장한다.
후보와 선택은 synthetic 분포를 실제로 계산한 뒤 확정한다 — **미확인 (분포 미산출)**.

---

## 5. GT 등급 — 절대 섞지 않는다

| 등급 | 정의 | 용도 | 현재 상태 |
|---|---|---|---|
| `physical_independent` | 촬영 시 자·줄자 등으로 독립 측정한 x,z,yaw | **절대 정확도 평가** | 아직 데이터 없음 |
| `manual_kp_geometry_derived` | 사람이 찍은 9 kp + intrinsics + 기하로 계산 | **학습 타깃** | 생성 가능 |
| `model_derived` | 현재 모델 예측 → PnP | **금지** | — |

`manual_kp_geometry_derived` 만 있는 동안 다음 표현을 쓰지 않는다:
`absolute 3DoF accuracy`, `true metric accuracy`, `real-world z error`.
보고 가능한 것은 target agreement / temporal stability / baseline consistency /
held-out session generalization 뿐이다.

같은 solver 로 만든 타깃으로 그 solver 와 direct model 의 절대 정확도를
비교하지 않는다 — circular evaluation 이다.

---

## 6. 입력
```
RGB only.  depth / aligned depth / depth 파생 x,z 사용 금지.
RealSense 는 RGB 취득 장치로만 취급한다.
```

---

## 7. synthetic exact GT — 있다

`objects[0]` 에 카메라 좌표계 6DoF 가 그대로 들어 있다 `[확인]`.

```
location         [x, y, z]   카메라 좌표계의 object 원점 (m)
pose_transform   4x4         [:3,:3]=R_obj→cam, [:3,3]==location
quaternion_xyzw  4           같은 회전
euler_angles     pitch/yaw/roll
```
검증: `u = fx·X/Z + cx` 로 `location` 을 투영하면 JSON 의
`projected_cuboid_centroid` 와 소수점까지 일치했다. **PnP 없이 x, z 를 바로 읽는다.**

- 단위 **m** (`dimensions_m`, `camera_distance_actual_m` 과 교차검증)
- 축 **OpenCV camera (+X right, +Y down, +Z forward)** — deployment 와 동일 `[확인]`
  (`scripts/data_prep/blender/blender_math.py:build_view_matrix` = `[right, -up, forward]`)

> ⚠️ `euler_angles.yaw` 를 쓰지 말 것. 그건 카메라 Y축 둘레 회전이고 지면 yaw 가 아니다.
> 이 트랙은 `pose_transform` 의 R 에서 `atan2(R[0,2], R[2,2])` 로만 yaw 를 뽑는다.

### ⚠️ 7.1 object origin 이 deployment 와 다르다 — 반드시 보정
```
synthetic / annotate  :  cuboid 8코너의 기하 중심 (mid-height)
deployment            :  전면(포크 진입면) 중심
```
검증: local 로 되돌린 8코너가 `±depth/2, ±height/2, ±width/2` 로 중심대칭이고,
`_box_object_points()` 는 face 를 z=0, back 을 z=depth 에 둔다.

보정 (중심 → 전면):
```
t_front = t_center + R @ [0, 0, -depth/2]
```
이 보정을 빠뜨리면 z 가 팔레트 깊이의 절반(약 0.55 m)만큼 계통적으로 어긋난다.

### ⚠️ 7.2 현재 R0 synthetic 은 치수가 프레임마다 랜덤 — direct z 학습 불가
`g38_legacy_v1v2_p0_tex20k` 의 G38 소스 600 프레임 실측:
```
width   0.675 ~ 1.660  (2.46x)  std/mean 16.4%
depth   0.631 ~ 1.597  (2.53x)
height  0.107 ~ 0.223  (2.09x)
z       1.457 ~ 9.843  (6.76x)
```
이미지가 결정하는 것은 `실제크기 / z` 비율 하나뿐이다. 치수가 16.4% 로 흔들리면
같은 픽셀 크기가 여러 z 에 대응해 **z 를 원리적으로 결정할 수 없다**.
z 오차 하한 ≈ 16.4% (평균 z 4.43 m 기준 약 0.73 m).

x 와 yaw 는 영향이 작다 (x 는 z 에 비례하는 만큼만, yaw 는 형상비에서 나옴).
**z 학습용 데이터는 별도로 정해야 한다** — §7.3.

### 7.3 고정 치수 대안
`challenge/data/02_synthetic/training/{v1,v3}` 는 `cuboid_dimensions_m = [1.1, 1.3, 0.12]`
전 프레임 고정이고 `challenge/data/INDEX.md` 가 **challenge 전용**이라고 명시한다.
현장 팔레트(1.10×1.10×0.15)와 depth·height 가 다르므로 z 스케일 보정이 필요하다.
어느 데이터를 z 학습에 쓸지는 **사용자 결정 사항** — 확정 전 loader 에 박지 않는다.

## 8. 그룹핑 키 (session split 용)
`scene_id` / `scenario_id` 라는 키는 generic/legacy 에 **없다**.
- generic(G38): `records.jsonl` 의 `seed`(프레임 고유), `_src_shard`, `pallet_type`,
  `scene_preset`, `background_asset`
- legacy(P0/TEX): `manifest.jsonl` 의 `shard`, `pallet_type`, `material_variant`
- 02_synthetic v1/v3: `frame_meta.scenario` (진짜 scenario ID)
real 은 `challenge/yolo_pose_one_model/manifests/all_samples.csv` 의 `session_id`.

## 9. 아직 미확인
- x/z 정규화 통계 (어느 데이터를 쓸지 정해진 뒤 산출)
- Jetson 실측 latency (보드에서 측정한 적 없음 — 기존 문서 수치도 전부 추정치)
