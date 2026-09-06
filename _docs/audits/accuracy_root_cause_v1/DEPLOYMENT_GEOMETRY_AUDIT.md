# 배포 기하 감사 — 평가가 허용하는 대칭 vs 지게차가 요구하는 방향 구분

작성 2026-09-06 · HEAD `2e5ec0e` · 읽기 전용 감사(학습·추론 미실행, registry 무수정)

---

## 한 화면 요약

```
CANONICAL_DIMENSIONS_MM
  물체가 하나가 아니다. 세 개의 서로 다른 물체 + 여러 개의 어긋난 사본이 병존한다.

  (A) 논문 정본 plastic_standard_110x130x11   x1100 / y110  / z1300
      challenge/real_gt_v2/OBJECT_GEOMETRY_REGISTRY.json
      sha256 0c7a10729b6db18cbe47fa4adb158e2f26ec7a7c9458f59ee60d023c282f0627
  (B) 논문 정본 wood_small_80x59x14           x800  / y140  / z590      (같은 파일·SHA)
  (C) 과제/현장 plastic_standard_110x110x15   x1100 / y150  / z1100
      challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json
      sha256 e923b44880b031a7d3a9e2fffb5a6bd287cfa0de758133eeb5a73770137eba86
  ★ 지게차가 실제로 집는 물체는 (C) 다. 논문 평가가 도는 물체는 (A)/(B) 다.

OBJECT_FRAME_ORIGIN
  두 가지가 공존한다.
  - 학습·어노테이션·합성·논문평가 : 8코너의 기하 중심 (mid-height centroid)
  - 배포(lifter FSM) solvePnP     : 전면(포크 진입면) 중심, z=0 평면
  둘의 차이는 R @ (0,0,-depth/2) = 약 0.55 m.

AXIS_ORDER
  X = width (화면 가로), Y = height (아래가 +, OpenCV), Z = depth (0~3 면 → 4~7 면).
  즉 width 가 x 다. 단 함수 시그니처는 (width, depth, height) 순서고
  registry JSON 은 x/y/z 이름 순서다 — 두 관례가 서로 다른 파일에 섞여 있다.

FRONT_FACE_DEFINITION
  camera-facing. 물체 고유 특징이 아니다.
  0~3 = "이 프레임에서 카메라에 가까운 면"(z = -d/2). 물리적 앞면이라는 개념은
  이 규약 안에 존재하지 않는다.

EVAL_ALLOWED_PERMUTATIONS
  함수마다 다르다. 통일된 집합이 없다.
  - 논문 평가(ADD-S/rot/yaw)      : {identity, yaw180}  — 2개
  - 논문 평가(ADD, translation)   : {identity}          — 1개
  - C4 track "fixed_index" 열     : {identity}          — 1개
  - C4 track "c4_equivalent" 열   : {0,90,180,270}      — 4개
  - direct_3dof (sin4psi,cos4psi) : {0,90,180,270}      — 4개(인코딩에 흡수)
  - model_compare kp_err          : 8! 전수 Hungarian   — 대칭 아님, 진단용

DEPLOYMENT_REQUIRED_ORIENTATION_DISTINCTION
  물리적으로는 "포크가 들어가는 면을 향해 수직 정렬" 뿐이다.
  코드적으로는 keypoint 0~3 이 어느 면인지를 배포가 그대로 믿는다
  (POSE_FACE_KPTS=(0,1,2,3), 원점=그 면 중심). 90° phase 가 바뀌면
  tvec 이 0.55*sqrt(2) = 0.778 m 점프하고 yaw 가 90° 점프한다.

FORK_POCKET_DIRECTIONALITY = 4-way
  (C) 현장 정사각 팔레트: 4-way. 근거 = 3DOF_CONTRACT.md §3 "사용자 확인" [확인].
  (A) 1.10x1.30 팔레트  : 4-way. 근거 = 2026-05-19 photogrammetry mesh 감사 [확인].
  (B) wood 80x59        : UNKNOWN (기록 없음).
  ※ (C) 의 근거는 사람의 진술이지 도면·실측 사진이 아니다. 저장소에 mesh/도면 없음.

TASK_SYMMETRY_CONTRACT_MISMATCH = YES
  단, 흔히 상상하는 방향의 반대다.
  평가가 90°를 오답으로 세는 자리에서 배포는 그 90°를 신경 쓰지 않고(물리 4-way),
  반대로 배포가 실제로 요구하는 index phase 안정성은 어느 평가 지표도 재지 않는다.
  게다가 지게차가 집는 물체 (C) 에는 **동결된 symmetry contract 자체가 없다**
  (`symmetry_status: "UNREVIEWED"`, `symmetry_contract: null`).
```

---

## 1. 정본 치수와 그 사본들 — 전수

### 1.1 정본 두 개

`challenge/real_gt_v2/OBJECT_GEOMETRY_REGISTRY.json`
sha256 `0c7a10729b6db18cbe47fa4adb158e2f26ec7a7c9458f59ee60d023c282f0627` [확인]

| object_type | x (m) | y (m) | z (m) | geometry_status | symmetry_status | symmetry_contract |
|---|---|---|---|---|---|---|
| `plastic_standard_110x130x11` | 1.1 | 0.11 | 1.3 | FROZEN | FROZEN | `challenge/real_gt_v2/SYMMETRY_CONTRACT.json` |
| `wood_small_80x59x14` | 0.8 | 0.14 | 0.59 | FROZEN | UNREVIEWED | null |

`challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json`
sha256 `e923b44880b031a7d3a9e2fffb5a6bd287cfa0de758133eeb5a73770137eba86` [확인]
— 위 둘을 축자 복사하고 아래 하나를 추가한 과제 전용 파일.

| object_type | x (m) | y (m) | z (m) | geometry_status | symmetry_status | symmetry_contract |
|---|---|---|---|---|---|---|
| `plastic_standard_110x110x15` | 1.1 | 0.15 | 1.1 | FROZEN | **UNREVIEWED** | **null** |

`source_measurement_order` 는 세 항목 모두 `width_depth_height` 이고,
`canonical_axis_semantics` 는 y 를 "positive down" 높이축으로 명시한다 [확인].

### 1.2 같은 물체를 다르게 적은 곳 — 전수

| 값 (m) | 위치 | 어느 물체를 가리키나 | 상태 |
|---|---|---|---|
| 1.1 / 0.11 / 1.3 | `challenge/real_gt_v2/OBJECT_GEOMETRY_REGISTRY.json` | (A) | 정본 |
| 1.1 / 0.15 / 1.1 | `challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json` | (C) | 정본(과제) |
| `CANONICAL_X_M=1.10 / Y=0.11 / Z=1.30` | `scripts/annotate/pallet_geometry.py:44` | (A) | 하드코딩 기본값 |
| `PALLET_DIMS = (1.1, 1.3, 0.11)` (W,D,H) | `scripts/annotate/annotate_pnp.py:86` | (A) | 호환 shim |
| `width 1.1 / depth 1.3 / height 0.11` | `challenge/config/task.yaml` (pallet 절) | (A) | `keypoint_convention: "y_up"` 로 표기 — 폐기된 규약 이름 |
| `edge_sets(width=1.1, depth=1.3, height=0.12)` | `Deep_Object_Pose/common/pallet_graph_geometry.py:70` | (A) 인데 높이 0.12 | 정본과 1 cm 불일치 |
| `make_pallet_keypoints_3d(width=1.1, depth=1.1, height=0.15)` | `scripts/self_training/pnp_solver.py:22` | (C) | docstring 은 "KS T 1002 1100x1100x150" |
| `make_pallet_keypoints_3d_isaac(1.1, 1.3, 0.11)` | 같은 파일 `:63` | (A) | 코너 순서가 또 다름(§4.5) |
| `width 1.1 / depth 1.1 / height 0.12` | `config/default.yaml:66-68` | 미정 | 주석이 "SLOT — 확정 전이라 부정확(정사각 잔재)" 이라고 스스로 밝힘 |
| `1.10 x 1.30 x 0.12` | `data/pallet/scan_cleanup/pallet_full.obj` (2026-05-19 기록) | (A) | 실측 mesh — 높이가 정본 0.11 과 1 cm 다름 |
| `PALLET_WIDTH_M 1.10 / DEPTH 1.30 / HEIGHT 0.12` | 배포 `depth_cam/calib/pose6d_adapter.py:35-37` | (A) | 지게차는 (C) 를 집는데 (A) 치수를 씀 |
| `PALLET_FACE_W 1.000 / FACE_H 0.150 / DEPTH 1.200` | 배포 `depth_cam/calib/config.py:22-24` | (C) 를 의도 | 주석은 "실물 100 x 120 x 15 cm" — **네 번째 값** |

배포 쪽 결론 [확인]: 지게차 스택은 현장 물체 (C) `1.10 / 0.15 / 1.10` 을
`solvePnP` 에 넣지 않는다. `geometry.py` 경로는 `1.000 / 0.150 / 1.200`,
`pose6d_adapter.py` 경로는 `1.10 / 1.30 / 0.12` 를 쓴다. 두 경로가 서로도 다르다.
크기 오차 자체는 x −9.1% / z +9.1% 수준이고 높이는 정확하다 — jitter 의 주원인으로
지목하면 과장이다(이전에 한 번 정정된 이력).

---

## 2. object frame — origin, 축 순서, width/depth

### 2.1 학습·평가 쪽 (하나의 계보)

`scripts/annotate/annotate_pnp.py:169-206` `make_pallet_keypoints_3d_diagram(width, depth, height)` [확인]

```
w, h, d = width/2, height/2, depth/2
0 (-w,-h,-d)  1 (+w,-h,-d)  2 (+w,+h,-d)  3 (-w,+h,-d)      near face  (z = -d/2)
4 (-w,-h,+d)  5 (+w,-h,+d)  6 (+w,+h,+d)  7 (-w,+h,+d)      far  face  (z = +d/2)
8 = 8코너 평균 = (0,0,0)
```

- ORIGIN = **cuboid 기하 중심**(centroid). 바닥 중앙이 아니다. `corners.mean(axis=0)` 이
  정확히 원점이고, index 8 이 그 점이다 [확인].
- X = width, Y = height(아래가 +, OpenCV), Z = depth [확인].
- **width 가 x 다** [확인]. 다만 함수 인자 순서는 `(width, depth, height)` 라
  좌표 순서(x,y,z)와 어긋난다 — `CameraFacingDimensionsWHD.as_legacy_wdh_tuple()`
  (`scripts/annotate/pallet_geometry.py`)이 이 어긋남을 흡수하는 전용 어댑터로 존재한다.

`Deep_Object_Pose/common/pallet_graph_geometry.py` 는 이 함수를 그대로 호출해 축 의미를
좌표에서 재유도한다(`_AXIS_TO_CLASS = {0: width, 1: vertical, 2: depth}`) [확인].

`challenge/evaluation_v2` 계열은 `pallet_geometry._diagram_points(width, height, depth)`
로 같은 배치를 다시 만든다. 여기서 canonical frame 은 x=1.10축 / y=높이 / z=1.30축이고,
camera-facing frame 으로 가는 변환이 `AxisAssignment` (YAW_0/90/180/270) 다 [확인].
문서화된 관계식:

```
P_cf[i] = A @ P_canonical[p[i]]
R_canonical = R_cf @ A
t_canonical = t_cf              (두 frame 이 keypoint 8 을 공유하기 때문)
```

### 2.2 배포 쪽 (다른 계보)

`25y_automatic_lifter-master/depth_cam/calib/geometry.py:43-58` `_box_object_points()` [확인]

```
face = [(-hw,-hh,0), (hw,-hh,0), (hw,hh,0), (-hw,hh,0)]    z = 0
back = 같은 순서, z = depth
"원점은 전면(포크 진입면) 중심 — tvec 이 그대로 전면 중심 3D 좌표"
```

ORIGIN = **전면 중심**. 학습 쪽 centroid 와 `R @ (0,0,-depth/2)` (약 0.55 m) 만큼 다르다.
`3DOF_CONTRACT.md §7.1` 이 이 보정을 명시적으로 요구한다 [확인]. 보정을 빠뜨리면
z 가 팔레트 깊이의 절반만큼 계통적으로 어긋난다.

배포 쪽 각도 정의 (`_angles_from_R`) [확인]:
```
n   = R @ (0,0,1)                yaw   = atan2(n[0], n[2]) = atan2(R[0,2], R[2,2])
u   = R @ (1,0,0)                roll  = atan2(-u[1], u[0])
```

---

## 3. front face 는 무엇으로 정해지는가

**카메라 기준이다. 물체 고유 특징이 아니다.** [확인]

- `scripts/annotate/annotate_draw.py:45-53` 주석: "Camera-facing convention (2026-05-22):
  0~3 = 카메라에 가까운 near face (운용 시 = fork pocket 면)". `KP_NAMES` 도
  `NearTopLeft / NearTopRight / NearBottomRight / NearBottomLeft / FarTopLeft ...` 다.
- `make_pallet_keypoints_3d_diagram` docstring: "★ near = Z_local 작은 쪽 = cam.z 작은쪽".

그래서 **"물체의 앞면" 이라는 개념은 이 규약 안에 존재하지 않는다.** 0~3 은 물체에
붙어 있는 면이 아니라 카메라가 지금 보고 있는 면이다.

이것이 관측 가능한 결과를 이미 냈다 [확인] — `audit_20260821T1716` 감사에서
라벨의 `dimensions_m` 이 (1.1,1.3) 89장 / (1.3,1.1) 72장으로 갈리고, 7 세션 중 6개에서
두 변종이 **한 세션 안에서 섞이며**, 0.37초 간격에 뒤집히는 사례가 27회 있었다.
즉 그 필드는 물리 속성이 아니라 "이번 프레임에서 어느 물리축이 화면 가로로 보이는가"다.

정사각 물체 (C) 에서는 이 모호성이 극단화된다. 네 면이 전부 같아서 45° 부근에서
사람마다 앞면 선택이 갈렸다 — 실측으로 한 사람 왼쪽 90%(n=30), 다른 사람 오른쪽
63.8%(n=116) [확인] (`fourfold-normalisation-must-match-deployed-convention` 메모리).
그래서 `scripts/annotate/canonicalize_fourfold_yaw.py` 가 사후에 규칙을 통일한다
(기본 dry-run, `--offset-deg -90` 이 배포 규약).

부수 효과 [확인]: PnP 가설 생성기(`_physical_wd_hypotheses`)는 정사각에서 가설을
**1개**(YAW_0)만 만든다. 이름이 `"square-face-front"` 다. 직사각에서 두 개를 만들던
short/long face 구분이 정사각에서는 정의 자체가 성립하지 않기 때문이다.

---

## 4. 평가가 허용하는 permutation — 함수 단위

### 4.1 동결 계약 (직사각 plastic 전용)

`challenge/real_gt_v2/SYMMETRY_CONTRACT.json` [확인]
```
metric_variant           "ADD-S"
canonical_axis           "+Y"
equivalent_yaw_degrees   [0, 180]
accepted_proper_rotations  I, diag(-1, 1, -1)
equivalence_basis.kind   "DECLARED_BENCHMARK_ASSUMPTION"
physical_inspection_claimed  false
inclusion_exclusion_rules[1]:
  "Only yaw angles separated by 180 degrees are equivalent;
   a 90-degree yaw is not equivalent because the canonical X and Z extents differ."
```

`scripts/annotate/pallet_symmetry.py` 가 이를 스키마로 강제한다 —
`_validate_yaws()` 는 `(0, 180)` 이외의 어떤 집합도 `SymmetryContractError` 로 거부한다 [확인].
즉 **이 코드 경로에서는 4-fold 대칭을 선언하는 것이 구조적으로 불가능하다.**

계약이 스스로 밝히듯 이것은 물리 검증이 아니라 벤치마크 선언이다
(`physical_inspection_claimed: false`, "not evidence that every physical pallet instance
was independently inspected"). 별도 감사에서도
`NO_VERIFIED_NONIDENTITY_SYMMETRY` 로 남아 있다 — mesh/도면이 저장소에 없어 확인 불가 [확인].

### 4.2 함수별 허용 집합

| 함수 / 열 | 파일 | 허용 permutation | 근거 |
|---|---|---|---|
| `add_error_m` | `challenge/evaluation_v2/pose_metrics.py:251` | **{identity}** | 대칭 인자 없음 [확인] |
| `translation_error_m` | 같은 파일 `:223` | **{identity}** | 순수 거리 [확인] |
| `rotation_error_degrees` | 같은 파일 `:201` | **{identity}** | 호출자가 min 을 취함 |
| `yaw_error_degrees` | 같은 파일 `:211` | **{identity}**, 0~180 범위 | 접기 없음 [확인] |
| `adds_error_m` | 같은 파일 `:263` | 무제한 최근접점 | docstring 이 "paper evaluator does not call this helper" 라고 명시 — cuboid 코너에서 pitch/roll 대칭까지 몰래 허용하기 때문 [확인] |
| `_pose_records` (실제 보고 경로) | `challenge/evaluation_v2/paper_real_eval.py:2239-2258` | **{identity, yaw180}** | `min(... for symmetry in object_context.equivalent_rotations)` 를 add/rotation/yaw 세 개에 각각 적용 [확인] |
| ↳ 같은 레코드의 `add_error_m` | 같은 곳 `:2259` | **{identity}** | `direct_add` 를 그대로 저장 [확인] |
| `symmetry_aware_pose_metrics.SYMMETRY_GROUP` | `scripts/paper/pose_metric_closure_v1/` | **{I, Ry(180)}** | 모듈 상수 [확인] |
| `PG.symmetry_permutation` | `Deep_Object_Pose/common/pallet_graph_geometry.py:88` | **{identity, (5,4,7,6,1,0,3,2,8)}** | 3D 좌표 매칭으로 유도 [확인] |
| `allowed_symmetries` | `Deep_Object_Pose/common/pallet_polarity_disambiguation.py:73` | **{I, Ry(180)}** | `[np.eye(3), PG.symmetry_rotation()]` [확인] |
| `YAW180_PERMUTATION` | `Deep_Object_Pose/common/corner_role_adapter.py:27` | 프레임 단위 전체 적용, per-corner 금지 | [확인] |
| G0 yaw 전역 탐색 | `_docs/audits/GLOBAL_YAW_IDENTIFIABILITY.md` | 탐색 구간 **[0°,180°)** = 180° 대칭 적용 | [확인] |
| C4 track `fixed_index` 열 | `challenge_c4_track/scripts/evaluate_c4_arms.py` | **{identity}** | `errs[0]` [확인] |
| C4 track `c4_equivalent` 열 | 같은 파일 | **{0,90,180,270}** | `errs[best]`, `C4_PERMUTATIONS.json` [확인] |
| `direct_3dof` yaw | `challenge/direct_3dof/pose3dof.py` | **{0,90,180,270}** | `(sin4psi, cos4psi)` 인코딩이 네 회전을 같은 타깃으로 접음 [확인] |
| `hungarian_median` | `scripts/stage0/model_compare/mc_score.py:36` 외 | **8! 전수** | 대칭 계약이 아니라 order-free 코너 오차 진단 [확인] |

### 4.3 C4 permutation 정본 (정사각 전용)

`challenge/yolo_pose_one_model/challenge_c4_track/C4_PERMUTATIONS.json` [확인]
`permutation_semantics: new[j] = old[perm[j]]`, 원점 = cuboid 중심, 회전축 = y

```
  0도  [0, 1, 2, 3, 4, 5, 6, 7, 8]
 90도  [1, 5, 6, 2, 0, 4, 7, 3, 8]
180도  [5, 4, 7, 6, 1, 0, 3, 2, 8]
270도  [4, 0, 3, 7, 5, 1, 2, 6, 8]
```
bijection / centroid 고정 / top-bottom 보존 / 12-edge 보존 / 군 폐포·역원 / 4회 항등 —
8항목 전수 검증 PASS 로 기록돼 있다. 90도 순열은
`canonicalize_fourfold_yaw.ROT90_PERMUTATION` 및 기존 라벨의 `perm_v4` 와 독립 유도로 일치.

`C4_SYMMETRY_CONTRACT.json` 은 적용 범위를 `"square real FT only"` 로 못 박고,
broad40k/generic synthetic 에 적용 금지라고 쓴다 — 그 풀에는 직사각 팔레트가 있어
90° 등가를 가르치면 거짓 등가를 학습시키기 때문이다 [확인].

### 4.4 정사각 물체에는 평가 계약이 아예 없다 ★

`plastic_standard_110x110x15` 는 `symmetry_status: "UNREVIEWED"`, `symmetry_contract: null` [확인].
논문 평가 게이트는 `symmetry_status != "FROZEN"` 이면 pose 지표를 내지 않고
(`build_pose_metric_gate` → `blocked_pose_metrics`), 사전등록 selector 모집단도
plastic(110x130) 과 wood 둘뿐이다 (`SELECTOR_DIAGNOSTIC_POPULATION_BY_OBJECT`) [확인].

→ **지게차가 실제로 집는 물체는 논문 평가 배터리에 존재하지 않는다.**
그 물체의 pose 를 재는 곳은 C4 track 의 2D keypoint 픽셀 오차(fixed / C4 두 열)와
direct_3dof 트랙뿐이고, 둘의 허용 집합이 서로 다르다.

### 4.5 부수 발견 — 코너 순서 사본이 셋

| 순서 | 정의 위치 | 0번이 무엇인가 |
|---|---|---|
| camera-facing v4 (정본) | `annotate_pnp.make_pallet_keypoints_3d_diagram` | near-top-**LEFT** |
| NDDS/DOPE legacy | `scripts/self_training/pnp_solver.make_pallet_keypoints_3d` | Front-top-**RIGHT** |
| Isaac canonical | 같은 파일 `make_pallet_keypoints_3d_isaac` | left-top-front, 단 front = **z 최대** |

`challenge/_docs/annotate_guide.md` §2 클릭 순서 표는 아직 NDDS 순서
(`0 FrontTopRight / 1 FrontTopLeft ...`)를 적고 있어 도구의 실제 `KP_NAMES`
(`NearTopLeft / NearTopRight / NearBottomRight / NearBottomLeft`)와 **좌우가 뒤집혀 있다** [확인].
같은 문서 §1 의 "110 면 라벨 / 130 면 skip" 규칙도 정사각 물체에는 적용되지 않는다.
코드가 정본이고 가이드가 뒤처진 것으로 보이나, 어느 라벨이 어느 시점 가이드로
찍혔는지는 이 감사 범위 밖이다 — **[추정]** 으로 남긴다.

또한 `pallet_graph_geometry.edge_sets()` 는 W/D/H 가 서로 달라야 한다고 요구하며
(`len({w,d,h}) != 3` → `ValueError`), 정사각 (1.1, 1.1, 0.15) 에서는 예외를 던진다 [확인].
PalletGraph 계열 모듈은 구조적으로 현장 물체에 쓸 수 없다.

---

## 5. fork pocket 방향성

### 5.1 확인된 것

| 물체 | 판정 | 근거 | 강도 |
|---|---|---|---|
| (C) `plastic_standard_110x110x15` (현장) | **4-way** | `challenge/direct_3dof/3DOF_CONTRACT.md §3` — "현장 팔레트는 4방향 포크 진입이라 90° 회전이 등가다 (사용자 확인)" | 사람의 진술. 도면·사진 근거는 저장소에 없음 |
| (A) `plastic_standard_110x130x11` | **4-way** | `_docs/history/2026-05-19.md` photogrammetry mesh 감사 — "포크 슬롯 : ✓ (4-way entry, 큰 슬롯 보존)", 좌우·앞뒤 mirror 대칭 | 실물 스캔 mesh 실측 [확인] |
| (B) `wood_small_80x59x14` | **UNKNOWN** | 기록 없음 | — |

`challenge/config/task.yaml` 의 `robot.fork` 는 "KS T-11 양면형: 포크 구멍 중심 간격
~0.4m, 바닥에서 ~0.05m" 로 **한 면분의 진입점만** 정의한다
(`left_entry [-0.20, 0.05, 0.55]`, `right_entry [+0.20, 0.05, 0.55]`, 둘 다 +Z 면) [확인].
즉 config 는 4-way 를 표현하지 않는다 — 진입면이 하나라고 가정한다.

### 5.2 미확정이 대칭 계약에 주는 영향

(A) 가 4-way 라는 것은 **90°가 그 물체의 대칭이라는 뜻이 아니다.** 1.10 ≠ 1.30 이므로
90° 회전은 형상을 자기 자신으로 보내지 않고, 90° yaw 오차는 진짜 pose 오차다.
`SYMMETRY_CONTRACT.json` 의 `[0,180]` 은 (A) 에 대해 여전히 옳다.

문제는 그 계약을 정당화하는 **문장**이다.
`scripts/paper/pose_metric_closure_v1/symmetry_aware_pose_metrics.py` docstring [확인]:

> "90 degrees is never absorbed by the symmetry group. For a forklift the difference
> between entering the pockets and hitting the deck is exactly that rotation."

이 배포 근거는 저장소의 자체 기록과 어긋난다. (A) 도 (C) 도 4-way 진입이므로
90° 회전한 팔레트에서 지게차가 만나는 것은 "deck" 이 아니라 **다른 쌍의 포켓**이다.
지표 값은 옳지만 그 지표를 배포 의미로 번역한 문장은 근거가 없다.
→ 논문에서 이 문장을 그대로 쓰면 검증되지 않은 배포 주장이 된다.

(B) wood 의 UNKNOWN 은 지금은 무해하다 — wood 는 `symmetry_status: UNREVIEWED` 라
어차피 identity 만 쓰고, 평가 게이트가 대칭 계약을 요구하는 자리에서 막힌다.
다만 wood 를 대칭 클래스로 승격하려면 4-way/2-way 확인이 선행돼야 한다.

---

## 6. 충돌 판정 — TASK_SYMMETRY_CONTRACT_MISMATCH = YES

세 갈래로 갈라서 본다.

### 6.1 평가는 90°를 오답으로 세는데 배포는 무관한가 → **그렇다 (현장 물체에서)**

현장 물체 (C) 는 정사각 + 4-way 이므로 90° 회전은 물리적으로 같은 배치다.
그런데 이 물체를 재는 자리에서 identity-only 열이 그대로 보고되고 있다 —
C4 track `fixed_index` 열에서 F1 모델의 62/155 프레임이 20px 초과로 잡혔고,
그 62장은 **전부 270° 순열**이며 같은 프레임의 C4 오차 중앙값은 1.97px 다 [확인].
즉 위치는 맞고 번호만 돌았는데 지표는 위치 실패로 셌다.

`SYMMETRY_CONTRACT.json` 이 90°를 배제한 이유("canonical X and Z extents differ")는
X = Z 인 (C) 에서 전제가 무너진다 — 3DOF_CONTRACT.md §3 이 이 점을 명시한다 [확인].

### 6.2 평가가 90°를 정답으로 인정하는데 배포는 구분해야 하는가 → **부분적으로 그렇다**

배포의 요구는 물리가 아니라 **코드 계약**에서 온다 [확인]:
```
depth_cam/calib/config.py     POSE_FACE_KPTS = (0, 1, 2, 3),  POSE_KPT_VIS_THR = 0.5
depth_cam/calib/perception.py sel = k[list(POSE_FACE_KPTS)]   네 점 다 visible 해야 통과
depth_cam/calib/geometry.py   _box_object_points 원점 = 전면 중심,  z=0 평면
                              1단계 SOLVEPNP_IPPE(전면 4점) → 2단계 8점 ITERATIVE
                              yaw = atan2(R[0,2], R[2,2])
```
0~3 의 phase 가 90° 바뀌면 tvec 이 인접 면 중심으로 이동한다 —
정사각 1.10 m 기준 0.55·√2 ≈ **0.778 m**, yaw 는 90° 점프 [확인].
FSM 은 `YAW_TOL_DEG = 2.00` 으로 절대 yaw 를 쓰므로 [확인] 이 점프를 흡수하지 않는다.

즉 4-fold 등가를 그대로 배포에 흘리면 물리적으로는 옳은 예측이 제어 루프에서는 사고가 된다.
C4 track 이 이 위험을 GT 없이 실측했고(연속 프레임 예측끼리 비교) 세션 **내부** 안정률은
F0 99.06% / F1 99.68% 였다. 판정은 `DEPLOYABLE_WITH_CAVEAT` 이고 release 를 덮지 않았다 —
세션 **사이**에는 갈린다(F1 이 val 155장에서 93장 0°, 62장 270°) [확인].

### 6.3 어느 지표도 배포가 실제로 요구하는 것을 재지 않는다 ★

배포가 요구하는 양은 "index phase 의 시간적 안정성" 이다. 위 §4.2 표의 어떤 지표도
이것을 재지 않는다. identity-only 지표는 phase 를 절대값으로 벌하고(과다 처벌),
C4 지표는 phase 를 완전히 무시한다(과소 처벌). 유일하게 이 양을 잰 것은
`challenge_c4_track/scripts/diagnose_downstream.py` 이며, 정식 지표 배터리에 없다.

### 6.4 종합

```
TASK_SYMMETRY_CONTRACT_MISMATCH = YES
```

세 축 모두에서 어긋난다.

1. **물체 축** — 동결 대칭 계약이 있는 물체(110x130, 직사각)와 지게차가 집는
   물체(110x110, 정사각)가 다르다. 후자는 `symmetry_status: UNREVIEWED` 라
   허용 permutation 이 정의된 적이 없다.
2. **원점 축** — 학습·평가는 centroid, 배포는 전면 중심. 0.55 m 차이이며
   보정은 `3DOF_CONTRACT.md §7.1` 에만 문서화돼 있고 계약으로 강제되지 않는다.
3. **의미 축** — 평가의 "90°는 오답" 근거로 적힌 배포 서술(포켓 vs deck)이
   저장소 자체 기록(양 물체 모두 4-way)과 모순된다.

따라서 **모델 학습보다 계약 수정이 먼저다** 라는 전제는 이 감사가 지지한다.
다만 수정 대상은 "평가를 4-fold 로 완화" 가 아니다 — 그러면 §6.2 의 배포 사고를
지표가 못 잡게 된다. 필요한 것은 물체별로 분리된 계약이다.

---

## 7. 조치 후보 (실행하지 않음, 사용자 결정 대상)

우선순위 순.

1. **현장 정사각 물체에 대칭 계약을 신설한다.** 파일은
   `challenge/yolo_pose_one_model/challenge_c4_track/C4_SYMMETRY_CONTRACT.json` 이
   이미 내용을 갖고 있으나 registry 의 `symmetry_contract` 필드가 `null` 이라
   평가 경로와 연결돼 있지 않다. 연결하려면
   `challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json` 만 고치면 되고
   **논문 정본 SHA `0c7a1072…` 는 건드리지 않는다.**
   단 `scripts/annotate/pallet_symmetry.py._validate_yaws()` 가 `(0,180)` 만 통과시키므로
   4-fold 를 스키마로 받으려면 v3 스키마가 필요하다 — 이건 논문 v2 검증기와
   분리해서 만들어야 한다.
2. **지표를 두 열로 고정 보고한다.** 정사각 물체에서는 identity-only 단일 수치를
   보고하지 않는다. `C4-equivalent`(기하 성능) + `phase stability`(배포 가능성) 두 축.
   후자는 `diagnose_downstream.py` 를 정식 지표로 승격하면 된다.
3. **논문의 배포 정당화 문장을 고친다.** `symmetry_aware_pose_metrics.py` docstring 의
   "entering the pockets vs hitting the deck" 은 4-way 팔레트에서 성립하지 않는다.
   지표 정의는 유지하되 근거를 "형상 extent 가 다르므로 90°는 형상 대칭이 아니다" 로
   바꾸면 검증 가능한 진술이 된다.
4. **배포 config 치수를 실측에 맞춘다** — `PALLET_FACE_W 1.000 → 1.100`,
   `PALLET_DEPTH 1.200 → 1.100`, `pose6d_adapter.PALLET_DEPTH_M 1.30 → 1.10`.
   두 줄짜리 수정이고 재학습보다 싸다. 다만 jitter 해결책으로 제시하지 말 것 —
   오차는 9% 수준이고 이 과장은 한 번 정정된 이력이 있다.
5. **`annotate_guide.md` §2 클릭 순서 표를 도구의 `KP_NAMES` 와 맞춘다.**
   현재 좌우가 뒤집혀 적혀 있다.

---

## 8. 미확정으로 남기는 것

- (C) 현장 팔레트의 4-way 진입은 **사람의 진술**이 근거다. mesh·도면·측면 사진이
  저장소에 없어 코드에서 확인할 수 없다. `NO_VERIFIED_NONIDENTITY_SYMMETRY` 감사 판정과
  같은 성격이다.
- (B) wood 팔레트의 포켓 방향성 — 기록 없음.
- `annotate_guide.md` 의 옛 NDDS 순서 표로 찍힌 라벨이 실제로 존재하는지, 존재한다면
  어느 세션인지 — **[추정]** 단계에서 멈췄다. 라벨 파일을 코너 순서 기준으로
  전수 검사하지 않았다.
- 로컬 `~/Documents/github/25y_automatic_lifter-master` 와
  `~/Documents/github/Korea-Railroad-project/25y_automatic_lifter-master` 두 사본이
  내용이 다르다(후자에만 `pose6d_adapter.py` / `yolo_inference.py` 존재).
  어느 쪽이 실제 배포본인지 이 감사에서 확정하지 않았다 — 위 인용은 두 사본 모두에
  존재하는 `config.py` / `geometry.py` 와, 후자에만 있는 `pose6d_adapter.py` 를 구분해 적었다.

---

## 부록 — 메인 세션 재검증 (2026-09-06)

위 감사는 위임 결과다. 판정을 바꿀 무게의 주장만 골라 메인 세션에서 다시 확인했다.
확증 4건, 정정 1건, 저장소 밖이라 검증 불가 1건, 새로 찾은 것 2건.

### 확증

1. [확인] registry 두 개의 SHA·치수가 보고와 정확히 일치한다.
   `challenge/real_gt_v2/OBJECT_GEOMETRY_REGISTRY.json` = `0c7a10729b6d…`,
   `challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json` = `e923b44880b0…`.
   후자의 `default_object_type` 은 `plastic_standard_110x110x15` 다.

2. [확인] `symmetry_status` 를 물체별로 직접 읽었다.

   ```
   plastic_standard_110x130x11   FROZEN      → real_gt_v2/SYMMETRY_CONTRACT.json
   wood_small_80x59x14           UNREVIEWED  → null
   plastic_standard_110x110x15   UNREVIEWED  → null      ← 지게차가 집는 물체
   ```

   즉 대칭 계약이 존재하는 물체는 셋 중 하나뿐이고, 그 하나는 배포 물체가 아니다.

3. [확인] C4 track 수치는 `challenge_c4_track/COMPARISON.md` §C·§D·§E 에서 그대로 확인된다.
   F1 의 fixed-index 실패 62 프레임은 `best_deg` 분포가 `{270: 62}` 로 전부 270도이고,
   같은 프레임의 C4 오차는 median 1.97px / max 5.92px. symmetry-rescued 는 F0 4 / F1 61,
   true collapse(C4>20px)는 F0 2 / F1 0.

4. [확인] 배포 원점이 전면 중심이라는 계약 진술은
   `challenge_c4_track/scripts/diagnose_downstream.py` 헤더에 실재한다.

### 정정

5. [확인] "어느 정식 지표도 phase 의 시간적 안정성을 재지 않는다" 는 과한 진술이다.
   `COMPARISON.md` §F 가 GT 없이 이웃 프레임의 **예측끼리만** 비교해 이미 측정했다 —
   5 세션에서 F0 638 pairs 99.06%, F1 627 pairs 99.68%.
   따라서 phase 오류는 프레임 단위 jitter 가 아니라 **세션 단위 상수 오프셋**이다.
   정확한 진술은 "측정된 적이 없다" 가 아니라 "정식 metric battery 에 포함돼 있지 않다" 다.
   이 구분은 중요하다 — 세션 상수 오프셋이면 배포에서 한 번의 phase 결정으로 해결되지만,
   프레임 jitter 면 시간 필터가 필요하다.

### 저장소 밖이라 검증 불가

6. [미확인] `geometry.py` (1.000/0.150/1.200) 와 `pose6d_adapter.py` (1.10/1.30/0.12) 의
   치수 불일치 주장은 이 저장소에서 확인할 수 없다.
   `git grep -l "POSE_FACE_KPTS|_box_object_points"` 는 파이썬 정의를 하나도 찾지 못하고,
   `diagnose_downstream.py` 헤더가 그것을 **`newauto`** 의 계약이라고 부른다 —
   즉 별도 배포 저장소다. 이 저장소만으로는 그 두 파일의 상수를 확인할 수 없으므로
   판정 근거로 쓰지 않는다.

### 새로 찾은 것

7. [확인] 대칭 계약이 스스로 밝힌 근거가 배포 물체에는 성립하지 않는다.
   `real_gt_v2/SYMMETRY_CONTRACT.json` 의 `inclusion_exclusion_rules[1]`:

   > "Only yaw angles separated by 180 degrees are equivalent; a 90-degree yaw is
   > not equivalent **because the canonical X and Z extents differ**."

   이 근거는 X=1.1, Z=1.3 인 (A) 에서만 성립한다. 배포 물체 (C) 는 X=Z=1.1 이라
   extent 가 같으므로 90도를 배제할 이 근거가 없고, 그런데도 (C) 에는 계약이 없다.
   같은 파일이 `physical_inspection_claimed: false`, `kind:
   "DECLARED_BENCHMARK_ASSUMPTION"` 이라고 스스로 적어 두었다 —
   물리 검사 결과가 아니라 평가 편의상의 선언이다.

8. ~~이 저장소 안의 온디바이스 추론 스크립트도 배포 물체와 다른 치수를 쓴다.~~
   ★**2026-09-06 정정 — 이 주장은 과했다.**
   `challenge/pallet_jetson_deploy/infer_fps.py` 는 배포 파이프라인이 아니라
   **Jetson 추론 속도(FPS) 측정 스크립트**다(파일 docstring). 문서화된 입력이
   `forklift_raw_20260528_163408.mp4` 인데, 그 세션의 GT 25장을 전수 확인하니
   `dimensions_m` 이 110×130×11 이다({w1.3,d1.1} 22장 / {w1.1,d1.3} 3장). [확인]
   따라서 `PALLET_DIMS = (1.1, 1.3, 0.11)` 은 **그 입력에 대해 맞는 값**이다.
   registry default(110×110×15)와 다른 것은 그 스크립트가 정사각 팔레트용이 아니기
   때문이지 불일치가 아니다. 정사각 팔레트 영상에 돌릴 때만 바꿔야 한다.

### 이 부록이 최종 판정에 넘기는 것

- 배포 물체 `plastic_standard_110x110x15` 에는 **대칭 계약이 정의된 적이 없다**.
  그 상태에서 identity-only 열이 보고되고 있고, C4 track 은 그 열의 실패가
  실제로는 위치가 맞는 순열임을 같은 표에서 보였다.
- 이건 모델을 더 학습해서 줄일 수 있는 오차가 아니다. 계약이 없어서 생긴 계상 오차다.
- 다만 이 증거는 challenge live-capture 155 프레임 · F0/F1/F2/v4 모델의 것이다.
  **논문 트랙의 PAPER_EVAL 319 · R0 에 같은 분해를 아직 하지 않았다.**
  그 분해가 이번 감사의 다음 필수 계산이다(§6.2 fixed-index vs allowed-symmetry vs order-free).
