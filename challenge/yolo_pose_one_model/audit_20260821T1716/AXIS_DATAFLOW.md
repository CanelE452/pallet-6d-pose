# AXIS_DATAFLOW — 현재 evaluator 가 실제로 무엇을 받는가

코드 호출을 따라 실측한 것이다. 주석·변수명 추론이 아니다. `[확인]`

## 흐름

```
IMAGE  (challenge/data/01_real/**/manual_gt/*.png)
  |
  v
YOLO26N_PAPER_GENERIC_V1 last.pt  ->  boxes + 9 keypoints
  |                                    scripts/stage0/model_compare/mc_dump_paper.py
  |                                    PAD=100 REFLECT_101, imgsz=640
  v
predicted keypoint 0..8  =  camera-facing 0123 convention
  |     0-3 near face / {0,1,4,5} top / {2,3,6,7} bottom / 8 centroid
  |     "near" = 카메라에 가까운 면.  학습 라벨이 그렇게 만들어졌다.
  v
3D model points
  |     scripts/stage0/model_compare/mc_geom.py:53  gt_of()
  |       APNP.make_pallet_keypoints_3d_diagram(width=dims["width"],
  |                                             depth=dims["depth"],
  |                                             height=dims["height"])
  |     scripts/annotate/annotate_pnp.py:72-105
  |       0..3 = -d/2 (near),  4..7 = +d/2 (far),  X=width, Y=height
  |     ★ 즉 local +Z(depth) 축이 "카메라에서 먼 쪽" 으로 **정의**된다.
  v
W / D / H
  |     mc_geom.py:48   dims = label["objects"][0]["dimensions_m"]
  |     = **GT annotation 의 per-frame 필드**
  |     실측 분포: 1.1x0.11x1.3  89 frames /  1.3x0.11x1.1  72 frames
  v
PnP  cv2.SOLVEPNP_SQPNP -> cv2.solvePnPRefineLM   (corner 0..7, centroid 제외)
  |     K = label["camera_data"]["intrinsics"]  (per-frame GT intrinsics)
  v
R, t
  v
GT R, t = label["objects"][0]["pose_transform"]
  |     scripts/annotate/convert_to_camera_facing_v4.py:42
  |       "보존: pose_transform, cuboid, keypoints_3d_world, location, quaternion"
  |     -> camera-facing 재정렬은 **2D projected_cuboid 순서만** 바꾸고
  |        pose_transform 은 원본 그대로 둔다.
  v
re_metrics.pose_error  (permutation 없음)
```

## 항목별 성질

```
항목                        SOURCE                         배포가능  GT전용  프레임의존  고정object frame
─────────────────────────────────────────────────────────────────────────────────────────────────
predicted kp 0..7 의미      모델 출력(camera-facing)         O        X       O          X
PnP 3D point 0..7 의미      annotate_pnp (camera-facing)     O        X       O          X
W / D                       label dimensions_m               X        O       O          X
H                           label dimensions_m (전부 0.11)   O        X       X          -
K                           label camera_data.intrinsics     △        O       O          -
GT R, t                     label pose_transform             -        O       O          X
```

`△` = 실제 배포 카메라는 자체 캘리브레이션이 있으므로 K 는 원리적으로 얻을 수
있다. W/D 와는 성질이 다르다.

## ★ 결정적 실측 — dimensions_m 은 물리 속성이 아니다 `[확인]`

같은 세션·같은 물리 팔레트인데 프레임마다 변종이 바뀐다.

```
set               n   1.1x1.3   1.3x1.1
─────────────────────────────────────────
eval_cad         22        18         4
eval_night08     17         1        16
eval_night09     25        10        15
eval_noapril     12         0        12
eval_outside     22        13         9
eval_pallet07    27        18         9
eval_pallet09    36        29         7
```

7 세션 중 **6 세션에서 두 변종이 섞인다.** 그리고 타임스탬프로 정렬하면
**0.37 초 간격에 변종이 뒤집힌다** (총 27 회 뒤집힘, 그 중 8 회가 2 초 이내).
물리 팔레트 교체로는 설명되지 않는다.

추가로, object 의 depth 축이 카메라 시선과 이루는 각은 **최대 75.2°** 이고
83% 가 45° 이내다. 좌표계가 세계에 고정되어 있고 카메라가 돌았다면 0~180° 전
범위가 나와야 한다.

→ `dimensions_m` 의 width/depth 는 **"이 프레임에서 어느 물리 축이 화면 가로로
보이는가"** 를 기록한 것이다. 즉 **시점 의존 축 배정**이며, 배포 시점에는 알 수
없는 정보다.

## 이것이 왜 정보 누수인가

팔레트가 1.1 × 1.3 이라는 것은 known-size 가정으로 받아들일 수 있다. 그러나
**둘 중 어느 쪽이 이번 프레임의 depth 인가** 는 6DoF pose 의 일부(90° yaw
구분)다. 현재 evaluator 는 그 답을 GT 에서 받아 PnP 에 넣는다.

`AXIS_90DEG_AMBIGUITY = PRESENT` 는 모델의 실패가 아니라 **평가가 대신 풀어주고
있던 문제**다.
