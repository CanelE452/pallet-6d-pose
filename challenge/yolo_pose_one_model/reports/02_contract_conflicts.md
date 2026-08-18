# 02 — Contract conflicts

범위: 이번 작업은 **합성 G + T 만** 학습한다(사용자 지시 2026-08-14). real 은 이후 별도
finetune 단계로 미뤘다. 아래에는 real 관련 충돌도 조사된 그대로 남긴다 — finetune 단계에서
그대로 쓰기 위해서다.

모든 수치는 재실행 가능한 명령과 함께 적었다.

---

## 결론 먼저

```
항목                                    판정        학습 진행 가능?
────────────────────────────────────────────────────────────────────
keypoint 0~8 의 물리적 의미 (G/T/R)      일치        예 — 실증됨
camera convention (OpenCV)              일치        예
near/far face 정의                      일치        예 — 실증됨
top/bottom 정의                         일치        예
T 의 dims 필드 순서                      해소        예 — 실증으로 확정
팔레트 width/depth 배정                  ★충돌       합성 학습에는 무영향 (PnP 후처리 이슈)
팔레트 height 0.11 vs 0.12              ★충돌       합성 학습에는 무영향
G 해상도 4종 vs padding 계약             ★충돌       해소 — 사용자 결정
PnP 방법 (프롬프트 SQPnP vs 배포 EPnP)    ★불일치     이번 범위 밖 (real 평가 시 결정)
```

**합성(G+T) 학습을 막는 중단 사유는 없다.**

---

## 1. keypoint 의미는 세 도메인에서 동일하다 [확인]

permutation 재계산만으로는 증명이 안 된다(§7 참조). **배포 PnP 로 실증**했다 —
`pose6d_adapter.keypoints9_to_align_vars` 의 `object_points` 순서를 그대로 쓰고,
near/far 를 일부러 뒤집은 대조군과 median reprojection error 를 비교했다.

```
도메인                 n     as-is median   near/far swap median   비율
──────────────────────────────────────────────────────────────────────────
G  paper_release      400        0.00 px           8.03 px         ∞
T  v1 palletobj       382        3.49 px          23.76 px         6.8x
T  v2 palletobj       373        3.75 px          23.14 px         6.2x
R  real manual_kps    243        1.53 px          31.16 px        20.4x
```

as-is 가 압도적으로 낮다 = 저장된 0~7 순서가 배포 3D 모델점 순서와 **같은 물체 코너**를
가리킨다. 뒤집으면 무너진다.

```
재현: python challenge/yolo_pose_one_model/scripts/verify_kp_contract.py --n 400
```

문서 근거도 일치한다.
```
challenge/scripts/annotate.py:11-17    (R 을 만든 도구)
  0 NearTopLeft   1 NearTopRight   2 NearBottomRight  3 NearBottomLeft
  4 FarTopLeft    5 FarTopRight    6 FarBottomRight   7 FarBottomLeft   8 Centroid
G paper_release JSON  objects[0].keypoint_convention = "camera_dynamic_0123_v4"
G cuboid 3D 좌표      {0,1,4,5} z=+height, {2,3,6,7} z=0  → top/bottom 계약 일치
```
→ 프롬프트가 지정한 9-keypoint 계약과 **완전히 같다**. 변환 함수는 필요 없다.

## 2. T 의 dims 필드 순서 = (depth, width, height) [확인] — 해소

`v1/v2` 의 `objects[0].cuboid_dimensions_m = [1.1, 1.3, 0.12]` 에는 순서 문서가 없다.
네 가지 해석을 모두 PnP 로 돌려 판별했다.

```
해석                         n     median reproj    <2px
────────────────────────────────────────────────────────
(w,h,d) = a[1],a[2],a[0]    283       0.00 px      70.7%   ← 정답
(w,h,d) = a[0],a[2],a[1]    283       3.82 px      30.4%
(w,h,d) = a[0],a[1],a[2]    283      31.61 px       0.0%
(w,h,d) = a[2],a[0],a[1]    283     109.05 px       0.0%
```
→ **`[depth, width, height]`**. 즉 T 팔레트는 width 1.30 × depth 1.10 × height 0.12.
registry 는 이 순서로 읽는다(`discover_and_audit.py` enrich()).

## 3. ★ 팔레트 width/depth 배정이 뷰에 따라 바뀐다 — 배포 설정과 불일치

물리 팔레트는 1.10 × 1.30 (정사각 아님). camera-facing convention 에서 width 는
"지금 보이는 near face 의 가로"이므로, 어느 면을 보느냐에 따라 1.10 이 되기도 1.30 이
되기도 한다. real GT 는 프레임마다 이 값을 다르게 기록한다(annotate 의 auto-selected dims).

배포는 고정값 하나만 쓴다.
```
challenge/25y_automatic_lifter-master/.../depth_cam/calib/config.py:153-155
  PALLET_WIDTH_M = 1.10   PALLET_DEPTH_M = 1.30   PALLET_HEIGHT_M = 0.12
```

그 결과 real 프레임 절반에서 배포 PnP 가 나빠진다.
```
GT (width,depth)     n     프레임 자기 dims   배포 1.10/1.30    배포 W/D swap
──────────────────────────────────────────────────────────────────────────────
(1.1, 1.3)         168          1.23 px          1.58 px          8.11 px
(1.3, 1.1)         183          1.55 px          7.59 px          1.63 px
──────────────────────────────────────────────────────────────────────────────
전체               351          1.43 px          4.23 px          4.02 px
```
→ 긴 변을 정면으로 보는 프레임(183/351 = 52%)에서 배포 고정 dims 는 **5배** 나쁜
reprojection 을 낸다.

**영향 범위**: RGB 모델 학습에는 무관하다(모델은 2D 키포인트만 예측). PnP 후처리에만
영향이 있고, 이번 합성 학습 범위 밖이다. real finetune 후 평가 단계에서 반드시
해결해야 한다 — 권고는 "런타임에 near-face 가로를 판정해 dims 를 고르기".

## 4. ★ 팔레트 height 0.11 vs 0.12 — 미해결, 실측 필요

```
0.12   배포 config.py / pose6d_adapter.py 기본값 (주석: "pallet_full.obj 1.10x1.30x0.12")
0.12   T (v1/v2) 합성 데이터 전체
0.11   R real manual GT 전체 (356 프레임 중 356)
```
1 cm 차이다. 어느 쪽이 실측인지 저장소만으로는 증명할 수 없다.
합성 학습에는 영향이 없다(T 는 0.12 로 일관). **real finetune 전에 사용자 실측 확인 필요.**

## 5. ★ real 안에 서로 다른 팔레트 3종이 섞여 있다

```
GT dims (w,h,d)           프레임    해석
────────────────────────────────────────────────────────────
(1.1/1.3, 0.11, ...)       356     과제 팔레트 (양쪽 뷰)
(1.1, 0.15, 1.1)           243     pallet11 — 정사각, manual_kps 없음
(0.8/0.59, 0.14, ...)       45     wood 소형, 해상도 1280x720
```
과제 팔레트만 h=0.11 로 판별 가능하다. finetune 시 이 필터가 필수다.

## 6. ★ G 해상도가 4종 — padding 계약이 840x680 으로 고정되지 않는다

```
G  640x480 / 960x540 / 720x480 / 560x560   (aspect 랜덤 렌더)
T  640x480 단일
R  640x480 (과제 팔레트) / 1280x720 (wood)
```
**사용자 결정(2026-08-14): 해상도와 무관하게 각 이미지에 +100px reflect.**
결과 크기는 제각각이지만 YOLO 정규화 좌표는 이미지별 기준이라 학습에 문제가 없고,
"경계 밖 문맥을 보여준다"는 padding 목적은 유지된다.
계약 파일의 `padded_width/height: 840/680` 은 **640x480 입력에만** 해당한다고 명시했다.

## 7. convert_to_camera_facing_v4 의 2D 휴리스틱은 G 에서 53% 오판 — 도구 한계

`challenge/scripts/convert_to_camera_facing_v4.py` 는 "image polygon area 가 큰 face =
FRONT" 로 near/far 를 정한다. 이 규칙을 G 에 적용하면 400 표본 중 159 개(40%)가
non-identity 로 나온다 — 그중에는 `[4,5,6,7,0,1,2,3]`(near/far 전면 반전)도 있다.

그러나 §1 에서 G 의 as-is reprojection 이 **0.00 px** 임이 확인됐다. 즉 G 는 렌더러의
정확한 투영이고, 어긋난 쪽은 저앙각(edge-on)에서 무너지는 2D area 휴리스틱이다.

→ **G/T 에 이 변환 스크립트를 다시 돌리면 안 된다.** 데이터가 망가진다.
   (T 는 이미 이 스크립트로 변환됐고, 그 결과는 §1 에서 정상 확인됨 — 재적용만 금지)

```
재현: python challenge/yolo_pose_one_model/scripts/verify_v4_conversion.py --n 600
      → v1 600/600 identity, v2 600/600 identity  (T 는 v4 확정)
```

## 8. PnP 방법이 프롬프트와 배포 코드에서 다르다

```
프롬프트 지시   SQPnP + LM refinement
배포 실제       n_vis >= 6 : EPnP + solvePnPRefineLM
                n_vis 4~5  : SQPnP (fallback)
   (pose6d_adapter.py:163-182)
```
프롬프트는 "새 PnP 를 만들지 말고 배포 계약을 재사용하라"고도 했으므로 두 지시가 충돌한다.
이번 합성 학습 범위 밖이라 **결정을 보류**한다. real 평가 evaluator 를 만들 때
사용자 확인이 필요하다. (위 §1/§3 검증은 배포 방식인 EPnP+LM 으로 수행했다.)

## 9. 이미지-annotation pair 결손

```
domain           samples   image 없음
generic_synth      40000          0
target_synth       19991         23
real                 644         62
```
`is_pair_ok` 는 registry 의 `image_exists` 열로 판별한다. 결손분은 split 에서 제외한다.

## 10. manual_kps 의 "클릭" 과 "외삽" 을 사후 구분할 수 없다

`annotate.py` 의 `x` 키(parallelogram 외삽)로 넣은 점도 `manual_kps` 에 좌표로 저장되어,
직접 클릭한 점과 구분되지 않는다. `f`/`g` 키의 auto-PnP fill 은 `manual_kps` 에 `None`
으로 남으므로 이것만 확실히 구분된다(`annotate_io.py:97`).

→ registry 는 9개 전부 채워진 프레임을 `manual_direct`, 하나라도 None 이 있으면
`manual_inferred` 로 적었다. 프롬프트가 요구한 "auto_pnp 를 manual 로 적지 않는다" 는
지켜진다(None 인 점은 v=0 으로 갈 것이다). 이번 범위 밖.

## 11. real 학습 가용은 133장뿐 — 161장이 봉인된 정본 평가셋

```
real 과제 팔레트 · 이미지 있음   294
  split == "eval"                161   ← 정본 평가셋. CLAUDE.md 가 학습 사용을 금지하고
                                        challenge/tests/test_eval_set_canonical.py 로 잠금
  split == "train"                25
  split 없음                     108
  ★ 학습 가용                    133   (13 세션)
```
이번 범위 밖이지만 finetune 설계 시 결정적 제약이므로 기록한다. 프롬프트의
"real train 4,000 effective" 는 133 x 12(반복 상한) = 1,596 이 실제 천장이다.
