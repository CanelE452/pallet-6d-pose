# Current Real Pallet Pose Dataset

작성 2026-08-27. 모든 수치는 **실제 artifact 재계산**이다 — 기억이나 기존 문서에서
복붙하지 않았다. 재계산 결과는 `DATASET_AUDIT.json` 에 기계 판독 형태로 있다.

## 목적

RGB monocular pallet 6D pose 평가. 단일 카메라 한 장에서 팔레트의 9 keypoint 를
예측하고, PnP 로 6D pose 를 복원하는 파이프라인을 실촬영 데이터로 평가한다.

## 구성

```
population                     count   DAY   NIGHT
────────────────────────────────────────────────────
reviewed positive (140)         140    112     28
common comparison positive      128    100     28
real negative                 2,689      —      —
```

- **positive** — 실촬영 팔레트 프레임. 전부 수동 annotation(`gt_source = "manual"`, 140/140).
- **negative** — 팔레트가 없는 실촬영 프레임. false-positive 측정 전용.
- **DAY / NIGHT** — `eval_night08` · `eval_night09` 를 NIGHT, 나머지를 DAY 로 본다.

## keypoint convention

`camera_dynamic_0123_v4`, 9 점.

```
0 NearTopLeft      4 FarTopLeft
1 NearTopRight     5 FarTopRight
2 NearBottomRight  6 FarBottomRight
3 NearBottomLeft   7 FarBottomLeft
8 Centroid
```

이름은 `scripts/annotate/annotate_draw.py::KP_NAMES` 실측이고, 3D 정의는
`scripts/annotate/annotate_pnp.py::make_pallet_keypoints_3d_diagram` 이다
(local frame X=right, Y=down, Z=forward; 0~3 = near face).

## ★ 현재 role — DEVELOPMENT / diagnostic

**이 데이터셋은 held-out final test 가 아니다.**

`REVIEWED_CLEAN_REALDEV_V2` 매니페스트 자신이 `role: "DEVELOPMENT — final test 아님"`,
`status: "CANDIDATE_PRIMARY_REAL_EVAL"` 로 선언한다.

그리고 실제로 이 140/128 셋은 모델·loss·데이터 선택에 **반복적으로** 사용됐다 —
architecture 비교, negative supervision 스크린, hard-negative 스크린, threshold 탐색 등.
따라서 논문 최종 수치를 이 셋에서 뽑으면 selection 에 쓴 셋에서 성능을 보고하는 것이 된다.

```
FINAL_TEST_STATUS = NOT_AVAILABLE
```

요건은 `FINAL_TEST_REQUIREMENTS.md` 에 정리했다.

## 문서 구성

```
README.md                        이 문서 — 개요 · 논문 평가 계획 · 비교 대상
CURRENT_REAL_DATASET_CONTRACT.md population 표 · 128 provenance · evaluator 별 모집단
INHOUSE_DATASET_VALIDITY.md      coverage / split integrity / annotation reliability
ANNOTATION_RELIABILITY_PLAN.md   제출 전 최소 검증 계획 (GT noise floor)
FINAL_TEST_REQUIREMENTS.md       final test 가 갖춰야 할 조건
DATASET_AUDIT.json               기계 판독 요약 (전부 실측, 미확인은 null)
```

---

# 논문용 평가 계획

## main table 지표 — 5개로 고정

```
Box AP50:95
ADD(-S) AUC
Rotation median
Translation median
Yaw median
```

## main table 에서 제외 (필요하면 diagnostic / appendix 에서만)

```
5cm5deg        AUROC        IoU3D        Gross20
cbox           candidate/frame          ranking margin
```

**`5cm5deg` 는 이후 paper-facing 문서에서 다시 제안하지 않는다.**

---

# 다른 method 와의 비교 계획

## 최소 비교 대상

```
method                RGB   target-specific   CAD at        bbox     real
                     input?   training?      inference?    input?  supervision?
──────────────────────────────────────────────────────────────────────────────
DOPE                   O          O              X           X       설정에 따름
PVNet                  O          O              X           O       O
SingleShotPose         O          O              X           X       O
YOLO26n-Pose (G38)     O          X *            X           X       X
MegaPose RGB (opt.)    O          X              O           O       X
```

\* G38 은 generic pallet 합성으로만 학습하고 평가 대상 팔레트를 학습에 넣지 않는다.

각 열은 **training / input 계약의 차이**다. 이 차이를 숨기고 한 표에 숫자만 나열하면
비교가 아니라 오도가 된다. 표에 그대로 싣는다.

## ★ 비교 규칙

- **published dataset 의 논문 숫자와 우리 in-house 숫자를 직접 비교하지 않는다.**
  (LINEMOD/YCB-V 의 ADD-S AUC 와 우리 파렛트 셋의 값은 같은 양이 아니다.)
- 가능한 method 는 **같은 final in-house dataset 에서 재평가**한다.
- 재평가가 불가능한 method 는 표에 넣되 "not re-evaluated — reported from original paper"
  로 명시하고 정량 비교에서 제외한다.
