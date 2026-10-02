# 이미지와 물리 치수를 같이 넣을지에 대한 결론

## 결론

ResNet18에는 이미지와 고정 물체축 `W,D,H`를 함께 입력하는 것이 맞다. 다만 논문에서는 **치수가 항상 성능을 올린다고 주장하지 않고**, 같은 구조의 상수 입력 대조군과 형상 비율 대조군을 함께 제시해야 한다. 현재 근거는 다음과 같다.

- 직사각형 실사에서는 치수 입력이 2D 오차를 작지만 일관되게 줄였다.
- T 중앙값과 꼬리 오차는 지표·seed에 따라 섞여 있어 T/R 동시 안정 개선은 아직 아니다.
- 정사각형 집단은 모든 프레임의 치수가 같으므로 그 집단만으로 치수 정보의 인과 효과를 식별할 수 없다.
- 최근 0918 정사각형 119장에서도 치수 모델의 우위는 작고 혼합되어 있다.

따라서 구현은 `RGB+FULL WDH`로 진행하되, 표에는 `CONSTANT`, `SHAPE`, `FULL`을 같은 학습 조건으로 비교한다. 치수는 카메라에서 보이는 폭/깊이 또는 GT pose로 바꾸지 않고 고정 물체축을 사용한다.

## 직사각형 DEV319

DEV319는 정사각형이 아니다. Plastic 194장은 110×130×11cm, Wood 125장은 80×59×14cm다. 최신 guided 모델에서 같은 구조·seed의 `CONSTANT`와 `DIMENSION`을 비교하면 다음과 같다.

| seed | CONSTANT median/P90 px | DIMENSION median/P90 px | CONSTANT→DIM PCK10 |
|---:|---:|---:|---:|
| 1 | 6.5370 / 42.7854 | 6.4043 / 42.0833 | 64.466%→65.026% |
| 2 | 6.5076 / 42.7416 | 6.4379 / 42.4728 | 64.466%→65.026% |
| 3 | 6.5141 / 43.0085 | 6.4022 / 42.6218 | 64.266%→64.786% |

2D 중앙값·P90·PCK10은 세 seed 모두 좋아졌다. 그러나 T 중앙값은 `7.6598→7.7326`, `7.6759→7.7132`, `7.5796→7.5479cm`로 두 seed가 나빠지고 한 seed만 좋아졌다. R 중앙값은 세 seed 모두 소폭 좋아졌다. R0와 비교하면 DIMENSION의 T 중앙값은 좋아지지만 T P90은 나빠진다. “직사각형이 나빠졌다”는 판단은 T 또는 꼬리 오차를 본 경우에는 맞고, 2D 전체 결과에는 맞지 않는다.

별도 N2 치수 보정기에서도 3-seed 평균 `N0→N2`는 median `5.9439→5.7777px`, PCK10 `67.53→68.59%`, T median `7.2604→7.0106cm`였다. 평균은 좋아졌지만 사전 안전 판정에서 good<5px→bad>10px 손상 1건이 있어 보편 성공 판정은 하지 않았다.

## 기존 정사각형 GREEN150

GREEN150은 110×110×15cm 한 종류뿐이다. guided 결과에서 치수 입력은 상수 입력보다 median이 두 seed에서 나빠지고 한 seed에서 좋아졌으며 P90도 혼합됐다. PCK10은 거의 같았다.

| seed | CONSTANT median/P90 px | DIMENSION median/P90 px | CONSTANT→DIM PCK10 |
|---:|---:|---:|---:|
| 1 | 4.1917 / 9.1162 | 4.2279 / 9.0667 | 82.085%→82.085% |
| 2 | 4.1984 / 9.1798 | 4.2332 / 9.2367 | 82.085%→82.085% |
| 3 | 4.2355 / 9.0617 | 4.1997 / 9.0713 | 81.938%→82.085% |

이 집단에서는 치수 벡터가 모든 이미지에 동일하므로 shuffle을 해도 입력이 변하지 않는다. 따라서 이 표는 모델 전이 결과이며 치수 정보 자체의 효용을 증명하지 않는다.

## 최근 0918 정사각형 119장 직접 평가

이 119장은 GREEN150과 encoded SHA 및 decoded RGB SHA가 모두 0장 겹치는 별도 자료다. 기존 GREEN150 수치를 이 자료의 결과로 재사용하지 않고, 동결 R0/N0/N2를 새로 추론했다. 주 분석은 declared-visible 수동 클릭 602점이다.

| seed | 치수 없는 N0 median/P90 px | 치수 N2 median/P90 px | N0→N2 PCK10 |
|---:|---:|---:|---:|
| 1 | 4.9620 / 10.0396 | 5.0293 / 9.9862 | 88.87%→89.20% |
| 2 | 4.9664 / 9.9147 | 4.8158 / 9.8933 | 89.37%→89.37% |
| 3 | 4.9612 / 9.8120 | 5.0017 / 9.7672 | 89.53%→90.20% |

N2는 P90이 세 seed 모두 조금 좋아졌지만 median은 한 seed만 좋아졌다. 전체 대칭 오차 E_sym은 seed 1·2가 좋아지고 seed 3은 나빠졌다. 치수 입력의 효과는 작고 일관되지 않다. 반면 R0 median 5.5257px/PCK10 85.22%에서 N0와 N2가 모두 좋아져, 이 자료가 강하게 보여주는 것은 **학습된 작은 보정기의 효과**이고 치수의 단독 효과는 아니다.

## 0918 주석 상태

- 119장 모두 `split=train`, `population_role=DEV`, `canonical_pose=null`, `UNCONFIRMED_SIGNED_AXIS`다.
- 수동 코너는 602점이고 이미지 안 점은 600점이다. PnP 생성 코너 350점과 자동 중심 119점이 있다.
- `029710`과 `029844`에는 visible 수동 클릭점이 이미지 밖에 있어 별도 민감도로 확인했다.
- 저장 reprojection error 중앙값은 0.9414px, 최대는 8.1448px이다.
- 이 자료로 3D T/R 정답을 만들거나 독립 TEST라고 부르지 않는다.

주석 예시는 [026500 overlay](../../../outputs/annotations/0918_dataset_square/_overlays/026500.png), [028924 overlay](../../../outputs/annotations/0918_dataset_square/_overlays/028924.png)에서 볼 수 있다.

## ResNet18 실행 결정

ResNet18은 동일 FiLM 구조에서 다음 세 입력만 바꾼다.

1. `CONSTANT`: 치수 context를 모두 0으로 고정한다.
2. `SHAPE`: `log(W/D)`, `log(H/√WD)`만 사용한다.
3. `FULL`: `logW`, `logD`, `logH`와 두 비율을 모두 사용한다.

세 모델은 같은 ImageNet 초기값, 배치 순서, augmentation, optimizer, 10-epoch 예산을 사용한다. 기능 진단 500 step에서는 세 모델 모두 고정 TRAIN16의 IoU≥0.5를 16/16 달성했다. 이는 이전 MSE ResNet의 zero-map 붕괴가 해소됐다는 실행 점검이며 일반화 결과는 아니다.

- [최근 119장 상세 결과](../../experiments/pallet_green0918_dimension_audit_v1/REPORT_KO.md)
- [최근 119장 개선·악화 이미지](../../experiments/pallet_green0918_dimension_audit_v1/GALLERY.md)
- [최근 119장 전수 지표 gzip](../../../data/pallet/results/pallet_green0918_dimension_audit_v1/METRICS.json.gz)
- [ResNet 기능 진단](../../experiments/pallet_resnet18_dimension_20261002_v1/DSNT_DIAGNOSTIC_RESULTS_KO.md)
- [데이터 동일성 전수 감사](DATASET_IDENTITY_AUDIT.json)
