# 이미지와 물리 치수 입력에 대한 최종 판단

## 답

물리 치수는 PnP와 T·R 계산에는 계속 필요하다. 그러나 현재 근거로는 치수 벡터를 신경망의 필수 입력으로 두거나 IEEE Sensors 원고의 핵심 기여로 주장하는 것이 맞지 않다. 논문의 주 방법은 치수 입력이 없는 작은 보정기 비교로 두고, 치수 조건은 별도 ablation으로 보고하는 편이 근거에 맞다.

[세 백본 보정 실험 최종 보고서](../../experiments/pallet_three_backbone_closeout_20261002_v1/REPORT_KO.md)에는 YOLO·DOPE·ResNet18 정확도, 대응 runtime, 치수 ablation, 실제 RGB 사례를 한 문서에 묶었다.

이 판단은 ResNet18 학습 전 작성한 [이전 실행 결정](CONCLUSION_KO.md)을 완료 결과로 갱신한 것이다.

## 왜 최종 판단이 바뀌었나

### 1. 기존 YOLO 계열

직사각형 DEV319에서 같은 구조·seed의 치수 입력은 3 seed 모두 2D median/P90/PCK10을 소폭 개선했다. 하지만 T 중앙값은 3 seed 중 2개에서 악화했고, 꼬리 오차와 안전 판정도 혼합됐다. 따라서 “직사각형에서 전부 나빠졌다”는 표현은 2D에는 맞지 않지만, T와 안정성까지 포함하면 치수 입력의 일관된 우위를 주장할 수 없다.

기존 GREEN150과 최근 0918 정사각형 자료에서는 모든 프레임에 같은 치수가 입력된다. 이 자료의 N0 대 N2 차이는 학습된 모델 전체의 전이 결과이며, 프레임마다 달라지는 치수 정보의 인과 효과가 아니다.

### 2. 이미지+치수를 처음부터 입력한 direct ResNet18

동일한 ResNet18에서 `CONSTANT`, `SHAPE`, `FULL`을 같은 초기값·학습 순서·10 epoch로 비교했다. `FULL`은 이미지와 `logW`, `logD`, `logH`, `log(W/D)`, `log(H/√WD)`를 함께 입력한다.

| 비교 | 2D median | T median | R median | 판단 |
|---|---:|---:|---:|---|
| DEV319 FULL−CONSTANT | -0.210 px | +1.269 cm | -0.760° | 2D·R 개선, T 악화 |
| Plastic 110×130×11 FULL−CONSTANT | +0.145 px | 별도 subgroup 표 참조 | 별도 subgroup 표 참조 | 2D 중앙값 악화 |
| Wood 80×59×14 FULL−CONSTANT | -1.196 px | 별도 subgroup 표 참조 | 별도 subgroup 표 참조 | source 치수 범위 밖 외삽 |
| GREEN150 FULL−CONSTANT | -0.959 px | pose GT 없음 | pose GT 없음 | 고정 치수 전이 |
| 0918-119 FULL−CONSTANT | -0.286 px | pose GT 없음 | pose GT 없음 | 고정 치수, paired 중앙 Δ +0.031 px |

DEV319 전체에서도 T와 R이 동시에 좋아지지 않았고, plastic의 2D 중앙값은 오히려 나빠졌다. Wood는 D=0.59 m가 합성 TRAIN의 D 최소 0.818 m보다 작아 치수 입력의 일반화 근거로 사용할 수 없다.

이 direct 표의 R은 W/D parity를 회전에 포함하는 fixed-physical-frame 평가다. 세 백본 주 표의 canonical MAIN camera-facing-frame R과 절대값을 섞지 않는다. 두 평가의 차이와 319장 행 단위 대조는 [pose frame 계약 감사](../../experiments/pallet_resnet18_dimension_20261002_v1/POSE_FRAME_CONTRACT_AUDIT.md)에 기록했다.

[direct ResNet18 상세 결과](../../experiments/pallet_resnet18_dimension_20261002_v1/REPORT_KO_V2.md) · [정사각형 포함 비교 그림](../../experiments/pallet_resnet18_dimension_20261002_v1/DSNT_REAL_COMPARISON_V2.png) · [RGB 사례와 치수](../../experiments/pallet_resnet18_dimension_20261002_v1/GALLERY.md)

### 3. 동결 ResNet18 뒤의 보정 head

치수의 순수 증분 효과는 같은 P5 구조에서 치수 5값만 0으로 고정한 `P5_CONSTANT`와 비교했다. 13세션 paired bootstrap 결과는 다음과 같다.

| P5−P5_CONSTANT | 관측 차이 | 95% CI | 판단 |
|---|---:|---:|---|
| 2D median | -0.074 px | [-0.116, +0.019] | 0 포함 |
| PCK10 | +0.177 pp | [+0.011, +0.397] | 작고 방향성 있음 |
| T median | -0.071 cm | [-0.115, +0.071] | 0 포함 |
| R median | -0.027° | [-0.067, +0.025] | 0 포함 |

치수 입력의 추가 효과는 PCK10에서만 작게 한 방향을 보였고, median·T·R 구간은 0을 포함했다. 반면 치수 없는 `P0` 보정기 자체는 FULL baseline 대비 2D median 8.001→7.143 px, PCK10 54.68→58.28%, T 11.007→10.001 cm, R 3.252→3.018°로 개선됐다. 논문의 백본 전이 비교에는 `P0`가 더 직접적인 대조다.

[ResNet18 보정기 상세 결과](../../experiments/pallet_resnet18_dim_refiner_report_20261002_v3/REPORT_KO.md) · [paired 비교 그림](../../experiments/pallet_resnet18_dim_refiner_report_20261002_v3/REFINER_PAIRED_CONTRASTS.png)

## 최근 초록 정사각형 자료의 정확한 신원

사용자가 기억한 “0913” 자료는 저장소에서 확인된 `0918_dataset_square`다.

- 119장, 기존 GREEN150과 encoded/decoded SHA 중복 0장
- 전부 1.10×1.10×0.15 m
- 수동 클릭 602점, 그중 이미지 안 600점
- canonical pose 없음 119/119, signed axis 미확정 119/119
- 2D 개발 감사 자료이며 독립 TEST나 T·R 정답 자료가 아님

기존 YOLO 계열 N0 대 N2에서 N2 P90은 3 seed 모두 소폭 개선됐지만 median은 1 seed만 개선됐다. direct ResNet18에서는 FULL aggregate median/P90이 CONSTANT보다 낮았지만 프레임 paired 기준 개선 59장, 악화 60장, 중앙 Δ가 +0.031 px였다. 두 결과 모두 “치수 입력이 정사각형에서 안정적으로 우수하다”는 결론을 지지하지 않는다.

[0918 YOLO 계열 상세 결과](../../experiments/pallet_green0918_dimension_audit_v1/REPORT_KO.md) · [0918 개선·악화 사례](../../experiments/pallet_green0918_dimension_audit_v1/GALLERY.md)

## 세 백본 원고에서 사용할 구조

주 비교는 각 frozen estimator의 보정 전후다.

1. YOLO: `R0 → P1`
2. DOPE: `baseline → P1`
3. ResNet18: `FULL → P0`

DOPE P1과 ResNet18 P0에는 치수 벡터를 신경망 입력으로 넣지 않았다. 치수는 동일한 PnP와 T·R 계산에 사용한다. ResNet18 FULL baseline은 이미지와 치수를 함께 학습한 모델이므로 전체 시스템 수준의 image-only 대 dimension 비교와도 구분한다.

치수 입력 결과는 `CONSTANT/SHAPE/FULL`과 `P5/P5_CONSTANT` ablation으로만 남긴다. 현재 자료로 허용되는 표현은 “일부 2D 지표에서 작고 방향성 있는 차이가 관찰됐지만 T·R의 안정적 동시 개선은 확인되지 않았다”이다.

## 최종 권고

- 논문의 핵심 기여: 합성 source로 학습한 작은 보정기가 서로 다른 세 frozen estimator에서 2D 위치 오차를 줄이는가.
- 치수 사용: PnP에는 유지하고, 신경망 입력은 선택적 ablation으로 유지한다.
- 쓰지 말아야 할 주장: 치수 입력이 직사각형·정사각형 모두에서 항상 개선한다, 또는 T와 R을 안정적으로 동시에 개선한다.
- 후속 치수 실험이 필요하면 동일 이미지와 동일 head에서 여러 실제 W·D·H가 변하는 독립 test를 확보한다. 고정 치수 정사각형 집단을 반복 평가해도 치수의 인과 효과는 식별되지 않는다.
