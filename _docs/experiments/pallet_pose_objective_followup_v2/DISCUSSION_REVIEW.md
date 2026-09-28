# 최종 해석 독립 검토

결론: C의 아주 작은 OLD_REF 대비 joint sign은 **결정론적으로 다시 계산되었을 뿐**, 유효한 학습 난수 변화로 재현되지 않았다. Plastic에서 C_REF는 matched C_RAW와 R0보다 주 T/R median이 모두 나쁘다. Wood45에서는 OLD_REF와 matched RAW 대비 pooled median 둘 다 작아졌지만 R0 대비는 T 개선/R 악화이고 난도·recording에 따른 절충도 남는다. 새 recipe의 실용적·독립적 자세 향상 또는 배포 승격을 지지하지 않는다. 유효 재현 상태는 `NOT_RUN_EFFECTIVE_TRAINING_VARIATION`, 실행 해석은 `PARTIAL_BUDGET`이다.

## 독립 계산 범위

[기계 판독 검토](DISCUSSION_REVIEW.json)는 동결된 per-frame pose metric과 기존 population lock을 직접 읽고 NumPy quantile·frame별 차이를 계산했다. 기존 evaluator의 paired helper를 호출하지 않았다. C42, baseline43, recipe43, Wood42의 모든 그룹·6arm T/R median/P90 **1,008개 집계 cell**을 원본 RESULTS와 재검산했다. 전체 Plastic128 metric record도 baseline42/43, C42/43의 RAW/REF에서 정확히 일치했다. point 좌표·K·bbox·pose 배열은 공개 파일에 추가하지 않았다. 새 fit/추론/GPU0, 이 계산 내부 CPU0.154초다.

## 명목 seed43의 유효성

[재현 정정](REPLICATION_VALIDITY_CORRECTION.md)에서 C42↔recipe43 두 arm의 complete320batch trace SHA와879개 model tensor, baseline42↔baseline43 두 arm의879개 tensor가 모두 일치함을 독립 확인했다. trainer 설정은43이었지만 data loader/worker 난수 흐름이 고정이었다. fit 전 실제 흐름이 바뀌는지 확인하지 못한 실행 검증 누락을 명시한다.

관측된 baseline42→43의 오차 변화0은 분산이 작다는 증거가 아니다. 같은 초기 checkpoint·같은 데이터/증강 흐름에서 같은 결과를 다시 낸 것이므로 **학습 변동성은 추정할 수 없다**. 네fit/1,280update 비용은 전부 계상하며 이를 삭제하거나 cap 밖에서 다시 실행하지 않는다.

## Plastic 주99: 같은 seed의 recipe 효과와 좌표 보정 효과

Δ는 after−before, 음수가 개선이다. `median 차이`는 조건별 median의 차이고 `paired median`은 동일 frame 변화의 median이다. 다른 통계의 부호를 골라 사전 선택 기준을 바꾸지 않는다. 모든 대조 common valid99/99, pose 유효→실패0/실패→유효0이다.

| 대조 | median 차이 ΔT cm / ΔR ° | paired median ΔT cm / ΔR ° | 해석 |
| --- | --- | --- | --- |
| C42_REF − OLD42_REF | −0.035730 / −0.012101 | −0.055007 / +0.019063 | pooled median의 미세 joint sign, 전형적 paired R은 악화 |
| C43_REF − BASE43_REF | −0.035730 / −0.012101 | −0.055007 / +0.019063 | 위 결과의 정확 재실행; 독립 재현 아님 |
| C42_RAW − OLD42_RAW | −0.106030 / −0.791084 | −0.011640 / +0.016637 | 같은 입력 노출의 RAW 대조에서도 pooled 변화 |
| C43_RAW − BASE43_RAW | −0.106030 / −0.791084 | −0.011640 / +0.016637 | 위 결과의 정확 재실행 |
| C42_REF − C42_RAW | +0.064303 / +0.399305 | −0.052887 / −0.065493 | 선택한 C에서 REF의 pooled 두 median이 더 나쁨 |
| C43_REF − C43_RAW | +0.064303 / +0.399305 | −0.052887 / −0.065493 | 독립적인 좌표 보정 가치 재확인 아님 |
| C42_REF − R0 | +0.239359 / +2.644651 | −0.195561 / −0.238403 | 원본 R0 초과 주 목표 불충족 |
| C43_REF − R0 | +0.239359 / +2.644651 | −0.195561 / −0.238403 | 정확 재실행 |
| BASE43_REF − OLD42_REF | 0 / 0 | 0 / 0 | 분산 측정이 아니라 무효 난수 개입 |
| BASE43_RAW − OLD42_RAW | 0 / 0 | 0 / 0 | 분산 측정이 아니라 무효 난수 개입 |

C_REF−OLD_REF의 두 축 개선18frame, T만 개선39, R만 개선13, 두 축 악화29다. 즉 R은31frame에서 개선·68frame에서 악화했다. T P90은127.143068→126.911220cm, R P90은89.197272→89.266298°다. pooled rotation median의0.012101° 개선을 rotation 전반·tail 개선이라고 확대하지 않는다. 약0.36mm의 translation median 차이도 현재 reference 정밀도나 운영 효용을 입증하지 않는다.

### Recording 하나 제외 민감도

아래는 `C_REF − 대응 baseline REF`다. seed42와 명목43 값이 모두 동일하므로 두 열을 나란히 보존한다. REC_044의 주99 구성원은0으로 제외 후99 그대로다. 독립적인 새 split이나 재학습이 아니다.

| 제외 recording | 남은 n | C42−OLD42 ΔT/ΔR | C43−BASE43 ΔT/ΔR | paired median ΔT/ΔR (동일) |
| --- | ---: | --- | --- | --- |
| REC_007 | 66 | +0.182852/+0.034645 | +0.182852/+0.034645 | +0.006563/+0.009091 |
| REC_021 | 97 | +0.117366/−0.018105 | +0.117366/−0.018105 | −0.055007/+0.015948 |
| REC_022 | 83 | −0.080414/−0.012101 | −0.080414/−0.012101 | −0.060868/+0.028902 |
| REC_025 | 72 | +0.049902/+0.076303 | +0.049902/+0.076303 | −0.063083/+0.015769 |
| REC_027 | 87 | −0.030536/−0.136238 | −0.030536/−0.136238 | −0.078662/+0.015948 |
| REC_041 | 90 | +0.039590/−0.074170 | +0.039590/−0.074170 | −0.066014/+0.022858 |
| REC_044 | 99 | −0.035730/−0.012101 | −0.035730/−0.012101 | −0.055007/+0.019063 |

REC_007 또는 REC_025를 제외하면 pooled 두 축 모두 악화한다. joint sign이 모든 recording 구성에 안정적이라고 할 수 없다. 각 seed의 corrected−RAW LORO도 JSON에 별도로 계산했으며 pooled 효과와 섞지 않았다.

## Wood45: 별도 재질 적용성

Wood는 기존361 고유 TRAIN 이미지와 같은 기존 source recipe를 사용한 별도 적용성이다. Plastic128와 합쳐173frame의 새 주목표로 만들지 않는다. 평가는 Clean38/Moderate7/Severe0이며 Severe는 `NA_EMPTY_POPULATION`, strict verified Wood는 `NA_REFERENCE_NOT_VERIFIED`다. 같은45frame에서 모든 arm pose valid45/45, 실패 전환0이다.

| 비교 | before T/R median | C_REF T/R median | median 차이 ΔT/ΔR | paired median ΔT/ΔR |
| --- | --- | --- | --- | --- |
| OLD_REF | 2.071426/1.604867 | 1.867061/1.593281 | −0.204365/−0.011585 | −0.126886/+0.071505 |
| matched C_RAW | 2.017550/1.633067 | 1.867061/1.593281 | −0.150489/−0.039785 | −0.025781/+0.005068 |
| R0 | 2.095130/1.544559 | 1.867061/1.593281 | −0.228069/+0.048722 | −0.088065/+0.048552 |

| 집합, C_REF−OLD_REF | n | median 차이 ΔT/ΔR | paired median ΔT/ΔR |
| --- | ---: | --- | --- |
| Clean | 38 | −0.115926/+0.074132 | −0.119533/+0.090893 |
| Moderate | 7 | −0.199589/+0.032418 | −0.199589/−0.009180 |
| Severe | 0 | NA/NA | NA/NA |
| REC_039 (REC_042 제외) | 25 | −0.084695/+0.057484 | −0.015362/+0.062133 |
| REC_042 (REC_039 제외) | 20 | −0.135510/−0.014999 | −0.163543/+0.090893 |

Clean과 Moderate 각각의 rotation median은 악화했는데 pooled45의 median은 조금 개선했다. median은 비선형 순위 통계이므로 모순이 아니며 난도 median을 평균하지 않았음을 다시 확인했다. OLD_REF 대비 두 축 개선10frame, T만 개선18, R만 개선4, 두 축 악화13이다. R은14개선/31악화이고 paired median도 악화한다. pooled OLD_REF/R0/matched RAW 세 대조를 분리해야 하며 Wood의 부호를 Plastic의 미완료 재현을 대체하는 성공으로 쓰지 않는다.

## 최종 보고 강한 주장 점검

읽은 draft는 이미 OLD_REF/R0/matched RAW와 median 차이/paired median을 구분하고, C−A tradeoff·반복 무효·추가 승격 없음·재사용 DEV 제한을 기술했다. JSON에는 **읽은 draft 시점 SHA**를 남겼으며 그 뒤 자동 재생성되는 최종 보고서를 미리 인증한 것은 아니다.

최종 재생성에서 필요한 사항은 다음과 같다.

- 정정 파일이 없다는 오래된 draft 표시를 제거하고 [재현 정정](REPLICATION_VALIDITY_CORRECTION.json)을 binding한다.
- 완료된 Wood 수치와 최종 그림을 반영하되 독립 seed 재현과 별도 적용성을 구분한다.
- nominal seed43을 “유효한 seed 반복 성공”으로 요약하지 않는다. 실행 검증 누락, 실제 비용4fit/1,280update, 추가 fit 없음과 최종 `PARTIAL_BUDGET` 의미를 보존한다.
- C의 작은 pooled joint sign, 다수 frame의 R 악화, R0/matched RAW 반증, LORO 및 큰 절대 tail을 함께 유지한다.
- geometry-derived reference·stored correspondence·반복 DEV와 독립 물리6D 검증을 구분한다. verified66과 source32는 이를 대체하지 않는다.

원인 판정은 여전히 경쟁 설명이다. B의 음성결과가 모든 강건 손실을 반증하지 않으며, A/C의 입력 가림은 잘못된 pseudo를 더 강하게 학습시켰을 가능성과 표현·기하·reference 문제를 분리하지 못했다. 이 배치에서는 새 방법이나 보정 재fit 없이 비용 상한 안에서 결과와 한계를 닫는 것이 맞다.
