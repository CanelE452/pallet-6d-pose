# 모의 active learning — 실제 학습 비교 완료

판정: `RETROSPECTIVE_AL_NO_SIGNAL`. 기존 데이터를 활용한 retrospective development 결과이며 독립 일반화·신규성 증명이 아니다.

정본319장을 원본 촬영 연결까지 고려해 선택용174장/평가용145장으로 분리했다. 평가용 GT는 모든 학습 완료 후에만 채점에 사용했다.
네 방법이 각각 기존 정답30장을 선택했고, 선택 합집합83장의 정답만 학습용으로 내보냈다. 4방법×3seed×300회, 총3600 optimizer update를 수행했다. 같은 R0/stock pose loss/학습량/최종 체크포인트 규칙을 사용했다.
실제 synthetic 증강 입력 hash는 같은 seed의 네 방법에서 모두 일치했다. Real은 선택 이미지가 다르지만 nominal batch slots와 위치별 증강 seed는 같았다. Mosaic의 추가 source 사용을 nominal batch-slot 수와 혼동하지 않는다.

## 세 seed 평균

| 선택 방법 | AP50-95 | Det | 공통 kp median/P90 px | translation cm | yaw deg | pose coverage |
|---|---:|---:|---|---:|---:|---:|
| random | 0.782465 | 0.993103 | 4.3502/25.9490 | 10.2538 | 1.3317 | 1.0000 |
| diversity | 0.769148 | 0.997701 | 4.2481/24.1507 | 8.8140 | 1.3941 | 1.0000 |
| instability_only | 0.731467 | 0.977011 | 4.3885/28.2699 | 10.7749 | 1.4889 | 1.0000 |
| geometry_weighted_diversity | 0.785874 | 0.993103 | 4.3452/20.6280 | 9.7425 | 1.2917 | 1.0000 |

키포인트 primary는 각 seed의 네 방법 공통 검출 프레임에서 계산한 pooled median이다. 세 seed 평균은 해당 통계의 평균이며 독립 세션 수를 세 배로 늘리지 않는다.

## 모든 seed

| Seed | 방법 | AP50-95 | 공통 median/P90 px | translation cm | yaw deg |
|---|---|---:|---|---:|---:|
| 1 | random | 0.789597 | 4.4317/24.6744 | 9.9646 | 1.3823 |
| 1 | diversity | 0.766142 | 4.3652/25.1649 | 9.0458 | 1.5902 |
| 1 | instability_only | 0.734254 | 4.1639/28.7877 | 10.1741 | 1.6311 |
| 1 | geometry_weighted_diversity | 0.781286 | 4.3590/20.3495 | 8.7120 | 1.2931 |
| 2 | random | 0.796765 | 4.3847/26.0534 | 10.6913 | 1.2904 |
| 2 | diversity | 0.793828 | 4.2013/25.8035 | 8.5092 | 1.2772 |
| 2 | instability_only | 0.723838 | 4.5714/32.0821 | 11.0383 | 1.4090 |
| 2 | geometry_weighted_diversity | 0.797373 | 4.4816/23.8496 | 10.3894 | 1.3542 |
| 3 | random | 0.761033 | 4.2342/27.1192 | 10.1054 | 1.3222 |
| 3 | diversity | 0.747475 | 4.1780/21.4837 | 8.8869 | 1.3150 |
| 3 | instability_only | 0.736310 | 4.4303/23.9400 | 11.1122 | 1.4264 |
| 3 | geometry_weighted_diversity | 0.778964 | 4.1951/17.6849 | 10.1261 | 1.2278 |

## 고정 판정 조건

- random: primary_improves=PASS, two_of_three_seeds=PASS, p90_nonworse=PASS, gross20_nonworse=PASS, Det_nonworse=PASS, pose_safe=PASS
- diversity: primary_improves=FAIL, two_of_three_seeds=FAIL, p90_nonworse=PASS, gross20_nonworse=PASS, Det_nonworse=FAIL, pose_safe=FAIL

## R0 참고값 및 해석 제한

같은 평가 subset의 R0: AP50-95=0.762226, 개별 matched kp median/P90=4.7298/38.6115px, translation=11.2414cm, yaw=1.3242°. R0 개별 matched 집계와 위 네 방법 공통 집계는 분모가 다르므로 직접 차감하지 않는다.
기존 GT-v2는 수동/기하 재구성 provenance가 섞인 정본이며 새 외부 센서 GT가 아니다. Stock visibility1은 가려진 좌표를 감독하며, 이전 pseudo TRUE_IGNORE loss는 사용하지 않았다. 이미지 밖으로 나간 점은 visibility0으로 내보내고 원본 라벨은 보존했다.
기존 optional Albumentations API 불일치로 해당 optional transform은 미적용이며 모든 arm에서 동일하다. cuBLAS 엄밀 결정론은 보장하지 않는다. 실제 입력 순서의 parity는 직접 확인했다.
평가 세션은 학습/선택에서 분리됐지만 과거 연구에서 이미 본 세션이다. Negatives도 기존 DEV이며 세션 독립 확인을 주장하지 않는다. 3seed 방향은 통계적 유의성이나 독립 일반화 증명이 아니다.
기하 불안정성은 단일 밝기 변화 heuristic이다. CLUE 재현이나 새 방법 novelty를 주장하지 않는다. 이번 실험은 하나의30-label 예산점 비교이며 완전한 label-efficiency curve가 아니다.
Acquisition seed는 하나이며 선택 집합은 학습 seed1/2/3에서 고정이다. 모든 학생의 시작 가중치는 동일한 R0다. 학생 학습 RNG 변동을 반복했을 뿐, 서로 다른 무작위 라벨 선택의 변동성까지 추정하지 않았다.
정본 pose evaluator의 일부 설명문은319장/학습0이라는 역사적 상수를 포함한다. 실제 입력/분모 및 새 checkpoint는 각 ACTUAL_POSE_BINDING.json과 이 보고서를 기준으로 읽는다. 수치 계산은 변경하지 않았다.
이전97장 라벨링 미리보기와 A/B/C/D/E 결과, 논문 final 문서는 변경하지 않았다. 이번 학생은 선택용으로 쓴 원래DEV 프레임을 전체319 평가에 다시 포함해 성능 주장하면 안 된다.
결과에 따른 추가 seed/학습량/밝기/가중치/선택 budget 탐색은 수행하지 않았다.
