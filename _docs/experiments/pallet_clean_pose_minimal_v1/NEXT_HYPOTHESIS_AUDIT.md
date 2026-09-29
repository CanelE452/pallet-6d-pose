# 남은 학습 예산에 대한 독립 가설 감사

## 판정

**누락된 CLEAR43 RAW/REF 두 학생으로 실제 두 흐름의 2×2 대조를 완성하는 것은 정당하다. 그 다음 남는 두 학생 학습을 지금 바로 다른 레시피에 쓰는 것은 정당화되지 않는다.**

이는 “가능한 해법이 모두 반박됐다”거나 “예산이 부족해서 방법을 찾을 수 없다”는 뜻이 아니다. 현재 자료에서 확인된 감독 추종·loss·국소 gradient·가림 노출이 어느 새 단일 변경을 선택해야 하는지 충분히 특정하지 못한다는 판단이다. 남은 예산은 사용 의무가 아니다.

이 문서는 [CLEAR43 결정](DECISION_CLEAR_S43.json)과 기존 TRAIN 근거 및 [새 GEO 결과](CURRENT_GEO_RESULTS.json)를 읽은 후 작성했다. CLEAR43 평가 결과는 아직 읽지 않은 상태의 판단이며, 새 fit/추론/GT 수정은 하지 않았다. 누적6회 뒤 누락 대조2회를 추가하면8회/2,560 student update가 될 예정이다. 실제 완료·비용은 최종 원장이 결정하며 계획을 완료로 세지 않는다. 선택기1회 한도는 이미 사용했다.

## 1. CLEAR43은 음성 결과의 재시도가 아니라 누락된 대조다

기존42에는 RAW/REF × CLEAR/OCC 네 학생이 있고, 기존43에는 OCC RAW/REF만 있다. 따라서 가림 제거 효과와 CLEAR 조건의 좌표 보정 효과가 학습 흐름을 바꾸어도 유지되는지를 아직 알 수 없다.

새 current-domain GEO는 같은 frozen 모델·동일 whole-pose 후보에서 다음과 같은 혼합 결과를 냈다. 자연99 전체, valid99/99다.

| 같은 NEWGEO | RAW T cm / R° | REF T cm / R° | REF−RAW T cm / R° |
|---|---:|---:|---:|
| OCC42 | 11.90328 / 5.18892 | 11.42760 / 4.94675 | −0.47569 / −0.24217 |
| OCC43 | 12.16715 / 5.21813 | 12.53851 / 5.53460 | +0.37136 / +0.31646 |

old GEO와 new GEO 중 seed마다 좋은 것을 골라 배포하는 규칙은 승인된 고정 구성이 아니다. REF CLEAR42+NEWGEO는 RAW CLEAR42보다 T가 높고 R가 낮아 역시 단일 성공 방향이 아니다. 강한 단순 대조인 OLD_REF217+old GEO도 계속 남겨야 한다.

CLEAR43 두 fit은 기존 seed43 OCC와 같은 초기값·데이터·teacher·support·기본 증강·학습량을 유지하고 추가 mask만 없는 학생을 보충한다. 새 donor/scorer·LR·loss·checkpoint 선택은 없다. 실제 입력 흐름 차이, 같은 seed RAW/REF의 비교 조건, 초기879/보호747 tensor,320 update와 prediction-before-score를 다시 확인해야 한다. 완성 후에는 양 seed에서 다음을 직접 비교할 수 있다.

- 같은 target/selector의 CLEAR−OCC: 입력 가림의 추가 가치.
- 같은 condition/selector의 REF−RAW: 보정 타깃의 추가 가치.
- 각 고정 구성−R0/OLD_REF217 및 호환되는 OLD_RAW217: 더 단순한 구성 대비 실용적 가치.

이는 두 학습 흐름의 반복이지 새로운 독립 평가가 아니며,42를 보고43을 추가한 적응적 이력도 유지한다.

## 2. TRAIN 근거는 “아무것도 못 배움”을 지지하지 않는다

### native TRAIN의 타깃 추종

[TRAIN_TARGET_FOLLOWING_S42.json](../pallet_clean_to_pose_transfer_v1/TRAIN_TARGET_FOLLOWING_S42.json)의 같은78영상/582지원코너에서 **REF 의사 타깃**에 대한 native2D 잔차다. 물리 GT 오차가 아니다.

| 모델 | mean px | median px | P90 px |
|---|---:|---:|---:|
| R0 | 3.35858 | 2.56619 | 6.98314 |
| RAW CLEAR42 | 3.48129 | 2.75580 | 6.84313 |
| REF CLEAR42 | 2.62424 | 2.10350 | 4.98399 |
| RAW OCC42 | 3.46310 | 2.71150 | 6.81890 |
| REF OCC42 | 2.64904 | 2.19381 | 5.02528 |

REF 학생이 같은 REF 타깃을 더 따른다는 신호는 있다. 완전히 맞춘 것은 아니지만 잔차가 남는다는 사실만으로 LR 부족·표현력 부족·source 간섭·label noise 중 하나를 원인으로 고를 수 없다. R0→RAW target 잔차가 거의0인 것은 RAW target이 R0 출력을 재사용하기 때문이지 R0의 실제 정확도가 완벽하다는 뜻이 아니다.

### augmented TRAIN은 별도 입력·별도 단위다

[TRAIN_AUGMENTED_FOLLOWING.json](../pallet_clean_to_pose_transfer_v1/TRAIN_AUGMENTED_FOLLOWING.json)은 사전 K8 batch의62 real occurrence/45 unique 영상,476지원코너 중 canonical covered21/unmasked455이다. 실제 변환 후640px 공간의 REF target 평균 L2는 다음과 같다.

| own-condition 입력·모델 | planned covered21 | unmasked455 |
|---|---:|---:|
| RAW CLEAR | 2.55278 | 4.15561 |
| REF CLEAR | 2.06290 | 3.70440 |
| RAW OCC | 11.90989 | 4.37520 |
| REF OCC | 11.27199 | 3.94075 |

CLEAR의 covered는 **계획상 위치**이고 실제로 지운 입력이 아니다. 따라서 CLEAR와 OCC의 이 표 차이를 학습 개입의 순효과로 해석할 수 없다.21점은 독립21영상도 아니며 native 잔차와 직접 비교할 수 없다. 가린 점이 어렵고 REF 추종이 부분적으로 좋아진다는 관측만 가능하다.

## 3. 새 paired fit를 바로 고를 수 없는 이유

| 제안할 법한 축 | 이미 있는 반증·제한 | 정직한 상태 |
|---|---|---|
| 더 큰 LR/더 오래 학습 | 기존217의 LR1e−4는 REF mixed loss를1e−5보다 낮췄지만 Severe T/R는 둘 다 나빴다. 같은640 update도 REF 고유 이득을 만들지 못했다. clean78 epoch loss는 다른 배치/증강 평균이고 마지막 상승만으로 발산을 확정할 수 없다. | 모든 schedule이 불가능한 것은 아님. 현재 로그만으로 새 LR/step 하나를 특정할 수 없어 sweep 근거가 안 됨. |
| L1/SmoothL1 또는 RLE clamp 변경 | 실제 RLE clamp 활성은 관측됐으나 location gradient가 남았다. C3 큰 잔차의 위치항 감쇠 때도 RLE/combined 신호가 있었다. TRAIN-calibrated SmoothL1 보조항 B는 해당 설정에서 T/R 이득 없음. | 전체 원인을 gradient 소실로 확정할 수 없음. clamp 제거·계수 재선택은 새로 입증해야 할 목적함수 변경. |
| source 제거/감소 또는 gradient surgery | 같은 full mixed graph의 REF 코너119에서 TOWARD 상실16/획득16,87→87. 네 전역 cosine은 양수. 과거 source-off는 실사 노출2배와 얽힘. | 국소 간섭은 있음. 지배적 source 실패 원인이나 실제 AdamW 장기 효과는 미확정. |
| 신뢰도/교사 일치 gating | AGREE는 큰 보정점을 버렸고 실제 다중교사 융합 tail이 나빠졌다. clean78 RGB-only 감사도 코너 정답 검수가 아님. | 독립적인 TRAIN 신뢰성 cue가 없음. 안정적으로 같은 오답일 가능성은 남지만 새 cue 없이 filter sweep은 부적절. |
| 실사 affine 제거 | C2에서 이미 실행했다. Plastic PCK/AUC는 악화, Full128 T/R에는 작은 이득이 있어 전면 실패로 쓰면 안 됨. 큰 TRAIN 잔차 일부는 잘못 선택된 detection instance였다. | cached 후보를 현재99/동일 GEO로 재평가할 수는 있음. 동일 설정의 새 fit은 새로운 질문이 아님. |
| 학습 범위 확대 | full-model 과거 음성과 C3 partial fit만으로 모든 partial unfreeze를 반박할 수 없음. 하지만 공유 neck을 풀면 detector head가 고정이어도 box/score·assignment가 달라질 수 있음. | current78에 pose-only capacity bottleneck이 입증되지 않음. 단순히 잔차가 남는다는 이유로 detector 보존 계약을 바꾸지 않음. |
| 새 selector/계수/feature | current-domain 공통 Linear94를 한 번 검증했고 결과는 모델/seed 의존적. 후보 두 개가 모두 나쁘면 선택기로 해결 못 함. | 모든 selector가 불가능한 것은 아님. 이번 신규 selector1회 한도 사용 완료; 결과를 보고 재학습/라우팅 금지. |

근거: [학습 곡선](../pallet_clean_to_pose_transfer_v1/TRAIN_CURVE_AUDIT.md), [actual loss](../pallet_pose_objective_followup_v2/LOSS_SIGNAL_AUDIT.md), [실제 파라미터 방향](../pallet_gradient_transfer_diagnostic_v1/REPORT_KO.md), [과거 감사](PRIOR_AND_STANDARD_AUDIT_KO.md).

## 4. 남아 있는 구체적 경고: 실제 가림 노출량

가림 정책의 현재 TRAIN 노출량은 확실히 작다. [전체320 batch 감사](../pallet_clean_to_pose_transfer_v1/PAIR_INTEGRITY_S42.json)에서 seed42 실사2,560 occurrence 중 예정1,282회, 실제 적용418회(**16.33%**),32 proposal 안에서 placement 실패864회다. REF의 가려진 감독코너는632/19,017 occurrence(**3.32%**); RAW는619로 좌표 차이에 따른 덮임 수 차이가 있다. 적용 시 bbox 겹침 면적 비율 평균은7.79%다. 따라서 “확률0.5이므로 실사 절반을 실제 가렸다”라고 쓰면 안 된다.

bbox/지원점 근처로 proposal 영역을 제한하면 같은 크기·aspect·fill·true-ignore에서 실제 covered 노출이 늘 수 있다. 이는 current78에서 아직 시험하지 않은 **입력 mask 위치분포** 변경이다. 하지만 아래 이유로 남은2fit에 즉시 배정할 근거는 부족하다.

1. applied 비율이 작다는 것은 구현된 정책의 정확한 특성이지 코드 오류가 아니다. 정책은 잠긴 full-canvas random32/cover≥1/leave≥2를 그대로 수행했다.
2. 더 많이 가리면 현 의사 타깃의 오차도 더 강하게 학습할 수 있다. 외부 clean과 정확한 amodal target은 다른 조건이다.
3. 이전217의 exposure 증가 C는 자연99의 robust joint gain을 주지 못했다. 이 결과가 current78의 새 placement를 완전히 반박하지는 않지만 자동 확장의 근거도 아니다.
4. current TRAIN 보고서는 각 학생을 자기 CLEAR/OCC 입력에서 측정해, 같은 masked 입력에서 OCC로 학습한 모델이 CLEAR로 학습한 모델보다 회복하는지를 직접 분리하지 않았다.
5. mask 입력분포와 실제 노출 빈도를 함께 바꾸므로 결과를 “placement 위치만의 인과 효과”라고 과장할 수 없다. 독립 처치 축은 sampling policy이되 매개변수인 적용률·covered-point 수·bbox 비율을 모두 보고해야 한다.

이 축을 나중에 재개할 때 필요한 최소 추가 근거는 **새 fit이 아닌 fixed-input TRAIN cross-evaluation**이다. 이미 저장된 CLEAR/OCC 두 모델을 동일 사전고정 clear/occluded 입력 양쪽에 평가해 learned restoration과 단순 입력 난도를 분리하고, covered/unmasked 및 정확한 assignment/deployment identity를 함께 본다. 새 mask를 고른다면 TRAIN geometry만으로 적용률/covered 분포를 잠그고 DEV T/R를 보며 크기나 영역을 조정하지 않는다. 이 감사는 그 추가 추론이나 실험을 실행하지 않았고, 이를 필수 완료 조건으로 새로 부과하지 않는다.

## 5. 남은 두 학생 fit을 보류하는 판단의 의미

CLEAR43 후2회가 남는다는 사실만으로 새 RAW/REF pair를 학습할 필요는 없다.2회로 한 고정 변경을 두 seed에서 시험하는 설계 자체는 가능하므로 “예산 때문에 어떠한 유효한 실험도 불가능”하다는 설명도 틀리다. 문제는 **현재 관측에서 그 변경을 선택할 검증된 TRAIN 원인 근거가 부족하다**는 점이다.

유효한 CLEAR43 결과까지 나왔을 때 다음을 완성하면 해당 배치를 마칠 수 있다.

- 두 실제 흐름의 같은-selector 2×2와 강한 단순 R0/기존217 대조를 누락 없이 제시한다.
- 안정적 공동 개선이 없으면 이를 인정하고, 필요성이 확인되지 않은 가림/보정은 최소 구성의 필수 요소로 남기지 않는다.
- T/R 중앙값 부호만이 아니라 cm·mm·°, P90, valid/전체, paired 프레임, 난도·촬영 기록과 예시를 보존한다.
- 새로운 합성 scorer의 음성/혼합 결과와 사용된1회 비용을 숨기지 않는다.
- partial unfreeze, current78 ROI placement, 더 나은 독립 TRAIN reliability cue와 geometric loss는 **미검증**으로 남긴다. 불가능하거나 모두 반증됐다고 쓰지 않는다.
- 다음 실행에는 지금의 frozen 산출물과 원장을 재사용하게 하고, 구체적인 새 모순·진단 근거 없이 유효한 음성 결과를 다시 학습하지 않는다.

현재 목적에서 충분히 더 단순한 구성이나 보수적인 trade-off가 남는 것과, 모든 가능한 연구 가설을 소진한 것은 다르다. 이 감사의 권고는 전자에 필요한 비교를 끝내고 증거 없는 추가 fit은 보류하자는 것이다.

