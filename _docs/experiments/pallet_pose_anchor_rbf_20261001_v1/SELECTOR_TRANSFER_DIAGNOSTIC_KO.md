# RBF 추가 후 고정 실사 선택·점수·입력 반응 진단

고정된 실제692개 선택만 직전 risk189 및 기존 anchored oracle와 비교했다. 새로운 argmin·부분특징 selector·문턱·width·센터 수 정책은 시험하지 않았다. raw GT·이미지·PnP·fit은0회이며 저장된 오차만 재사용했다.

| 모집단 | 모델 | 이전과 다른 후보 | anchor | safe 개선 | unsafe | oracle 기회 | 놓친 기회 |
|---|---|---:|---:|---:|---:|---:|---:|
| NATURAL99 | R0_ONLY | 0 | 84 | 5 | 10 | NA | NA |
| NATURAL99 | UNION_s1 | 1 | 94 | 2 | 3 | 51 | 49 |
| NATURAL99 | UNION_s2 | 2 | 90 | 4 | 5 | 51 | 47 |
| NATURAL99 | UNION_s3 | 4 | 93 | 3 | 3 | 53 | 50 |
| CLEAN29 | R0_ONLY | 0 | 29 | 0 | 0 | NA | NA |
| CLEAN29 | UNION_s1 | 1 | 25 | 0 | 4 | 8 | 8 |
| CLEAN29 | UNION_s2 | 0 | 26 | 1 | 2 | 9 | 8 |
| CLEAN29 | UNION_s3 | 2 | 26 | 2 | 1 | 12 | 10 |
| WOOD45 | R0_ONLY | 0 | 45 | 0 | 0 | NA | NA |
| WOOD45 | UNION_s1 | 0 | 45 | 0 | 0 | 14 | 14 |
| WOOD45 | UNION_s2 | 1 | 44 | 1 | 0 | 13 | 12 |
| WOOD45 | UNION_s3 | 0 | 43 | 1 | 1 | 17 | 16 |

R0_ONLY의2후보 pool에는4후보 oracle 회수율을 적용하지 않는다. 다른 oracle identity라도 실제 pointwise-safe 개선이면 기회를 회수했다고 센다. 모든 recording·seed·692행 및 정확 동률은 JSON에 보존했다.

## 실제 non-anchor 선택의 점수 기여

아래는 실제로 선택한 후보에서 R0 anchor를 뺀 점수다. 음수는 선택 후보에 유리하다. anchor를 그대로 고른 행은 차이가0이라 별도로 분모를 표시한다. 기존189만으로 다시 argmin하거나 RBF를 제거한 정책을 평가하지 않았다.

| 모델 | 실사 non-anchor/173 | prefix189 차이 중앙값 | RBF64 차이 중앙값 | RBF가 선택에 유리/불리 | 가산 검산 최대오차 |
|---|---:|---:|---:|---:|---:|
| R0_ONLY | 15/173 | -1.6590423 | -0.108287368 | 10/5 | 1.42e-14 |
| UNION_s1 | 9/173 | -1.02279438 | 0.246388083 | 0/9 | 1.04e-14 |
| UNION_s2 | 13/173 | -0.580413876 | 0.263893653 | 2/11 | 8.88e-15 |
| UNION_s3 | 11/173 | -0.445505747 | 0.29373159 | 0/11 | 6.58e-15 |

## TRAIN과 실사의 kernel 반응

각 후보의64개 RBF 중 최대 활성값과, 같은 W/D의 non-anchor 후보−anchor kernel L2 차이 및 가중 점수차이의 절댓값을 요약했다. 임계값 기반 OOD 분류나 선택 filter는 만들지 않았다. 활성은 거리의 함수이며 정답 확률이 아니다.

| 모델 | 모집단 | 유효 후보 | 최대활성 중앙값 | 동일 W/D Δkernel L2 중앙값 | 동일 W/D abs(RBF 점수차) 중앙값 | 동일 W/D abs(prefix 점수차) 중앙값 |
|---|---|---:|---:|---:|---:|---:|
| R0_ONLY | TRAIN2598 | 5194 | 0.915910486 | NA | NA | NA |
| R0_ONLY | NATURAL99 | 198 | 0.828177806 | NA | NA | NA |
| R0_ONLY | CLEAN29 | 58 | 0.684267431 | NA | NA | NA |
| R0_ONLY | WOOD45 | 90 | 0.703426584 | NA | NA | NA |
| UNION_s1 | TRAIN2598 | 10388 | 0.913170332 | 0.15467876 | 0.331352179 | 1.49270338 |
| UNION_s1 | NATURAL99 | 396 | 0.825219994 | 0.237022578 | 0.409739194 | 2.14168656 |
| UNION_s1 | CLEAN29 | 116 | 0.691498565 | 0.102339897 | 0.301082692 | 0.79251641 |
| UNION_s1 | WOOD45 | 180 | 0.699286646 | 0.220673701 | 0.418329576 | 1.96954777 |
| UNION_s2 | TRAIN2598 | 10388 | 0.913269464 | 0.154540654 | 0.26488669 | 1.5519931 |
| UNION_s2 | NATURAL99 | 396 | 0.826094664 | 0.288705044 | 0.334851123 | 2.16271431 |
| UNION_s2 | CLEAN29 | 116 | 0.692561204 | 0.100296254 | 0.206396739 | 0.864913252 |
| UNION_s2 | WOOD45 | 180 | 0.694322279 | 0.205594405 | 0.295527094 | 2.08218697 |
| UNION_s3 | TRAIN2598 | 10388 | 0.913171814 | 0.145872533 | 0.359658152 | 1.48675239 |
| UNION_s3 | NATURAL99 | 396 | 0.828910842 | 0.228232418 | 0.463651657 | 1.96332296 |
| UNION_s3 | CLEAN29 | 116 | 0.69263757 | 0.101438629 | 0.257631024 | 1.04765445 |
| UNION_s3 | WOOD45 | 180 | 0.695805933 | 0.196437364 | 0.377447728 | 1.63811609 |

**관측:** natural에서 이전 risk 대비 후보가 바뀐 행은 seed별1·2·4개뿐이었다. 그러나 kernel 자체가 꺼졌다는 설명은 이 고정 결과와 맞지 않는다. 최대활성 중앙값은 TRAIN 약0.913, natural 약0.825–0.829였고, 동일 W/D 후보 사이 kernel L2 대조는 오히려 natural에서 더 컸다. 실제 non-anchor 선택의 RBF 점수항은 UNION1의9/9, UNION2의11/13, UNION3의11/11에서 양수여서 그 선택을 anchor보다 비싸게 만들었다. 이 항을 제외하거나 약하게 한 selector를 새로 평가했다는 뜻은 아니다. 일부 선택이 바뀌어도 공동 중앙값·불확실성·recording 조건을 통과하지 못했고, 단순 입력활성 소실보다 최종 점수의 보수성과 놓친 개선 기회가 직접 관측된다.

TRAIN2,598행 중 실패1행은 유지하며 kernel 수치는 유효 후보에서 계산했다. 실사173행은 모두 유효하다. 앞189 가중치도 새로 학습됐으므로 RBF 항의 부호나 크기를 이전 모델 대비 변화의 인과 효과라고 해석할 수 없다. 합성 지원범위, 학습 감독, 기하 표현 및 RGB 단서 부족은 분리되지 않은 경쟁 설명이다. 활성 분포 차이만으로 어떤 설명이 원인인지 확정하지 않는다.

## 기존 RGB router의 반대 증거

`pallet_selector_recovery_v1/router_features.py:7–43`은 두 expert의 confidence·잔차·box/점/pose 차이와 frozen S1 GAP448을 MLP64/32에 넣었다. `train_router.py`와 Stage4 보고서에서 source clean/occluded 쌍 TRAIN8192·VAL2048·TEST2048, exact C2 ADDnorm의 더 나은 expert target, VAL 정확도로 epoch5를 선택했음을 확인했다. TEST 정확도0.5415, 전체 mean ADDnorm routed0.097688은 S0의0.096217보다 나빴고, 실사 판정은 CLEAN_RECOVERY_BUT_HARD_LOSS였다. 현재 R0→PoseFix 네 후보·physical T/R 감독과 같은 실험은 아니지만, 단순히 global RGB feature와 MLP를 추가하면 해결된다는 주장을 반박한다.

## 다음 학습 문제 하나의 제안 — 두 축의 anchor 대비 변화 회귀

같은 RBF의 폭·센터 수·margin 배율을 다시 고르지 않는다. 다음 한 가지 후보는 현재 고정253 입력과 후보를 유지한 채,4-way best-label CE 대신 각 후보의 signed physical T/R anchor-excess를 두 출력으로 직접 학습하는 것이다. 현재의 one-hot target와 max-risk margin은 어떤 축이 얼마나 좋아지거나 나빠지는지 전체 signed 두 축을 연속 target으로 전달하지 않는다. 이 정보는 기존 TRAIN 참조에 이미 있으며 runtime에서 참조를 추가하지 않는다. 다만 이번 진단이 감독 문제의 원인을 증명한 것은 아니므로 성공을 예측하지 않는다.

구체적인 다음 계약 후보는 e_T=(T_c−T_anchor)/sT, e_R=(R_c−R_anchor)/sR의 각 축에 sign(e)*log1p(abs(e))를 적용한2-output target이다. 원래 TRAIN 고정 sT/sR과2598행을 쓴다. 특징은 기존253의 후보−anchor 차이로 만들고 공유253×2 가중치, bias0을 사용해 anchor의 예측을 정확히(0,0)으로 둔다. 일반적인 shared linear scalar에서 anchor 차분이 argmin에 상쇄되는 사실은 그대로다. 여기서는 두 개의 signed 회귀 target과 max-of-two 출력 판정으로 학습 문제를 바꾸며, 차분 자체를 새 표현력이라고 주장하지 않는다.

TRAIN valid 후보의 두 축 Huber(delta=1) 평균을 frame마다 평균하고 전체2598행으로 나눈 뒤 λ/2||W||²(λ=1e−4)를 더하는 단일 고정 목적식을 제안한다. invalid/all-invalid는 차분을 계산하지 않고0loss로 원래 분모에 남긴다. λ-strong convexity의 gradient-gap 인증과 기존 단일 solver 예산을 유지하고 네 모델 각1fit만 허용한다. 이 λ와 손실값은 이전 CE 목적과 수치상 동등하다고 주장하지 않는다. 새 Huber폭·변환·가중치 후보를 비교하는 sweep은 하지 않는다.

추론은 예측된 두 변화의 max를 score로 하여 anchor를 포함한 기존 whole-pose 중 선택하고 정확 동률은R0를 우선한다. reference-safe mask나 실제risk를 넣지 않으며0은 학습한 변화의 부호 기준이지 실사에서 맞춘 threshold가 아니다. 이미 고정된 source45와 원래/개입 실사5개 AND·세seed·실패행을 유지하고 하나라도 실패하면 중단한다. TRAIN에서 두 축 예측이 맞아도 실사 전이·보수성·중앙값 개선은 보장되지 않는다.

이 제안은 새 RGB 정보를 추가하는 방법이 아니며 이미 실패한 일반 RGB MLP를 반복하지 않는다. 기존 utility/TrackD의 signed2D gain 회귀 및 DHT의2D cost 회귀 음성 결과도 유지한다. 차이는 현재 네 physical whole-pose에 대한 R0-anchor-relative T/R 두 축을 직접 감독한다는 범위이며, 회귀 일반의 새 발명이나 해결책으로 주장하지 않는다. 별도 prefit와 사전 protocol 없이 실행하지 않으며 이번 문서에는 새 fit·새 정책 성능이 없다.

[전체 고정 진단](SELECTOR_TRANSFER_DIAGNOSTIC.json) · [TRAIN 검산](TRAIN_CONVERGENCE_KO.md) · [실사 결과](REAL_RESULTS.json) · [직전 risk 진단](../pallet_pose_anchor_risk_20261001_v1/SELECTOR_TRANSFER_DIAGNOSTIC_KO.md)
