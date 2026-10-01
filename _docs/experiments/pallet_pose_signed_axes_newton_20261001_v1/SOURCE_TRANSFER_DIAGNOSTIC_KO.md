# 수렴한 두 축 회귀의 source 선택 진단

**수렴 검산은 네 모델 모두 PASS이나 source 조건은43/45이며 전체 FAIL이다.** 이 진단은 이미 잠긴 선택·예측·오차만 다시 요약한다. 새 fit, checkpoint forward, 후보 argmin, threshold sweep, 물리 참조 재채점, 실사 조회·routing은0이다. 진단 PASS는 방법의 gate PASS가 아니다.

UNION_s3의 T 중앙값1.682086665cm가 R0_GEO 및 R0_ONLY의1.674594149cm보다 커 두 strict-median 조건을 실패했다. 세 모델 중 좋은 것만 선택하거나 이 차이를 tolerance로 지우지 않는다. 나머지43조건이 통과했다는 사실도 전체 실패를 대체하지 않는다.

## 실제 선택과 이전 RBF의 비교

다음은 각1,024행 전체의 실제 선택이다. anchor는 후보 정체성이 운영 R0 GEO와 같은 경우이고, safe 개선은 두 실제 오차가 비증가하면서 적어도 한 축이 엄격 개선된 다른 후보다. 실패행은 현재0이며, 근사 허용값으로 safe/unsafe를 나누지 않았다.

| 모델 | anchor 이전→현재 | safe 개선 이전→현재 | unsafe 이전→현재 | 바뀐 후보 |
|---|---:|---:|---:|---:|
| R0_ONLY | 1014 → 1023 | 10 → 1 | 0 → 0 | 9 |
| UNION_s1 | 985 → 922 | 15 → 30 | 24 → 72 | 81 |
| UNION_s2 | 984 → 919 | 14 → 33 | 26 → 72 | 89 |
| UNION_s3 | 992 → 925 | 14 → 26 | 18 → 73 | 87 |

## 두 축 예측 부호와 실제 선택의 위험

선택된 score는 max(predicted T,predicted R)이고 anchor 예측은0이므로 선택된 예측 두 축은 모두0 이하이다. 이것은 실제 물리 변화의 부호를 보장하지 않는다. 아래는 anchor를 제외한 **실제 선택**에서 실제 T 또는 R이 증가한 횟수다. 부호 오차는 고정 경계0으로만 계산하며 별도 calibration이나 threshold를 만들지 않았다.

| 모델 | 선택 nonanchor | T 악화 | R 악화 | 양축 악화 | unsafe 위험 P90 |
|---|---:|---:|---:|---:|---:|
| R0_ONLY | 1 | 0 | 0 | 0 | 해당 없음 |
| UNION_s1 | 102 | 58 | 40 | 26 | 1.642679 |
| UNION_s2 | 105 | 56 | 36 | 20 | 1.023923 |
| UNION_s3 | 99 | 55 | 43 | 25 | 1.532739 |

위험은 `max((T−T_anchor)/sT,(R−R_anchor)/sR,0)`이다. 회귀 MAE/RMSE는 signed-log1p 단위이며 cm/degree가 아니다. JSON에는 anchor를 포함한 선택과 제외한 선택을 나누어 두 축 confusion·MAE·RMSE를 기록했다. anchor의 예측·정답0을 학습된 sign 정확도로 해석하지 않는다.

## 이미 증명된 개선 기회와 놓친 기회의 하한

현재 결과의 오차 NPZ에는 실제 선택한 자세만 있다. 과거 source oracle CSV도 일부 선택만 남겼다. 따라서 기존 RBF의 실제 선택, 기존 oracle/learned CSV 및 현재 실제 선택의 **이미 채점된 후보**를 모아 확인했다. 전체 네 후보를 새로 채점하거나 새 oracle를 만들지 않았다. 기회 수와 miss는 이 부분 캐시가 증명하는 하한이며, 전체 pool recall/false-negative 수가 아니다. 다른 safe 후보를 선택해도 capture로 센다.

| 모델 | 증명된 기회 | 실제 safe capture | miss 하한 | miss 중 anchor | 알려진 nonanchor/가능 수 |
|---|---:|---:|---:|---:|---:|
| R0_ONLY | 10 | 1 | 9 | 9 | 10/1024 |
| UNION_s1 | 226 | 30 | 196 | 193 | 558/3072 |
| UNION_s2 | 238 | 33 | 205 | 204 | 575/3072 |
| UNION_s3 | 219 | 26 | 193 | 191 | 576/3072 |

## TRAIN과 source에서 관측한 것

TRAIN은 전체2,598행 중 유효2,597행·실패1행을 유지한 기존 독립 검산 값을 그대로 인용했다. source는1,024행 모두 유효하다. 다음 값은 각각 다른 모집단의 실제 고정 선택이며, 차이가 곧 도메인 이동의 인과 추정은 아니다.

| 모델 | TRAIN safe/unsafe (유효2597) | source safe/unsafe (1024) | TRAIN nonanchor T/R sign 일치율 |
|---|---:|---:|---:|
| R0_ONLY | 4/0 | 1/0 | 90.72% / 92.80% |
| UNION_s1 | 135/171 | 30/72 | 79.68% / 81.66% |
| UNION_s2 | 120/183 | 33/72 | 79.98% / 81.34% |
| UNION_s3 | 106/171 | 26/73 | 79.98% / 81.16% |

모든 네 모델은 같은 Huber+ridge 목적식의 강볼록 수렴 인증을 통과했다. 그 목적식은 후보의 연속 변화량 적합을 최적화하며, source population의 두 중앙값이나 위험한 선택의 부호 정확도를 직접 보장하지 않는다. 실제 unsafe 선택은 예측 부호 오류가 남았다는 관측이다. 표현력·합성 지원범위·감독 손실 가운데 무엇이 유일 원인인지는 이 진단으로 확정할 수 없다. 반복 사용된 source VAL이므로 새로운 독립 일반화 시험으로 부르지 않는다.

## 다음 개입 한 가지 제안 — 미실행

**기존 두 축 Huber에 각 물리 축의 sign logistic 항 하나를 추가하는 TRAIN-only 목적함수 비교**를 제안한다. 입력253·RBF·후보·두 축 출력·runtime max·기존 tie·λ·zero 초기화를 유지한다. `y!=0`인 각 scalar에 `softplus(−sign(y)·prediction)`을 계수1로 더하고, Huber와 동일한 원래 유효 후보×2축 평균 및 전체2,598행 분모를 유지한다. target0인 anchor/정확 동률은 logistic 항0이다. 계수·sign 경계·margin·seed를 VAL에서 탐색하지 않는다.

근거는 기존 TRAIN에서도 실제 개선을 양수로 예측하거나 실제 악화를 음수로 예측하는 오류가 남고, 운영 규칙이 두 예측의0 경계에 직접 의존한다는 점이다. Huber는 작은 물리 변화의 잘못된 부호에 작은 기울기를 줄 수 있지만 logistic 항은 이 경계에서 분류 신호를 유지한다. 기존 연속 Huber는 남기므로 변화 크기 감독을 버리지 않는다. 추가 항은 선형 예측에 대해 볼록하고 ridge의 강볼록성은 유지된다. 이는 기대 효과의 보장이 아니다.

이전4way CE는 한 후보 winner, pairwise는 후보 간 순서, risk-margin CE는 hard target 보호를 학습했다. 이번 제안은 각 T/R 축의 물리 변화 sign을 보조 감독한다는 차이가 있다. 그러나 분류+회귀를 섞는 일반 아이디어는 새롭지 않고, Structured DHT의2D 혼합 손실·signed2D gain·기존 RGB MLP 전이 실패를 반대 증거로 유지한다. 작은 개선의 수락이 tail을 해치거나 더 보수적인 anchor 유지로 돌아갈 수 있다. 표현력 부족이 해결된다고 주장하지 않는다.

별도 namespace에서4개 최종 fit만 허용하고 기존 Newton 예산·인증·source45/all3seed·실사 original/matched5 조건을 전부 유지해야 한다. 어느 source 조건이라도 실패하면 실사 routing은 계속 금지한다. 이번 진단에서 이 손실을 구현·학습하거나 새 선택 규칙을 평가하지 않았다.

[진단 JSON](SOURCE_TRANSFER_DIAGNOSTIC.json) · [현재 source 판정](SOURCE_VAL_GATE.json) · [독립 TRAIN 검산](TRAIN_CONVERGENCE_KO.md) · [선행 목적함수 감사](../pallet_pose_selector_objective_audit_20261001_v1/PRIOR_OBJECTIVE_AUDIT_KO.md)
