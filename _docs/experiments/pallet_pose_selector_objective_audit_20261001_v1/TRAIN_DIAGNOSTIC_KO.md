# 동결 TRAIN scorer 진단

기존6개 fit을 수렴 완료나 Linear94 표현력의 한계로 판정할 근거는 부족하다. **새로운 cost-weighted loss보다, 같은 후보·target·feature에서 수렴 여부를 증명할 수 있는 한 번의 통제가 먼저다.** 동시에 CE가 pose 오류의 크기를 반영하지 않는 현상은 확인됐다. 그것만으로 mean-regret 목적함수가 T/R 중앙값을 함께 회복한다고 결론 내릴 수는 없다.

[전체 측정값과 모든 TRAIN frame 선택](TRAIN_DIAGNOSTIC.json)은 원래 checkpoint·trace·feature·TRAIN label·protocol의 hash를 검증하고 계산했다. 새 fit, optimizer step, 이미지 모델 forward, 실제 영상 routing, 실사 GT 읽기는0회다. 이전에 완료된 source VAL summary만 읽었다. 단일 RGB 이미지와 팔레트 치수라는 입력 조건을 유지하며, 기존 K/기하 계약이 사라졌다고 주장하지 않는다.

## 분모와 계산

TRAIN2598행을 모두 유지했다. 모든 arm에 후보가 없는 동일1행이 있으며 R0 fallback도 실패한다. T/R 성능은 이1행을 양축+∞로 포함한다. `P50/P90`은 원래 extended-real 보간을 사용하고 `finite_quantiles`만 명시적으로 조건부다. regret는 양쪽 cost가 정의되는2597행에서 계산하며, 실패1행을 성공처럼0 regret로 보고하지 않는다. JSON의 frame vector에는 해당 행이 남고 요약의 실패 수는1이다.

공통 cost는 이미 동결된 `max(T/2.4636887551191258 cm, R/1.113474019956766°)`다. 실제 예측 선택은 원래 float32 scorer 그대로 계산했다. gradient/Hessian은 같은 float32 정규화 입력과 저장 weight를 float64 실수 함수로 해석한 CE에서 계산했다. score의 float32/64 최대 차이는9.60e−6 이하이며, 미분을 위해 예측 선택을 바꾸지 않았다.

| 모델 | 정확한 target 일치율 | 최종 full-TRAIN CE | 오답 수 | 정의된 mean regret | T 중앙값 cm | R 중앙값 ° |
|---|---:|---:|---:|---:|---:|---:|
| R0_ONLY s1 | 92.22% | 0.207705 | 202 | 5.43934 | 2.47034 | 1.11748 |
| R0_ONLY s2 | 92.03% | 0.208843 | 207 | 5.56375 | 2.46680 | 1.11748 |
| R0_ONLY s3 | 91.99% | 0.213817 | 208 | 5.58645 | 2.47604 | 1.11937 |
| UNION s1 | 57.18% | 0.854337 | 1112 | 5.81993 | 2.26569 | 0.98509 |
| UNION s2 | 56.57% | 0.848326 | 1128 | 6.26905 | 2.28686 | 1.03190 |
| UNION s3 | 56.41% | 0.872825 | 1132 | 6.06567 | 2.30534 | 1.02171 |

2후보와4후보의 CE 크기를 같은 난이도로 직접 비교하면 안 된다. UNION은 TRAIN 중앙값을 개선하면서도 R0 operational 선택의 같은-pool mean regret보다 나쁘다. 같은 UNION pool 기준 원래 R0+GEO mean regret는 각각5.77063/5.79812/5.76553이다. R0_ONLY도 원래 R0+GEO regret5.10428보다 모두 크다. target 정확도와 pose utility는 같은 지표가 아니다.

## 오답의 대부분과 위험의 대부분이 다르다

| UNION seed | 잘못된 long/short 가설 | 같은 가설에서 expert 오선택 | 가설 오류의 전체 regret 비중 | 가설 오류의 CE 비중 |
|---|---:|---:|---:|---:|
| s1 | 199 | 913 | 93.09% | 14.98% |
| s2 | 212 | 916 | 92.59% | 16.57% |
| s3 | 207 | 925 | 93.07% | 16.34% |

regret가 큰 상위260행은 총 regret의95.4–95.7%를 차지하지만 CE의19.2–20.1%만 차지한다. 반면 오답의 조건부 median regret는 약0.80이며, 같은 long/short 안의 작은 expert 차이를 잘못 분류한 사례가 많다. 현재 one-hot CE에는 cost 간격을 반영하는 항이 없다. 따라서 거의 동등한 expert 사이의 선호와 큰 W/D 오류를 같은 형태의 분류 문제로 취급한다.

그러나 “두 번째 좋은 후보와 cost 차이가 작다”는 것이 그 frame의 실제 오선택 피해도 작다는 뜻은 아니다. 실제 scorer가 멀리 떨어진 나쁜 long/short 가설을 선택할 수 있다. JSON은 best–second margin과 실제 selected–best regret를 분리해 기록한다.

TRAIN의 DIVERSE target 비중은44.0–44.9%지만 실제 DIVERSE 선택은25.4–27.4%다. class collapse로 DIVERSE를 전혀 쓰지 않는 상태는 아니지만 R0를 더 자주 택한다. 이 차이가 명시적 expert-ID feature 때문인 것은 아니다. 모든 후보는 동일 shared Linear94를 사용한다.

## 정규화 오류나 수치 발산인가

6개 checkpoint의 mean/std는 bitwise 동일하며 적격 TRAIN의 유효 R0 후보에서 재계산한 값과 일치한다. 정규화 입력의 NaN/∞는0이다. |z| 최대32.85로 큰 값이 있지만, 가장 큰 값들은 모든 expert에서 공유되는 confidence 특징이다. 공통 선형 항은 후보 간 차이에서 상쇄되므로 이 최대값만으로 ranking 실패 원인이라고 할 수 없다. DIVERSE residual 분포의 변화는 실제로 존재하지만 seed3만의 NaN, 다른 scale, 폭주 증거는 없다.

각 epoch의 online loss는11개 batch를 단순 평균하지 않고 실제 batch 행 수로 가중했다. 마지막38행 batch도 포함한다. UNION 마지막5epoch의 감소율은0.47/0.47/0.40%, R0_ONLY는약3.2%다. loss는 초기값 대비 감소했고 finite지만, 낮은 변화율만으로 정지점 도달을 증명할 수 없다.

## 수렴 여부의 직접 증거

| 모델 | CE gradient L2 | Newton decrement | local quadratic gap 추정 | 양의 Hessian 방향 조건수 |
|---|---:|---:|---:|---:|
| R0_ONLY s1 | 0.08844 | 0.34975 | 0.06116 | 1.76e6 |
| R0_ONLY s2 | 0.08237 | 0.37374 | 0.06984 | 1.92e6 |
| R0_ONLY s3 | 0.09153 | 0.38652 | 0.07470 | 1.70e6 |
| UNION s1 | 0.05710 | 0.36183 | 0.06546 | 1.62e6 |
| UNION s2 | 0.05319 | 0.40133 | 0.08053 | 1.63e6 |
| UNION s3 | 0.05570 | 0.41591 | 0.08649 | 1.56e6 |

Hessian의 식별 가능한 양의 방향 rank는R0_ONLY60, UNION61이다. rank threshold는 λmax의1e−10이며 diagnostic 용도다. 나머지 방향의 gradient norm은약1e−9다. Newton decrement는 양의 고유공간에서 `sqrt(gᵀH⁺g)`로 계산했다. **local quadratic gap은 전역 optimization gap의 상·하한이 아니다.** 큰 조건수와 현재 gradient는 아직 남은 최적화 여지를 확인하기 전에 표현력 부족으로 단정할 수 없음을 보여준다.

별도로 `H ≤ (1/4N)ΣframeΣa<b (x_a−x_b)(x_a−x_b)ᵀ`라는 전역 softmax covariance 상한을 사용했다. 이 상한의 최대 고유값 L로 descent lemma가 보장하는 가능한 CE 감소 하한 `||g||²/(2L)`은 UNION1.75e−5–2.11e−5다. 보수적인 양수 하한이며, hypothetical `−g/L` update를 실제 적용하거나 채점하지 않았다. T/R, 특히 중앙값 개선에 대한 보장은 아니다.

기존 학습은 명시적 L2 loss가 아니라 **AdamW의 decoupled weight decay1e−4**였다. 따라서 이를 곧바로 `CE+λ||w||²/2`를 수렴시킨 실험이라고 부르면 안 된다. 참고로 `λ||w||/||g_CE||`는 UNION0.17–0.21%에 불과하다. 단순 penalty cancellation이 현재 gradient를 설명한다는 증거는 없다. 중간 TRACE에는 state hash만 있고 파라미터 snapshot이 없으므로 step별 update norm을 복원하지 않았다.

## seed3와 다음 결정

같은 frozen UNION weight를 다른 TRAIN 후보 pool에 교차 적용한3×3 진단도 저장했다. 세 weight 모두 pool3에서 pool1보다 CE가 높지만, weight3를 pool1에 적용했을 때 정확도와 regret가 일방적으로 가장 나쁘지는 않다. 이는 원래 각 weight가 자기 pool과 seed의 영향을 함께 받아 만들어졌다는 사실을 없애지 않는다. **refiner data와 optimizer initialization의 인과 효과를 분리한 실험이 아니다.** 새로운 seed를 고르거나 실사 모델을 선택하지 않았다.

이미 완료된 VAL에서 s3는 R0_ONLY 대비 T 중앙값이 같고 R 중앙값이 커서 실패했다. P90은 좋아질 수 있으므로 scalar regret 또는 tail 개선만으로 중앙값 gate를 대체할 수 없다. root의 별도 VAL oracle 진단과 함께 해석해야 하며, 이번 JSON은 새 VAL oracle 채점을 수행하지 않았다.

[기존 objective 실험 감사](PRIOR_OBJECTIVE_AUDIT_KO.md)는 DHT의2D soft-cost/regression, Hough gain regressor, TrackD signed-gain 학습의 음성 결과를 구분한다. 현재 physical T/R 후보의 exact regret와 동일한 실험은 아니지만, cost-sensitive나 relative target이라는 이유만으로 전이가 해결된다는 주장은 지지하지 않는다.

따라서 지금은 새로운 regret-weighted objective를 추가로 권하지 않는다. 다음 한정 통제는 같은 target의 **명시적 CE+ridge λ1e−4**를 충분히 수렴시키고, 강볼록 objective gap 상한으로 수렴을 인증하는 것이다. 이 ridge objective는 기존 AdamW와 다르다는 점을 공개해야 한다. 수렴 후에도 원래 source/실사 joint T/R·tail·failure gate를 그대로 유지해야 하며, 수렴 인증 자체는 사용자 목표 달성이 아니다.
