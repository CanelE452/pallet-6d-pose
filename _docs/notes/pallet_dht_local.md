# 고해상도 국소 점·Hough 보정 — v3/v3b 완료 결과

2026-09-09 기준 실험·독립 검증은 완료됐고 전체 개발 목표는 외부에서 일시정지됐다. **v3/v3b 모두 사전 합성 진입 기준을 통과하지 못했고 실사에는 새로 적용하지 않았다.** 전체 실사 개선 목표는 미달성이다. 앞선 실사 v2는 [structured 기록](pallet_dht_structured.md)에 있다.

## v3: 실제 학습한 국소 영상·선 보정

원본 영상으로 보존한640×640 grayscale canvas에서 예측12구조선 주변을 읽는다. canonical reflect100/rectangular LetterBox는 재현하되, 반사·패딩은 원래 영상 content mask와 Sobel support 검사로 선 근거에서 제외했다. 선 방향32점, 법선 방향33점(±16 input pixels)의 corridor를 작은 학습 CNN으로 읽고, rho/angle 분포에서 얻은 선을 point anchor가 있는 WLS에 넣는다. 코너 이동 상한4 input pixels, centroid 복사이며 숨은 GT를 보이는 물리 edge로 가정한 지도는 없다.

hough/no_hough를 seed1, 각1,000 optimizer step/batch16, 같은 초기값·배치·jitter로 학습했다. 전체 등록 파라미터는 둘 다3,243개이며 arm별 사용하지 않는 head도 구조 안에 남는다. no_hough는 같은 corridor CNN의 pooled feature로 선 위치·각도를 직접 예측한다. Hough entropy precision과 direct active precision도 달라서 오직 누적 한 연산만 바꾼 대조군이라고 주장하지 않는다.

학습 합성1,792장의 배치 노출에 50% 확률로 input-pixel sigma2/clip6 jitter를 주고 GT는 유지했다. 감독8코너의 input-pixel SmoothL1(beta1)만 사용했다. 실제 CNN에 점 손실 gradient가 전달됐으며 별도 GT 선 손실은 없다. signed tanh gate가 음수면 이동을 반전할 수 있어 항상 선에 더 가까워지는 투영은 아니다.

감독 코너는 train14,175/calibration2,022/검증4,021개다. 실제 반복 학습 노출은 arm당126,565감독 코너였다. 새 YOLO backbone forward는 없고, 새 국소 CNN은 실제 학습했다. 초기 gate0에서는 baseline이 정확히 유지된다. [프로토콜](../../data/pallet/results/pallet_dht_local_v3/PROTOCOL.json), [구현 계약](../../data/pallet/results/pallet_dht_local_v3/IMPLEMENTATION_SPEC.json), [파이프라인 검사](../../data/pallet/results/pallet_dht_local_v3/provenance/pipeline_review/PIPELINE_REVIEW.json).

## 같은 합성 검증의 결과

calibration256에서 scale `[0,.125,.25,.5,1]` 중 median/P90/mean 비악화와 양호점(≤10px)의 >10px 전환≤1%를 만족하는 값을 선택했다. 목적은 감독 코너 pooled error/raw-diagonal 평균, 동률은 작은 scale이다. 선택 저장 후 검증512장을 열었다. 표의 분모는 감독을 가진509장/4,021코너이며 centroid 제외다.

| 방법 | mean px | median px | P90 px | ≤10→>10 전환 | 사전 기준 |
|---|---:|---:|---:|---:|---|
| baseline | 4.957429 | 2.117257 | 7.799472 | 0/3,743 | 기준값 |
| v3 no_hough, scale1 | 4.953924 | 2.111652 | 7.804086 | 0/3,743 | 미통과 |
| v3 hough, scale1 | 4.965718 | 2.136389 | 7.861490 | 1/3,743 | 미통과 |
| v3b MAP prior, scale1 | 4.943098 | 2.095173 | 7.809183 | 0/3,743 | 미통과 |

사전 진입 조건은 mean1% 이상 감소, median/P90 비악화, 양호점 전환≤1%, scale>0을 모두 만족하는 것이다. v3 direct는 평균0.0707% 감소에 그쳤고 P90이 증가했다. v3 Hough는 평균0.1672% 증가했다. 실제1,000step/AdamW state/같은 trace/마스크/선택 grid와 지표를 별도 CPU 코드로 재계산했다. [독립 감사](../../data/pallet/results/pallet_dht_local_v3/provenance/independent_audit/INDEPENDENT_AUDIT.json), [완료된 합성 결과](../../data/pallet/results/pallet_dht_local_v3/PILOT_RESULTS.json).

## v3b: 고정 가중치 MAP prior 진단

v3는 선 분포를 고른 뒤 WLS point anchor를 적용한다. v3b는 선을 고르기 전에 원래 점 쌍에서 먼 선을 덜 선호하도록 logits 한 곳에 다음 고정 prior를 추가했다.

`logits = score/.05 − rho²/(2×2²) − alpha²/(2×atan(2√2/edge_length)²)`

rho/edge length는 input pixels, alpha는 radians다. 식·sigma는 결과 전 고정했고 다른 모델 연산은 동일하다. 원래 v3 Hough 최종 checkpoint를 strict load해 가중치가 정확히 같으며 추론 전후 파라미터 차이의 최댓값·분산은0이었다. 별도 폴더만 사용해 원래 소스·결과를 보존했다.

합성 forward는 calibration256+검증512=768장, optimizer0회다. scale1에서 평균0.2891% 감소와 중앙값 개선을 보였지만 P90은7.799472→7.809183px로 늘어, P90와 평균1% 조건으로 미통과했다. **원래 v3 Hough보다 나아졌지만 기준을 넘은 학습 모델 또는 실사 개선 증거는 아니다.** Prior는 posterior 위치와 entropy precision을 함께 바꾸므로 둘의 효과를 따로 입증하지 않는다. [결과](../../data/pallet/results/pallet_dht_local_map_v3b/RESULTS.md), [사전 프로토콜](../../data/pallet/results/pallet_dht_local_map_v3b/PROTOCOL.json), [정확한 코드 변경](../../data/pallet/results/pallet_dht_local_map_v3b/MODEL_CHANGE.json), [독립 수치 검증](../../data/pallet/results/pallet_dht_local_map_v3b/INDEPENDENT_REPLAY.json).

## 적용 범위와 GT 한계

두 방식은 semantic ID를 유지하는 국소 위치 보정이다. 의자가 겹친 원래 사례처럼 번호배치 때문에 수백 픽셀 오차가 날 때 4 input-pixel 상한으로 해결할 수 없다. 해당640×480 원본 gain16/21에서 최대5.25 raw pixels이고 실제 gate/scale은 더 줄인다. v3/v3b의 해당 실사 해결 여부를 측정하지 않았으며 기존 사례 결과도 바꾸지 않았다.

수치는 기존 감독 좌표에 대한 오차이며 GT 전체의 픽셀 정확성을 인증하지 않았다. amodal·가림·화면 밖·출처 미상 한계는 [기존 GT 감사](../../data/pallet/results/pallet_dht_gt_audit_v1/AUDIT_CONCLUSION.json)에 따른다. 이번 결과는 불가능함의 증명이 아니지만, 현재 결합이 실사에 유용하다는 결론에도 부족하다. 일시정지 상태에서 새 학습·실사 추론·threshold 조정·GT 수정은 하지 않는다.
