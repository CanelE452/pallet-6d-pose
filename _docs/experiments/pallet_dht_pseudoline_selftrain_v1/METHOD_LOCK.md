# 사전 고정 — Stage A 및 조건부 학생 실험

## 선과 loss

semantic edge 12개, center8 제외. endpoint 각각의 signed residual을 계산한다.
정규화 선 l=(n,c), ||n||=1. L_line = weighted mean over available edges of
SmoothL1(d_a/D,beta=.01)+SmoothL1(d_b/D,beta=.01).
weight=clamp(1-ambiguity,0,1). mode_mass를 곱하지 않는다: ambiguity와 같은
분포의 중복 confidence 가중을 추가하지 않는 단일 deterministic 정의다.
available/finite/nondegenerate가 아니면 weight0. confidence cutoff는 없다.
캐시의 utility>0은 sigmoid×available의 available 복원에만 쓰며 utility값을
weight로 사용하지 않는다. 테스트에서 fresh decode available과 일치를 검증한다.
기하 augmentation은 homogeneous A의 inverse-transpose를 적용하고 normal을
재정규화한다. flip/resize/translation/composition 불변성을 검사한다.

## Mechanism gate — 결과 확인 전 고정

calibration256은 진단, synth_val512는 단 하나의 Stage A gate 모집단이다.
P가 선택한 C1/C2 whole-object assignment를 모든 teacher/GT 비교에 공유한다.
GT를 보고 선을 고르거나 role을 permutation하지 않는다. C4는 허용하지 않는다.

다음 세 조건을 모두 요구한다:

1. 모든 frame의 output-coordinate gradient cosine 평균 >0.
   grad L_line 대 grad(frame mean 8-corner Euclidean GT error / D).
   zero norm은 cosine0으로 보존하며 별도 개수를 보고한다.
2. 동일 common eligible edges와 동일 DHT soft weight로 측정한
   GT-endpoint-to-DHT-line 가중 평균 px < GT-endpoint-to-P-line 가중 평균 px.
3. DHT line error < P-line error인 edge 비율의 frame-cluster bootstrap
   95% two-sided interval 하한 >.5 (4096 draws, RNG seed=1701).
   edge를 독립 표본으로 bootstrap하지 않는다. tie는 better로 세지 않는다.

common eligible: 두 GT/Point endpoint 유효, P/GT 선 길이>1e-8,
DHT available/finite/unit-normal. quality의 공통 mask는 GT 진단에서만 쓰며
학습 선 weight의 GT filter로 사용하지 않는다.
가상 step은 최대 coordinate norm=.001px 방향 -grad L_line, optimizer update0.
derivative=-dot(gradGT,gradLine)/||gradLine|| 및 실제 GT loss 차이를 함께 기록한다.
line-normal 개선 가능성은 teacher normal축에서 (P-Y)·n과 teacher residual의
부호 일치 비율로 보고하며 tangent 위치 정답 생성으로 확대하지 않는다.

## 조건부 학생 예산

gate 실패: DHT_PSEUDOLINE_MECHANISM_FAIL, 9-fit/real cache/λ calibration 미실행.
gate 통과: C0/C1/C2 × seeds1/2/3, 각900 updates, last-step, 동일 R0 init.
V3A true-ignore 기존273개 pseudo labels를 C1/C2가 byte-identical하게 재사용.
SGD lr=.002, momentum=.937, wd=.0005, cosine lrf=.01, warmup1epoch.
기존 V3 augmentation recipe 그대로, horizontal flip0. geometry에 line을 함께 변환.
V3 예산의21600 synthetic exposures를 모든 arm에 동일하게 유지한다.
각 update의 paired plan: synthetic24 + real8. C0의 real8 slot은 gradient 없는
padding이며 synthetic loss의 scaling은 C1/C2와 동일하다. C0에 extra synthetic을
채우면 synthetic exposure가 달라지므로 기존 R0-CONT substitution과 구별한다.
따라서 C0는 optimizer/synthetic exposure control이지 동일 FLOP control은 아니다.
λ는 합성 calibration 첫8frame의 동일 학생 parameter gradient norm으로 1회,
target ratio=.25, clamp=[0,1000], beta=.01. gate 통과 전에는 계산하지 않는다.
test/실사 성능으로 λ·threshold·예산을 바꾸지 않는다.

평가 시 real DEV/PAPER_EVAL은 POSTHOC_DEVELOPMENT_ONLY이며 confirmatory가 아니다.
never-consulted target population은 정본 provenance로 증명된 경우에만 별도 사용한다.
현재 paper 문서/claim/table/abstract는 성공해도 자동 수정하지 않는다.
