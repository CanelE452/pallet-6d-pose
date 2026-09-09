# Stage 0C/0D — 7.90 cm 가 translation sensitivity 로 설명되는가

새 추론 0 회, 새 학습 0 회. 입력은 동결된 R0 예측 캐시와 geometry-resolved GT
이며 둘 다 sha256 을 산출 JSON 에 기록했다.
산출: `TRANSLATION_SURROGATE_AUDIT.json`, `TRANSLATION_SURROGATE_PER_FRAME.csv`.

## 0. 파이프라인 검증 [확인]

이 감사가 자체 계산한 ORACLE 경로 depth median 은 **7.25368570362197 cm** 로,
`POSE_EVALUATION_R0.json` 의 `paths.ORACLE.ALL.depth_median_cm` 과
소수점 끝까지 일치한다. 평가자와 같은 모델점·같은 solver 를 쓰고 있음이 확인된다.

## 1. Jacobian 게이트 (§13) — PASS

```
analytic vs central finite difference, 무작위 200 케이스, 열 단위 상대오차
median 5.22e-10      p99 1.82e-09      max 1.93e-09
조건수 median 7.98   p99 19.36
게이트 median<1e-4, p99<1e-2  ->  PASS
```

★ 첫 구현은 **FAIL(상대오차 3.7)** 이었다. 회전 블록에
`dP/d(dr) = -skew(R X + t)` 를 썼는데 옳은 것은 `-skew(R X)` 다.
게이트가 없었으면 그대로 loss 로 들어갔을 오류다.

pinv 절단값 `rcond` 는 1e-3 ~ 1e-8 에서 Spearman 이 **소수점 이하까지 동일**하다
(조건수가 8 수준이라 절단이 발동하지 않는다). §16 의 `[추정]` 상수 하나가
살아있는 하이퍼파라미터가 아님이 확인됐다.

## 2. 본 결과 (n=319, ORACLE 축 공급 — 축 선택기가 교란하지 않도록)

```
                        n   L_TR    대조군    대조군    L_TR    부분     부분    linearised
                          ~depth  RMS~depth 거리~depth ~거리  |RMS    |거리    ~depth
ALL                   319  0.828    0.163     0.589   0.583   0.829   0.738    0.929
corner med <5px       104  0.813   -0.160     0.672   0.772   0.816   0.624    0.988
corner med 5-10px      86  0.845   -0.285     0.798   0.882   0.838   0.498    0.998
corner med >=10px     129  0.776    0.545     0.676   0.676   0.664   0.588    0.814
plastic               194  0.810    0.195     0.507   0.486   0.812   0.748    0.929
wood                  125  0.830    0.128     0.567   0.614   0.827   0.741    0.939
```

사전등록 게이트 — depth Spearman >= 0.4 — **PASS (0.828).**

## 3. ★ 진짜 정보는 대조군에 있다

`L_TR` 이 depth 와 상관하는 것 자체는 일부 자명하다. PnP 는 `L_TR` 이 선형화한
바로 그 사상의 비선형판이므로 어느 정도 상관은 구조적으로 보장된다.
자명하지 **않은** 것은 두 가지다.

**(a) 순수 2D 잔차 크기는 depth 오차를 예측하지 못한다.**
plain residual RMS 대 실제 depth 오차의 Spearman 은 전체 +0.163 이고,
저오차 구간에서는 **음수**다 (-0.160, -0.285). 반면 부분상관
(RMS 를 순위공간에서 제거한 뒤) `L_TR` 은 여전히 0.816~0.838 이다.
→ 신호를 나르는 것은 **오차의 크기가 아니라 오차의 방향**이다.
   이것이 §1.3 가설의 직접 확증이다.

**(b) 국소 선형화가 관측 오차 규모에서 유효하다.**
linearised depth 대 actual depth 는 <10px 구간에서 0.988~0.998,
전체 0.929, >=10px 에서 0.814 로 예상대로 열화한다(§17 의 경고가 맞다).
크기도 맞는다 — linearised median 7.579 cm vs actual 7.254 cm.
즉 **oracle depth 오차 7.25 cm 는 GT 기준 2D 잔차의 1 차 전파로 거의 전부 설명된다.**

## 4. surrogate 를 두 인자로 쪼개면 [확인]

`L_TR = sqrt(sum_j s_j r_j^2)` 에서 기하 인자 `sqrt(mean s)` 와 잔차 인자
`RMS(r)` 를 따로 보면

```
Spearman(기하 인자, 실제 depth 오차) = +0.537
Spearman(잔차 인자, 실제 depth 오차) = +0.163
기하 인자 프레임 간 폭 p05..p95 = x26.0
잔차 인자 프레임 간 폭 p05..p95 = x60.9
```

프레임 순위를 정하는 힘의 대부분은 **Jacobian(시점 기하)** 에서 온다.
그리고 그 인자는 GT·K·치수만으로 학습 전에 계산되는 **고정 가중치**다.

이 사실의 두 얼굴을 모두 적는다.

- 좋은 쪽: 튜닝 없이 기하로부터 유도되는 hard-regime 가중이다. 저앙각·원거리
  프레임의 코너 오차에 자동으로 더 큰 비용을 매기며, 이는
  `FINAL_DECISION.md` 가 병목으로 지목한 저앙각 코너 위치추정과 정확히 겹친다.
- 나쁜 쪽: 그렇다면 이 항의 상당 부분은 **프레임 단위 재가중**과 구별되지 않는다.
  그래서 Stage A 에 등방 대조군 `A2b` 를 반드시 넣는다 (METHOD_LOCK_DRAFT §arm).

## 5. ★ 이 감사가 증명하지 않는 것

**surrogate 가 오차와 상관한다는 것은, 그 surrogate 로 학습하면 오차가 준다는
뜻이 아니다.** 이 저장소는 정확히 그 반대를 네 번 이상 독립적으로 재현했다.

```
diffpnp-track-closed-both-sides            "예측 2D 에 더 잘 맞출수록 실제 pose 가 나빠진다" 4 회
aggregate-scale-statistic-does-not-transfer-to-pose   3 회 독립 재현
corner-residual-systematic-not-noise       오차를 줄여도 pose 로 전이 안 됨
scale-lever-real-but-unpredictable         레버는 실재하나 예측 보정은 악화
```

그리고 `FINAL_DECISION.md` 의 `DO_NOT_RUN` 은 **"새 loss 항"** 을 명시적으로
금지한다. Stage 0 의 결과는 그 금지를 뒤집지 않는다 — 다만 금지가 세워질 때
없었던 증거(잔차 크기 대조군이 음수라는 것)를 하나 추가할 뿐이다.
어느 쪽을 택할지는 사용자 결정이다.
