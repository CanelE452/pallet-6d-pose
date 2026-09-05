# GATE B — 국소 코너 증거 감사

학습 0. R0 예측 좌표를 중심으로 반경 **12 px** patch 안에서 고전 CV 후보를 만들고,
GT 와 가장 가까운 후보(oracle)와 GT 없이 고른 후보(prediction-only selector)를 잰다.
primary 는 supervised **visible**(visibility==2) 코너 1,594 개.
반경·생성 파라미터는 METHOD_LOCK 에 잠겨 있고 합성 도메인에서만 골랐다.

## 판정 (사전등록 임계 기준)

```
LOCAL_CORNER_HEADROOM            = STRONG
CLASSICAL_LOCAL_SELECTOR_SIGNAL  = ORACLE_ONLY_HEADROOM
```

```
기준                                    임계      실측     통과
────────────────────────────────────────────────────────────────
oracle 후보의 visible p90 개선           >= 15%    22.8%    예
   (또는 gross20 상대감소)               >= 20%    25.9%    예
GT 5px 이내 후보 존재율                  >= 60%    80.7%    예
```

## 수치 (visible 코너 n=1,594)

```
arm                          median     p90   gross20  gross40
────────────────────────────────────────────────────────────────
R0                             6.36   43.89    0.157    0.102
ORACLE_CANDIDATE               2.00   33.88    0.117    0.094
PREDICTION_ONLY_SELECTOR       7.07   45.03    0.168    0.104
```

후보 근접률 3px 0.730 / 5px 0.807 / 10px 0.857. 후보가 하나도 없는 patch 는 0.8%.
patch 당 후보 중앙값 48개.

## ★ 그런데 그 근접률은 대부분 밀도의 산물이다

반경 12 patch 는 약 25x25 = 625 px^2 이고 후보가 48 개다.
같은 patch 에 같은 개수의 **균등난수**를 뿌려 다시 재면:

```
임계        고전 CV    균등난수      lift
──────────────────────────────────────────
 3 px        0.736      0.667     +0.069
 5 px        0.814      0.799     +0.015
10 px        0.864      0.860     +0.004

최근접 거리 중앙값   고전 2.00 px   난수 2.18 px
```

lift 가 5px 에서 +0.015 다. 즉 "GT 5px 이내에 후보가 있다" 는 사실은
**Shi-Tomasi·Harris·LSD·junction 이 semantic corner 를 찾아서가 아니라,
좁은 patch 를 촘촘히 덮었기 때문**이다.

따라서 Gate B 의 oracle headroom 은
"실제 RGB 에 더 정확한 semantic corner 증거가 있다" 로 읽으면 안 되고
"반경 12 안을 촘촘히 뒤지면 GT 근처 점이 있다" 로 읽어야 한다.

사전등록 임계를 결과를 보고 바꾸지 않았다. 판정은 STRONG 그대로 두고,
그 STRONG 을 어떻게 읽어야 하는지를 여기 적는다.

## prediction-only selector 는 R0 를 악화시킨다

가중치를 학습하지 않고 네 term(coarse 거리 · corner response · 두 line 교차각 ·
예상 투영 cuboid 변 방향 일치도)의 rank 를 동일 비중 평균한 선택기다.

```
selector vs R0 (visible)   median -11.2%   p90 -2.6%   gross20 -6.4%
R0 가 >20px 인 코너 구제율   0.000
R0 가 <=10px 인 코너 훼손율  0.137
```

전부 악화다. 특히 구제율이 정확히 0 이다 — 이 선택기는 R0 의 큰 실패를 하나도 못 고쳤다.
밀도 통제와 합치면 결론이 하나로 모인다: **후보 집합 안에 정답 근처 점은 있지만,
그 점이 "코너처럼 생겨서" 있는 것이 아니라서 코너다움으로는 고를 수 없다.**

## occluded 코너 (진단용, primary 아님)

```
arm                          median     p90   gross20
──────────────────────────────────────────────────────
R0                             8.35   82.90    0.273
ORACLE_CANDIDATE               3.17   65.38    0.192
PREDICTION_ONLY_SELECTOR      11.09   78.97    0.299
```
근접률 5px 0.622 로 visible 보다 낮고, 선택기 훼손율은 0.249 로 더 높다.

## 코너별 (visible)

```
kp     n   R0 med   ORACLE med   후보 5px
──────────────────────────────────────────
 0   279     5.99         1.59      0.835
 1   269     7.32         1.91      0.829
 2   224     6.28         2.00      0.853
 3   209     7.19         2.00      0.789
 4   252     5.06         2.00      0.786
 5   247     6.39         2.00      0.789
 6    50     6.41         1.48      0.740
 7    64     6.51         2.19      0.703
```
oracle median 이 1.5~2.2 px 에서 멈추는 것은 Harris·junction 후보가 정수 픽셀 격자라
생기는 양자화 바닥이다.

## 하류에 넘기는 것

METHOD_LOCK 규칙상 LOCAL_CORNER_HEADROOM = STRONG 이므로 Gate C(국소 전문가 학습)를
진행한다. 다만 Gate C 의 결과는 위 밀도 통제를 함께 놓고 읽어야 한다 —
Gate C 가 성공한다면 그것은 "고전 CV 가 못 본 것을 학습된 국소 모델이 본다" 는 뜻이고,
실패한다면 "반경 12 안의 국소 외형에는 애초에 쓸 신호가 별로 없다" 는 뜻이다.
