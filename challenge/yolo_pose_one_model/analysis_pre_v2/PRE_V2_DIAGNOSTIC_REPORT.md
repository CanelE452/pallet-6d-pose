# PRE-V2 진단 — WEAK_PASS 가 정말 data gap 인가

학습 0 / 새 architecture 0. 기존 checkpoint 와 REAL DEV 만 사용.

## ★ STOP RULE 이 걸렸다 — V2 spec 을 자동으로 바꾸지 않는다

```
D1  availability 잠재 +18.6pp   (기준 +10pp)
D3  gross frame 의 54.2% 가 semantic 문제로 설명됨   (기준 30%)
```

**VERDICT = `INFERENCE_PIPELINE_ISSUE_FOUND` + `LABEL_SEMANTIC_ISSUE_FOUND`**

data gap 이 **유일한** 원인이라는 가설은 약해졌다. 데이터와 무관하게 존재하는
문제가 둘 있다.

---

## D1 — NO_BOX 의 2/3 는 검출 실패가 아니다

원래 NO_BOX 45/161 (640, conf 0.4). 같은 checkpoint 로 conf 만 낮추면:

```
conf      imgsz640   imgsz960   imgsz1280      ← IoU>=0.5 올바른 후보가 뜨는 수
0.001         30         29         26
0.01          24         26         25
0.05          19         22         20
0.10          18         21         20
0.40           0         13         11
```

```
LOW_CONF_CORRECT_CANDIDATE     30   66.7%   ← 모델은 찾았는데 0.4 아래로 점수를 줬다
WRONG_OBJECT_TOP1               8   17.8%
MULTIPLE_CANDIDATE_AMBIGUITY    7   15.6%
```

**confidence calibration 문제다.** 다만 positive DEV 뿐이라 여기서 threshold 를
내리면 FP 를 못 보고 정하는 것이다 — real negative 와 함께 정해야 한다.

## D2 — 해상도는 병목이 아니다

```
imgsz   OPEN56 avail / IoU>=0.5     CHALLENGE avail / IoU>=0.5
 640        0.893 / 0.786               0.629 / 0.486
 960        0.839 / 0.750               0.676 / 0.467
1280        0.696 / 0.446               0.533 / 0.229
```
960 은 availability 를 +4.7pp 올리지만 **올바른 검출 비율은 오히려 −1.9pp**,
1280 은 전면 붕괴. 학습 데이터 이전에 추론 해상도 병목이 있다는 가설은 기각된다.

## D3 — near/far role 혼동이 절반이다 ★

gross(R>10°) 24 장에서 각 permutation 의 최선을 세면:

```
identity        11   45.8%
near_far_swap   11   45.8%
top_bottom       2    8.3%
role confusion  54.2%
```

**앞면/뒷면을 뒤집으면 오차가 줄어드는 프레임이 절반**이다. 저앙각에서 앞뒤 면이
겹쳐 보이는 것과 방향이 맞는다.

★ oracle permutation 은 **진단 전용**이다. main metric 으로 쓰지 않는다.

이 결과는 V2 spec 을 바꿀 수 있다 — 단순 appearance 확대가 아니라
**role 을 구분 가능하게 하는 geometry/viewpoint 다양성**이 필요하다는 뜻이다.

## D4 — 나쁜 점 하나가 오염시키는 게 아니다

```
                  OPEN56   CHALLENGE
all 8 corners      2.254      5.629
top7 (conf 상위)    2.186      5.548
top6               2.603      6.979
top5               3.143      6.927
top4              14.308     17.399
near only         11.761     14.268
far only          23.131     29.033
top only           2.266      5.628
bottom only        2.269      5.547
```
top7 은 전체와 사실상 같다. 한 점을 빼서 회복되지 않는다.
**far only 가 near only 보다 2 배 나쁘다** — D3 의 near/far 혼동과 정합한다.

## D5 — solver 는 멀쩡하다

```
              예측 점    GT 점
SQPNP          3.484     0.038
SQPNP+refLM    3.276     0.000
ITERATIVE      3.276     0.000
EPNP+refLM     3.420     0.000
```
GT 점을 주면 네 solver 모두 0° 다. **문제는 전적으로 point prediction 이다.**

## D6 — 평가 자체는 믿을 만하다

```
set         n    GT 재투영 median / p90    GT2D->pose R 불일치
outside    22        0.47 /  1.94 px            0.000 deg
noapril    12        0.42 /  1.65
cad        22        1.99 /  4.54
pallet07   27        2.46 /  4.73
pallet09   36        1.05 /  2.52
night08    17        1.55 /  3.75
night09    25        1.91 /  6.83
```
**night/target 세션만 residual 이 큰 패턴은 없다.** 모델 실패를 calibration 이나
annotation 탓으로 돌릴 수 없다. 평가 노이즈 바닥은 ~2px 다.

### D6D — 치수가 곧 translation 바닥이다

```
치수 오차   R median      t median
   +1%      0.0000 deg    0.0217 m
   +2%      0.0000 deg    0.0435 m
   +5%      0.0000 deg    0.1086 m
```
`5cm5deg` 의 t 기준이 0.05m 인데 **치수 2% 오차만으로 0.0435m** 가 생긴다.
회전은 전혀 영향받지 않는다. 실측 치수 정확도가 t 성능의 상한을 정한다.

## D7 — 낮은 신뢰도를 버리면 tail 이 줄어든다

```
Spearman vs R 오차
  box_conf            -0.433   ← 가장 강함
  kp_conf_4th_lowest  -0.333
  kp_conf_mean        -0.278
  pnp_reproj          +0.255

pnp_reproj 로 coverage 를 줄이면 (CHALLENGE)
  100%  R 5.63  5cm5 0.24
   90%  R 5.13  5cm5 0.27
   80%  R 4.13  5cm5 0.31
   70%  R 3.87  5cm5 0.33
```
selective prediction 이 작동한다. 다만 **real negative 없이 threshold 를 확정하지
않는다.**

## D8 — real negative 후보는 있다, 측정은 안 했다

배포영상 911 장 중 FT 가 negative 로 쓴 259 장. **PAPER_GENERIC_V1 은 real 을 한
장도 안 봤으므로 미사용** → `REAL_NEG_DEV_CANDIDATE`.

FP/image 는 이번에 재지 않았다 — threshold 확정과 묶이면 이 셋이 소진되므로
사용자 승인 후 별도로 한다. 그리고 `yolo26n_ft` / `yolo26m_ft` 는 이 negative 를
학습했으므로 그 둘의 FP 와 섞어 비교하면 안 된다.

---

## 종합

```
검출 실패    아니다 — 2/3 는 confidence 문제 (D1)
해상도       아니다 (D2)
solver       아니다 (D5)
GT/K         아니다 (D6)
나쁜 점 하나  아니다 (D4)

role 혼동    ★ 맞다. gross 의 54.2% (D3)
confidence   ★ 맞다. NO_BOX 의 66.7% (D1)
data gap     여전히 남아 있다 — 위 둘을 고쳐도 target/night 격차가 사라진다는
             증거는 없다
```

## NEXT — 하나만

**`D3 role 혼동을 V2 spec 에 반영`.**

지금 V2 는 geometry(형태 다양성) × appearance(조명) 2x2 인데, D3 는 세 번째 축이
필요하다고 말한다 — **앞뒤 면을 구분 가능하게 만드는 viewpoint/구조 다양성**.
저앙각에서 near/far 가 겹쳐 보이는 것이 role 혼동의 유력한 기전이고 [추정],
그렇다면 렌더 spec 의 elevation 분포와 비대칭 구조(한쪽 면에만 있는 특징)가
appearance 확대보다 우선한다.

D1(confidence)은 데이터가 아니라 추론 계약 문제라 V2 와 별개 트랙이고,
real negative 확보 후에 다뤄야 한다. 지금 threshold 를 내리면 FP 를 못 보고 정하는 것이다.
