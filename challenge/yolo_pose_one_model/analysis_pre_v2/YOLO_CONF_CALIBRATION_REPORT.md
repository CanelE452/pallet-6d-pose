# YOLO26N_PAPER_GENERIC_V1 — conf operating point 보정

**VERDICT = `CONF_CALIBRATION_UNRESOLVED`**  ·  **THRESHOLD = `UNRESOLVED`**

질문 하나: *conf=0.4 때문에 숨겨진 올바른 detection 을 real negative FP 를
폭증시키지 않고 얼마나 회수 가능한가.*

답: **회수되는 것의 대부분은 올바른 detection 이 아니다.** conf 를 낮추면
availability 는 오르지만 pose 는 무너진다. 그리고 recall≥0.98 은 어떤
threshold 로도 도달 불가능하다.

Hough/CIGM/F3 는 `HOUGH_TRACK_CLOSED` 판정대로 실행하지 않았다. 새 학습 0.

---

## PHASE 0 — NEGATIVE DEV CONTRACT → `REAL_NEG_DEV_V1` **유효**

```
항목                              값
────────────────────────────────────────────────────────────────
n_frames                          259
sequence                          forklift_raw_20260528_163408 (911 장 중)
pallet_present == false           전 프레임 (빈 라벨 + 전수 육안 검수)
paper_generic 학습 사용            False  (학습셋 = BROAD 합성 40,000, real 0장)
positive DEV session overlap       False
  positive DEV 촬영일               2026-04-03 / 05-13 / 05-22
  negative 촬영일                   2026-05-28
yolo26n_ft / yolo26m_ft 학습 사용   True  → 그 두 모델 FP 와 섞어 비교 금지
```

역할 고정: 이 259 장은 이 순간부터 **threshold calibration DEV** 이며
**final test 로 재사용 금지**.

### ★ 반드시 같이 읽어야 할 표본 편향

`prepare_real_ft.py` 는 negative 를 이렇게 골랐다:

```python
frames = sorted(r["frame"] for r in rows if r["max_conf"] < 0.20)
```

911 장 중 **이전 모델이 이미 conf<0.20 을 준 259 장**만 채택됐다. 팔레트가 없는데
이전 모델이 conf≥0.20 으로 오검출한 프레임은 **체계적으로 배제**됐다.

→ 여기서 잰 **FP/image 는 하한(lower bound)** 이다. 학습 누수는 아니다
(paper_generic 은 real 을 한 장도 안 봤다) — 표본 편향이다.
나머지 652 장은 팔레트 유무가 판정되지 않았다(시퀀스 median max_conf 0.81 —
대부분 실제로 팔레트가 있다).

---

## PHASE 2 — CONF SWEEP (동일 epoch60 checkpoint, PAD=100 REFLECT_101, imgsz=640)

conf=0.001 로 **한 번만** 추론해 전 후보를 덤프하고 threshold 는 오프라인으로
쓸었다 — threshold 마다 재추론하면 NMS 가 달라져 비교가 흐려진다.

```
   conf   avail   recall   r_avail   corner   R med    5cm5   FP/img  frac FP  cand
────────────────────────────────────────────────────────────────────────────────────
  0.001    1.000    0.708     0.857    17.04    4.33   0.304    5.510    0.873   1427
  0.005    0.994    0.708     0.826    17.03    4.29   0.306    2.672    0.846    692
   0.01    0.988    0.702     0.795    17.04    4.33   0.308    2.039    0.842    528
   0.02    0.963    0.689     0.764    16.77    4.23   0.316    1.587    0.799    411
   0.05    0.925    0.689     0.727    14.96    3.78   0.329    1.004    0.649    260
    0.1    0.888    0.689     0.714    14.61    3.72   0.343    0.734    0.533    190
    0.2    0.807    0.640     0.658    13.74    3.42   0.369    0.459    0.421    119
    0.3    0.770    0.615     0.627    13.46    3.34   0.387    0.305    0.286     79
    0.4    0.720    0.584     0.590    12.69    3.28   0.414    0.208    0.201     54
```

- `avail` survivor 를 하나라도 가진 positive 프레임 비율
- `recall` **top-1 survivor 의 IoU≥0.5** — 배포가 실제로 쓰는 것 (1차 지표)
- `r_avail` survivor 중 아무거나 IoU≥0.5 — 진단용 천장
- `FP/img` `REAL_NEG_DEV_V1` 기준 **하한**

conf 0.40 → 0.001 로 내렸을 때:
**availability +28.0pp**(0.720→1.000) · **recall +12.4pp**(0.584→0.708) ·
**FP/image 26배**(0.208→5.510) · **FP 있는 negative 프레임 20.1%→87.3%**.

---

## PHASE 3 — THRESHOLD RULE → **UNRESOLVED**

사전등록 규칙: *tau\* = FP/image 최소, subject to positive recall ≥ 0.98.*

**recall ≥ 0.98 을 만족하는 threshold 가 grid 에 없다.**
최대 달성 recall = **0.7081** @ conf=0.001,
그 지점의 FP/image = 5.510 (하한).

게이트를 0.95 등으로 낮추지 않는다. tradeoff 를 그대로 보고한다.
conf=0.001 에서조차 recall 0.708 이라는 것은 **threshold 가 병목이 아니라는 뜻**이다.

---

## PHASE 4 — METRICS

```
지표                           값
────────────────────────────────────────
AUROC (frame-level presence)   0.8399
AUPRC                          0.837
n positive / n negative        161 / 259
```

```
threshold        TP    FP    TN    FN   presence R   presence P   FP/img
─────────────────────────────────────────────────────────────────────────
tau=0.001        161   226    33     0      1.000        0.416    5.510
현재 0.40        116    52   207    45      0.720        0.690    0.208
```

population 별 (tau=0.001):
```
population                  n    availability   recall
────────────────────────────────────────────────────────
OPEN_56                    56       1.000      0.839
CHALLENGE_DEV_105         105       1.000      0.638
negative (REAL_NEG_DEV_V1) 259   FP 보유 87.3%
```

---

## PHASE 5 — 회수된 low-conf positive 의 pose 품질

conf 0.40 에서는 안 잡히고 0.001 에서 잡히는 **45 장**만 따로 봤다.

```
지표                기존 검출(n=116)   회수분(n=45)      배율
──────────────────────────────────────────────────────────────────────
IoU>=0.5 비율          0.810              0.444
corner median (px)      12.69               84.72          6.7x
corner p90 (px)        210.00              175.96
R median (deg)           3.28               10.81          3.3x
R p90 (deg)             60.68               39.44
t median (m)           0.0682             2.1231         31x
t p90 (m)              1.0765            18.0921
5cm5                   0.414              0.022
gross R>10 비율        0.207              0.533        +32.6pp
```

**"detect 는 회수하지만 pose 가 쓰레기" 가 확증됐다.** 회수분 45 장 중 IoU≥0.5 는
44% 뿐이고, translation median 이 **31배**(0.068m → 2.12m),
5cm5 는 0.414 → 0.022 로 사실상 0 이다.
사전등록 pose gate(gross R>10 비율 +20pp 이내)를 **+32.6pp 로 위반**.

→ brief 가 예고한 조건이 성립한다: **presence gate 와 pose-quality gate 를
분리해야 한다.** 낮은 conf 로 "팔레트가 있다" 를 알리는 것과 그 프레임의 pose 를
쓰는 것은 다른 결정이어야 한다.

---

## ★ 예상 밖 소득 — 병목은 threshold 가 아니라 **top-1 랭킹**

```
conf=0.001   recall 0.708   천장 0.857   격차  14.9pp = 24장
conf=0.01    recall 0.702   천장 0.795   격차   9.3pp = 15장
conf=0.05    recall 0.689   천장 0.727   격차   3.7pp = 6장
conf=0.2     recall 0.640   천장 0.658   격차   1.9pp = 3장
conf=0.4     recall 0.584   천장 0.590   격차   0.6pp = 1장
```

conf=0.001 에서 recall(0.708)과 천장(0.857) 사이 **14.9pp = 24 장**이,
*올바른 후보가 목록 안에 있는데 top-1 이 아니라서* 버려진다.
threshold 를 아무리 내려도 이 24 장은 회수되지 않는다 — 이건 **box ranking/NMS**
문제지 threshold 문제가 아니다.

브리프의 전제였던 "NO_BOX 45 장 중 30 장(66.7%)에 correct candidate 존재 →
+18.6pp 회수 가능" 은 *후보 존재* 기준이었다. 실제로 top-1 을 통과해 배포에
쓰이는 것은 **+12.4pp** 이고, 그 중 pose 가 쓸 만한 것은 거의 없다.

---

## PHASE 6 — ROLE CONFUSION (D3 그대로, 새 학습 0)

```
gross R>10 프레임              24
best permutation = identity     11
best permutation = near_far_swap 11   -> 회수 비율 0.4583
best permutation = top_bottom    2
role confusion rate             0.5417
```

→ BROAD_FAMILY_V2 spec 에 **`LOW_ANGLE_ROLE_DISAMBIGUATION_COVERAGE`** 항목 추가.
단 **near/far swap oracle 을 main inference 에 사용하지 않는다** (진단 전용).

---

## PHASE 7 — VERDICT

```
조건                                              결과
──────────────────────────────────────────────────────────────
positive recall >= 0.98                           미달 (최대 0.708)
FP/image deployment-safe 수준 유지                 판정 불가 (prevalence 미상)
recovered pose 가 catastrophic 하지 않음            위반 (+32.6pp)
```

**`CONF_CALIBRATION_UNRESOLVED`**

FP 기준을 결과를 보고 만들지 않았다. real negative 의 natural prevalence 를
모르므로 FP/image 자체만 보고하고 final deployment threshold 는 별도로 둔다.

### conf=0.4 는 정당화되는가

브리프의 출발 전제("conf=0.4 는 final operating threshold 로 정당화되지 않음")는
**부분적으로만 맞다.** 0.4 가 최적이라는 증거도 없지만, 낮춰서 얻는 것이
없다는 것이 이번에 확인됐다 — 0.4→0.05 로 낮추면 recall +10.5pp 를 얻고
FP/image 는 0.208→1.004(하한 기준 4.8배)로 오른다. 그 recall 증가분의 pose 는
쓸 수 없다. **threshold 는 이 모델의 레버가 아니다.**

---

## 적용 범위 (넘어서 주장하지 말 것)

- positive = `REAL_DEV_POS` 161 장(OPEN56 + CHALLENGE105), 이미 development 로
  사용됐으므로 calibration 에 사용 가능. **final test 아님.**
- negative = `REAL_NEG_DEV_V1` 259 장. **FP/image 는 하한**이며 절대값을 배포
  기준으로 쓰면 안 된다.
- synthetic 결과 아님. 단, negative 는 단일 시퀀스(outside/주간)라 야간·실내
  FP 는 측정되지 않았다.
- `yolo26n_ft` / `yolo26m_ft` 는 이 negative 를 학습했으므로 FP 비교 금지.

## 산출물

`REAL_NEG_DEV_AUDIT.json` · `YOLO_CONF_SWEEP.json` ·
`YOLO_CONF_SWEEP_PER_FRAME.csv` · `plots/PR_CURVE.png` ·
`plots/FP_VS_RECALL.png` · `plots/RECOVERED_LOWCONF_POSE.png` ·
스크립트 `cc_dump.py` · `cc_sweep.py`
