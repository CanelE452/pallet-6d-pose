# PAPER_STAGE_A · PART 1 — paper_base_v2 preprocess (padding) ablation

**목적**: paper_base_v2(무패딩 학습: aspect resize + `truncation_aug_prob=0.0`)의
논문 정량 **main protocol** 을 전처리 4-way ablation 으로 확정.

- weights: `weights/paper_base_v2/final_net_epoch_0060.pth` (scratch 60ep, camera_dynamic_0123_v4)
- eval infra: `eval_capturecad_b2.eval_frame` (order-free Hungarian corner, `solve_pose`
  order-free W/D, honest full-8 reproj vs GT `projected_cuboid`, per-frame K). 모든 variant 동일.
- 실행: `scripts/stage0/paper_stage_a_pad_ablation.py` → `eval.json` / `pad_ablation_summary.md`
- **A_nopad = pad0** (`pad_frame` no-op → `preprocess` 400/min **aspect** resize, squash 아님)
  / **B_pad50 / C_pad100 / D_pad150** = reflect-pad, belief→orig 역매핑 `*(W+2P)/W` 후 `-P`.
- 평가셋: **filterval N=123 (주 신호, split-lock, final-test 봉인)** + handannot17 N=17 (고앙각 정성).
- ★ **cross-check 통과**: C_pad100 filterval overall = STAGE25 와 완전 일치
  (det79 / corner22.6 / rear33.5 / honest25.1 / good21.8 / gross51.6). 인프라 정합 확인.

good%=corner<10px, gross%=corner>20px, 둘 다 order-free Hungarian 매칭된 코너 per-corner 기준.

---

## filterval (N=123) — 주 신호

overall
```
variant    det%  front  rear  corner  worst2  pnp%  honest8  good%  gross%
────────────────────────────────────────────────────────────────────────
A_nopad     68   16.4  34.7   27.5    54.3    71    31.7    28.7    48.1
B_pad50     74   16.9  30.9   22.2    46.2    75    24.2    26.0    47.8
C_pad100    79   16.2  33.5   22.6    50.9    79    25.1    21.8    51.6   (=STAGE25)
D_pad150    71   18.6  34.8   23.4    55.0    72    32.6    17.4    51.9
```

per-metric winner (filterval overall):
```
metric     winner        note
──────────────────────────────────────────────────────────────
good%      A_nopad 28.7   순도(정밀 코너 비율) — no-pad 최고
gross%     A_nopad 48.1   B와 동률(47.8), C/D 최악
corner_med B_pad50 22.2   A 최악(27.5)
rear_med   B_pad50 30.9
worst2     B_pad50 46.2   A 54.3
honest8    B_pad50 24.2   A/D 최악(31.7/32.6)
det%       C_pad100 79    pad 가 검출↑
pnp%       C_pad100 79
```

truncation split (V=8 full-view N=106 / V<8 truncated N=17)
```
variant       det%  rear  corner  honest8  good%  gross%
────────────────────────────────────────────────────────
A_nopad  V8    77   34.2   27.2    28.4    28.6   47.7
B_pad50  V8    83   28.5   22.0    23.3    25.7   47.3
C_pad100 V8    86   32.4   21.8    23.7    22.1   50.5
D_pad150 V8    75   31.5   23.1    27.0    17.8   49.9
────────────────────────────────────────────────────────
A_nopad  V<8   12   64.6   60.0    70.2    31.2   62.5   ← 절단 프레임: no-pad 검출 붕괴(12%)
B_pad50  V<8   18   56.0   31.1    96.0    33.3   61.9
C_pad100 V<8   35   69.6   37.6    64.3    17.8   68.9   ← pad 검출↑(35%)
D_pad150 V<8   41   72.3   71.5    86.3    11.8   76.5
```

per-domain (핵심: 도메인마다 상반됨)
```
domain=outside (N=44)      det%  corner  good%  gross%
  A_nopad                  73    18.1    39.3   43.4   ← no-pad 압도(모든 축 최고)
  B_pad50                  84    21.8    31.1   45.9
  C_pad100                 89    30.9    26.3   51.0
domain=night (N=43)        det%  corner  good%  gross%
  A_nopad                  42    22.1    29.2   41.6   ← no-pad 검출 최악(42%)
  B_pad50                  53    15.6    33.1   37.3   ← pad50 최고
  C_pad100                 74    23.4    18.4   54.0
domain=manual (N=36)       det%  corner  good%  gross%
  A_nopad                  94    39.7    18.2   56.1   ← no-pad 검출多·median 나쁨(꼬리)
  C_pad100                 72    20.1    19.2   49.5
```

---

## handannot17 (N=17, 고앙각 편향 → 정성)
```
variant    det%  corner  honest8  good%  gross%
────────────────────────────────────────────────
A_nopad    24    7.6     28.0     76.9    3.8   ← 검출한 것은 매우 정밀(good77%/gross4%)
B_pad50    24    7.6     16.9     63.0   14.8
C_pad100   47    13.2    18.9     45.5   21.8
D_pad150   53    13.8    20.6     31.7   30.2
```
handannot17 은 no-pad 의 "검출↓ / 검출한 코너는 초정밀" 성향을 극단적으로 보여줌
(det 24% 인데 good 77% · gross 4%). N=17 소표본 → 방향성만.

---

## 판정 (honest — 예단 금지, 데이터 기준)

**클린한 "no-pad 우세" 는 데이터가 확정하지 않음.** 실제는 metric-축별 tradeoff:

- **순도(good% / gross%) 와 train/infer parity → A_nopad.** no-pad 는 정밀 코너 비율 최고
  (good 28.7, gross 48.1)이고 outside(가장 깨끗한 real 도메인)에서 전 축 압도. 이는 모델이
  무패딩(aspect)으로 학습된 parity 와 정합 — reflect-pad 는 학습 때 못 본 zoom-out+border 를
  주입해 순도를 깎는다. [확인: pad0=aspect parity, C_pad100=STAGE25 재현으로 인프라 검증]
- **중심경향(median corner / rear / honest8 / worst2) → B_pad50.** pad50 이 절단·근접 프레임의
  코너를 일부 복구해 median 을 낮춤(특히 night). 단 이는 parity 밖 test-time trick.
- **검출/PnP 커버리지 → C_pad100.** pad100 이 절단 프레임 검출률을 35%까지 끌어올림(A는 12%).
- **D_pad150 = 열등(dominated).** good% 최저·gross% 최악·C 대비 검출 이득 없음 → **폐기**.

### 권고 (paper main protocol)
1. **main 정량 protocol = A_nopad (no-pad aspect).** 근거 = (a) train/infer parity 원칙,
   (b) 코너 정밀 순도(good%↑ / gross%↓)가 "포즈가 정확할 때 얼마나 정확한가"를 직접 표현,
   (c) outside real 에서 결정적 우위. **STAGE25(pad100) 대비 filterval overall**:
   good% 21.8→**28.7**, gross% 51.6→**48.1** 개선 / 단 det% 79→68, corner_med 22.6→27.5 악화.
2. **honest caveat (반드시 병기)**: A_nopad 는 검출률(68 vs 79)과 median corner(27.5 vs 22.2)를
   비용으로 치른다 — reflect-pad 가 절단/근접(V<8, night/manual 근접) 코너를 실제로 복구하기 때문.
   즉 no-pad 는 **정밀도-우선**, pad 는 **검출/커버리지-우선**. 표본도 소규모(outside44 night43
   manual36 handannot17)라 단정 금지.
3. **C_pad100 = detection/demo 보조 variant 로만 별도 유지** (검출·PnP 커버리지 데모용). 논문
   정량 main 표에는 A_nopad, 부록/데모에 pad100 병기.
4. B_pad50 은 median-metric 최강이나 parity 밖이라 main 으로는 부적합 — robustness ablation
   행으로 보고 가능.

### STAGE25(pad100) 대비 결론
STAGE25 는 무패딩 학습 모델에 pad100 을 강제(부정합)해 paper_base_v2 를 과소평가했음이 확인됨
(good% 21.8, gross% 51.6). no-pad 재측정 시 순도 축이 개선(good 28.7, gross 48.1)되어 paper_base_v2
의 정직한 출발점은 STAGE25 숫자보다 낫다. 단 B2 와의 gap(B2 는 padding+v3/addon 학습) 자체는 여전히
존재 — 그건 학습 데이터(v3/addon replay) 차이지 전처리 문제가 아님.
```
```
