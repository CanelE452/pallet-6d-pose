# PGBC COMPONENT DECISION — 세 gate 전부 FAIL, 구현 중단

> 기존 DOPE 전체 frozen.  학습 0회.  세 gate 는 기존 mechanism cache(87 frame,
> checkpoint SHA c0055fe7...) 위에서 계산했고 G1 만 shared feature 를 위해 frozen
> forward 87회를 추가했다.  edge/mask/line/vector/offset/voting branch 없음.
> N87 은 mechanism capability screen 이며 final-test 가 아니다.

## [관찰] — gate 결과

```
gate                       metric                                값       기준     판정
──────────────────────────────────────────────────────────────────────────────────────
G0 RESIDUAL_CAPACITY       far corner 중 error 50% 감소 비율      0.283    >=0.80   FAIL
G1 FEATURE_OBSERVABILITY   fold 최소 AUC (GT vs wrong peak)      0.591    >=0.75   FAIL
                           fold 최소 GT>wrong pair 비율           0.600    >=0.70   FAIL
G2 GRAPH_INFORMATION       far median error 감소                  0.018    >=0.20   FAIL
                           far signed bias 감소                  -0.023    >=0.20   FAIL
```

baseline 재현 [확인]: strict 87 / GT-2D PnP 87 / predicted PnP 70 /
yaw median 6.025216° / fixed indexed reproj 23.161629px — 전부 기대값 일치.

## [G0] fixed ±0.25 additive residual — 산술적으로 불가능

초기 오차 구간별로 나누면 실패 지점이 정확히 드러난다.

```
 bin      n   belief@wrong  belief@GT   ±0.25 후 GT 승   argmax→GT   err median
──────────────────────────────────────────────────────────────────────────────
 <5px     4      0.953        0.953        1.000          1.000     2.1 → 2.1
 5-20px  24      0.900        0.727        1.000          1.000    11.5 → 10.5
 20-50px 49      0.822        0.407        0.735          0.980    32.7 → 14.8
 >50px   61      0.778        0.003        0.033          0.131    76.3 → 76.3
```

[확인] **>50px 구간에서 GT 위치의 belief 는 0.003 = 사실상 0 이다.**
±0.25 를 더하고 빼면 GT 0.253 vs wrong 0.528 이므로 wrong 이 그대로 이긴다.
이 구간이 뒤집히려면 amplitude > (0.778-0.003)/2 = **0.39** 가 필요하다.
0.25 는 통계적으로 부족한 게 아니라 **산술적으로 불가능**하다.
그리고 이 구간(61/110)이 바로 PGBC 가 겨냥하는 confidently-wrong 레짐이다.

[확인] 20-50px 구간은 98% 가 argmax 를 GT 로 옮긴다.  즉 residual 자체는 작동한다.
못 하는 건 **belief 가 GT 에서 아예 반응이 없는 경우**다.

[확인] 격자 한계: 50×50 grid 의 1 cell = 12.8×9.6px 이라, top1 read-out 은
GT cell 에 도달해도 오차가 ~1 cell 남는다(20-50px 구간 refined median 14.8px).

### ★ Phase C1 spec 결함 (구현 전에 알아야 함)

지시문 C1 은 `T_global = 1.0` 의 full-map spatial softmax 를 규정한다.
belief 값이 [0,1] 이므로 exp(0.82)/exp(0.05) = **2.16배**뿐이고, softmax 는 거의 균등해진다.

```
T_global    base belief 의 global coordinate median error (F2 far)
──────────────────────────────────────────────────────────────────
 1.00                144.5 px      <- 지시문 spec
 0.30                136.3 px
 0.10                 54.7 px
 0.05                 45.5 px
 0.01                 42.1 px      <- top1(42.8px)로 수렴, 이득 없음
```

[확인] T=1.0 은 base 에서조차 144.5px 이다.  G0 의 global read-out 이 전 구간 0.000 인 것은
residual 무능이 아니라 **이 spec 때문**이다.  T 를 낮춰도 top1 로 수렴할 뿐 이득이 없다.

## [G1] frozen feature 는 "어느 corner 인지"를 모른다 — 종료 사유

F2 프레임에서 base 가 20px 이상 틀린 corner 159 쌍(far 110 / near 49).
positive = GT 위치의 F50(128ch), negative = 현재 wrong peak 위치의 F50.
session-grouped 3-fold, corner ID one-hot + W,D,H 동반.

```
probe                          fold AUC                  GT>wrong
────────────────────────────────────────────────────────────────────────
linear (지시문 spec)            0.613 / 0.642 / 0.591     0.600 / 0.691 / 0.633
MLP 64-hidden (진단)            0.585 / 0.642 / 0.581     0.618 / 0.673 / 0.592
linear, GT vs 무작위 위치 (진단)  0.864 / 0.769 / 0.858     0.873 / 0.745 / 0.857
control: feature 제거            0.501 / 0.497 / 0.502     —
```

[확인] control 이 정확히 우연(0.50)이다.  corner ID 와 dims 는 pair 안에서 동일하므로
구조상 판별력이 0 이어야 하고, 실제로 그렇다.  따라서 0.59~0.64 는 **전부 F50 에서 온다** —
probe 는 정직하다.

[확인] **비선형으로 올려도 개선되지 않는다** (MLP 0.581~0.642 ≈ linear).
선형 probe 가 약해서 나온 결과가 아니다.

[확인] 반면 **GT vs 무작위 위치는 0.77~0.86** 으로 분명히 우연 위다.
→ frozen feature 는 "팔레트 부근인가 배경인가"는 알지만
**"이 위치가 corner k 인가"는 모른다.**  wrong peak 도 팔레트 위에 있기 때문이다.

[확인] 이것이 G1 FAIL 의 의미다.  frozen F50 을 읽는 corrector 는
**어디로 옮겨야 하는지 알 수 있는 정보 자체가 없다.**

## [G2] 나머지 7 corner 도 여덟 번째를 모른다

predicted 7 corner(GT 미사용)로 PnP → 제외한 corner 재투영.

```
far corner 138    median 44.59 → 43.80 px  (감소 1.8%,  기준 20%)
                  signed bias 20.58 → 21.06 px  (오히려 2.3% 증가)
```

[확인] F2 프레임은 나머지 7개도 함께 틀려 있어서, 이웃으로부터 복원할 신호가 없다.
graph message passing 이 기여할 여지가 이 데이터에서는 확인되지 않는다.

## [Component ablation]

```
component                        판정
────────────────────────────────────────────────────────────────
fixed ±0.25 additive residual    REJECT  (G0: >50px 구간 산술적 불가)
global spatial softmax T=1.0     REJECT  (spec 결함, base 조차 144.5px)
frozen-F50 global proposal       REJECT  (G1: AUC 0.59~0.64, 비선형도 동일)
12-edge graph message passing    REJECT  (G2: 1.8% 감소)
Pallet Graph Belief Corrector    REJECT
```

### 사용자 사전 지정 fallback 에 대하여

G0 FAIL 시 `P_ref = (1-g)·P_base + g·Q_global` 로 전환하도록 사전 지정돼 있었다.
[확인] 이 구조는 **G0 은 우회하지만 G1 은 우회하지 못한다.**
convex blend 가 confident-wrong peak 를 완전히 덮을 수 있다는 것은 맞지만,
`Q_global` 을 만들 재료가 frozen F50 이고 G1 이 그 F50 에 corner 판별 정보가 없다고
말하기 때문이다.  **따라서 blend 도 구현하지 않는다.**
(결과를 보고 component 를 되살리지 않는다는 규칙에 따른 판단.)

## [현재 판정]

```
Local residual C1                REJECT
Global-only C2                   REJECT
Global+graph C3                  REJECT
Pallet Graph Belief Corrector    REJECT
Predicted-seed DiffPnP           NOT RUN  (PGBC 미통과)
```

Final path: **base DOPE only** (변경 없음).

16-frame overfit / session-grouped 3-fold / DiffPnP micro-screen 는
"통과한 component 만 구현한다"는 규칙에 따라 **실행하지 않았다**.

## [지지 증거]

- [확인] baseline 완전 재현(yaw 6.025216, reproj 23.161629) — 측정 경로 정상.
- [확인] G1 control 이 정확히 0.50 — probe 무결성 확인.
- [확인] G0 20-50px 구간은 argmax 98% 이동 — residual 메커니즘 자체는 작동.
  즉 "아무것도 안 움직였다"가 아니라 **특정 레짐에서만 불가능**하다는 국소적 결론.

## [반증 증거 / 한계]

- [확인] 표본이 얇다: G0 far 138 corner(F2 35 frame), G1 159 pair, G2 far 138.
- [확인] G0 는 GT 를 쓰는 oracle 이다.  capacity 상한이지 학습 가능성 증명이 아니다.
  상한이 못 넘었다는 것이 결론이므로 방향은 보수적이다.
- [확인] G2 는 F2 프레임 한정이라, 나머지 7개도 틀린 최악 조건이다.
  clean frame 에서 graph 가 유용할 가능성은 이 시험이 부정하지 않는다.
- [추정] G1 은 vgg trunk 출력(50×50, 128ch) 하나만 조사했다.  더 얕은 층의
  고해상도 feature 에는 다른 답이 있을 수 있다.

## [다음 admissible experiment]

1. **frozen 전제를 깨는 것이 정공법** — G1 이 말하는 것은 "지금 얼어 있는 표현에
   답이 없다"이다.  backbone 을 푸는 fine-tune 이 이 결론과 모순되지 않는 유일한 방향.
2. G1 을 얕은 층(고해상도) feature 로 반복 — frozen 을 유지하려면 이것이 선결.
   현재 결론은 vgg trunk 50×50 에 한정된다.
3. >50px 레짐의 belief@GT = 0.003 은 **표현이 아니라 학습 데이터** 문제일 수 있다
   (기존 진단의 저앙각 flat-view 결론과 정합).  데이터 쪽 레버.
4. 위 없이 amplitude 를 0.39 이상으로 올리거나 gate 를 재조정하지 않는다.
