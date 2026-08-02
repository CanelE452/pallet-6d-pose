# CORNER REPLACEMENT GATE — STOP (12개 중 10개 FAIL)

> scratch 가 아니라 ep57 initialization.  canonical Stage-B 6 roots + 60:40 sampler 그대로.
> last VGG block + belief stage 4~6 + proposal branch 공동 학습.  정확히 5 epoch,
> checkpoint selection 없음, epoch5 고정.  graph/edge/mask/DiffPnP 미사용.
> N87 은 epoch0/epoch5 두 번만 본 mechanism screen 이며 final-test 가 아니다.

```
FAIL  1 F2 far median -15%                          -0.0075
FAIL  2 F2 far signed bias -20%                      0.0329
FAIL  3 >50px tail -20%                             -0.1417
FAIL  4 paired improved > worsened                 -12.0000
FAIL  5 PnP success >= 72/87                        69.0000
FAIL  6 reproj -10%                                 -0.1041
FAIL  7 near median <= +5%                           0.1147
FAIL  8 no new PnP failure                           2.0000
FAIL  9 no new >50px                                17.0000
PASS  10 no new NaN                                -20.0000
FAIL  11 gate not collapsed                          0.0000
PASS  12 C1-base no catastrophic regression         -1.0000

GO/STOP: STOP
```

## [관찰] 네 경로 비교 (N87)

```
arm            F2 far   signed   near     >20px  >50px  PnP    reproj    yaw
               median   bias     median   tail   tail   /87    median    median
──────────────────────────────────────────────────────────────────────────────
C0 (ep57)       44.59    20.58    6.88     203    120    70    23.16     6.03
C1-base         44.93    19.91    7.67     215    137    69    25.57     5.97
C1-proposal    160.34    71.57   18.97     462    387    87   108.30    42.59
C1-refined      44.93    19.91    7.67     215    137    69    25.57     5.97
```

## [현상 1] refined = base — replacement 가 한 번도 작동하지 않았다

```
epoch5 gate:  min 1.13e-09   median 3.53e-09   max 1.99e-08
max |base - refined| : corner error 3.3e-08 px, reproj 3.1e-05 px
```

[확인] g ~ 1e-9 이므로 `H_ref = (1-g)H_base + g*Q` 는 수치적으로 H_base 다.
C1-refined 와 C1-base 가 모든 지표에서 같은 이유이며, **이번 screen 은 replacement
mechanism 을 측정하지 못했다.**

### 왜 붕괴했나 — 사전 고정 상수들의 산술적 귀결

calibration(사전 규칙: weighted ≈ L_DOPE 의 20%)이 낸 값:

```
median  L_DOPE 0.001111   L_proposal 6.0525   L_refined 0.7233
lambda_proposal = 3.67e-05      lambda_refined = 3.07e-04
lambda_gate     = 1.00e-02  (사전 지정 상수)
```

[확인] gate 패널티 계수 0.01 이 refined 항 계수 3.07e-04 보다 **33배** 크다.
g 를 키워서 얻는 L_refined 이득(3.07e-04 x O(1))이 g 를 키우는 비용(0.01 x g)을
이길 수 없으므로 **g→0 은 최적해**다.  실제로 epoch 1 의 step 400 에서 이미 0 이 됐다.
학습이 실패한 게 아니라 지정된 목적함수가 그 답을 요구했다.

[확인] 결과를 보고 loss weight 를 바꾸는 것은 금지 항목(8·F2)이므로 조정하지 않고
그대로 5 epoch 을 완주한 뒤 보고한다.

## [현상 2] proposal branch 는 학습됐지만 decoder 와 맞지 않는다

```
raw L_proposal : 6.05(calib) -> 1.30(ep1) -> 1.15 -> 1.08 -> 1.04 -> 1.02(ep5)
proposal peak  : min 0.435   median 0.520   max 0.695
base peak      : 0 근처 ~ 1.0
```

[확인] loss 는 6.05 → 1.02 로 실제로 내려갔다.  branch 가 아무것도 못 배운 게 아니다.
[확인] 그런데 `sigmoid(Q)` 는 **0.435 아래로 내려가지 않는다**.  decoder threshold 가
0.3 이므로 50x50 전 cell 이 threshold 를 통과하고, NMS 가 배경에서 peak 를 잡는다.
그래서 C1-proposal 의 far median 이 160px, reproj 108px 이면서도 PnP 는 87/87 "성공"한다
— 항상 8 점을 내놓기 때문이지 맞아서가 아니다.

★ 교훈: **operating range 를 맞추는 것과 operating point 를 맞추는 것은 다르다.**
belief range 감사(min -0.030, max 1.004)는 통과했지만, decoder 가 실제로 요구하는
"배경이 0 근처인 희소한 map" 은 sigmoid 로 만들어지지 않는다.

## [현상 3] 후반 fine-tuning 자체가 real 에서 소폭 악화

```
C0 -> C1-base :  reproj 23.16 -> 25.57 (+10.4%)   PnP 70 -> 69
                 >50px tail 120 -> 137            near median 6.88 -> 7.67 (+11.5%)
                 F2 far median 44.59 -> 44.93     signed bias 20.58 -> 19.91 (-3.3%)
```

[확인] 5 epoch 학습으로 signed bias 만 3.3% 줄었고 나머지는 모두 악화됐다.
synthetic 학습 loss 는 계속 내려갔다(L_DOPE 0.001088 → 0.001018).
**synthetic 에서 내려간 loss 가 real N87 로 전이되지 않았다** — PPD long-run 과 같은 형태.

## [Component 판정]

Phase F1 의 해석 규칙을 적용한다.

```
Trainable corner feature   REJECT        (C1-base 가 C0 대비 악화)
Global corner proposal     REJECT        (decoder 와 호환되지 않는 operating point)
Replacement gate           INCONCLUSIVE  (g~1e-9, 한 번도 작동하지 않아 시험되지 않음)
Corner proposal replacement STOP
```

Final path: **base DOPE** (ep57 유지, 변경 없음).

Phase F2 에 따라 즉시 STOP.  epoch 추가·LR/loss/gate 수정 없음.

## [지지 증거]

- [확인] baseline 완전 재현 후 학습 시작(strict 87 / 87 / 70 / yaw 6.025216 / reproj 23.161629).
- [확인] canonical 6 roots·60:40 sampler 를 trainer 의 `build_training_loader` 로 그대로 재사용.
- [확인] 초기 identity: max|refined-base| = 0.0078, gate mean 0.00995.
- [확인] L_proposal 이 6.05→1.02 로 내려갔으므로 branch 는 학습 가능했다.

## [반증 증거 / 한계]

- [확인] **gate 가 붕괴해 replacement 를 시험하지 못했다.**  "replacement 가 나쁘다"가
  아니라 "이 목적함수 아래서는 켜지지 않는다"까지만 말할 수 있다.
- [확인] 5 epoch 는 방향성만 본다.  더 긴 학습에서 달라질 가능성을 부정하지 않는다.
- [확인] N87(87 frame, F2 35)은 architecture go/no-go 전용이며 일반화 수치가 아니다.
- [확인] mixed_v8_train 은 dimensions_m 이 없어 query 의 dims 가 0 으로 들어갔다
  (전체 draw 의 약 28%).  corner-ID embedding 은 유지되지만 aspect 조건화는 그만큼 약하다.

## [다음 admissible experiment]

1. **proposal transform 을 decoder operating point 에 맞추기** — sigmoid 대신
   peak-normalized spatial softmax 등 배경이 0 으로 가는 변환.  이것 없이는
   C1-proposal 수치가 branch 품질을 반영하지 않는다.
2. **gate 상수 재설계** — λ_gate 와 λ_ref 를 같은 스케일에 두거나, gate 패널티를
   L_refined 개선량 대비 상대값으로 정의.  현재 조합은 g=0 을 최적해로 만든다.
3. 위 둘을 고친 뒤 **동일 프로토콜 재실행**.  이번 결과로 threshold 를 바꾸지 않는다.
4. C1-base 악화는 synthetic→real 전이 문제이므로, 학습을 늘리는 방향보다
   real 분포(저앙각) 확보가 선결일 가능성이 높다 [추정].
