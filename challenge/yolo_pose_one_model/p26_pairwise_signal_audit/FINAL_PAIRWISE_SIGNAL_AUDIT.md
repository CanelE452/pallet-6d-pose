# PAIRWISE SIGNAL AUDIT — Y0 frozen, training-0

[CONTRACT]
checkpoint    runs_posecls_g38/Y26_G38_Y0_VANILLA_30EP_SEED42/weights/last.pt
sha256        37f904b975db3e95297af5acb51f6e99360f4b59245cef04d0511af3f5a189b1   기대값과 동일 [확인]
commit        96ddf1967ecee2759e5d36578a84f2e4eb021efe   기대값과 동일 [확인]
ultralytics   8.4.60 · torch 2.1.1+cu118 · RTX 3080
guards        training 0 · optimizer 0 · backward 0 · autograd 0 · model.train() 0 · fuse 0
              gradient 는 해석식(g_rank=sigmoid(-delta), g_bce=sigmoid(s)-target)만 사용

[PAIR COUNTS]
train5k    5,000 / 5,000   (sha256(stem) 정렬 앞 5,000, membership sha16 eb8e73c090b92fb4)
val1998    1,998 / 1,998
realdev128   128 / 128     ★ SECONDARY diagnostic only — 선택에 사용 안 함

[DELTA]   delta = s_pos - s_neg (raw logit)
```
                      train5k   val1998   realdev
delta < 0              0.0182    0.0235    0.1797
delta <= 0.5           0.0276    0.0345    0.1953
delta <= 1             0.0366    0.0450    0.2266
delta <= 2             0.0626    0.0656    0.2812
median                12.0407   12.0637    5.1219
p10                    3.6671    3.7326   -2.1385
p90                   14.3866   14.3856   11.4506
```
source 의 positive 는 최고 unassigned anchor 보다 **중앙값 12 logit** 높다. real 은 5.12 이고
10 분위가 음수다 — 같은 모델인데 source 에서는 경쟁이 거의 없다.

[VIRTUAL RANK GRADIENT]   g_rank = sigmoid(-delta)
```
                      train5k   val1998   realdev
g >= 0.50              0.0182    0.0235    0.1797
g >= 0.25              0.0384    0.0465    0.2266
g >= 0.10              0.0682    0.0706    0.2969
g >= 0.05              0.0836    0.0846    0.3672
mean                   0.0285    0.0325    0.1978
median                 0.0000    0.0000    0.0060

gradient mass  rank/stock-BCE   train5k 0.619 · val1998 0.671 · realdev 1.301
               mean g_rank      0.0285 / 0.0325 / 0.1978   (real 이 6~7 배)
```

[NEGATIVE TYPE]   ★ 이 audit 의 핵심
```
                          DUPLICATE   NEAR    FAR    NEAR+FAR      n
train5k  overall              0.427   0.071  0.501     0.573    5000
         hard delta<=2        0.952   0.016  0.032     0.048     313
         inverted delta<0     0.945   0.000  0.055     0.055      91
val1998  overall              0.412   0.068  0.520     0.588    1998
         hard delta<=2        0.870   0.076  0.053     0.130     131
         inverted delta<0     0.766   0.170  0.064     0.234      47
realdev  overall              0.516   0.125  0.359     0.484     128
         hard delta<=2        0.250   0.139  0.611     0.750      36
         inverted delta<0     0.130   0.174  0.696     0.870      23
```
**source 의 hard pair 는 87~95% 가 같은 물체의 duplicate 이고, real 의 hard pair 는
75~87% 가 진짜 distractor(NEAR/FAR)다.** 종류가 반대다.

[SOURCE LEVEL]   frequency 만 확인 (T5 는 이미 기각, 다시 열지 않음)
```
train5k  same n=2886 delta_med 12.02 · cross n=2114 delta_med 12.05
val1998  same n=1109 delta_med 12.10 · cross n= 889 delta_med 12.04
combo (val)  P5->P5 928 · P5->P4 383 · P4->P4 172 · P5->P3 164 · P4->P3 164 · P4->P5 159
```
same/cross 의 delta 분포가 사실상 같다 — level 은 이 신호의 축이 아니다.

[SYNTHETIC POSTPROCESS 대조]
TAL audit artifact: postprocess 까지 살아남은 S+/SW 쌍 210/1,998 = 10.5%, rank failure 5.2%.
raw anchor 수준: delta<=2 가 **6.6%**, delta<0 이 **2.4%**.
→ "postprocess 로는 적어 보이지만 raw anchor 에는 많다" 는 기대는 **성립하지 않는다.**
   raw 수준에서도 신호가 그만큼 얇다.

[TRAIN/VAL STABILITY]
```
              G1      G2      G3
train5k     False   False   False
val1998     False   True    True
```
G1 은 두 split 모두 실패(0.0682 / 0.0706 < 0.10). G2·G3 는 서로 반대다.

[GATE]
G1  frac(g_rank>=0.10) >= 0.10        val 0.0706  → FAIL   (train 0.0682 도 FAIL)
G2  frac(delta<=1)>=0.05 OR <0 >=0.02 val 0.0450 / 0.0235 → PASS (train 0.0366 / 0.0182 FAIL)
G3  hard(delta<=2) NEAR+FAR >= 0.10   val 0.130 → PASS      (train 0.048 FAIL)

**PAIRWISE_SIGNAL_TOO_WEAK_OR_MISMATCHED**
(부가: SOURCE_SIGNAL_UNSTABLE — G2·G3 가 두 split 에서 반대)

→ **pairwise loss 30ep STOP.** G1 이 두 split 모두 실패하므로 안정성과 무관하게 결론은 같다.

[NEXT]  GO 가 아니므로 loss 설계안을 내지 않는다.

이번 audit 이 새로 확정한 것은 **원인이 loss 가 아니라 source 데이터에 있다**는 것이다.

```
representation   가른다      (cls_pen 0.910)          — 병목 아님
readout          안 버린다   (linear 0.951 vs actual 0.972) — 병목 아님
TAL target       옳다        (RW target 항상 0)        — 병목 아님
source 신호      거의 없다   delta 중앙값 12, hard 6.6%, 그 중 87~95% 가 duplicate
```

즉 pointwise BCE 가 within-image ranking 을 못 가르친 게 아니라,
**source 가 그 선택을 거의 제시하지 않고, 제시할 때조차 real 과 다른 종류를 제시한다.**

다음 한 걸음(설계·학습 아님, training-0 특성화):
real 의 hard pair 를 지배하는 FAR_DISTRACTOR(IoU<0.1, real hard 의 61%, inverted 의 70%)가
무엇인지 특성화한다 — 어떤 배경·구조물이 pallet 으로 오인되는가, 그 분포가 G38 synthetic
배경에 존재하는가. 이 결과가 나와야 데이터 축의 개입 지점을 정할 수 있다.

[VERIFY]
training = 0 · optimizer = 0 · backward = 0 · autograd 미사용
checkpoint sha before/after   37f904b975db3e95297af5acb51f6e99360f4b59245cef04d0511af3f5a189b1 / 37f904b975db3e95297af5acb51f6e99360f4b59245cef04d0511af3f5a189b1   동일
mtime before/after            2026-08-25 16:07:28.570120660 +0900 / 2026-08-25 16:07:28.570120660 +0900   동일
