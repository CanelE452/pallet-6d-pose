# PPD real N87 one-shot — FAIL (transfer 실패, 방향이 반대)

untouched PASS arm(L0, M1)만 각각 **checkpoint 하나로 한 번** 평가했다.
N87 결과로 epoch·threshold·loss 를 바꾸지 않았다.

```
arm   pair    conditional     inversion   indexed    point-fail
              polarity acc                reproj     correct
──────────────────────────────────────────────────────────────
L0    86/86     0.023          84/86     60.69 px      1/17
M1    86/86     0.012          85/86     64.65 px      0/17

참고  S0 unsigned              30/86    155.6 px        —
      H0 frozen heatmap        26/86     16.5 px       8/17
      O0 oracle 5-class         3/86      7.39 px     17/17
```

gate 전 항목 FAIL (inversion ≤8/86 기준에 84~85/86).

## ★ 0.023 은 "성능 저하"가 아니다

우연이면 ~0.5 여야 한다.  0.023 은 **97.7% 를 반대로** 고른다는 뜻이므로
계통적 원인이 있다.

### 결정적 실험 — 평가 경로는 정상

동일 N87·동일 candidate set 에서 **oracle 5-class map 을 long-run scorer 로** 채점:

```
oracle map + long-run scorer : 86/86 = 1.000
(O0 원본 scorer = 0.965,  learned map = 0.023)
```

[확인] scorer·candidate·reference pose·좌표 규약 모두 정상이다.
**learned map 자체가 real 에서 틀린다.**

### 원인 — top↔base 스왑이 아니라 base 로의 붕괴

L0(best ep4) 예측을 GT 위치에서 읽으면:

```
GT top_width  위치 → pred base_width 0.683  vs  pred top_width 0.206   뒤집힘
GT top_depth  위치 → pred base_depth 0.781  vs  pred top_depth 0.146   뒤집힘
GT base_width 위치 → pred base_width 0.350  vs  pred top_width 0.435   혼재
GT base_depth 위치 → pred base_depth 0.667  vs  pred top_depth 0.537   OK

predicted positive 면적: base_depth 0.295 > base_width 0.242 > top_* 0.183~0.187
```

[확인] 깨끗한 스왑이 아니라 **거의 모든 곳에서 base 로 예측**한다.
그러면 base edge 가 실제 top 선에 얹히는 **inverted candidate 가 더 낮은 에너지**를
받으므로, 계통적 inversion 이 나온다.

[추정] 원인은 target 자체의 클래스 불균형으로 보인다.  target audit 에서
positive-frame rate 가 base_width 0.985 / base_depth 0.975 vs
top_width 0.685 / top_depth 0.620 이었다.  synthetic 에서는 top evidence 가
충분해 문제가 드러나지 않았고, real(94% 저앙각 edge-on)에서는 top 근거가 약해져
학습된 base-우세 prior 로 붕괴한 것으로 보인다.

[확인] 재학습·threshold 조정은 하지 않았다 (무결성 규칙 8).
