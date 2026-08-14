# Base / proposal 상보성

P-LOCAL(primary) vs C0(ep57 anchor).  유효 corner 519.

```
group             n     proposal    >3px      >10px     base med   prop med
                        better      better    better
──────────────────────────────────────────────────────────────────────────────
all              519    23.5%       18.1%      9.8%      13.24      58.75
far              270    17.8%       16.3%      8.9%      22.08     113.39
near             249    29.7%       20.1%     10.8%       6.88      11.78
F2               268    25.4%       21.6%     13.1%      27.23      96.10
weak_corner        -    (n<1 이라 행 없음)
confident_wrong  181    34.3%       32.0%     22.7%      61.58     140.23
```

[확인] confident-wrong corner(base peak>=0.5 & base err>20px)에서 proposal 이
**34.3%** 이기고 **22.7%** 는 10px 이상 이긴다.  이것이 상보성의 실체다.
[확인] 그러나 같은 그룹에서 proposal 의 median 은 140.2px 로 base 61.6px 보다 훨씬 나쁘다.
즉 proposal 은 **가끔 크게 맞고 대체로 크게 틀린다** — 고분산 신호다.
[확인] weak_corner(base peak<0.1) 행이 없다.  이 harness 의 C0 belief 에서
peak<0.1 인 corner 가 없었기 때문이며, no-response 는 belief 가 낮아서가 아니라
decoder 가 좌표를 못 내는 형태로 나타난다(C0 nan_err 177).
