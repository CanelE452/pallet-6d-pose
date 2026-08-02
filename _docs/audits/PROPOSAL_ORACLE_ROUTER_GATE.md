# Oracle router 상한 — FAIL, learned router NOT RUN

corner 별로 base/proposal 중 GT 에 가까운 쪽을 **GT 를 보고** 고른 상한.

```
arm             median  near    far     F2 far  signed  >20px >50px  PnP   reproj  prop share
C0               13.24   6.88   22.08   44.59   20.58    203   120    70   24.91      -
oracle_exact     10.69   5.51   19.00   36.94   18.18    168    95    70   20.71    17.5%
oracle_margin    10.77   5.72   19.10   36.94   18.12    170    95    70   20.53    13.5%
```

## Gate (사전 고정, margin 3px)

```
FAIL  1 f2_far -20%
FAIL  2 f2 signed bias -20%
PASS  3 tail_gt50 -20%
FAIL  4 PnP >= 74
PASS  5 reproj -10%
PASS  6 near <= +2%
PASS  8 no new NaN
```

```
1 F2 far      44.59 -> 36.94  = -17.2%   (기준 -20%)   FAIL
2 signed bias 20.58 -> 18.12  = -12.0%   (기준 -20%)   FAIL
3 >50px tail    120 -> 95     = -20.8%   (기준 -20%)   PASS
4 PnP            70 -> 70                (기준 >=74)   FAIL
5 reproj      24.91 -> 20.53  = -17.6%   (기준 -10%)   PASS
6 near         6.88 -> 5.72               (기준 +2%)   PASS
8 NaN           177 -> 177                             PASS
```

[확인] 상한은 **실재한다** — tail -20.8%, reproj -17.6%, near 도 개선.
[확인] 그러나 사전 고정 기준 4/7 을 통과하지 못했고, 특히 **PnP 가 70 → 70 으로 전혀
구조되지 않는다**.  좌표를 골라 개선해도 PnP 실패 프레임은 그대로다.
[확인] proposal 채택률은 exact 17.5% / margin 13.5% 로, 상한조차 소수 corner 에서만 이긴다.

★ 이것은 **GT 를 보고 고른 상한**이다.  학습된 router 는 이보다 나쁠 수밖에 없다.
상한이 기준에 못 미치므로 router 학습은 규정대로 금지한다.

[판정] Oracle gate **FAIL** → learned router **NOT RUN**,
proposal / replacement architecture **REJECT**, final path **base DOPE**.
