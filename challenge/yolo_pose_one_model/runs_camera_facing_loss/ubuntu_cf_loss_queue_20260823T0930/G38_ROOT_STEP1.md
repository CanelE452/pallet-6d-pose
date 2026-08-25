# G38 GENERIC-ONLY — ROOT CAUSE STEP 1

```
model  composition                  scope     n   cbox     med     p90  gross
----------------------------------------------------------------------------
A42    generic 10K                  ALL     128  0.422   53.46   87.91  0.917
A42    generic 10K                  DAY     100  0.540   53.46   87.91  0.917
A42    generic 10K                  NIGHT    28  0.000     n/a     n/a    n/a

G38    generic 38K                  ALL     128  0.852   12.03   66.66  0.314
G38    generic 38K                  DAY     100  0.940   11.61   63.90  0.298
G38    generic 38K                  NIGHT    28  0.536   16.38   78.16  0.417

OLD    generic38K + target17.9K x2  ALL     128  0.969    9.68   40.99  0.222
OLD    generic38K + target17.9K x2  DAY     100  0.980    9.61   54.56  0.247
OLD    generic38K + target17.9K x2  NIGHT    28  0.929    9.84   22.76  0.125

C43    V2 10K                       ALL     128  0.797   14.28   91.98  0.401
C43    V2 10K                       DAY     100  0.840   13.61   71.76  0.387
C43    V2 10K                       NIGHT    28  0.643   17.30  145.66  0.465

FT     OLD + real FT                ALL     128  0.984    6.47   25.40  0.135
FT     OLD + real FT                DAY     100  0.990    6.55   27.53  0.141
FT     OLD + real FT                NIGHT    28  0.964    6.31   21.37  0.111

```

## G38 NIGHT candidate
```
any-cbox 0.821  top1-cbox 0.536  cand/frame 7.93  wrong 존재 86%  margin med +0.0362
```

**GENERIC_SCALE_EFFECT = STRONG**
**TARGET_ADDITION_REQUIRED = False**
median 회수율 (A42→OLD 구간에서 G38 위치) = 94.6%

★ 60ep 고정 = update 수도 함께 감소. target 제거의 순수 인과효과로 읽지 않는다.

## ★ 판정 문구 정정 (2026-08-23)

`TARGET_ADDITION_REQUIRED = False` 를 전역 결론으로 쓰지 않는다.

```
TARGET_ADDITION_REQUIRED_FOR_ALL_MEDIAN      = FALSE_BY_PREVIOUS_GATE
TARGET_CONTENT_REQUIRED_FOR_NIGHT_RANKING    = UNRESOLVED
```

ALL median 회수율 94.6% 는 DAY(100/128) 지배 지표다. NIGHT ranking 은 회수되지 않았고
(top1-cbox 0.536 vs 0.929, margin +0.036 vs +0.879), 이를 G38_EXP73916 이 판정한다.
