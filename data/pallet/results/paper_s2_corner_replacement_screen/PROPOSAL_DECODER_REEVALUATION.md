# Raw-Q decoder 재평가 — interface 는 원인의 일부일 뿐이었다

> 재학습 0 step.  epoch-5 checkpoint(SHA aad97f6b...) 고정, base ep57(c0055fe7...) 고정.
> 이전 평가가 `sigmoid(Q)` + 절대 threshold 0.3 을 썼는데 Q 는 spatial-softmax objective 로
> 학습됐으므로 절대값에 의미가 없었다.  여기서는 **raw Q 만** 읽는다(sigmoid·threshold 없음).
> N87 은 mechanism screen 이며 final-test 가 아니다.

## [Existing interface failure]

이전 screen: `sigmoid(Q)` 최소값 0.435 > decoder threshold 0.3 → 전 cell 통과 →
배경에서 peak, PnP 87/87 은 강제검출 artifact.  이 진단은 옳았다.

## [Raw-Q decoder comparison]

```
arm       median   near     far      F2 far   signed   p90     >20px >50px >100px  PnP   reproj
C0        13.24     6.88    22.08     44.59    20.58   101.2    203   120    54     70    24.91
C1base    13.95     7.67    24.32     44.93    19.91   103.2    215   137    59     68    25.03
Pargmax   80.07    17.87   119.91    161.92    71.57   332.2    453   387   317     87   112.11
Plocal    80.15    16.29   117.94    162.39    71.26   329.6    455   388   314     87   115.07
Pdsnt    115.89   109.93   119.65    135.08    82.08   254.6    611   533   384     86   121.47
```

[확인] **interface 를 고쳐도 proposal 은 회복되지 않는다.**
P-LOCAL median 80.1px vs C0 13.2px, far 117.9 vs 22.1px.
이전 sigmoid 경로(far median 160.3)보다 far 는 나아졌지만(117.9) 여전히 base 의 5배다.

[확인] P-ARGMAX 와 P-LOCAL 이 거의 같다(80.07 vs 80.15) = subpixel 정제의 문제가 아니라
**top-1 위치 자체가 틀렸다**.

[확인] P-DSNT 는 near 가 109.9px 로 최악이다.  full-map expectation 이므로 분포가 넓으면
좌표가 화면 중앙으로 끌린다 — Q 가 뾰족하지 않다는 뜻이다.

[확인] proposal 의 PnP 87/87 은 여전히 성능이 아니다.  세 decoder 모두 "미검출" 경로가 없어
항상 8 점을 내놓으므로 PnP 가 항상 성립한다(yaw 28.8~47.2°).

## [C3 proposal decoder gate]

```
A  F2 far median -10%                      FAIL   (-264.2%)
B  >50px tail -10%                         FAIL   (-223.3%)
D  confident-wrong 중 10px 이상 우세 >=20%   PASS   (22.65%)
-> GO (조건 D 단독)
```

[확인] 전체 평균으로는 완패지만 **confident-wrong 부분집합에서는 상보성이 있다**.
그래서 Phase D oracle 을 규정대로 실행했다.

## ★ Harness parity 주의

이 표의 C0 reproj 는 **24.91px** 이고 canonical baseline 은 23.161629px 다.
차이의 원인을 실측으로 확인했다:

```
centroid 포함 (canonical 경로) : PnP 70/87  reproj 23.161629 px
centroid 제외 (이 harness)     : PnP 70/87  reproj 24.913784 px
```

[확인] canonical DOPE PnP 는 **centroid 를 correspondence 로 포함**한다.
이번 지시문 금지항목 13("centroid 를 PnP correspondence 로 사용")을 따라 제외했으므로
C0 앵커가 canonical 과 다르다.  **모든 arm 이 동일하게 제외**되어 상대 비교는 공정하지만,
이 표의 절대 reproj 를 canonical 23.162px 와 직접 비교하면 안 된다.
