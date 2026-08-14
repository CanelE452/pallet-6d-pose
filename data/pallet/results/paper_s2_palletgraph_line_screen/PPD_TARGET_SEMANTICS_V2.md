# PPD target semantics v2 — T0/T1/T2

## 왜 바꿨나

기존 T0 target 은 `self-visible projection ∩ dilated mask_rle` 였다.
파렛트는 포크 슬롯이 뚫려 있어 `mask_rle` 면적이 cuboid hull 의 **0.52~0.80** 이고,
유효한 top edge 가 구조적으로 mask 밖에 놓인다.  mask 는 파렛트 foreground 이지
**어떤 cuboid line 이 유효한가의 정의가 아니다**.

이번엔 dilation 을 키우거나 gate 를 낮추지 않고, target 정의를 oracle O0 와 동일하게 바꿨다.

## 세 모드 (역할은 실행 전 고정)

```
T0 mask_filtered      self-visible ∩ dilated mask_rle        실패 control (main 금지)
T1 self_visible_full  self-visible only                      geometry upper bound (main 금지)
T2 observed_fragment  self-visible ∩ O0 gradient association  ★ MAIN (사전 확정)
                      mask filtering 없음
```

[확인] T2 의 gradient association 은 O0 helper 를 **그대로 추출**해 쓴다.
새 Canny/threshold 를 만들지 않았고, 실제 N87 20 프레임에서 runner 의
`association_keep_mask` 와 신규 `gradient_association_mask` 의 출력이 **완전히 일치**한다
(canny (100,200), radius 4.0).

## 비교 (동일 200 train frames)

```
metric              T0(mask)  T1(visible)  T2(observed)
nonempty rate         1.000     1.000        1.000
top&base 모두          0.895     1.000        0.865
retained fraction     0.788     1.000        0.418
class derivation      0         0            0
yaw+180 invariance    0         0            0

class 별 positive-frame rate
  top_width           0.755     0.755        0.685
  top_depth           0.620     0.815        0.620
  base_width          0.995     1.000        0.985
  base_depth          1.000     1.000        0.975
  vertical            0.990     1.000        0.815
```

[확인] T2 는 gradient 가 없는 구간을 버리므로 retained 가 0.418 로 가장 낮다.
[주의] T1/T2 에는 mask-supported fraction 을 gate 로 쓰지 않는다.
**해당 없음 — mask occupancy 는 semantic cuboid line validity 의 정의가 아니다.**

## Target polarity utility (핵심 검증)

동일 unsigned SAI-U candidate set, 동일 scorer 정의, 원본 해상도 support.

```
mode                  n    polarity accuracy   inversion
mask_filtered        48    0.979               1/48
self_visible_full    48    0.979               1/48
observed_fragment    48    1.000               0/48
candidate>=2 frame: 48/60      upright 후보 존재: 48/48
```

[확인] **T2 는 polarity 를 완벽히 보존한다 (1.000, inversion 0)**.
[확인] T0/T1 도 0.979 로 높다 — polarity 정보 자체는 세 정의 모두 담고 있다.

### 측정 중 발견한 결함 (수정함)

첫 측정에서 candidate 가 0 개였다.  원인은 target 이 아니라 **측정 경로**였다:
100x100 target 을 원본 해상도로 nearest-resize 하면 픽셀이 6.4x4.8 배로 뭉쳐
(105 -> 3200 px) TLS component 가 전부 파괴된다.  원본 해상도 support 로 후보를
만들면 component 3/2/0, candidate 2, top3 rotation error 0.90°, upright 후보 존재.

## T2 gate 결과

```
  PASS  nonempty >=0.95                  1.000
  FAIL  top&base 모두 >=0.90               0.865
  PASS  class derivation mismatch 0      0
  PASS  yaw+180 invariance mismatch 0    0
  PASS  NaN/Inf 0                        0
  PASS  candidate polarity acc >=0.95    1.000
  PASS  vertical inversion <=0.05        0.000

  T2 TARGET GATE: FAIL
```

[확인] 미달 항목은 **`top&base 모두 존재 >=0.90` 하나뿐**(0.865)이고,
목적에 직결되는 **candidate polarity accuracy 는 1.000, inversion 0.000** 이다.

[추정] top 과 base 가 **둘 다** 있어야 polarity 가 결정되는 것은 아니다.
한쪽 클래스만 관측돼도 그것이 top 인지 base 인지 알면 방향이 정해진다.
따라서 `top&base >=0.90` 기준은 polarity 목적에 대해 과도할 수 있다.

## 판정

- [확인] **T0 (mask-filtered) 는 REJECT** — mask 가 유효 line 을 구조적으로 삭제한다.
- [확인] **T2 는 polarity utility 를 완벽히 보존**하므로 representation 으로서 유효하다.
- [판정] 그러나 지시문이 사전 고정한 gate 항목 하나가 미달이므로, **학습을 임의로
  진행하지 않았다**.  무결성 규칙 5(gate 완화 금지)와 D4(FAIL 시 학습 금지)를 따랐다.

## 사용자 결정이 필요한 지점

`top&base 모두 존재 >=0.90` 을 유지할지, polarity utility(1.000)를 근거로
이 항목을 polarity 목적에 맞게 재정의할지.  전자면 학습 중단, 후자면 32-frame
overfit 부터 진행한다.  어느 쪽도 이번 실행에서 임의로 고르지 않았다.
