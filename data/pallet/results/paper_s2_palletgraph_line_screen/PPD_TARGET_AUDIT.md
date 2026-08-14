# PPD 5-class target audit — GATE FAIL, 학습 미실행

## 결과 (deterministic 200 train frames)

```
검사                          값       기준      판정
nonempty target              1.000    >=0.95    PASS
top&base 모두 존재             0.895    >=0.90    FAIL
mask-supported 비율           0.788    >=0.95    FAIL
class derivation mismatch    0        0         PASS
yaw+180 semantic mismatch    0        0         PASS
all finite                   True     -         PASS
no full-frame target         1.000    -         PASS
```

class 별 positive frame 비율:

```
top_width    0.755   (median 148 samples)
top_depth    0.620   (median 102)
base_width   0.995   (median 295)
base_depth   1.000   (median 267)
vertical     0.990   (median  58)
```

## 원인 — 구현 버그가 아니라 데이터 구조

[확인] self-visible 필터는 정상이다.  60 프레임에서 **top edge 가 self-visibility 로
탈락한 경우 0건**, mask 로 top 이 전부 탈락한 경우 2건.

[확인] `mask_rle` 면적 / cuboid hull 면적 = **0.52 ~ 0.80**.  팔레트는 포크 슬롯이
뚫린 구조라 실제 실루엣이 외곽 박스보다 훨씬 작다.  따라서 cuboid edge 의 일부는
필연적으로 mask 밖에 놓인다.

[확인] mask-supported 비율은 camera_mode 와 무관하게 일정하다 (0.767~0.803).
특정 시점 문제가 아니라 **구조적**이다.

```
mode            n    mask_frac   top_w   top_d
close_crop     33     0.798      0.818   0.667
close_full     32     0.791      0.719   0.531
far            34     0.767      0.735   0.618
mid            35     0.788      0.686   0.600
top_down       34     0.781      0.735   0.676
top_down_crop  32     0.803      0.844   0.625
```

[추정] top edge 가 base 보다 더 많이 탈락하는 것은, 상판 가장자리가 실루엣 경계와
겹쳐 dilation 2 cells(100x100 격자 기준, 원본 약 13 px)로 덮이지 않기 때문이다.

## 판정

- [확인] 지시문 Phase E4 "Gate FAIL 이면 학습하지 않는다" 에 따라 **32-image overfit 이하
  전 학습 단계를 실행하지 않았다**.
- [확인] dilation 을 키우거나 mask 필터를 완화하면 gate 를 통과시킬 수 있으나, 그것은
  **gate 결과를 보고 설정을 바꾸는 것**이라 무결성 규칙에 어긋나므로 하지 않았다.
- [판정] 이 FAIL 은 **target 설계와 데이터의 불일치**이지 head/loss 구현 문제가 아니다.
  현 상태로 학습하면 top class 의 25~38% 프레임이 positive 없이 학습되어,
  "top/base 를 구분한다" 는 능력을 측정할 수 없다.

## 사용자 결정이 필요한 선택지

1. `MASK_DILATION_CELLS` 를 2 -> 4~6 으로 올린다 (원본 ~26~38 px).  가장 단순.
2. mask 필터를 self-visible 필터로 대체한다 (oracle O0 evidence 와 동일한 조건이 된다).
   oracle 은 mask 를 쓰지 않았으므로 **learned 와 oracle 의 조건을 맞추는 효과**도 있다.
3. gate 기준 `mask-supported >=0.95` 를 이 데이터 구조에 맞게 재설정한다.

어느 쪽이든 **결정 후 gate 를 다시 돌려야** 하며, 이번 실행에서는 임의로 고르지 않았다.
