# PCR Gate A (32-frame capacity) — FAIL, Gate B 미실행

> heatmap signed bias 의 존재는 이미 확인됐고, role-constant / viewpoint offset 보정은
> 앞선 실험에서 실패했다.  이번 방법은 post-hoc offset correction 이 아니라
> **feature identity 를 직접 학습**한다.  belief/affinity representation, decoder,
> centroid 포함 canonical PnP 는 그대로다.  N87 은 이번 작업에서 **한 번도 열지 않았다**
> (Gate A FAIL → Gate B real one-shot 금지).  final-test 미사용.

## [Architecture]

```
trainable   vgg_last          5,014,912
            belief stage4~6   12,567,579
            role encoder      248,736   (32ch embedding + 8 prototypes)
            FiLM (C3만)       119,616
            total             17,950,843
```

- role score = cosine(normalize(embedding), normalize(prototype)) / 0.10.  sigmoid 없음.
- FiLM 은 belief 가 아니라 **shared feature 를 modulate** 한다.
  zero-init 이라 step0 에서 ep57 과 **max|delta| = 0.000e+00** [확인].
- flag-off forward 가 legacy forward 와 **max|delta| = 0.0** 로 일치 [확인].

## [Capacity gate] — 두 arm 모두 FAIL

### C2 (role auxiliary, FiLM 없음)
```
  FAIL  1 proto accuracy >=0.95           0.4688
  FAIL  2 structural GT>other >=0.95      0.8080
  PASS  3 GT>student wrong >=0.90         1.0000
  PASS  4 GT>teacher wrong >=0.90         1.0000
  FAIL  5 corner error -50%               0.2129
  PASS  6 centroid <= +10%               -0.4588
  PASS  7 no NaN                          0.0000
```
### C3 (role + FiLM)
```
  FAIL  1 proto accuracy >=0.95           0.4688
  FAIL  2 structural GT>other >=0.95      0.8304
  PASS  3 GT>student wrong >=0.90         1.0000
  PASS  4 GT>teacher wrong >=0.90         1.0000
  FAIL  5 corner error -50%               0.2129
  PASS  6 centroid <= +10%               -0.4588
  PASS  7 no NaN                          0.0000
```

## 학습은 실제로 일어났다

```
arm  proto acc            structural GT>other      corner
C2   0.188 -> 0.469       0.693 -> 0.873 -> 0.808  6.26 -> 4.93 px
C3   0.188 -> 0.609(peak) 0.693 -> 0.925 -> 0.830  6.26 -> 4.93 px
                -> 0.469
```

[확인] chance 는 1/8 = 0.125 다.  proto accuracy 가 0.47~0.61 까지 올라갔고
structural 은 0.93 까지 갔으므로 **objective 는 작동하고 gradient 도 흐른다**.
[확인] 그러나 32 frame 을 600 step 외우는 조건에서도 기준(0.95)에 크게 못 미친다.
이것이 capacity 판정이다 — 일반화 이전에 **암기조차 되지 않는다**.

[확인] C2 와 C3 의 최종 proto accuracy 가 **동일(0.469)** 하다.
FiLM 은 role identity 학습 자체에 기여하지 않는다.

## ★ 기준 3·4 의 PASS 는 증거로 쓸 수 없다

```
gt>student-wrong / gt>teacher-wrong 이 측정된 시점: 7 개 로그 중 1 개
나머지 6 개는 nan = GT 에서 4 cell 초과 떨어진 wrong peak 가 하나도 없음
```

[확인] 선정된 32 frame 은 8 corner 가 전부 valid 한 clean frame 이라
ep57 이 이미 잘 맞힌다(초기 corner error 6.26px = 0.5 cell).
따라서 hard negative 가 거의 생기지 않고, 기준 3·4 는 **극소 표본에서 PASS** 했다.
이 두 항목을 "role feature 가 wrong peak 를 이겼다" 는 증거로 인용하면 안 된다.

## ★ 발견한 구현 결함 (판정 전 수정)

첫 실행에서 structural 지표가 전 구간 **정확히 0.000**, λ_cross 가 최대값 10 에 clamp 됐다.
원인은 `cross_location_loss` 의 gather 축 오류다.

```
Before : at_other = sampled.transpose(1,2).gather(1, index.transpose(1,2))
         -> [b,i,j] 전부가 own 과 동일해짐
         -> softplus(own - own + 0.2) = 상수 -> gradient 0 -> λ clamp
After  : index = labels[:, None, :]        # [b, j, i]
         at_other = sampled.gather(2, index).transpose(1,2)
```

[확인] 수정 후 structural 이 0.000 → 0.93 으로 살아났다.
이 회귀를 잡는 테스트 2 건을 추가했다(off-diagonal 값 검증 + gradient 부호).

## ★ 선정기 defect 와 재실행 (두 실행 모두 보고)

첫 실행의 frame 선정은 명세된 기준(8 corner valid 우선, scale·azimuth 분산)을
구현하지 않고 index stride 만 사용했다.  그 결과 초기 corner error 가 5.6px 로
기준 5(−50%)가 구조적으로 도달 불가였다.  선정기를 명세대로 고쳐 **한 번** 재실행했다.
threshold 는 바꾸지 않았다.

```
arm  run1(결함 선정) -> run2(명세 선정)
C2   acc 0.250 -> 0.469    struct 0.632 -> 0.808
C3   acc 0.391 -> 0.469    struct 0.750 -> 0.830
```

두 실행 모두 FAIL 이며 판정은 바뀌지 않는다.
run1 산출물은 `pcr_gate_a_run1_defective_selector.json`, `pcr_capacity_history_run1.csv`.

## [Gradient calibration]

reference = `vgg.25.weight` (VGG last block 마지막 conv weight), 8 batch, update 없음.

```
term            |grad| median      lambda       clamped
proto           2.1123e+00       3.5403e-04   False
cross           9.6859e-01       7.7205e-04   False
wrong           0.0000e+00       1.0000e+01   True
teacher_wrong   0.0000e+00       1.0000e+01   True
local           8.7650e-01       2.1329e-04   False
anchor          0.0000e+00       1.0000e+01   True
legacy          9.3475e-03       (기준)
```

[확인] `wrong` / `teacher_wrong` / `anchor` 는 clamp 상한에 걸렸다.
gradient 가 0 에 가깝기 때문이며, 위에서 본 대로 hard negative 가 없고
student 가 teacher 와 동일하게 시작하기 때문이다.  기전이 설명된 clamp 다.

## [현재 판정]

```
Corner-role auxiliary loss   REJECT (Gate A capacity 미달)
Role-conditioned FiLM        REJECT (role 학습에 기여 없음, C2==C3)
Final path                   base ep57 DOPE
```

사전 규칙에 따라 Gate B(group-disjoint 3k, N87 real one-shot), Gate C(full canonical
5 epoch)를 **실행하지 않았다**.  N87 접근 횟수 **0**.

## [지지 증거]

- [확인] baseline 재현(87/87/70, yaw 6.025216, reproj 23.161629) 후 시작.
- [확인] FiLM zero-init 과 flag-off forward 가 ep57 과 bit-exact.
- [확인] role objective 는 작동한다: structural 0.69 → 0.93, proto 0.125 → 0.61.
- [확인] 유닛 20 건 통과(frame 단위 symmetry, yaw180 involution, top-bottom 비허용,
  wrong-peak gradient 부호, close-pair 제외, teacher detach).

## [반증 증거 / 한계]

- [확인] 32 frame 암기에서도 proto accuracy 0.47 로 0.95 에 크게 못 미친다.
- [확인] 기준 3·4 는 표본 1/7 시점에서만 측정돼 PASS 를 증거로 쓸 수 없다.
- [확인] 선정 frame 이 clean 이라 corner error 가 6.26px 에서 시작한다.
  "wrong peak 를 고치는 능력" 은 이 표본으로 충분히 시험되지 않았다.
- [확인] Gate A 는 600 step 고정이다.  더 긴 최적화에서 달라질 가능성은 배제되지 않는다.

## [다음 admissible experiment]

1. capacity 가 미달이므로 **먼저 표현력**이다 — 32ch embedding / 8 prototype /
   temperature 0.10 조합이 8-class 를 담을 수 있는지 자체를 별도로 확인.
   이번 결과를 보고 이 값들을 바꾸지 않았다.
2. hard negative 가 없는 clean frame 대신 **ep57 이 실제로 틀리는 frame** 으로
   capacity 를 시험해야 기준 3·4 가 의미를 갖는다.  단 그 선정은 사전 고정해야 한다.
3. 위 둘 없이 Gate B 로 진행하거나 threshold 를 낮추지 않는다.
