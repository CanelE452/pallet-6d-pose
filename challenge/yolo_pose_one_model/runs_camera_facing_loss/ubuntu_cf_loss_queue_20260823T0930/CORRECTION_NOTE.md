# CORRECTION NOTE — "night correct-box = 0" 은 틀렸다

작성 2026-08-23. **기존 파일은 삭제·수정하지 않는다.** 이 노트만 추가한다.

## OLD STATEMENT (틀림)

> A/C/E 모두 night correct-box = 0 → night median/p90 계산 불가

`COVERAGE_EFFECT_10K.md`, `EXTRA_V2_SIGNAL.md`, 그리고 그 시점의 내 보고에
`real NIGHT p90 = nan` 으로 적혀 있고, 나는 이를 "night 에서 모델이 correct-box 를
하나도 못 만든다" 로 해석해 보고했다. **그 해석이 틀렸다.**

## CORRECTED (실측, NIGHT 28장 · IoU>=0.5 · conf=0.001 · pad100)

```
model   top1-cbox      any-cbox     best IoU median
A42       0 / 28  0.000    0.071            0.116
C42      13 / 28  0.464    0.714            0.633
C43      18 / 28  0.643    0.964            0.772
E42      13 / 28  0.464    0.786            0.695
FT       27 / 28  0.964    0.964            0.865
```

출처: `NIGHT_FAILURE_DECOMPOSITION.json` / `NIGHT_FAILURE_PER_FRAME.csv`.

## NaN 의 진짜 원인

`cf_real_eval.py` 의 게이트 모집단 `cbox_paired` 는
**"A0(=A42, V1-10K) 와 공통인 correct-box 프레임"** 이다.

```
A0 correct-box 58장 중 NIGHT = 0
  -> 어떤 모델을 넣어도 cbox_paired 에 night 표본이 0
  -> night_p90 = NaN

실제 자체 correct-box:  C42 95장(night 13),  E42 81장(night 13)
```

즉 NaN 은 **night 실패가 아니라 A0 기준 pairing 아티팩트**다.

## 결과 해석에 미치는 영향

- 기존 `COVERAGE_EFFECT_10K` / `EXTRA_V2_SIGNAL` 의 real 지표는
  **night generalization metric 이 아니다.** 사실상 **DAY-dominated comparison** 이다.
  (같은 모집단을 양쪽에 적용했으므로 A vs C, C vs E 비교 자체는 유효하다.)
- night 성능을 논하려면 A0 pairing 이 아닌 **모델 자체 correct-box** 를 써야 한다.
- 판정값(`COVERAGE_EFFECT_10K = POSITIVE`, `EXTRA_V2_SIGNAL = NULL_OR_WORSE`)은
  그대로 유효하되, **적용범위를 DAY 로 한정해 읽는다.**

## 남기는 이유

수치를 고쳐 덮으면 "처음부터 맞았던 것"처럼 보인다. 잘못된 해석이 어떤 경로로
나왔는지(모집단 정의 → NaN → 오독)를 남겨야 같은 함정을 다시 밟지 않는다.
