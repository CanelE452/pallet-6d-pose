# FINAL TRAIN PURPOSE

architecture search 가 아니다. 확정된 것을 한 번 깨끗하게 다시 학습한다.

```
architecture   SPLIT_LATE_2HEAD                (변경 없음)
pose route     corner -> Point PnP -> line theta rotation-only refine
               -> R 고정 -> point-only translation refit          (= F3)
rejection      score_4kp                        (inference 전용, 학습 안 함)
바뀌는 것      training pool 33,758 -> 40,000   ← 이것 하나뿐
```

## 왜 40,000 인가

최종 성능 주장이 **REAL IN-HOUSE DEV/TEST** 로 옮겨갔다. synthetic holdout 을
final test 로 보존할 이유가 없어졌으므로, 쓸 수 있는 broad synthetic support 를
전부 학습에 넣는다.

## 대가 — 명시적으로 기록한다

historical MH_DEV 6,242 가 학습 pool 안에 있다. 따라서 이 checkpoint 로는
**MH_DEV 를 unseen/held-out 이라고 부를 수 없다.** 학습 중 synthetic validation
지표로 checkpoint 를 고르지도 않는다 (`--no-eval`). 최종 checkpoint 는
사전에 고정된 step 25,000 이다.

## 판정 지표 (성능 아님)

```
seed 당 step_25000.pth 실재
meta 의 pool == 40000, evaluated_every_mark == False
FINAL_SYNTH_TRAIN_V1 manifest sha256 일치
구조 스모크: corner/line/score_4kp 유한, F0·F3 가 실제로 풀림
```
성능 비교는 하지 않는다 — 모든 synthetic 프레임이 in-train 이라 의미가 없다.
