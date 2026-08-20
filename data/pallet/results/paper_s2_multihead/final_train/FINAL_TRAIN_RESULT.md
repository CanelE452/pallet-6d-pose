# FINAL TRAIN RESULT

architecture search 가 아니다. 확정된 것을 40,000 으로 한 번 다시 학습했다.

## 완료 판정 (산출물로만. exit code 안 씀)

```
seed1  checkpoint True  pool 40000  eval False  COMPLETE True
seed2  checkpoint True  pool 40000  eval False  COMPLETE True
manifest sha256 일치  True
ALL_COMPLETE          True
```

## 구조 스모크 (성능 아님)

모든 프레임이 in-train 이라 수치 비교는 하지 않는다. 물어본 건 하나 —
체크포인트가 구조적으로 멀쩡한가.

```
seed1 D2_MH_DEV512   n= 64  corner/theta/score4kp finite True/True/True  F0 64/64  F3 64/64
seed1 D0_MH_SEEN512  n= 64  corner/theta/score4kp finite True/True/True  F0 64/64  F3 64/64
seed2 D2_MH_DEV512   n= 64  corner/theta/score4kp finite True/True/True  F0 64/64  F3 64/64
seed2 D0_MH_SEEN512  n= 64  corner/theta/score4kp finite True/True/True  F0 64/64  F3 64/64
STRUCTURAL_OK_ALL  True
```

## config — E3 에서 바뀐 것은 pool 하나

```
architecture   SPLIT_LATE_2HEAD (--split-late)
arm            A1_CORNER_LINE
frozen early   FIRST_TRAINABLE_VGG = 19
optimizer      AdamW  lr 0.001  wd 0.0001  batch 8
ramp           500
marks          [6000, 12000, 18000, 25000]
lambda_corner  0.03518006215468158
pool           33758 -> 40000   <- 유일한 변경
```

## 학습하지 않은 것

```
L_neg 0 / negative batch 0 / dense zero suppression 없음 /
detached linear presence 없음
EDGE_HARD sampling weight 0   CORNER_LA sampling weight 0
```

## threshold

`PRESENCE_THRESHOLD = UNSET_PENDING_REAL_DEV`

합성 dev 에서 얻은 값은 initial range/reference artifact 다.
최종 threshold 는 REAL_DEV 에서 정한다 (`_delivery/README_DELIVERY_20260820.txt` 계약).

## 이 checkpoint 로 하면 안 되는 말

historical MH_DEV 6,242 가 학습 pool 안에 있다.
**MH_DEV 를 unseen / held-out 이라고 부를 수 없다.**
synthetic 지표로 checkpoint 를 고르지도 않았다 — step 25,000 사전 고정.

## 다음

REAL IN-HOUSE DEV / TEST 구축 및 annotation.
`real_eval/` 의 프로토콜·스키마·manifest 템플릿과 `scripts/stage0/real_eval/re_metrics.py` 평가기가 준비돼 있다.
