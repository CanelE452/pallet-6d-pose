# NEGATIVE-READY TRAIN CONTRACT

`NEGATIVE_SYNTH_V1` **미도착**. 이 문서는 loader contract 만 정의한다.
도착 전 학습 금지, dummy negative 생성 금지.

## arm

```
N0  BASELINE
    Corner positive = BROAD MH_TRAIN
    Line   positive = BROAD MH_TRAIN
    Negative        = NONE

N1  NEGATIVE
    Corner positive = N0 와 EXACT SAME BROAD trajectory
    Line   positive = N0 와 EXACT SAME BROAD trajectory
    Negative stream = NEGATIVE_SYNTH_V1   (별도 stream)
```

## 핵심 — negative 가 positive 노출을 밀어내면 안 된다

C1 에서 배운 것: corner batch 를 `7 BROAD + 1 NEW` 로 만들면 **모든 cell 의 positive
노출이 균일하게 12.5% 줄어든다**(R1_EXPOSURE_AUDIT 실측). 그 희석 자체가 손상을
만들었다. negative 는 같은 실수를 반복하면 안 된다.

따라서 negative 는 **positive batch 를 대체하지 않고 별도 batch 로 추가**한다.

```
step 마다:
  1. LINE   stream 에서 B_L 로드   (N0/N1 동일)
  2. CORNER stream 에서 B_C 로드   (N0/N1 동일, positive 만)
  3. NEGATIVE stream 에서 B_N 로드 (N1 에서만)
  4. L = L_line(B_L) + λ_corner·L_corner(B_C) + λ_neg·L_neg(B_N)
```

positive exposure 는 N0 와 N1 에서 **frame id 단위로 동일**해야 한다. 검증:

```
set(corner positive draws N0) == set(corner positive draws N1)   순서까지 동일
line param max|diff| == 0                                        (배선 불변식)
```

## 이미 확보된 재사용 자산

```
two-stream trainer      scripts/stage0/multihead/mh_curriculum.py
                        early frozen + late 분리라 branch 별 다른 batch 가능
                        line exact parity 가 3,000 step 학습 후에도 0.000e+00 로 검증됨
root-aware loader       load_frame_from(root, stem) — 데이터셋 root 만 주입하면 됨
결정적 스케줄            C1_RESCUE 가 slot 단위 치환·SHA lock 으로 이미 검증
```

negative stream 은 `Stream` 클래스에 root 만 다르게 주면 그대로 붙는다.

## 아직 정하지 않은 것 (도착 후 사전등록 대상)

```
λ_neg              negative loss 가중치 — gradient 기준으로 anchor 할 것.
                   ★loss 값 기준으로 잡으면 안 된다. pose-aware 실험에서 값 기준으로
                   잡았다가 실제 gradient 비가 253~297배라 전부 발산한 이력이 있다.
B_N 크기            negative batch 크기
L_neg 형태          기존 head 로 표현 가능한 것만. 새 head/loss 금지.
평가                negative 는 positive-only dev 로 평가할 수 없다.
                   FP rate 주장은 negative eval split 이 생긴 뒤에만.
```

## 금지

도착 전 학습, dummy negative, positive 노출 대체, 새 head/loss.
