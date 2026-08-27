# PURPOSE — FIXED_OBJECT 60EP ADAPTIVE_CONVERGENCE_CONFIRMATION

[소비처] 논문 §실험. 다만 이 run 의 1차 소비처는 **의사결정**이다 —
fixed-object 표현을 논문 평가로 끌고 갈지 말지.

[문장] "5ep 실패는 표현 자체의 실패가 아니라 optimization budget 부족이었다"
— 이 [추정] 을 검증한다. 반대로 나오면 fixed-object assignment 가 학습되지
않는다는 것을 확증한다.

## ★ 이 run 은 5ep 게이트를 사후 수정하는 것이 아니다

```
ORIGINAL_5EP_GATE      = FAIL              ← 변경 금지, 그대로 보존
ADAPTIVE_60EP_CONFIRM  = USER_AUTHORIZED   ← 별도 트랙
```
결과를 보고 5ep 게이트를 PASS 로 바꾸지 않는다.

## 5ep 확인 사실 (실측)

```
COMPLETE True · NaN 0 · channel collapse 0
box  mAP50      CF 0.9850  FIXED 0.9834     <- 사실상 동일
pose mAP50      CF 0.9095  FIXED 0.5936
pose mAP50-95   CF 0.8512  FIXED 0.0875
train pose loss FIXED 8.44 -> 4.89 (5ep 종료 시점에도 하강 중)
val diagnostic  identity best 56.4% · yaw180 로 더 잘 맞는 프레임 42%
```

## 관찰과 추정의 분리

```
[관찰] detection 이 아니라 fixed physical corner identity assignment 가
       5ep 에서 아직 충분히 학습되지 않았다.
[추정] 더 긴 학습에서 assignment 가 개선될 가능성이 있다.
```

이 [추정] 을 검증하려고 60ep 를 한다. **5ep checkpoint 에서 resume 하지 않는다** —
5ep run 은 5ep 의 LR 스케줄을 이미 소비했으므로 이어붙인 것을 "처음부터 60ep" 와
같다고 볼 수 없다.

## 판정은 결과를 본 뒤 threshold 를 만들지 않는다

`H1 5EP_WAS_INSUFFICIENT_BUDGET` / `H2 FIXED_OBJECT_ASSIGNMENT_NOT_LEARNED` /
`H3 PARTIAL_FIXED_SEMANTIC_LEARNING` 중 하나로, **관찰과 판정을 분리해** 보고한다.
permutation oracle 은 진단 전용이며 main metric 이 아니다.
paper main checkpoint 는 사전 규칙대로 **last.pt** 이고, epoch30 이 우연히 좋아도
바꾸지 않는다.
