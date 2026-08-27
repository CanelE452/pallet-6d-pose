# METHOD SPEC — pairwise signal audit (training-0)

## 질문

G38 source 의 실제 one2one training anchor 에서, assigned positive 보다 높거나 가까운
unassigned hard anchor 가 **학습할 만큼 존재하는가.** pairwise loss 착수 전 마지막 HARD GATE.

## pair 정의 (실행 전 고정)

```
POS   one2one fg_mask=1 인 assigned anchor (topk2=1 이므로 GT 당 1 개)
NEG   같은 image·같은 class 에서 fg_mask=0 인 anchor 중 raw class logit 최대
      ★ IoU<0.5 로 제한하지 않는다 — one2one 에서 unassigned duplicate 도
        실제 inference 경쟁자이므로 primary hard-negative pool 에 포함한다
delta = s_pos - s_neg          (raw logit 차)
```

negative category (실행 전 고정):
```
DUPLICATE        IoU >= 0.5
NEAR_DISTRACTOR  0.1 <= IoU < 0.5
FAR_DISTRACTOR   IoU < 0.1
```

## margin 을 정하지 않는다

margin 이라는 새 hyperparameter 없이 **delta 분포 자체**와, margin-free logistic
pairwise 를 가정했을 때의 해석적 gradient 크기만 본다.

```
L_rank = softplus(s_neg - s_pos)   가정
g_rank = |dL/d delta| = sigmoid(-delta)
g_bce  = sigmoid(s) - target        (BCEWithLogits 의 logit 도함수)
```

autograd 를 쓰지 않는다. backward·optimizer·model.train()·fuse 전부 0.

## GATE (결과 보기 전 고정)

```
G1  val1998 에서 frac(g_rank >= 0.10) >= 0.10
G2  val1998 에서 frac(delta <= 1.0) >= 0.05  OR  frac(delta < 0) >= 0.02
G3  delta <= 2 hard subset 에서 NEAR + FAR >= 0.10
train5k 와 val1998 이 반대 방향이면 SOURCE_SIGNAL_UNSTABLE -> STOP
셋 다 통과 -> PAIRWISE_SIGNAL_PRESENT, 하나라도 실패 -> TOO_WEAK_OR_MISMATCHED
```

## real 은 secondary only

real DEV128 에 같은 정의를 적용해 비교만 한다. margin·lambda·pair sampling·threshold
중 어떤 것도 real 로 선택하지 않는다.
