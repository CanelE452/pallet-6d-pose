# Architecture

```
RGB 400x400
  -> VGG19 (early blocks frozen | layers after vgg[17] trainable)
       |- F100 = vgg[17], 256ch, 100x100   (runtime search, not hardcoded)
       `- F50  = vgg[26], 128ch,  50x50
  -> DOPE belief stages 1-3 frozen / 4-6 trainable, affinity + seg frozen
  -> H_base = stage6[:8]
  -> proposal branch:  F100 conv3x3 s2 -> 64 | F50 conv1x1 -> 64
                       concat 128 -> conv3x3 -> GroupNorm -> ReLU -> F*
                       pixel 1x1 -> 64 ;  8 corner queries (ID embed + canonical
                       corner signs + normalised W,D,H) -> Q = q.p / sqrt(64)
  -> gate g_i = sigmoid(MLP([query, pooled F*, H4/H5/H6 stats])), bias -4.6
  -> H_ref = (1-g) H_base + g sigmoid(Q)        centroid untouched
  -> existing decoder -> OpenCV PnP
```

BatchNorm 미사용(GroupNorm).  graph / edge / mask / line / vector / DiffPnP /
centroid proposal / direct pose regression 없음.

## Belief operating range audit

cached ep57 stage-6 corner belief over the mechanism set:
**min -0.030, max 1.004** = raw conv output regressed onto a Gaussian with peak 1.0.
sigmoid 를 골라 range 를 맞췄다.

★ 사후 확인된 결함: range 는 맞았지만 **operating point 는 맞지 않는다.**
학습 후 sigmoid(Q) 의 실측 peak 는 0.435~0.695 로 배경이 0 근처로 내려가지 않아,
decoder threshold 0.3 을 모든 cell 이 통과한다.  `CORNER_REPLACEMENT_GATE.md` §현상 2.

## Trainable / frozen

```
trainable   last VGG block   5,014,912
            belief 4-6       12,567,579
            proposal branch  334,081
            total            17,916,572
frozen      early VGG, belief 1-3, affinity 전체, segmentation, centroid path
```

## Dual output

forward 는 base / proposal / proposal_transformed / gate / refined 를 모두 반환하므로
같은 checkpoint 에서 C1-base, C1-proposal, C1-refined 를 분리 평가할 수 있다.
