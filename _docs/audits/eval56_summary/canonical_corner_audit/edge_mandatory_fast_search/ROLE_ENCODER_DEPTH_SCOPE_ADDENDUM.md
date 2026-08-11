# What the role-depth result is allowed to be called

`df1c2d0` stands unchanged.  `ROLE_ENCODER_DEPTH_INSUFFICIENT` is the verdict.

## A phrase to correct first

The result document and history entry describe the second block as having
"worked at maximum" and "engaged completely".  That overstates what was
measured.  The relative descriptor change reaching 1.60 says the block was
**strongly exercised** -- it was given a large share of the representation to
rewrite and it used it.  It does not say the block reached any ceiling, and the
plateau in `beta` alongside a still-growing body is evidence against reading it
that way.

```
was written    "the block worked at maximum" / "engaged completely"
correct        LOW_RANK-style wording: the block was STRONGLY_EXERCISED
```

## Established

```
the second role refinement block was strongly exercised
    relative descriptor change reached 1.59989, cosine fell to 0.6623
train CE improved
    6.190232 -> 5.792634 against F2's trajectory
D2 improvement over F2 stayed small
    angle +5.68%, offset +20.80%
the 40% threshold was not reached
no large D0/D2 specialization appeared
    1.0461 / 1.0283, sitting with F2 rather than with the broad-unfreeze arm
```

## Not established

```
role encoder capacity is sufficient
role encoder capacity is exhausted
one block is the optimal depth
attention is looking at the wrong place
broader role decoding cannot help
```

Nothing in the run distinguishes "the decoder has enough capacity" from "one
block of this shape does not convert it".  The attention entropy was recorded
and no claim about where attention looks follows from it.

## The next question

```
CAN_CONSTRAINED_IN_BLOCK_LATE_A1_ADAPTATION
RECOVER_THE_F1_SIGNAL_WITHOUT_FULL_UNFREEZE ?
```

Three arms have now been placed around one axis and the remaining gap on it is
specific:

```
F1   full parameter adaptation inside net.vgg[19:27]
     strongest accuracy so far, and a 42.5% D0/D2 gap
F2   residual adaptation outside the feature extractor, after F50
     no specialization, and 17-21% against a 40% threshold
R1   F2 plus a deeper role decoder
     no specialization, and 5.68% / 20.80%
```

So the untested cell is adaptation *inside* the late convolutions -- where F1's
effect actually came from -- but constrained rather than free.  Base weights go
back to frozen and only a low-rank additive delta is learned.  One factor.

A null result there will belong to `rank = 8` and this one formulation.  It will
not mean low-rank adaptation cannot work, and it will not mean full unfreezing
is required; the F1 signal stands either way.
