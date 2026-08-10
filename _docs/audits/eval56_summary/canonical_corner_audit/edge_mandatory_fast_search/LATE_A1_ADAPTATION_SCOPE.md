# Does line supervision have to reach the A1 feature extractor?

Phase A closed the step axis.  Fifteen passes bought 15.69% and 18.78% off the
D2 medians and left the model 3.7x and 3.9x from the gate, so exposure was not
the primary limit -- but the cross-entropy was still falling when it stopped, so
`LONG_SCHEDULE_STILL_OPTIMIZING_BUT_TASK_FAIL` is explicitly not a feature-limit
confirmation.  This screen asks the feature question directly, once, with one
factor moving.

```
DOES_LINE_SUPERVISED_LATE_FEATURE_ADAPTATION_HELP ?
```

## The two arms

```
F0_FROZEN_A1          DIRECT_HOUGH_TOKEN_XY_V0 exactly as Phase A ran it
F1_LATE_A1_TRAINABLE  the same, with net.vgg[19:27] receiving gradient
```

Identical in both: the role-query encoder, `DirectHoughHead`, the theta/rho
lattice, the target, the cross-entropy, the token XY construction, the query
count, the attention depth, batch 8, weight decay 1e-4, seed 1, the 13,618-frame
pool, the marks, and every gate.  The dead `self.position` stays dead and stays
constructed.

## What "the last feature block" actually is

Read out of the model, not guessed from a name.

```
A1                    DopeNetwork, 54,814,296 params, 15 children
F50 producer          net.vgg, a 27-entry Sequential, 7,340,480 params
                      output 128 x 50 x 50, taken today by a forward hook
last MaxPool          net.vgg[18]
last block            net.vgg[19:27]  -- the only stage at F50 resolution
                      conv 256->512, 512->512, 512->256, 256->128
                      5,014,912 params
```

```
trainable / net.vgg   68.3%
trainable / A1 total   9.15%
```

That first number deserves saying out loud: "the last block" is two thirds of
the feature extractor's parameters, because DOPE's VGG trunk puts most of its
weight in the final stage.  The factor being moved is thicker than the phrase
suggests.  It is still the correct block boundary -- there is no pooling after
index 18, so 19 through 26 is the last resolution stage and splitting it would
be an invented boundary -- but the screen is scoped as
`LATE_A1_BLOCK19_26` and a null result belongs to that block, not to "A1
adaptation" in general.

## Normalisation policy, fixed in advance

```
BatchNorm / GroupNorm / LayerNorm / InstanceNorm anywhere in A1    0
Dropout in net.vgg                                                0
```

Measured, not assumed.  There is no running statistic to have a policy about,
and train-versus-eval mode cannot change `net.vgg`'s forward.  A1 therefore
stays in `.eval()` in both arms, which removes a difference rather than adding
one.  This is recorded now so it cannot be revisited after a number is read.

## The gradient path

`FrozenA1.forward` is decorated `@torch.no_grad()` and detaches, so F1 needs a
wrapper that does not.  Two facts make that safe:

```
net.vgg(x) against the hooked feature from net(x)    max abs diff 0.000e+00
belief and affinity                                  unused by DirectHough
```

So the Phase B wrapper calls `net.vgg` directly and skips the fifteen downstream
stages, which is the same tensor by measurement and avoids building a graph
through 47M parameters that no loss touches.

## F0 is reused only if it is provably the same code path

Phase A's 25,545-step run *is* F0 -- same architecture, same schedule, same
budget.  Reusing it saves two hours and risks comparing two arms across
different code paths, which is the failure mode this whole screen exists to
avoid.  So it is reused only on proof:

```
deterministic mode, 20 steps, locked DH.train_network against the Phase B
trainer with unfreeze disabled                       must be 0.000e+00
```

Zero means the Phase B trainer with F0 settings is bit-identically the Phase A
trainer, and the recorded 25,545-step result stands as the F0 arm.  Anything
else and F0 is re-run fresh through the Phase B trainer.  The choice is made by
that measurement, not by convenience, and the measurement happens before F1
starts.

## Learning rate, fixed before the run

```
head and role encoder    CAP.LR = 1e-3, unchanged
net.vgg[19:27]           CAP.LR x 0.1 = 1e-4
```

One value, pre-registered, no sweep.  This is a feature-adaptation screen, not
an optimizer search.  If F1 fails, "a different A1 learning rate might have
worked" is a real limitation of the screen and will be stated as one rather than
chased.

## Budget and decision

```
both arms      fresh from 0 to 25,545 steps, the budget Phase A settled on
marks          1,703 / 5,000 / 8,515 / 17,030 / 25,545
decision       25,545, D2_LINE_DEV512, primary and only
D0_SEEN512     diagnostic, never selection -- including for the overfit read
per-role       8,515 / 17,030 / 25,545, recorded, never selection
gates          angle median <= 1.0, offset median <= 0.5
               angle p90 <= 2.0, offset p90 <= 1.0
```

## Verdict labels, fixed before the run

```
F1 task and safety PASS
    LATE_A1_FEATURE_ADAPTATION_RESCUES_DIRECT_HOUGH
    RGB -> partially adapted A1 -> token XY -> role queries -> direct Hough
    becomes a line-stage architecture candidate for the first time

F1 clears the 40% reduction threshold but fails the task
    LATE_A1_FEATURE_ADAPTATION_SIGNAL
    next screen is role-encoder capacity, not implemented here

F1 below 40% and F1 ~ F0
    FROZEN_A1_NOT_PRIMARY_LIMIT
    next is ROLE_ENCODER_CAPACITY_SCREEN, not implemented here

F1 train improves and D2 degrades
    LATE_A1_ADAPTATION_OVERFITS
    do not widen the unfreeze
```

The instruction leaves "F0 ~ F1" unquantified, so it is pre-registered here as
`SIMILAR_TO_F0`: both D2 medians within 5% of F0's, relative.
`LATE_A1_ADAPTATION_OVERFITS` is read as F1's train CE below F0's at 25,545
while either D2 median is above F0's.
If the observed combination matches no label, it is
`LATE_A1_ADAPTATION_INCONCLUSIVE` and every condition is reported as it fell.

The four labels are not mutually exclusive by construction -- a run could both
overfit and sit within 5% of F0 -- so they are evaluated in the order written,
except that the overfit condition is always reported whether or not it decides
the label.

## What a pass would and would not settle

A pass makes this a *candidate*, nothing more.  `LINE_STAGE_ARCHITECTURE_LOCKED`
needs four things and this screen is the first:

```
1  line-stage D2 task and safety PASS
2  same-protocol replicate in the same qualification class
3  fixed role-shuffle causal PASS
4  whole LINE_DEV, 2,393 frames and 27,684 roles, PASS
```

Until all four, nothing is called `ARCHITECTURE_COMPLETE`, and CIGM stays
blocked.  The integration screen -- twelve lines to CIGM to eight corners to
known-dimension PnP to 6D pose -- is a separate screen that does not open here.

## Forbidden for the duration

```
CIGM, PnP, known dimensions, additional K use, MAP200, image-map return,
GAP-FiLM, RGB stem, data filtering, target sigma, lattice, query count,
attention depth, loss, LR sweep, widening the unfreeze
validation512, untouched, eval56, wood45, final-test
```
