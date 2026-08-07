# SUPPORTING_LINE_MAP_GENERALIZATION_FAIL

Neither arm reaches the task budget on unseen frames, and neither approaches it.
`confirm6k` is blocked.

```
LINE_DEV512, 5,921 supported roles, epoch 5 only

arm                overfit32          e1              e3              e5
M0_F50_SLINE     0.1268 / 0.0586  9.4189 / 3.9292  7.2438 / 3.0222  6.8450 / 2.7717
M1_F50_RGB_SLINE 0.1426 / 0.0614 11.4823 / 4.8387  8.2026 / 3.5537  7.3931 / 3.0916

budget            <=1.0 deg / <=0.5 cell      APPROACH  <=1.5 / <=0.75
safety p90        <=2.0 deg / <=1.0 cell
M0 e5 p90          52.2924 / 24.7414
M1 e5 p90          53.0513 / 26.1205
```

The gap is not precision.  M0 goes from 0.1268 degree on memorised frames to
6.8450 on unseen ones -- **54 times worse** -- and its p90 is 52 degrees, which is
not a slightly misplaced line but a line pointing somewhere else entirely on a
large minority of roles.  This is a generalisation collapse.

## The training signal is nearly flat

```
arm    train map loss   e1        e3        e5
M0                    0.08698   0.07993   0.07897
M1                    0.08940   0.08121   0.07999
```

Four epochs of training move the map loss by about 9%, and the decoded error by
27%.  Both curves are still descending at epoch 5 but nowhere near a scale that
would close a 54x gap.

## What the model did learn

Role identity, and unambiguously.

```
arm    normal e5   shuffled    margin    gate
M0        6.8450    50.8995    +44.05    >= +5
M1        7.3931    50.1872    +42.79    >= +5
```

Both pass `ROLE_SEMANTICS_LEARNED` by a factor of nine.  Permuting the twelve
channels destroys the result, so channel k really is role k -- the model knows
*which* structural line each output is responsible for.  It does not know
*where* to put it.

Reload parity is exactly 0.0 for both arms, and the finite fraction is 1.000.

## Map diagnostics

Reported, not used for selection.

```
arm    positive MSE   negative MSE   map NCC   mass      peak
M0          0.1345         0.0233     0.2912   1468.5    0.820
M1          0.1372         0.0220     0.2824   1438.0    0.787
```

Against the overfit values (positive 0.042, NCC 0.775, peak 0.982) every one has
degraded.  NCC is a correlation diagnostic only and does not partition the map
into right and wrong fractions; what it does say is that the predicted and
target maps are far less related on unseen frames than on memorised ones.  The
probability mass roughly doubles while the peak drops, which is consistent with
a broader, less committed ridge -- consistent with, not proof of.

## Full versus partial

```
arm    IN_FRAME_FULL (n=5,109)      IN_FRAME_PARTIAL (n=812)
M0     6.6390 / p90 49.7442         8.1009 / p90 60.6331
M1     7.2783 / p90 50.9258         8.3258 / p90 61.3433
```

Partial roles are about 20% worse in the median, not an order of magnitude.  The
extent-mismatch correction still holds at n=812 -- this is not a return of the
old collapse, and it must not be explained as `TARGET_SEMANTICS_MISMATCH`.  What
remains is an ordinary and modest generalisation residual on clipped roles.

## The RGB stem

M0 beats M1 at every epoch and on every metric, now on unseen data, which is the
comparison the overfit run could not make.  One seed and one configuration, so
this orders the two arms rather than closing the question; but nothing here
supports `LINE_SUPERVISED_RGB_ADAPTATION_REQUIRED`.

## Standing

```
SUPPORTING_LINE_MAP_GENERALIZATION_FAIL       both arms, no approach
confirm6k                                     BLOCKED
role semantics                                LEARNED (both)
reload parity                                 0.0 (both)
STRUCTURAL_LINE_MAP_CAPACITY                  measured, and it fails
off-frame                                     NOT TESTED
CIGM / PnP                                    NOT BUILT
```

Per the locked protocol this is the condition that opens an
architecture-capacity screen, and its two candidates are recorded here and not
implemented, one factor each:

```
A  head capacity        deeper or wider map decoder
B  spatial resolution   MAP100 against MAP200
```

They must not move together.  The evidence available so far does not choose
between them: the head fits 32 frames to 0.13 degree, which argues its capacity
is not the binding constraint on *fitting*, while the collapse to 6.8 degrees on
unseen frames is about what it can infer from a frozen 50x50 feature upsampled
to 100x100 -- which is as much a resolution question as a depth one.

No PnP, no CIGM, no dimensions, no `validation512`.  `untouched`, `eval56`,
`wood45` and final-test remain unopened.
