# Why the step axis closes before the encoder changes

`9e1ad5e` records `DIRECT_HOUGH_TOKEN_XY_FULL_FAIL` and that result stands
unchanged.  This document does not revise it.  It states what the failure is
*not yet* allowed to be called, and pre-registers the one screen that decides.

## The claims that are blocked right now

```
FROZEN_A1_FEATURE_LIMIT_CONFIRMED     not claimed
ROLE_ENCODER_LIMIT_CONFIRMED          not claimed
LINE_NATIVE_REPRESENTATION_FAIL       not claimed
```

Three facts in the recorded FULL run block all three.

**The cross-entropy was still falling at the decision step.**

```
step     train CE (last-250 mean)
1,250    8.436930
2,500    7.977469
5,000    7.224050
8,515    6.841119
```

**The geometry was still improving at the decision step.**

```
step     D2 angle median   D2 offset median
5,000       5.448242           2.887970
8,515       4.429565           2.445153
```

**The seen/unseen gap is small.** `D0_SEEN512` 4.218506 against `D2_LINE_DEV512`
4.429565 is a 5.0% difference on angle.  A model that has exhausted its features
does not usually sit 5% away from its own training distribution while its loss
is still dropping.

Against that, `DIRECT_HOUGH_OVERFIT_EXTENDED_PASS` shows the same architecture
reaching 0.338 degree and 0.230 cell on 32 frames with safety cleared, so the
lattice, the target, the scorer and the head can express the answer.  What has
not been shown is that the FULL run was given enough optimizer exposure to find
it.

## The question this screen asks

```
FULL_OPTIMIZER_EXPOSURE_INSUFFICIENT ?
```

Nothing else.  The architecture does not move by one character: frozen A1, token
XY, twelve role queries, the same cross-attention, the same `DirectHoughHead`,
the same theta/rho lattice, the same target, the same cross-entropy, batch,
learning rate, weight decay and seed.  The dead `self.position` stays dead and
stays constructed -- it consumes RNG before the encoder and the head, so
deleting it would change every initialisation and this would stop being a
step-count experiment.

## Exposure

```
current FULL     13,618 frames, batch 8, 1,703 step per pass, 8,515 = 5 passes
this screen      the same pool, 0 -> 25,545 = 15 passes, exactly 3x
```

Fresh from step 0.  The recorded FULL checkpoints store `tag, step, model` and
provenance and **no optimizer state**, so resuming from `step_08515.pth` would
continue a different AdamW trajectory than the one being extended.  Verified by
loading the file, not assumed from the code.

This extension happens once.  There is no 34,060 and no 42,575 whatever the
numbers say.

## Marks and decision

```
1,703    1 pass
5,000    diagnostic only, kept for comparability with the recorded ladder
8,515    5 passes -- the recorded FULL decision point
17,030   10 passes
25,545   15 passes  <- decision, and the only decision
```

No best-step selection.  `D2_LINE_DEV512` is primary; `D0_SEEN512` is
diagnostic and may not be used for any selection, including the observation
above that the gap is small.

Per-role metrics are recorded at 8,515, 17,030 and 25,545 only, and never enter
selection.

## Gates, unchanged

```
task     angle median <= 1.0 degree     offset median <= 0.5 cell
safety   angle p90    <= 2.0 degree     offset p90    <= 1.0 cell
```

The 40% reduction threshold is recomputed from the recorded Q1 image-map
baseline at full precision, from the JSON, not transcribed.

## Verdict labels, fixed before the run

```
A1  task and safety PASS at 25,545
    DIRECT_HOUGH_LONG_SCHEDULE_VALID_CANDIDATE
    reading: FULL_OPTIMIZER_EXPOSURE_WAS_PRIMARY_LIMIT
    does NOT license the claim that frozen A1 is optimal

A2  absolute FAIL, both medians clear the 40% reduction threshold
    DIRECT_HOUGH_LONG_SCHEDULE_SIGNAL
    plus OPTIMIZATION_EXPOSURE_REMAINS_ACTIVE if the last interval still falls

A3  absolute FAIL, weak improvement, CE plateau, geometry plateau
    DIRECT_HOUGH_FULL_OPTIMIZATION_PLATEAU
    this is the only branch that opens the frozen-A1 screen

A4  absolute FAIL, CE still falling strongly
    LONG_SCHEDULE_STILL_OPTIMIZING_BUT_TASK_FAIL
    opens the architecture screen but does not confirm a feature limit
```

The instruction fixes the labels and leaves "weak", "plateau" and "strongly"
unquantified, so those are pre-registered here and are not adjustable after any
number is read:

```
WEAK_IMPROVEMENT     both D2 medians improve < 20% from 8,515 to 25,545
CE_PLATEAU           CE(17,030) - CE(25,545) < 0.02  AND  the last-pass linear
                     slope over the final 1,703 steps is >= -1e-5 per step
CE_STRONG_DROP       CE(17,030) - CE(25,545) >= 0.10
GEOMETRY_PLATEAU     both D2 medians improve < 5% from 17,030 to 25,545
A2 "40% improvement" means the locked Q1 reduction_40 threshold, the same one
   the recorded FULL verdict used -- not an improvement over this run's 8,515
```

If the observed combination matches none of A1-A4, the label is
`DIRECT_HOUGH_LONG_SCHEDULE_INCONCLUSIVE` and every condition is reported as
it fell.  Forcing a branch would be the same error as reading a median-only
`PASS` field as an overall verdict.

## Phase B is not open

Phase B -- unfreezing the last A1 feature block -- runs only if Phase A is not
A1, and only after Phase A's result is committed and a separate
pre-registration is committed.  No Phase B training starts before that.

## The training path is not bit-reproducible, and that is a finding

The long runner re-composes `DH.train_network` from the same pieces rather than
importing it, because the slope bookkeeping and the per-role gating need a loss
history the original does not return.  That re-composition has to be proved
harmless, so `parity` trains both paths for 20 steps and compares state dicts.

The first version of that check demanded bit equality under the default kernels.
It failed at 7.2e-04 -- and then the locked runner failed the same check
**against itself**:

```
locked runner vs locked runner, 20 steps, default kernels
  max abs delta   1.435e-03      21 of 28 tensors differ
  largest         head.hypothesis.body.2.bias
  decoded metrics identical at the mark (9.9706 deg / 7.6035 cell)
```

So the assumption the check encoded was false, and the instrument was replaced
before the run rather than the threshold being loosened after it.  Parity is now
asked with `torch.use_deterministic_algorithms(True)`, where the locked runner
*is* exactly reproducible, and a control run proves that before the comparison
is believed:

```
deterministic_control   locked vs locked      0.000e+00
structural_parity       locked vs recomposed  0.000e+00
```

The real run never enables deterministic mode, so it stays in the same numerical
regime as the recorded FULL.

This also bears on the 16-21% component drift already on record as `UNRESOLVED`:
the gradient of `head.hypothesis` is a reduction over 24k hypotheses times batch
times twelve roles, and reordering that accumulation is enough to move
parameters at 1e-03 after twenty steps.  That is a mechanism, not a measurement
-- it is not tested here and is recorded as `[추정]`.

## Reproduction diagnostic

The fresh long trajectory passes through 8,515.  That point is compared with the
recorded FULL at full precision from the JSON.  A component drift of 16-21% is
already on record as `UNRESOLVED`, so **no tolerance is invented here and no
hard block is attached**.  If the two differ, the difference is reported as the
finding.

## Forbidden for the duration

```
CIGM, PnP, known dimensions, additional K use, MAP200, image-map return,
GAP-FiLM, RGB stem, data filtering, target sigma, lattice, query count,
attention depth, loss, LR sweep
validation512, untouched, eval56, wood45, final-test
```
