# Closure plan

Ordered, and the order is not negotiable. Each step gates the next.

```text
STEP 1   selector closure                    <- current step, and it decides everything
STEP 2   wood symmetry + canonical closure
STEP 3   signed-axis GT closure
STEP 4   evaluator unit tests                 DONE — 19 tests pass
STEP 5   final manifest freeze
STEP 6   POSE_METRICS_STATUS = READY
STEP 7   only then evaluate R0 ... R5 with one frozen evaluator
```

If STEP 1 fails, the track ends at `BLOCKED_SELECTOR`. Training is not expanded to
force it through.

## STEP 1 — the selector, and why priority A is already answered

`CURRENT_BLOCKERS.md` carries the measurements. The short form:

```text
both hypotheses geometrically valid    138 / 138  (100%)
|score_margin| AUC for correctness     0.5562
current selector accuracy              0.6014
constant "always short-face-front"     0.5725
best fail-closed operating point       coverage 0.145 at accuracy 0.750
```

### Priority A — deterministic selector from existing outputs

```text
VERDICT   insufficient as it stands, and unlikely to be fixed by reweighting
```

The projective fit accepts **both** hypotheses in every one of 138 frames, so there
is no hard geometric residual for a better weighting to exploit. Pooled, the margin
carries 0.556 AUC and no threshold converts coverage into accuracy.

One qualification, and it matters. The signal is **not** uniformly absent:

```text
eval_cad       18 / 18   with both classes present (long 3, short 15)
               a constant predictor caps at 15 / 18 there
eval_night08    4 / 12   with both classes present (long 11, short 1)
```

So the geometric score genuinely discriminates in at least one session and collapses
in others. The honest statement is not "no information exists" but "the information
is condition-dependent and absent exactly where it is needed". Reweighting the same
five terms is unlikely to recover it, because the frames where it fails are the ones
where both hypotheses fit equally well.

The reason is the object:

```text
plastic footprint 1.10 m x 1.30 m    aspect ratio 1.182
```

Swapping 1.10 and 1.30 is an 18 percent change that unknown yaw and depth largely
absorb. The unit test `test_set_distance_shrinks_as_the_footprint_approaches_square`
records the limiting behaviour: as the footprint approaches square the cost of the
wrong hypothesis goes to zero continuously, and 1.182 is close to that limit.

Priority A is therefore closed as insufficient, on measurement rather than on
opinion. It could be reopened only by a cue not currently in the selector's input
set — and the geometric inputs are exhausted.

### Priority B — synthetic-trained semantic-axis head

```text
STATUS   not attempted. Design and cost reported below; no training started.
```

The one deployment-available signal the geometric selector never consumes is
**appearance**. A pallet's fork-entry face and its closed side are visually
different, and that difference does not depend on the 18 percent aspect ratio.

Shape of the proposal, stated so it can be approved or rejected without ambiguity:

```text
input        the RGB crop the keypoint model already produces, plus the predicted
             keypoints. No GT of any kind.
output       binary: which footprint axis faces the camera, with a confidence
             that supports POSE_UNRESOLVED
supervision  synthetic only. The renderer knows the object axes exactly, so the
             label is free and no real target pose annotation is touched.
scope        a small auxiliary module. R0 through R5 are NOT retrained and NOT
             modified; the head is evaluated as a separate pose-disambiguation
             stage on top of frozen keypoint predictions.
gates        the same four, pre-registered: overall 0.95, night 0.90,
             per-session 0.85, coverage 0.95
```

Risks that must be stated before anyone spends time on it:

```text
sim-to-real   this is exactly the transfer that failed for keypoints. An appearance
              classifier trained on renders may not survive the real night imagery
              where the selector is currently weakest (eval_night08 at 0.333).
night         the sessions with the worst selector accuracy are the ones with the
              least appearance information. The gate requires 0.90 there.
prior work    this repository has a recorded negative result for a learned
              polarity predictor: synthetic accuracy 0.95 collapsing to 0.023 on
              real data, while the oracle stayed healthy. That is the same shape of
              task. It does not prove failure here, but it is not encouraging and
              it must not be omitted from the decision.
```

**No training starts without explicit approval.** That is the boundary of this
first pass.

### Priority C — temporal / multi-frame

```text
STATUS   deferred, not attempted
```

A vehicle approaching a pallet sees it from angles that break the ambiguity; a
single frame near the degenerate viewpoint does not. This is the only route with a
clear information argument in its favour, and it is also the largest change to the
deployment assumption. It is future work, and it is honest to say so.

## STEP 2 — wood

Delegated audit in progress. Two questions, answered independently:

```text
symmetry     can wood's symmetry_status become FROZEN, with an equivalence set
             justified by geometry rather than convention?
canonical    does keypoint ordering agree with canonical 3D, physical W/D/H, and
             the fork-entry axis?
```

Registry state at the time of writing:

```text
plastic   geometry FROZEN   symmetry FROZEN      contract present
wood      geometry FROZEN   symmetry UNREVIEWED  contract null
```

If either fails, wood is excluded and the pose table is plastic-only. Wood is not
merged in to make the table look complete.

A constraint the audit was given explicitly: a rotation that changes which face the
forks enter is **not** the same pose, however symmetric the shape looks. The unit
test `test_unrestricted_adds_forgives_the_square_swap` shows what happens if that
rule is relaxed — unrestricted ADD-S returns exactly 0.0 for a 90-degree swap on a
square footprint, which would erase the selector failure from the metric.

## STEP 3 — signed-axis GT

```text
146 / 146 frames    pose_status UNCONFIRMED_SIGNED_AXIS
                    axis_assignment_confirmed False
```

This is evaluation-GT work. Before any review begins, three things are written down:
the review protocol, the inter-annotator rule, and the ambiguity policy — because a
reviewer who can see model predictions will unconsciously confirm them.

Hard rule: confirmed axis labels are **GT reference only**. They never become
selector inputs and never become training targets.

## STEP 4 — evaluator unit tests

```text
DONE   scripts/paper/pose_metric_closure_v1/test_pose_metric_contract.py
       19 tests, all passing, no model loaded
```

Covers identity, known translation in centimetres, known yaw, the axis about which
yaw is measured, the 90-degree swap cost, ADD versus unrestricted ADD-S, AUC on
analytic sequences, model diameter, and that a blocked gate returns nulls rather
than numbers.

Two implementation gaps found while writing them are recorded in
`METRIC_DEFINITIONS.md` and must close before any pose number is produced:

```text
IoU3D      not implemented in challenge/evaluation_v2/; only in
           scripts/stage0/real_eval/re_metrics.py
pose_auc   two implementations with different discretisation (1001 vs 100 steps);
           one must be declared canonical
```

## STEP 5 — freeze

`POSE_FINAL_EVALUATOR_LOCK.json`, created only when steps 1-4 pass, pinning:
selector implementation and checkpoint hashes, PnP solver, intrinsics rule, registry
hash, symmetry contract hash, ADD model points, AUC integration range, IoU3D
implementation, failure and unresolved handling, population manifest hash, and
subgroup definitions. After that, nothing changes.

## STEP 6 and 7

`POSE_METRICS_STATUS = READY`, then one evaluation pass over R0, R0-CONT, R1, R2,
R3, R4, R5 with the frozen evaluator, on a population opened for the first time.

A READY evaluator does not license the sentence "self-training improves 6D pose".
Four conditions must hold together — evaluator ready, comparison fair, direction
favourable, uncertainty acceptable — and if pose does not improve, that is reported.

## Honest statement of the likely outcome

Priority A is closed as insufficient. Priority B is unproven and carries a recorded
negative precedent for the same shape of task. Priority C is out of scope.

The realistic outcome of this track is `BLOCKED_SELECTOR`, and the paper reports why
the pose layer could not be opened. That is a legitimate result: it explains a
concrete failure mode of monocular pose estimation on near-square industrial objects,
and it is more useful than a pose table produced by an evaluator that quietly solves
the ambiguity on the model's behalf.
