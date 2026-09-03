# Temporal evaluation closure

The refinement coordinates are untouched. Nothing was inferred again, no flow was
recomputed, no threshold was chosen. Only the scoring contract changed — and it
turns out the scoring cannot be done at all.

```text
FORMAL_TEMPORAL_PILOT = POPULATION_LIMITED
```

## Why the pilot needed re-scoring

Four defects, all in my evaluation rather than in the method.

```text
1 population   the lock excluded evaluation-ineligible centres; the builder
               filtered only FT_OVERLAP and admitted DEV_UNVERIFIED
2 2D metric    median of per-frame medians and of per-frame p90s, not the
               corner error distribution; a two-corner frame weighed as much
               as an eight-corner frame
3 6D path      a private width/depth selector instead of the frozen
               predict_pose_without_gt that every other pose result uses
4 coverage     ACCEPTABLE declared at 0.90, chosen while writing the evaluator
```

## Eligibility audit

```text
original centres     109
formal eligible      0
formal recordings    0

usage_role of all 109   DEV_SUPPORT
exclusion reasons       evaluation-ineligible (109)
                        workspace UNVERIFIED_LEGACY (109)
```

Every centre the pilot used carries legacy annotation the workspace preserves but
does not treat as paper eligible. Under the population contract the lock actually
stated, none of them qualifies.

## This zero is structural, not bad luck

```text
paper-eligible frames (EVAL_LABELED)            402
of those, inside PAPER_EVAL 319                 319
of those, outside PAPER_EVAL                     83
where the 83 live                    plastic_night_01
their source recording               real_unlabeled_night_20260830
does that recording feed PAPER_EVAL  yes
```

So every paper-eligible plastic positive frame sits inside a recording the lock
excludes whole. The lock asked for centres that are both paper-eligible and drawn
from recordings that supply nothing to PAPER_EVAL, and in this repository no plastic
frame satisfies both. It is not that too few eligible centres were sampled; none can
exist under the contract as written.

There is a separate reachability gap worth recording: the pilot indexed only
`raw_data/outside` and `raw_data/night`, so the incoming pool holding
`plastic_night_01` was never searched, and 84 centres were dropped as unresolvable.
Closing that gap would not have changed the verdict, because the recording-level
exclusion removes those frames anyway.

## What this means for the earlier result

The earlier `FAILED_TO_IMPROVE` was computed on a population the lock excluded, with
a summary-of-summaries metric and a private pose selector. It is kept, unedited, and
reclassified.

```text
original pilot   N = 109, K = 8
classification   EXPLORATORY_DIAGNOSTIC_ONLY
not              a preregistered formal result
```

## Exploratory diagnostics retained

These describe mechanism, not performance, and are not performance claims.

```text
observations per corner median      7.0
forward-backward rejection          8.1%
geometry coverage                   94.5%
consensus displacement              1.81 px
raw teacher error against GT        7.90 px
displacement / error                0.23
corners moving toward GT            50.0%
```

The temporal median moves each coordinate about a quarter of the distance the
teacher is wrong by, in a direction uncorrelated with the truth. That is what
neighbouring predictions sharing a single error looks like. It remains the most
useful thing the pilot produced, and it is an observation about why a fusion of
correlated observations cannot help — not a measured performance result.

The earlier note that the geometry step returned identical 6D numbers is also kept as
an observation. It was seen under the pilot's own selector; whether the frozen path
reproduces it was never checked, because there is no formal population to check it on.

## What would make a formal run possible

- relaxing the whole-recording exclusion to a frame-level disjointness rule, which is
  a contract change and the user's decision, not one to make while holding a result
- or annotating centres in a recording that supplies nothing to PAPER_EVAL

Neither is done here.

## Final

```text
FORMAL_TEMPORAL_PILOT = POPULATION_LIMITED
original 109-centre run = EXPLORATORY_DIAGNOSTIC_ONLY
refinement modified      NO
new inference            0
student training         0
```

`NEXT_ACTION = PAPER_FRAMING_DECISION`

