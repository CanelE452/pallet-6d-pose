# Canonical corner audit — PCS-Net gate

Zero training. A0 reproduces the canonical baseline exactly (eval56 50/56,
54/56 centroid, R4 47, 11.5578px; wood 44/45, 45/45, 44, 9.2839px), so the
harness is the same one every canonical number came from.

## Failure funnel

```
eval56   centroid 54/56  ->  centroid + 4 corners 47/56  ->  PnP 50/56
wood     centroid 45/45  ->  44/45                      ->  PnP 44/45
```

Corner level, GT in-frame corners over both sets, n = 778:

```
C-USABLE   (<= 20px)        490   63.0%
C-MISSED   (peak <= 0.30)   145   18.6%
C-WRONG50  (> 50px)          80   10.3%   of which > 100px: 48
C-WEAK     (20-50px)         63    8.1%
```

## Two different corner failures, split by role

```
ID role   n   det  recall   <=20   >50  >100  median err  peak median
 0 near   98   77   78.6%     72     3     1      5.99      0.8762
 1 near   94   61   64.9%     52     6     1      6.42      0.8100
 2 near   94   63   67.0%     55     7     2      6.01      0.8300
 3 near   94   71   75.5%     67     3     2      4.64      0.8810
 4 far    98   83   84.7%     63    12     8      7.30      0.8982
 5 far   101   94   93.1%     68    19    13     10.31      0.8582
 6 far   101   94   93.1%     60    16    12     14.26      0.8299
 7 far    98   90   91.8%     53    14     9     17.49      0.8755

near  recall 71.6%   <=20px 64.7%   >50px 19   median 5.69
far   recall 90.7%   <=20px 61.3%   >50px 61   median 12.36
```

**Near corners fail by not being detected** (recall 64.9-78.6%, worst at IDs 1
and 2) while being accurate when they are (median 5.7px, only 19 gross errors).
**Far corners are detected** (90.7%) **and land in the wrong place** (61 gross
errors, median 12.4px, IDs 5-7 carrying 13/12/9 beyond 100px).  These need
different fixes and the funnel loses 9 eval56 frames at the four-corner step
mostly to the near-side recall.

## The gate: is the right candidate in the map at all?

Top-5 local maxima per channel, NMS radius 3, no threshold applied at
extraction:

```
                              n     top5 has <= 20px   <= 50px
C-MISSED                     145        17.2%          42.1%
C-WRONG50                     80        25.0%          53.8%
C-WRONG100                    48        14.6%          35.4%
C-WEAK                        63        38.1%          98.4%
all failures (MISSED+WRONG50) 225        20.0%          46.2%
  near                       127        16.5%
  far                         98        24.5%
```

**REPRESENTATION_FAILURE, not selection failure.**  On four failures out of
five the correct position is not among the top five candidates at all.  The gate
for PCS-Net asks for at least 50% top-5 availability on C-WRONG50/C-WRONG100;
the measurement is 25.0% and 14.6%.

A structural head can only re-rank candidates that exist.  On 80% of the failing
corners there is nothing correct to re-rank.

## Stage trajectory

```
T5_STABLE_USABLE (H4,H5,H6 all <= 20px)   463   59.5%
T0_EARLY_NO_RESPONSE                      112   14.4%
T2_LATE_CORRUPTION                         26    3.3%
T3_MID_BEST (H4/H5 beats H6 by 10px+)      19    2.4%
T4_LATE_RESCUE                              1    0.1%

of the 225 failing corners: late corruption 26, mid-stage best 11
```

Late-stage corruption is 26 of 225 = 11.6%, well under the 40% the
stage-preserving refinement branch requires.  The refinement idea is not the
main lever either.

## Decision

```
PCS_NET_JUSTIFIED                 NO   gate 2 fails: 25.0% and 14.6% against >= 50%
STAGE_PRESERVING_REFINEMENT_FIRST NO   late corruption 11.6% against >= 40%
LOCAL_REPRESENTATION_FIRST        YES  80% of failures have no correct top-5 candidate
```

**Decision: LOCAL_REPRESENTATION_FIRST.  PCS-Net STOP.**

The structural-line direction is not refuted -- it is untestable here, because
its input would be candidates that do not exist.  Phases G and H (PPD canonical
re-evaluation and oracle incidence re-ranking) were not run: with 20% top-5
availability the oracle re-ranking upper bound is capped at 20% before any line
evidence is considered, so running it would measure the cap rather than the
line.

## What the evidence points at instead

Two separate problems, both in the local representation:

```
near corners   detection recall 71.6%, accurate when found
               -> the response is missing, not misplaced
far corners    detection 90.7%, 61 gross errors, median 12.4px
               -> the response exists and sits in the wrong place
```

Candidate directions, none implemented and none measured here: higher-resolution
local branch or multi-scale features for the near recall, and corner-identity
disambiguation for the far placement, since far corners being confidently wrong
is the signature this programme has recorded before.

## Not run

Phase F (affinity/PnP oracle decomposition), Phase G (PPD canonical
re-evaluation), Phase H (oracle incidence), Phase J (A1 formal paired gate),
figures, and the full test list.  A1's canonical numbers stand from the previous
re-evaluation (eval56 reprojection -12.1%, R4 +3 and PnP +2 with none lost) but
its formal gate is unjudged.

Final-test: UNOPENED.  Zero training steps, zero optimizers, no checkpoint
modified.
