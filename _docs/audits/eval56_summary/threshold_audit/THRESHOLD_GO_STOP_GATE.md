# Decision — the canonical threshold stays at 0.30

Nine pre-registered arms, both evaluation sets, zero training steps, ep57 read
only.  An arm is accepted only if it passes all ten checks on **both** sets.

```
arm  eval56  wood  verdict     eval56 failed  wood failed
─────────────────────────────────────────────────────────
 T1    FAIL  PASS   REJECT               1,4             
 T2    FAIL  FAIL   REJECT               1,4    5,6,7,8,9
 T3    FAIL  FAIL   REJECT  1,4,5,6,7,8,9,10    5,6,7,8,9
 T4    FAIL  FAIL   REJECT  1,4,5,6,7,8,9,10  4,5,6,7,8,9
 R1    FAIL  PASS   REJECT               1,4             
 R2    FAIL  FAIL   REJECT                 1    5,6,7,8,9
 R3    FAIL  FAIL   REJECT  1,4,5,6,7,8,9,10    5,6,7,8,9
 C1    FAIL  PASS   REJECT               1,4             
```

**No arm passes both sets.  The threshold stays at 0.30.**

wood T1, R1 and C1 pass, but each fails eval56 on check 1 (PnP never reaches
52) and check 4 (worsened >= improved).  A change that helps one pallet and
regresses the canonical set is exactly the cross-pallet case Phase J forbids.

## The premise this audit was built on turned out to be wrong

The audit was set up to confirm that PFDR N3's eval56 PnP 50 -> 52 was a
threshold-crossing artifact.  It is not, and the evidence is unambiguous.

The 14 near corners N3 newly detected had these **baseline raw peaks**:

```
0.0087  0.0095  0.0103  0.0108  0.0111  0.0124  0.0127
0.0156  0.0156  0.0169  0.0210  0.0284  0.0437  0.0448
```

Their peaks after N3:

```
0.3239  0.3358  0.4118  0.4427  0.4637  0.4712  0.4767
0.5760  0.5917  0.5928  0.6934  0.8298  0.8382  0.9488
```

The residual lifted them by **+0.31 to +0.94**.  These corners were not sitting
just under the gate; the base produced essentially no response at all.  For
comparison, the whole eval56 near population holds just **six** corners in
[0.20, 0.30) -- 1 in [0.275,0.30), 1 in [0.25,0.275), 3 in [0.225,0.25),
1 in [0.20,0.225) -- against 75 below 0.20 and 143 already accepted.  That is
why dropping the gate to 0.20 recovers 8 corners while N3 recovered 14 that no
threshold in this range can reach, and why the two do not rescue the same
frames.

**Correction to the earlier reading.**  The +0.0014 median peak change quoted
for N3 was computed across all 224 near corners, where 143 already-detected
corners dominate.  The corners that actually crossed moved by two to three
orders of magnitude more than that median.  Taking the population median as the
per-corner effect was the error; it produced the threshold-artifact hypothesis
that this audit has now falsified.

## What N3 actually did, and why it still does not qualify

N3 created a strong response where ep57 had none.  That is a representational
change, not a decision-boundary shift.  But the corners it created are not
accurate:

```
eval56  14 new corners   median 21.7px   within 20px  5/14 (36%)   beyond 50px 0
wood     1 new corner    median 39.7px   within 20px  0/1          beyond 50px 0
```

Against this audit's own precision bar (70% within 20px) N3 would fail check 8
as well.  It buys correspondences of roughly 20px quality, which is enough to
clear `valid < 4` and not enough to place a pose -- consistent with N3's near
error rising 4.676 -> 5.309px and its common-success reprojection worsening.

So the corrected statement is: **N3's PnP gain is a detection-recall gain of
mediocre corners, not a threshold artifact and not a pose improvement.**  The
conclusion about N3 is unchanged; the mechanism attributed to it was wrong.

## Near-only against global (Phase I)

- R2 is not worse than T2 on common-success reprojection (both -2.53%, R2
  1 improved / 1 worsened against T2 1/2).  Condition met.
- R arms must rescue at least two more eval56 frames than the far-only control
  C1.  R1 and R2 rescue 0, R3 rescues 1, C1 rescues 0.  **Condition failed.**
- wood shows regression in every R arm at 0.25 and below (>50px 40 -> 42,
  >100px 36 -> 38, new corners at 402px median).  **Condition failed.**

**A near-specific acceptance threshold is REJECTED.**  Where near-only and
global differ at all, the difference is one corner.

## Verdict

```
threshold 0.30                     KEEP
global threshold candidate         REJECTED  (fails eval56 on every arm)
near-specific threshold candidate  REJECTED  (Phase I conditions 2 and 3)
N3 = threshold-crossing artifact   FALSIFIED (base peaks 0.009-0.045, not ~0.30)
N3 = pose improvement              still REJECTED (36% of new corners within 20px,
                                   common-success reprojection worsens)
base ep57                          UNCHANGED
```

No further threshold is tried.  The grid was fixed in advance and is not
extended after seeing these numbers.
