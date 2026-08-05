# Decision

**GLOBAL_RESPONSE_PROBLEM.  The target-width track stops here.**

The 13 frames that failed the compatibility gate are not centroid failures.  On
12 of them not a single corner channel clears 0.30 either; the model has no
response to the pallet at all.  Widening the centroid target cannot recover a
signal that is not there, and the counterfactuals confirm it: an oracle centroid
at the true position builds an object on all 13 frames and still solves **zero**
poses, because a median of one corner is available to associate.

## Gate

```
                 condition  value
─────────────────────────────────
                        T1      0
                        T2     13
                        T4      0
          U1_objects_built      0
          U0_objects_built     13
            U0_pnp_success      0
          U2_objects_built     11
        BASE_objects_built      0
           U1_catastrophic      0
role_specific_target_width  False
       dual_bandwidth_head  False
         width_not_primary   True
             target_defect  False
```

```
ROLE_SPECIFIC_TARGET_WIDTH   not supported   T1 = 0 (needs >= 8), U1 builds 0 objects (needs >= 10)
DUAL_BANDWIDTH_HEAD          not supported   T1 = 0; the premise fails, and the oracle centroid solves 0 poses
WIDTH_NOT_PRIMARY            supported       T2 = 13 (needs >= 7), and U0 oracle -> 0 PnP
TARGET_DEFECT                not supported   T4 = 0
```

Both clauses of WIDTH_NOT_PRIMARY hold independently.

## What this changes

The previous audit concluded CONFIG_ONLY_FAIL and proposed role-specific target
widths (corner ~2.0, centroid >= 2.5) as the direction.  That proposal was made
before these 13 frames were opened, and it rested on a premise this audit has now
falsified.  **The width numbers themselves still stand** -- the deployment gate
does require a centroid target of 2.5 and a corner target of 2.0, and ep57's
centroid is 0.41 sigma short -- but width is not what is blocking these frames,
so a width-only run would not restore deployment compatibility.

To be precise about scope: widening remains the correct fix for the *74 frames
where the response exists but the blur kills it*.  It is not a fix for the 13
where there is no response.  Those are two different problems that the
compatibility gate happened to fail on the second.

## Where this connects

The no-response population is near, large and cut by the frame edge -- 10 of 13
truncated against 1 of 13 in the matched controls, and no pair where the dead
frame has more in-frame GT corners than its control (3 tied, 10 fewer).  That is the
truncation failure mode already recorded twice in this programme (the V<8
population, and the near-face border cut), not a new phenomenon.

## Next admissible experiment

1. Do not train a role-specific target run yet.  It would be scored on a set
   whose blocking frames it cannot address.
2. Establish first whether the truncated no-response population is recoverable
   at all by this architecture: measure ep57 on the truncated frames with the
   reflect-padding inference path that this project already uses for truncation,
   and see whether the response returns.  That is a zero-training measurement.
3. If padding restores the response, the deployment path needs the padding step,
   not a new head, and the width question returns to the 74 frames where it
   actually applies.
4. If it does not, the question is the truncation policy in supervision, which
   is a data and labelling question rather than a bandwidth one.

## Confirmatory set

N87 shares 12 frames with eval56.  Any confirmatory evaluation from here uses
**eval44-clean** (eval56 minus those 12) **plus wood**, fixed now, before any
run.
