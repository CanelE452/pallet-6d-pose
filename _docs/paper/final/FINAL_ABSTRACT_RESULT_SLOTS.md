# Final abstract result slots

Which numbers may appear in the abstract, and which may not. Every value here was
read from a frozen artifact; the traceable map is
`generated/RESULT_SOURCE_MAP.json`.

The earlier `_docs/paper/ABSTRACT_RESULT_SLOTS.md` is preserved as a historical
generator output. It must not be used: its `Improvement` slot holds **−9.0 %**, so
copying that template into an abstract would carry a premise the data refutes.

## CONFIRMATORY — Tier A, may appear in the abstract

```text
slot                          value                    source
──────────────────────────────────────────────────────────────────────────────────
Synthetic-only detection      0.975  (n = 319)         R0.subgroups.ALL
Synthetic-only AUROC          0.9921 (319 vs 2,689)    R0.subgroups.ALL.auroc
Synthetic-only keypoint       6.616 px median          R0.subgroups.ALL.corner_median_px

Night detection R0            0.840  (n = 50)          R0.subgroups.Nighttime
Night detection naive ST      0.960  (n = 50)          R1_NAIVE.subgroups.Nighttime
Night detection confidence    0.980  (n = 50)          R2_CONF.subgroups.Nighttime
Night detection full filter   0.960  (n = 50)          R5_PROPOSED.subgroups.Nighttime

AUROC full filter             0.9953                   R5_PROPOSED.subgroups.ALL.auroc
FPR95 R0                      0.0417                   R0.subgroups.ALL.fpr95
FPR95 full filter             0.0283                   R5_PROPOSED.subgroups.ALL.fpr95
Night FPR95 R0                0.1949                   R0.subgroups.Nighttime.fpr95
Night FPR95 full filter       0.0588                   R5_PROPOSED.subgroups.Nighttime.fpr95

Keypoint, full filter         7.210 px median          R5_PROPOSED.subgroups.ALL
Keypoint, best adapted arm    6.999 px (R4)            R4_CONF_REMOVE.subgroups.ALL
                              — still above R0's 6.616
```

All from `data/pallet/results/paper_eval_v1/arms/ARM_RESULTS.json`.

### Two mandatory caveats on the night-detection number

```text
1  0.840 -> 0.960 is NOT the geometry filter's achievement.
   Naive self-training reaches 0.960 unaided, and confidence-only reaches 0.980.
   Permitted:  "self-training raises nighttime detection from 0.840 to 0.960"
   Forbidden:  "our geometric filter raises nighttime detection"

2  The overall detection gain is not separated from noise.
   R0 vs full filter, paired bootstrap:
     detection   p_better = 0.121 (frame)   0.244 (session-clustered)
   Source: data/pallet/results/paper_eval_v1/PAIRED_UNCERTAINTY.json
```

The nighttime subgroup has N = 50 and is plastic-only. Every use of it carries that N.

### The localisation result, with its uncertainty

```text
R0 vs full filter, paired bootstrap, probability the filter is better
  corner                 0.028 (frame)   0.065 (session-clustered)
  pooled corner median   0.006 (frame)   0.095 (session-clustered)
```

Statable: localisation **did not improve**; the direction is a small degradation,
frame-level significant and not session-level significant.
Not statable: "self-training harms localisation" without that caveat.

## DIAGNOSTIC ONLY — Tier B, never in the abstract

May appear in Discussion or Appendix, always labelled development evidence.

```text
slot                              value            source
──────────────────────────────────────────────────────────────────────────────────
Axis permutation R0               0.047            AXIS_FAILURES.json
Axis permutation V3-B             0.041            V3_DEV_METRICS.json
Axis permutation, ambiguous       0.096 -> 0.084   q >= 0.75 subgroup, n = 83
Reliability score AUC             0.7625 (n = 79)  V5_MECHANISM_CHECK.json
Frame-level separability AUC      0.8116 (n = 194) FILTER_SEPARABILITY.json
Corner-level separability AUC     0.7259 (n=1,979) FILTER_SEPARABILITY.json
Corners removed by conf floor     0                FILTER_SEPARABILITY.json
Pseudo corner-gross reduction     0.2078 -> 0.1823 V5_MECHANISM_CHECK.json
Repair candidates                 ~1.2 % of corners V4_REPAIR_PROXY.json
Teacher consensus tail            p90 worsens in all three probes
```

The axis-permutation improvement 0.047 → 0.041 belongs to V3-B, a post-hoc
development variant. It is **not** promoted to an abstract number and V3-B is not
retroactively the proposed method.

## BLOCKED — may not be written at all

```text
slot                    status
────────────────────────────────
yaw                     BLOCKED
rotation error          BLOCKED
translation error       BLOCKED
6D pose AUC             BLOCKED
ADD / ADD-S             BLOCKED
3D IoU                  BLOCKED
5cm5deg                 BLOCKED
```

```text
POSE_METRICS_STATUS = BLOCKED
blocked_reason      = POSE_METRICS_BLOCKED_NO_RELIABLE_AXIS_SELECTOR;
                      FINAL_MANIFEST_NOT_FROZEN;
                      wood: CANONICAL_MIGRATION_NOT_PASS;
                      SYMMETRY_NOT_FROZEN
declaration         _docs/paper/POSE_METRIC_READINESS.md
machine verdict     data/pallet/results/paper_eval_v1/EVALUATOR_CONTRACT.json
```

The blocker is algorithmic, not a matter of unfinished labelling: the best axis
selector measured reaches 0.65 against a gate of 0.95, so additional manual review
would not open these slots.

## Unavailable

```text
session-cluster bootstrap 95% CI    UNAVAILABLE_FOR_CURRENT_DEV_NEGATIVE_CAPTURE
NME for R0-CONT, R1, R2, R3, R4     never computed by the V1 evaluator
catastrophic (>40 px) rate on the   only exists on the n = 1,979 teacher-probe
  full 319 population                 sub-population
```

Report these as unavailable. Do not substitute a value from a different population.

## Subgroup hazard

```text
subgroups.Nighttime        N = 50    plastic only
subgroups.Lighting_night   N = 106   plastic + wood
```

Two different nighttime populations. Never interchange them, and always print N.
