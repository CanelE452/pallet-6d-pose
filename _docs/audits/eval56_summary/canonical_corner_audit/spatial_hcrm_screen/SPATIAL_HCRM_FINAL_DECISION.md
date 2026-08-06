# Spatial HCRM screen: HCRM_UNSTABLE

Holdouts are whole source groups from the keyed roots; the 13,069
frames with no scene metadata are train-only, so validation and
untouched are a KEYED_SOURCE_GROUP_HOLDOUT and not an unbiased sample
of the synthetic distribution.  The adapter trained on unaugmented
source frames, which keeps the arm comparison single-variable and
leaves its robustness untested.

```
{
 "decision": "HCRM_UNSTABLE",
 "spatial_arm": "H2_seed1",
 "pointwise_arm": "H1_seed1",
 "gates": {
  "SYNTHETIC_HCRM_SIGNAL": {
   "1_id12+8pp": false,
   "2_near_le20+5pp": false,
   "3_near_gt50<=+5pp": true,
   "4_far_gt50<=+5pp": true,
   "5_two_of_three_seeds": false,
   "6_H2ZERO_parity": true
  },
  "SPATIAL_MODULE_VALUE": {
   "id12_or_le20_+3pp": false,
   "better_than_shuffle": false
  },
  "CANONICAL_NEAR_GAIN": {
   "eval56_id12+8pp": false,
   "eval56_R4+2": false,
   "wood_id12_no_drop": true,
   "wood_R4_drop<=1": true
  },
  "POSE_SAFETY": {
   "eval56_PnP_no_drop": true,
   "wood_PnP_no_drop": true,
   "reproj_degradation<=5%": true
  },
  "FAR_SAFETY": {
   "far_gt50<=+10%": true,
   "far_recall_drop<=5pp": true
  },
  "STABILITY": {
   "two_of_three_seeds": false,
   "id12_range<=10pp": true
  }
 },
 "gates_passed": {
  "SYNTHETIC_HCRM_SIGNAL": false,
  "SPATIAL_MODULE_VALUE": false,
  "CANONICAL_NEAR_GAIN": false,
  "POSE_SAFETY": true,
  "FAR_SAFETY": true,
  "STABILITY": false
 },
 "module_status": {
  "A1_BALCH": "KEEP",
  "Spatial_HCRM": "STOP",
  "dense_line": "STOP",
  "twelve_edge": "VALID_BUT_DEFERRED",
  "CIGM": "VALID_BUT_BLOCKED",
  "fusion": "STOP",
  "RCIM": "DEFERRED"
 },
 "holdout_caveat": "validation and untouched are KEYED_SOURCE_GROUP_HOLDOUT; 13,069 unkeyed frames are train-only, so they are not an unbiased sample of the synthetic distribution",
 "train_input_mode": "NO_AUG_SOURCE"
}
```
