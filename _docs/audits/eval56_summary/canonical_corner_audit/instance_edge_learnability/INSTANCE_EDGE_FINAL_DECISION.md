# Final decision: DIRECT_12EDGE_HEAD_NOT_LEARNABLE

```
{
 "decision": "DIRECT_12EDGE_HEAD_NOT_LEARNABLE",
 "primary_arm": "L12-MS",
 "detail": {
  "taxonomy": [
   "LOCALIZATION_FAILURE"
  ]
 },
 "multiscale": {
  "verdict": "F50_SUFFICIENT",
  "checks": {
   "1_untouched>=10pp": false,
   "2_canonical>=10pp": false,
   "3_no_pnp_drop": true,
   "4_two_of_three_seeds": true
  },
  "untouched_gain_pp": 1.1367478025693045,
  "canonical_gain_pp": {
   "eval56": -2.455357142857143,
   "wood": -0.5555555555555556
  },
  "seeds_same_direction": 3
 },
 "architecture": {
  "IAEH": "STOP",
  "CIGM": "STOP",
  "fusion": "STOP",
  "next_case": "DIRECT_INSTANCE_CHANNEL_STOP"
 },
 "oracle_reference": {
  "eval56": {
   "set": "eval56",
   "mode": "O12",
   "n_frames": 56,
   "le20": 0.9866071428571429,
   "le50": 0.9955357142857143,
   "gt100": 0.0,
   "median": 4.683842999069036,
   "pnp": 56,
   "reference_le20": 0.987,
   "reference_pnp": 56,
   "le20_delta": -0.00039285714285708373,
   "parity": true
  },
  "wood": {
   "set": "wood",
   "mode": "O12",
   "n_frames": 45,
   "le20": 0.9611111111111111,
   "le50": 0.9722222222222222,
   "gt100": 0.013888888888888888,
   "median": 7.992492955535445,
   "pnp": 45,
   "reference_le20": 0.961,
   "reference_pnp": 45,
   "le20_delta": 0.00011111111111117289,
   "parity": true
  }
 },
 "note": "O12 is a representation-capacity oracle using ground-truth geometry; the learned arms are a separate claim."
}
```
