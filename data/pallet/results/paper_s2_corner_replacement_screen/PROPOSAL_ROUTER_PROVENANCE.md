# Provenance

```
{
 "base_ckpt_sha256": "c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896",
 "epoch5_ckpt_sha256": "aad97f6bead6067d58ae178e99e404c738f63708e38de622ca3a6f07087da4e5",
 "retraining_steps": 0,
 "decoders": [
  "argmax",
  "local",
  "dsnt"
 ],
 "primary_decoder": "local",
 "temperature": 0.1,
 "window": 7,
 "router_margin_px": 3.0,
 "seed": 1,
 "centroid_excluded_from_pnp": true,
 "canonical_reproj_with_centroid": 23.161629,
 "harness_reproj_without_centroid": 24.913784,
 "interface_gate": {
  "primary_decoder": "Plocal",
  "conditions": {
   "A f2_far_median -10%": false,
   "B tail_gt50 -10%": false,
   "D confident_wrong better_by_10px >= 20%": true
  },
  "values": {
   "f2_far_drop": -2.6415381886000757,
   "tail50_drop": -2.2333333333333334,
   "c0_pnp": 70,
   "plocal_pnp": 87
  },
  "passed": true
 },
 "oracle_gate": {
  "conditions": {
   "1 f2_far -20%": false,
   "2 f2 signed bias -20%": false,
   "3 tail_gt50 -20%": true,
   "4 PnP >= 74": false,
   "5 reproj -10%": true,
   "6 near <= +2%": true,
   "8 no new NaN": true
  },
  "values": {
   "f2_far_median_px": 36.93595286217345,
   "f2_far_signed_bias_px": 18.117022644313455,
   "tail_gt50": 95,
   "pose_success": 70,
   "reproj_median_px": 20.528222616344543,
   "near_median_px": 5.715460960815219,
   "nan_err": 177,
   "proposal_share": 0.13505747126436782
  },
  "c0": {
   "f2_far_median_px": 44.59261722264212,
   "f2_far_signed_bias_px": 20.58383941627344,
   "tail_gt50": 120,
   "pose_success": 70,
   "reproj_median_px": 24.913783596875888,
   "near_median_px": 6.884772017265821,
   "nan_err": 177
  },
  "passed": false
 },
 "learned_router": "NOT RUN"
}
```
