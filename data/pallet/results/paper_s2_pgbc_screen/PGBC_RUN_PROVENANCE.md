# PGBC run provenance

```
{
 "head": "1643d70d54910a7ebd9c7074bb671edb440c37f2",
 "checkpoint_sha256": "c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896",
 "baseline_gate": {
  "strict_n": 87,
  "gt2d_pose_success": 87,
  "pred_pose_success": 70,
  "yaw_median_deg": 6.02521554515198,
  "fixed_gt_reproj_median_px": 23.161628981707747,
  "expected": {
   "strict_n": 87,
   "gt2d_pose_success": 87,
   "pred_pose_success": 70,
   "yaw_median_deg": 6.025,
   "fixed_gt_reproj_median_px": 23.162
  },
  "tolerance": {
   "yaw_median_deg": 0.1,
   "fixed_gt_reproj_median_px": 0.25
  },
  "problems": [],
  "passed": true
 },
 "amplitude": 0.25,
 "seed": 1
}
```

## gate verdicts

```
{
 "G0": {
  "n_f2_frames": 35,
  "n_corners_far": 138,
  "n_corners_all": 278,
  "far_share_50pct_top1": 0.2826086956521739,
  "far_median_base_top1": 42.84343685957403,
  "far_median_ref_top1": 17.618058702669153,
  "far_share_50pct_glob": 0.0,
  "far_median_base_glob": 144.47436744800459,
  "far_median_ref_glob": 144.10671786494305,
  "argmax_moved_to_gt_rate": 0.6086956521739131,
  "gt_outside_grid": 2,
  "belief_cell_px": "1 cell = 12.8 x 9.6 px, so a top1 read-out cannot beat ~1 cell",
  "threshold": {
   "share_of_far_corners_with_50pct_reduction": 0.8
  },
  "passed": false,
  "note": "PASS uses the top1 read-out because the fixed additive residual has to win the argmax to change the decoded point; the global read-out is reported alongside."
 },
 "G1": {
  "fold_auc": [
   0.6132231404958678,
   0.6419834710743801,
   0.5910037484381507
  ],
  "fold_accuracy": [
   0.5636363636363636,
   0.5909090909090909,
   0.5714285714285714
  ],
  "fold_gt_beats_wrong": [
   0.6,
   0.6909090909090909,
   0.6326530612244898
  ],
  "control_no_feature_auc": [
   0.5008264462809917,
   0.4965289256198347,
   0.5022907122032486
  ],
  "control_no_feature_gt_beats_wrong": [
   0.0,
   0.0,
   0.0
  ],
  "threshold": {
   "every_fold_auc_or_accuracy": 0.75,
   "gt_beats_wrong": 0.7
  },
  "passed": false,
  "note": "the control drops the 50x50 feature and keeps only corner ID and dimensions, which are identical within a pair, so it must sit at chance; any discrimination therefore comes from the feature."
 },
 "G2": {
  "n_far": 138,
  "n_far_unsolved": 0,
  "median_err_base_px": 44.59261722264212,
  "median_err_graph_px": 43.79973602629235,
  "error_reduction": 0.017780548569084176,
  "signed_bias_base_px": 20.58383941627344,
  "signed_bias_graph_px": 21.05845997724165,
  "bias_reduction": -0.02305792186626654,
  "paired_improved": 67,
  "paired_worsened": 71,
  "threshold": {
   "error_reduction": 0.2,
   "bias_reduction": 0.2
  },
  "passed": false
 }
}
```

## G1 probe diagnostics

```
{
 "linear": [
  [
   0.6132231404958678,
   0.6
  ],
  [
   0.6419834710743801,
   0.6909090909090909
  ],
  [
   0.5910037484381507,
   0.6326530612244898
  ]
 ],
 "mlp": [
  [
   0.5851239669421487,
   0.6181818181818182
  ],
  [
   0.6423140495867768,
   0.6727272727272727
  ],
  [
   0.5805914202415661,
   0.5918367346938775
  ]
 ],
 "gt_vs_random": [
  [
   0.8641322314049587,
   0.8727272727272727
  ],
  [
   0.7689256198347107,
   0.7454545454545455
  ],
  [
   0.8575593502707205,
   0.8571428571428571
  ]
 ],
 "f50_dim": 128,
 "n_pairs": 159
}
```
