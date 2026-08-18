# Gate 0 provenance

```
{
 "head": "c3557c0aed749f00e15f6f19905f0b0a99f0a328",
 "checkpoint_sha256": "c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896",
 "training_steps": 0,
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
 "n_corner_rows": 519,
 "n_sessions": 8,
 "ridge_lambda": 0.001,
 "feature_basis_B2": [
  "bias",
  "cos2psi",
  "sin2psi",
  "sin_elev",
  "cos_elev"
 ],
 "feature_basis_B3": [
  "bias",
  "cos2psi",
  "sin2psi",
  "sin_elev",
  "cos_elev",
  "log_scale",
  "cos2psi_sin_elev",
  "sin2psi_sin_elev"
 ],
 "protocol": "leave-one-session-out, 8 folds, centroid kept predicted"
}
```

## gate

```
{
 "conditions": {
  "B2": {
   "checks": [
    {
     "name": "1 F2 signed bias -25%",
     "passed": false,
     "value": -0.28341043844924374
    },
    {
     "name": "2 F2 far median -15%",
     "passed": false,
     "value": -0.08857201081710886
    },
    {
     "name": "3 >50px tail -15%",
     "passed": false,
     "value": -0.925
    },
    {
     "name": "4 paired improved > worsened",
     "passed": false,
     "value": -161.0
    },
    {
     "name": "5 near <= +5%",
     "passed": false,
     "value": 4.6044015785498305
    },
    {
     "name": "6 PnP >= 72 or reproj -8%",
     "passed": false,
     "value": 70.0
    },
    {
     "name": "7 no new >100px",
     "passed": false,
     "value": 34.0
    }
   ],
   "passed": false,
   "paired_improved": 179,
   "paired_worsened": 340
  },
  "B3": {
   "checks": [
    {
     "name": "1 F2 signed bias -25%",
     "passed": false,
     "value": 0.026877921064841104
    },
    {
     "name": "2 F2 far median -15%",
     "passed": false,
     "value": -0.5453514923771752
    },
    {
     "name": "3 >50px tail -15%",
     "passed": false,
     "value": -1.25
    },
    {
     "name": "4 paired improved > worsened",
     "passed": false,
     "value": -223.0
    },
    {
     "name": "5 near <= +5%",
     "passed": false,
     "value": 6.488556640003886
    },
    {
     "name": "6 PnP >= 72 or reproj -8%",
     "passed": false,
     "value": 70.0
    },
    {
     "name": "7 no new >100px",
     "passed": false,
     "value": 101.0
    }
   ],
   "passed": false,
   "paired_improved": 148,
   "paired_worsened": 371
  }
 },
 "view_necessity": {
  "B2": {
   "checks": {
    "signed bias -10% vs B1": false,
    "F2 far -7.5% vs B1": false,
    ">50px tail -5% vs B1": false,
    "PnP rescue >= 2 vs B1": false
   },
   "passed": false,
   "rescue_vs_B1": 0
  },
  "B3": {
   "checks": {
    "signed bias -10% vs B1": false,
    "F2 far -7.5% vs B1": false,
    ">50px tail -5% vs B1": false,
    "PnP rescue >= 2 vs B1": false
   },
   "passed": false,
   "rescue_vs_B1": 0
  }
 },
 "passed": false,
 "passing_arm": null
}
```
