# Run provenance

```
{
 "identity": {
  "head": "8dd5d064b8376ff56104f80a38ed7238daf2501a",
  "origin_main": "8dd5d064b8376ff56104f80a38ed7238daf2501a",
  "dirty_files": 10,
  "checkpoint_sha256": "c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896",
  "python": "3.10.20",
  "torch": "2.1.1+cu118",
  "cuda": "11.8",
  "gpu": "NVIDIA GeForce RTX 3080",
  "gpu_free_mb": 8787
 },
 "roots": [
  "/home/minjae/Documents/github/pallet-pose/data/pallet/training_data/mixed_v8_train",
  "/home/minjae/Documents/github/pallet-pose/data/pallet/training_data/v4_split_base",
  "/home/minjae/Documents/github/pallet-pose/data/pallet/training_data/aug_squash_v2",
  "/home/minjae/Documents/github/pallet-pose/data/pallet/training_data/aug_trunc_v2",
  "/home/minjae/Documents/github/pallet-pose/data/pallet/training_data/aug_scale_v2",
  "/home/minjae/Documents/github/pallet-pose/data/pallet/training_data/paper_4pallet_mask_v1"
 ],
 "balance_groups": "mixed_v8_train|v4_split_base|aug_squash_v2|aug_trunc_v2|aug_scale_v2:60,paper_4pallet_mask_v1:40",
 "dataset_frames": 29308,
 "batches": 2443,
 "batch_size": 12,
 "epochs": 5,
 "seed": 1,
 "features": {
  "index_high": 17,
  "channels_high": 256,
  "index_low": 26,
  "channels_low": 128
 },
 "param_groups": {
  "vgg_last": 5014912,
  "belief_stage4_6": 12567579,
  "proposal_branch": 334081,
  "total": 17916572
 },
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
 "calibration": {
  "median": {
   "dope": 0.001111264580272292,
   "proposal": 6.052473545074463,
   "refined": 0.7233118116855621
  },
  "lambda_proposal": 3.6721005783714514e-05,
  "lambda_refined": 0.00030727123830113275,
  "target_share": 0.2,
  "batches": 20
 },
 "gate_decision": {
  "conditions": [
   {
    "name": "1 F2 far median -15%",
    "passed": false,
    "value": -0.007461251250512557
   },
   {
    "name": "2 F2 far signed bias -20%",
    "passed": false,
    "value": 0.03285398673898776
   },
   {
    "name": "3 >50px tail -20%",
    "passed": false,
    "value": -0.1416666666666666
   },
   {
    "name": "4 paired improved > worsened",
    "passed": false,
    "value": -12.0
   },
   {
    "name": "5 PnP success >= 72/87",
    "passed": false,
    "value": 69.0
   },
   {
    "name": "6 reproj -10%",
    "passed": false,
    "value": -0.10406734833858056
   },
   {
    "name": "7 near median <= +5%",
    "passed": false,
    "value": 0.11474124851804124
   },
   {
    "name": "8 no new PnP failure",
    "passed": false,
    "value": 2.0
   },
   {
    "name": "9 no new >50px",
    "passed": false,
    "value": 17.0
   },
   {
    "name": "10 no new NaN",
    "passed": true,
    "value": -20.0
   },
   {
    "name": "11 gate not collapsed",
    "passed": false,
    "value": 3.5303807655040487e-09
   },
   {
    "name": "12 C1-base no catastrophic regression",
    "passed": true,
    "value": -1.0
   }
  ],
  "passed": false,
  "paired_improved": 11,
  "paired_worsened": 23,
  "gate_median": 3.5303807655040487e-09
 }
}
```
