# Run provenance

```
{
 "head": "60ee10281a2c8e430348f8e8d42f75593f0241fd",
 "checkpoint_sha256": "c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896",
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
 "lr": 5e-05,
 "trainable_tensors": 42,
 "trainable_params": 12567579,
 "frozen_audit": {
  "vgg_trainable": 0,
  "belief123_trainable": 0,
  "affinity_trainable": 0
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
 }
}
```

## gate

```
{
 "conditions": [
  {
   "name": "1 F2 far median -15%",
   "passed": false,
   "value": 0.009365255729120059
  },
  {
   "name": "2 F2 signed bias -20%",
   "passed": false,
   "value": 0.01733852005443537
  },
  {
   "name": "3 >50px tail -15%",
   "passed": false,
   "value": 0.09999999999999998
  },
  {
   "name": "4 sharpen-no-correct -30%",
   "passed": false,
   "value": -0.8224299065420562
  },
  {
   "name": "5 F2 improved > worsened",
   "passed": false,
   "value": -7.0
  },
  {
   "name": "6 canonical PnP >= 72",
   "passed": false,
   "value": 67.0
  },
  {
   "name": "7 reproj -10%",
   "passed": false,
   "value": 0.05121857142755637
  },
  {
   "name": "8 near <= +5%",
   "passed": true,
   "value": -0.010304368781970719
  },
  {
   "name": "9 no new PnP failure",
   "passed": false,
   "value": 3.0
  },
  {
   "name": "10 no new >100px",
   "passed": true,
   "value": -9.0
  },
  {
   "name": "11 no new NaN",
   "passed": false,
   "value": 5.0
  },
  {
   "name": "12 no stage collapse",
   "passed": true,
   "value": 0.0
  }
 ],
 "passed": false,
 "paired_improved": 14,
 "paired_worsened": 21
}
```
