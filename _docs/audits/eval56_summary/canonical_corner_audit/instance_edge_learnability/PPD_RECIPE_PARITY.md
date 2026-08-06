# PPD recipe trace

```
{
 "source": "scripts/stage0/paper_s2_ppd_long_run.py",
 "feature_source": "ep57 VGG tap found at runtime for a 100x100 output",
 "output_resolution": 100,
 "head": "PolarityLineHead: 1x1 stem over [f, gate, gate*f], 2 residual blocks, 1x1 out",
 "target": "exp(-d^2/2s^2) soft distance field, sigma 1.5 cells, observed_fragment mode",
 "loss": "BCEWithLogits(pos_weight) + lambda_pol * polarity_contrast",
 "optimizer": "AdamW",
 "lr": 0.0003,
 "weight_decay": 0.0001,
 "scheduler": null,
 "epochs": 20,
 "batch": 8,
 "seed": 1,
 "split": {
  "train": 3039,
  "val": 1045,
  "untouched": 5916,
  "group_key": "hdri|background|floor"
 },
 "augmentation": "none",
 "best_checkpoint_policy": "polarity_acc, then inversion_rate, then indexed reprojection, then macro F1, then earliest epoch",
 "calibration": {
  "pos_weight": [
   162.682861328125,
   200.0,
   58.41329574584961,
   63.529144287109375,
   200.0
  ],
  "L_line_median": 1.8359133005142212,
  "L_pol_median": 1.4578145146369934,
  "L_mask_median": 1.5119938254356384,
  "L_out_median": 0.754817008972168,
  "lambda_pol": 0.12593600091650736,
  "lambda_mask": 0.6071166659643119,
  "lambda_out": 0.12161313793221082,
  "source": "train split only, 20 batches, no update"
 },
 "calibration_rule": "lambda_x = c_x * median(L_line)/median(L_x) over 20 train batches with no update; c = 0.1 (pol), 0.5 (mask), 0.05 (outside)",
 "ppd_l0_epochs_recorded": 20,
 "ppd_l0_best_macro_f1": 0.4056716859340668,
 "changed_for_12_edge": [
  "output channels 5 -> 12",
  "target: semantic class -> physical edge instance",
  "decoder: O5 -> O12 incidence",
  "pos_weight recomputed by the same rule for 12 channels"
 ],
 "unchanged": [
  "optimizer",
  "lr",
  "weight_decay",
  "scheduler",
  "epochs",
  "batch",
  "seed policy",
  "target field scale",
  "loss form",
  "trunk",
  "split"
 ],
 "deviations": [
  "L12-MS uses the same PPD trunk as L12-F50 rather than the plain conv stack named in the instruction, so Phase H varies feature scale alone.",
  "L12 arms decode at 50x50 (the O12 oracle grid and the F50 tap); L5-CTRL stays at PPD's 100x100.  Each arm matches its own reference."
 ],
 "lambda_rule_checks": {
  "lambda_pol": {
   "stored": 0.12593600091650736,
   "reconstructed": 0.12593600091650736,
   "match": true
  },
  "lambda_mask": {
   "stored": 0.6071166659643119,
   "reconstructed": 0.6071166659643119,
   "match": true
  },
  "lambda_out": {
   "stored": 0.12161313793221082,
   "reconstructed": 0.12161313793221082,
   "match": true
  }
 },
 "lambda_rule_reproduced": true
}
```
