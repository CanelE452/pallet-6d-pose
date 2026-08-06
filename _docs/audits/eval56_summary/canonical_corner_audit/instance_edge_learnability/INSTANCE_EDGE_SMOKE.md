# Smoke

```
{
 "arms": [
  {
   "arm": "L5-CTRL",
   "seed": 1,
   "steps": 100,
   "loss_finite": true,
   "mean_loss": 0.7908743399381638,
   "head_gradient_l1": 340.01781817153096,
   "trainable_parameters": 181573,
   "output_channels": 5,
   "output_grid": 100,
   "all_channels_active": true,
   "decoder_finite": true,
   "calibration": {
    "pos_weight": [
     162.682861328125,
     200.0,
     58.41329574584961,
     63.529144287109375,
     200.0
    ],
    "lambda_pol": 0.12593600091650736,
    "source": "ppd_t2_loss_calibration.json (exact reuse)"
   }
  },
  {
   "arm": "L12-F50",
   "seed": 1,
   "steps": 100,
   "loss_finite": true,
   "mean_loss": 0.8873810535669326,
   "head_gradient_l1": 259.1399279907346,
   "trainable_parameters": 165644,
   "output_channels": 12,
   "output_grid": 50,
   "all_channels_active": true,
   "decoder_finite": true,
   "calibration": {
    "pos_weight": [
     40.19464469618949,
     144.19056261343013,
     39.03603242918627,
     142.16392269148176,
     39.72490327835472,
     40.241365089184455,
     39.53095551727632,
     38.53741227636651,
     40.53686396677051,
     142.7297879985627,
     141.9081814933905,
     40.29245380406731
    ],
    "L_line_median": 0.8246203660964966,
    "L_pol_median": 1.710140883922577,
    "lambda_pol": 0.048219440506272915,
    "source": "recomputed by the PPD rule on the train split, 20 batches, no update"
   }
  },
  {
   "arm": "L12-MS",
   "seed": 1,
   "steps": 100,
   "loss_finite": true,
   "mean_loss": 0.8807482159137726,
   "head_gradient_l1": 364.01988842338324,
   "trainable_parameters": 206796,
   "output_channels": 12,
   "output_grid": 50,
   "all_channels_active": true,
   "decoder_finite": true,
   "calibration": {
    "pos_weight": [
     40.19464469618949,
     144.19056261343013,
     39.03603242918627,
     142.16392269148176,
     39.72490327835472,
     40.241365089184455,
     39.53095551727632,
     38.53741227636651,
     40.53686396677051,
     142.7297879985627,
     141.9081814933905,
     40.29245380406731
    ],
    "L_line_median": 0.881749302148819,
    "L_pol_median": 1.3832498788833618,
    "lambda_pol": 0.06374475903519453,
    "source": "recomputed by the PPD rule on the train split, 20 batches, no update"
   }
  }
 ],
 "a1_checksum_before": "5b8a3f651120648377327d33ae7089f458194212678f977a6d49df29d30c1c7f",
 "a1_checksum_after": "5b8a3f651120648377327d33ae7089f458194212678f977a6d49df29d30c1c7f",
 "a1_unchanged": true,
 "passed": true
}
```
