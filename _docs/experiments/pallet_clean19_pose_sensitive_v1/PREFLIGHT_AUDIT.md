# Pose-sensitive preflight

HEAD: `7752974f39fdc2fd680c708b471bbb36b5b73db6`; branch: main

Original artifacts verified; old worktree preserved. Phase A no training, D8 changes only residual point set. Phase B requires exact512 binding and <=0.05px projection agreement before any fit. Four fits max, original R0, seed42, 320 steps each. No rescues.

Existing args:
```json
{
  "epochs": 5,
  "imgsz": 640,
  "batch": 16,
  "nbs": 16,
  "device": "0",
  "optimizer": "AdamW",
  "lr0": 0.0001,
  "lrf": 0.1,
  "cos_lr": true,
  "weight_decay": 0.0001,
  "warmup_epochs": 0.0,
  "patience": 0,
  "seed": 42,
  "deterministic": true,
  "box": 7.5,
  "cls": 0.5,
  "dfl": 1.5,
  "pose": 12.0,
  "kobj": 1.0,
  "hsv_h": 0.015,
  "hsv_s": 0.5,
  "hsv_v": 0.35,
  "degrees": 0.0,
  "translate": 0.1,
  "scale": 0.25,
  "shear": 0.0,
  "perspective": 0.0,
  "flipud": 0.0,
  "fliplr": 0.0,
  "mosaic": 0.0,
  "close_mosaic": 0,
  "mixup": 0.0,
  "copy_paste": 0.0,
  "erasing": 0.0,
  "workers": 2,
  "cache": false,
  "amp": false,
  "val": false,
  "plots": false,
  "save": true,
  "save_period": -1,
  "verbose": false
}
```
