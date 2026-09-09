# Target-model comparison

This namespace compares the new clean-start 60k base model with two historical
targets on reused development data:

- `FT_REFERENCE`: the repository-declared `TARGET`, retained only as a
  real-supervised upper bound;
- `OLD_STAGE_A`: the target-specific synthetic milestone that the legacy
  fine-tuning work attempted to recover.

The primary population is DEV128 after removing the 12 DAY frames listed in
`FT_EVAL_LEAK.json`. NIGHT has no such overlap. DEV140 is emitted only as a
reference diagnostic. These artifacts are not FINAL evidence and do not change
the G38 paper baseline.

Rebuild with:

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python \
  challenge/yolo_pose_one_model/spatial_concat_scratch/target_model_comparison/build_target_comparison.py
```

The script binds every checkpoint and evaluation input by SHA-256 and writes
`TARGET_MODEL_COMPARISON.json` plus a concise Markdown report.
