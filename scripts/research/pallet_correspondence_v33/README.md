# V33 point-specific line / endpoint correspondence

From the repository root, with GPU access:

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_correspondence_v33/run33.py
```

Requires completed V32 data/checkpoints/metrics. The driver skips complete stages:
train → GT-free inference → score → train diagnostics → independent audit →
MD/figures/HTML → verified Chrome display and artifact binding.
`train33.py --precheck-only` checks geometry/gradient/forward paths without training.
`eval33.py --phase infer|score|diagnose` supports individual stages.

6 arms × 3 seeds × 2,000 updates = 18 models / 36,000 optimizer updates.
Same V32 physical boundary supervision and batches. Controls: pooled, query,
no_geometry (explicit geometry removed), line_only, end_only, point_aux.
Unused branches remain in all models; equal total parameters do not imply equal
active capacity or runtime. Hough top4 bins may represent duplicate modes and
their discrete indices are not differentiated. Both parent points remain queries.

Primary acceptance is registered before training in PROTOCOL.json. Evaluation is
the user-authorized reused real297 DEV lineage plus prospective/fresh/render checks,
not canonical140, paper or sealed final evaluation. Original manual GT stays fixed.
Use a new result directory for changed protocols; never overwrite completed V32
or V33 results. Any completion Discord notification is a separate, explicitly
authorized step and must not be resent for the same completion evidence.
