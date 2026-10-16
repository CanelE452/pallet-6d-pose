# V32 visible geometry supervision

Completed controlled research: 4 primary arms × 3 seeds × 2,000 updates,
then one post-result auxiliary-only control × 3 seeds × 2,000 updates.
15 final checkpoints / 30,000 actual optimizer updates. Reused real297 DEV;
not canonical140 or a sealed final evaluation. No manual GT edits.

Run from the repository root. Rendering needs USD/pxr and the pallet-pose env:

```bash
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json /home/minjae/anaconda3/envs/pallet-pose/bin/python scripts/research/pallet_visible_geometry_v32/render32.py
```

Use `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python` for the following
sequential stages. Data preparation, training, inference, diagnosis, cross donor
and audit need GPU access. CPU scoring/reporting do not. Completed results must
remain immutable; reproduce in a separate result directory when modifying inputs.

1. `prepare32.py`
2. `train32.py`
3. `eval32.py --phase infer`
4. `eval32.py --phase score`
5. `eval32.py --phase diagnose`
6. `aux_control32.py` (post-result exploratory control)
7. `metadata32.py` (historical reference-metadata repair; point predictions fixed)
8. `eval32.py --phase score`, then import and call `aux_control32.score()` to merge the auxiliary results.
9. `cross_donor32.py` (post-result, cross-session role-map intervention)
10. `audit32.py`
11. `report32.py`
12. `deliver32.py` (verify HTML, open visible Chrome window and bind results)

`model32_training_source.py.txt` and `INFERENCE_ABLATION_AMENDMENT.json` preserve
the initial training implementation and the two inference-only ablation switches.
Normal-forward equality is audited. The metadata repair changes only the stock
YOLO reference used for new-catastrophe counts, then all scores/hashes are refreshed.

Artifact directory: `data/pallet/results/pallet_visible_geometry_v32/`.
See `PROTOCOL.json`, `AUX_CONTROL_PROTOCOL.json`, `CROSS_DONOR.json`, `AUDIT.json`,
`SUMMARY.json`, `REPORT_KO.md`, and `index.html` for scope, evidence and limitations.
Discord is an explicitly requested, separate completion-only notification; do not
send it again when inspecting these results.
