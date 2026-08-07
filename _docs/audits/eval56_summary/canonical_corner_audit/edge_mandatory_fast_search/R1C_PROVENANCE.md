# R1C provenance

The code that produced R1C is preserved as
`scripts/stage0/r1c_fusion_capacity.py` rather than rewritten from the result.

```
script                scripts/stage0/r1c_fusion_capacity.py
search2k / validation512 / A1 / PnP helper / topology     sha in r1c_provenance.json
optimizer AdamW   lr 1e-3   weight decay 1e-4   steps 166   seed 1
GT edges          refine_keypoints[:8] -> fixed 12 corner pairs
CIGM epsilon      1e-4        proposal sigma 2.0
decoder           raw peak >= 0.30 then argmax (canonical DOPE readout)
PnP               challenge/scripts/annotate_pnp.py::solve_pose  (unchanged)
```

Per-frame records are in `r1c_validation_per_frame.csv`; the five direct-PnP
failures are listed with their CIGM error and condition number in
`r1c_pnp_failures.csv`.

```
run1|001775   run1|008845   run1|008338   run2|015514   run2|014556
```

The R1C document reports 508/512 solved.  Re-running through the preserved
script gives 507/512 -- the five above.  The one-frame difference comes from a
borderline solve and is recorded rather than reconciled away; both runs sit at a
reprojection median of 0.033px.
