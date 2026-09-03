# Line ingredient audit — S5

`S5_STATUS = BLOCKED_INCOMPATIBLE_PROVENANCE`

Every line artifact in this repository was produced on the synthetic multihead
populations.

```text
paper_s2_multihead/line_uncertainty.json          population D2_MH_DEV512
paper_s2_multihead/point_line_solver_*.json       calibration D0_MH_SEEN512
                                                  decision    D2_MH_DEV512
paper_s2_multihead/line_signed_bias.json          UNCERTAINTY_CANNOT_FIX_SYSTEMATIC_RHO_BIAS true
```

There is no line prediction cache on the real 319 frames and no canonical adapter
mapping that model onto this population. Running it anyway would mean inventing a
correspondence between a synthetic-population model and the real evaluation set,
which is exactly what the lock forbids. New line training is also forbidden.

The arm is therefore not attempted, and this is recorded as a status rather than
an empty result file.
