# PDG-Net Stage 1 — final report

## Decision: FOUNDATION_STOP

```
{
 "A0|D13|D0": {
  "frames": 13,
  "centroid": 0,
  "R4": 0,
  "R6": 0,
  "pnp": 0,
  "reproj_median": NaN,
  "t100": 0
 },
 "A0|C13|D0": {
  "frames": 13,
  "centroid": 13,
  "R4": 11,
  "R6": 9,
  "pnp": 11,
  "reproj_median": 37.998577040512586,
  "t100": 7
 },
 "A1|D13|D0": {
  "frames": 13,
  "centroid": 2,
  "R4": 0,
  "R6": 0,
  "pnp": 0,
  "reproj_median": NaN,
  "t100": 0
 },
 "A1|C13|D0": {
  "frames": 13,
  "centroid": 13,
  "R4": 10,
  "R6": 8,
  "pnp": 11,
  "reproj_median": 42.59741882007237,
  "t100": 10
 },
 "A2|D13|D0": {
  "frames": 13,
  "centroid": 2,
  "R4": 0,
  "R6": 0,
  "pnp": 0,
  "reproj_median": NaN,
  "t100": 3
 },
 "A2|D13|D0V": {
  "frames": 13,
  "centroid": 2,
  "R4": 0,
  "R6": 0,
  "pnp": 0,
  "reproj_median": NaN,
  "t100": 3
 },
 "A2|C13|D0": {
  "frames": 13,
  "centroid": 11,
  "R4": 9,
  "R6": 8,
  "pnp": 10,
  "reproj_median": 39.965952037200125,
  "t100": 6
 },
 "A2|C13|D0V": {
  "frames": 13,
  "centroid": 11,
  "R4": 9,
  "R6": 8,
  "pnp": 10,
  "reproj_median": 39.965952037200125,
  "t100": 6
 }
}
```

## Wrapper parity

```
{
 "wrapper": {
  "img": 0.0,
  "kp": 0.0,
  "belief": 6.242451465743315e-07,
  "affinity": 0.0,
  "mask": 0.0
 },
 "a1_corner_delta": 0.0,
 "step0": {
  "A1": {
   "h6": 0.0,
   "a6": 0.0
  },
  "A2": {
   "h6": 0.0,
   "a6": 0.0
  }
 }
}
```

## Gradient calibration

```
{
 "lambda": {
  "palletness": 0.2035721109706831,
  "visibility": 0.09481360228591426,
  "truncation": 0.7914765160219902
 },
 "clamped": {
  "palletness": false,
  "visibility": false,
  "truncation": false
 },
 "achieved_ratio": {
  "palletness": 0.1,
  "visibility": 0.15,
  "truncation": 0.05
 },
 "grad_norm_median": {
  "local": 0.0057891254539508465,
  "palletness": 0.0028437713920373663,
  "visibility": 0.009158694503284725,
  "truncation": 0.0003657168176667672
 }
}
```

## Training

```
A1: [{"total": 0.005603853093343549, "local": 0.005603853093343549, "palletness": 0.0, "visibility": 0.0, "truncation": 0.0, "epoch": 1, "steps": 2443, "arm": "A1"}, {"total": 0.005590751064902021, "local": 0.005590751064902021, "palletness": 0.0, "visibility": 0.0, "truncation": 0.0, "epoch": 2, "steps": 4886, "arm": "A1"}, {"total": 0.005640039463194695, "local": 0.005640039463194695, "palletness": 0.0, "visibility": 0.0, "truncation": 0.0, "epoch": 3, "steps": 7329, "arm": "A1"}]
A1 runtime_s: 2349.559789928957
A2: [{"total": 0.40517059250952425, "local": 0.00742440687693012, "palletness": 0.009260389793548325, "visibility": 0.027255547808902967, "truncation": 0.4968901005037193, "epoch": 1, "steps": 2443, "arm": "A2"}, {"total": 0.35268632247220577, "local": 0.007277753011781358, "palletness": 0.007901673750866874, "visibility": 0.021777380305235444, "truncation": 0.4317692594112454, "epoch": 2, "steps": 4886, "arm": "A2"}, {"total": 0.3391470507128874, "local": 0.007243278541103176, "palletness": 0.0075418726849223695, "visibility": 0.020608226500126194, "truncation": 0.41493906990163304, "epoch": 3, "steps": 7329, "arm": "A2"}]
A2 runtime_s: 2794.5784590689
```

## Holdout

```
{
 "e44_open": 0,
 "w45_open": 0,
 "final_test_open": 0
}
```
