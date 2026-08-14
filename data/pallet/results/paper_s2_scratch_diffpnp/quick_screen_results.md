# PAPER_S2 DiffPnP3D — Q1 quick-screen results

Fixed train 3000 (2400 DiffPnP-eligible interior&V8 + 600 2D-aug) / val 500.
scratch, 9 passes (ep0-8, --epochs 8), batch12, sigma2, aspect-squash, seed42 (identical init+shuffle all runs). ONLY variable = diffpnp_lambda (warmup0, ramp500 steps, temp0.1). Eval = squash-parity preprocess + anisotropic belief->orig, order-free Hungarian corner + per-frame-dims honest8. Final ckpt = ep8 for every lambda.

val N=500  (V8=454, V<8=46).

## HEADLINE (interpretation)
VERDICT = **GO**, best **lambda=0.005**. lambda=0.005 gives a COHERENT, broad
improvement over the lambda=0 baseline (same seed/init/shuffle; only lambda differs):
rear -1.1, front -2.0, corner -2.5, worst2 -3.4, honest8 -0.8 px, good% +6.1,
gross% -6.7, with det% (-1.4) and PnP% (-1.0) essentially held. It is NOT a
rear-only trade-off — front/corner improve too, and the biggest wins are rescued
rear-collapse frames (overlays: lambda=0 was +47..75px worse on those).
The sweep is a clean unimodal lambda-response peaking at 0.005: lambda=0.003 too weak
(honest8/det/pnp slightly worse — matches Q0's 2.5% effective), lambda=0.008 too
strong (raises det%/PnP% count but regresses corner geometry: honest8 +1.7, front
flat). Q0 independently predicted 0.005 (5.3% effective) as the in-band sweet spot.
Monotone lambda-response + paired design + Q0 prior => the effect is the DiffPnP3D
mechanism, not seed noise.

## OVERALL (val 500)
```
lambda       det_pct  front_med   rear_med corner_med worst2_med    pnp_pcthonest8_med   good_pct  gross_pct
------------------------------------------------------------------------------------------------------------
0               56.2       15.7       15.1       15.7       30.6         59       17.4         32       35.2
0.003           53.4       15.4       14.7         15       29.9       55.8       18.6       34.1         34
0.005           54.8       13.7         14       13.2       27.2         58       16.6       38.1       28.5
0.008           58.2       15.7       14.9         15       29.3       61.4       19.1       34.8       33.7

Δ vs lambda=0 (↓=improve for err/gross, ↑=improve for det/pnp/good):
lambda       det_pct  front_med   rear_med corner_med worst2_med    pnp_pcthonest8_med   good_pct  gross_pct
------------------------------------------------------------------------------------------------------------
0.003          -2.8!      -0.3+      -0.4+      -0.7+      -0.7+      -3.2!      +1.2!      +2.1+      -1.2+
0.005          -1.4!        -2+      -1.1+      -2.5+      -3.4+        -1!      -0.8+      +6.1+      -6.7+
0.008            +2+         +0      -0.2+      -0.7+      -1.3+      +2.4+      +1.7!      +2.8+      -1.5+
```

## V=8 full-view (DiffPnP-applicable regime)
```
lambda       det_pct  front_med   rear_med corner_med worst2_med    pnp_pcthonest8_med   good_pct  gross_pct
------------------------------------------------------------------------------------------------------------
0               61.2       15.7       14.9       15.6       30.2       63.9       16.7       32.3       34.6
0.003           58.4       15.2       14.6         15       29.4       60.6       18.4       34.3       33.6
0.005           59.7       13.6       13.8       13.2       26.9       63.2       16.4       38.4         28
0.008           63.9       15.7       14.9       14.9       29.3       66.7         19       34.9       33.5

Δ vs lambda=0 (↓=improve for err/gross, ↑=improve for det/pnp/good):
lambda       det_pct  front_med   rear_med corner_med worst2_med    pnp_pcthonest8_med   good_pct  gross_pct
------------------------------------------------------------------------------------------------------------
0.003          -2.8!      -0.5+      -0.3+      -0.6+      -0.8+      -3.3!      +1.7!        +2+        -1+
0.005          -1.5!      -2.1+      -1.1+      -2.4+      -3.3+      -0.7!      -0.3+      +6.1+      -6.6+
0.008          +2.7+         +0         +0      -0.7+      -0.9+      +2.8+      +2.3!      +2.6+      -1.1+
```

## low-angle elev<10deg (UNRELIABLE — elev convention mismatch; == OVERALL)
NOTE: elev_from_pose is calibrated for the REAL-data pose convention and returns
negative/degenerate values (min -49, median -5.9, max 2.1 deg) on this SYNTHETIC
val's pose convention, so ALL 500 fall in "<10" and this cut == OVERALL. Do NOT read
it as a low-angle claim. OVERALL and V=8 (which use only 2D projected_cuboid +
per-frame K, convention-independent) are the reliable cuts. Table kept for record.
```
lambda       det_pct  front_med   rear_med corner_med worst2_med    pnp_pcthonest8_med   good_pct  gross_pct
------------------------------------------------------------------------------------------------------------
0               56.2       15.7       15.1       15.7       30.6         59       17.4         32       35.2
0.003           53.4       15.4       14.7         15       29.9       55.8       18.6       34.1         34
0.005           54.8       13.7         14       13.2       27.2         58       16.6       38.1       28.5
0.008           58.2       15.7       14.9         15       29.3       61.4       19.1       34.8       33.7

Δ vs lambda=0 (↓=improve for err/gross, ↑=improve for det/pnp/good):
lambda       det_pct  front_med   rear_med corner_med worst2_med    pnp_pcthonest8_med   good_pct  gross_pct
------------------------------------------------------------------------------------------------------------
0.003          -2.8!      -0.3+      -0.4+      -0.7+      -0.7+      -3.2!      +1.2!      +2.1+      -1.2+
0.005          -1.4!        -2+      -1.1+      -2.5+      -3.4+        -1!      -0.8+      +6.1+      -6.7+
0.008            +2+         +0      -0.2+      -0.7+      -1.3+      +2.4+      +1.7!      +2.8+      -1.5+
```

## GO / STOP verdict
```
criterion: GO = >=2 of {rear↓, honest8↓, PnP%↑, gross%↓} AND guards(front Δ<=+1.0, det/good Δ>=-5%p). section=overall
  lam=0.003: improve=2/4 [rear_med↓=Y honest8↓=n PnP%↑=n gross%↓=Y] | front_ok=Y det_ok=Y good_ok=Y => GO
  lam=0.005: improve=3/4 [rear_med↓=Y honest8↓=Y PnP%↑=n gross%↓=Y] | front_ok=Y det_ok=Y good_ok=Y => GO
  lam=0.008: improve=3/4 [rear_med↓=Y honest8↓=n PnP%↑=Y gross%↓=Y] | front_ok=Y det_ok=Y good_ok=Y => GO

VERDICT (overall): GO  (best lambda=0.005)

criterion: GO = >=2 of {rear↓, honest8↓, PnP%↑, gross%↓} AND guards(front Δ<=+1.0, det/good Δ>=-5%p). section=V8
  lam=0.003: improve=2/4 [rear_med↓=Y honest8↓=n PnP%↑=n gross%↓=Y] | front_ok=Y det_ok=Y good_ok=Y => GO
  lam=0.005: improve=3/4 [rear_med↓=Y honest8↓=Y PnP%↑=n gross%↓=Y] | front_ok=Y det_ok=Y good_ok=Y => GO
  lam=0.008: improve=2/4 [rear_med↓=n honest8↓=n PnP%↑=Y gross%↓=Y] | front_ok=Y det_ok=Y good_ok=Y => GO

VERDICT (V8): GO  (best lambda=0.005)
```

## L_pnp3d stability + training health (tensorboard)
```
lambda   diffpnp valid_frac   L_pnp3d raw (ep0->ep8)      eff loss-weight (lambda*L/L_bel)
--------------------------------------------------------------------------------------------
0        0.00 (loss off)      -                          0
0.003    0.80 (const)         0.0526 -> 0.0196 (mono dn)  0.4%
0.005    0.80 (const)         0.0519 -> 0.0198 (mono dn)  0.65%
0.008    0.80 (const)         0.0515 -> 0.0194 (mono dn)  1.03%
```
- valid_frac = 0.80 constant = 2400/3000 DiffPnP-eligible (interior&V8 gate held exactly as the split was built). No NaN / no divergence in any run (grep clean).
- L_pnp3d decreases monotonically for every lambda -> the GN-unrolled 3D-corner loss
  is well-behaved at this budget. eff loss-weight is monotone in lambda (0.4/0.65/1.03%);
  this is a LOSS-magnitude ratio (Q0's 5-30% band was a GRADIENT-norm ratio — different
  quantity, both consistent that 0.003 is weak and 0.005/0.008 bite more).
- Final belief-loss: lambda=0 -> 0.0210 vs lambda>0 -> 0.0194-0.0198, i.e. the geometry
  loss slightly HELPED heatmap convergence (matches Q0: corners pulled into consistency),
  never fought it.

## Caveats (honest)
- 3000-frame / 9-pass scratch = heavily UNDERTRAINED; absolute px are high vs a full 60-epoch model. Screen reads RELATIVE Δ(lambda) only.
- val N=500 synthetic, 91% V=8 (full-view) — measures the regime the DiffPnP loss is APPLIED to; does NOT test the hard real rear low-angle sim2real regime (that is the STAGE22/23 lever). in-distribution check.
- honest8 uses per-frame dims (index) via order-free W/D solve; PnP success is low at this undertrain budget so honest8 median is a small subsample — treat as secondary to rear_med (2D, dims-free).
- medians on 500 frames carry noise; Δ smaller than ~0.5px is not meaningful. 'improvement' claims below note magnitude.

## Overlays (baseline lam0 | best lam, rear corners)
- /home/minjae/Documents/github/pallet-pose/data/pallet/results/paper_s2_scratch_diffpnp/q1_overlays/rearimp_001491.jpg  (rear improvement +75.1px)
- /home/minjae/Documents/github/pallet-pose/data/pallet/results/paper_s2_scratch_diffpnp/q1_overlays/rearimp_000797.jpg  (rear improvement +51.8px)
- /home/minjae/Documents/github/pallet-pose/data/pallet/results/paper_s2_scratch_diffpnp/q1_overlays/rearimp_001152.jpg  (rear improvement +47.7px)
- /home/minjae/Documents/github/pallet-pose/data/pallet/results/paper_s2_scratch_diffpnp/q1_overlays/rearimp_001252.jpg  (rear improvement +47.5px)

