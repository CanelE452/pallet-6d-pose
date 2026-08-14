# measC — does self-training reduce YAW error? (MEASUREMENT ONLY)

```
R0     = weights/paper_s2_stageB/net_epoch_0057_noseg.pth
self outside = data/pallet/results/ralph_selftrain/h6_s2_outside/round_02.pth
self night   = data/pallet/results/ralph_selftrain/h3_s2_night/round_02.pth
self noapril = data/pallet/results/ralph_selftrain/h7_s2_noapril/round_02.pth
self cad     = data/pallet/results/ralph_selftrain/h4_s2_combined/round_02.pth   [combined self (NO cad-specific self weight) — FLAG]
decode = infer_squash (Stage B squash400, belief50, subpixel THRESH=0.3)
pnp    = APNP.solve_pose auto_swap_dims=False, per-frame GT dims (W,D,H)
yaw    = _yaw_deg(R)=deg(atan2(R00,R20)) [reused run_measB]
gate   = n_det>=6 both & PnP ok; frame GT reproj>5.0 excluded
seed   = none (deterministic argmax/subpixel + PnP)
leakage: infer(model,img)=PNG only; GT used for yaw_gt(ref)+err only
```

## Exclusion bookkeeping (dedup by fid)
```
dom       nJson valid reproj>5 gtfail under<6 pnpfail  dup
outside     117    71        4      0      42       0    0
night        43    28        3      0      12       0    0
noapril      18    15        0      0       3       0    0
cad          44     1        1      0      42       0    0

sanity: infer_squash(subpixel) vs measA raw-argmax(grid) per-corner px, R0
  outside   median=6.9255416565522925px  (100/100 frames <=13px grid)
  night     median=7.063736878988065px  (33/33 frames <=13px grid)
  noapril   median=6.667587965789432px  (18/18 frames <=13px grid)
  cad       median=6.954039594166014px  (36/36 frames <=13px grid)
```
noapril N is small -> report as COUNTS not %.

## yaw_gt vs stored pose_transform yaw (cross-check)
```
dom          n  raw_med(deg)  fold180_med
outside     71           0.0          0.0
night       28           0.0          0.0
noapril     15           0.0          0.0
cad          1           0.0          0.0
```
(constant offset possible: stored pose uses a different 3D-corner
 convention; what matters is that they TRACK. large fold180 residual = flag)

## 1. MAIN — yaw err_180 (primary). R0 median FIRST (how big is yaw err at all?)
```
dom         N  R0_med         R0_IQR self_med       self_IQR   Delta  Bfloor  verdict
------------------------------------------------------------------------------------------------
outside    71    6.62   [2.45,12.80]     7.03   [2.29,16.50]   -0.41    0.41  NO CHANGE (<=floor or not sig)
night      28    6.04   [2.69,11.60]     6.51    [3.07,9.81]   -0.47    0.69  NO CHANGE (<=floor or not sig)
noapril    15    0.89    [0.55,2.08]     0.76    [0.23,1.24]    0.12    0.25  NO CHANGE (<=floor or not sig)
cad         1    1.77    [1.77,1.77]     1.27    [1.27,1.27]    0.50    0.37  NO CHANGE (<=floor or not sig)
```
Delta = median(err_R0) - median(err_self).  Delta>0 = self better.

## 2. paired Wilcoxon (err_R0 vs err_self, err_180) + HL effect
```
dom         N medDelta  HL_derr      dir          p  eff_r  impr  wors   eq
--------------------------------------------------------------------------------
outside    71    -0.41     0.50   worsen   2.94e-01   0.12    27    44    0
night      28    -0.47    -0.31  improve   4.65e-01   0.14    15    13    0
noapril    15     0.12    -0.35  improve   2.15e-02   0.59    11     4    0
cad         1   n<5 or all-zero
```
HL_derr = Hodges-Lehmann pseudomedian of (err_self-err_R0); <0=self better.
noapril N small -> Wilcoxon underpowered (flag).

## 5. err_raw / err_180 / err_90180 (swap-harmlessness check)
```
dom          ver   R0_med  self_med   Delta  impr  wors
outside      raw     6.62      7.03   -0.41    27    44
outside      180     6.62      7.03   -0.41    27    44
outside    90180     5.69      6.58   -0.88    26    45
-------------------------------------------------------
night        raw     6.04      6.51   -0.47    15    13
night        180     6.04      6.51   -0.47    15    13
night      90180     6.04      6.51   -0.47    16    12
-------------------------------------------------------
noapril      raw     0.89      0.76    0.12    11     4
noapril      180     0.89      0.76    0.12    11     4
noapril    90180     0.89      0.76    0.12    11     4
-------------------------------------------------------
cad          raw     1.77      1.27    0.50     1     0
cad          180     1.77      1.27    0.50     1     0
cad        90180     1.77      1.27    0.50     1     0
-------------------------------------------------------
```
If raw / 180 / 90180 give the SAME verdict per domain -> W/D swap harmless.
If they diverge (esp. raw >> 180) -> swap IS polluting = FLAG.

## figures
- fig1_yaw_box_R0_vs_self.png  (full range, outliers shown)
- fig1b_yaw_box_zoom.png       (y 0-20 deg, medians near floor)
- fig2_scatter_R0_vs_self.png  (err_R0 vs err_self, below diag=self better)
- per-frame: measC_perframe.csv
