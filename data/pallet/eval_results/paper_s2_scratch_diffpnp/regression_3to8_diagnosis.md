# PAPER_S2 — 3-8deg regression diagnosis (raw heatmap vs PnP)

Q: Stage B improves rear at <3deg but regresses corner/honest8 at 3-8deg.
   Is the regression in the RAW heatmap keypoint, or in the PnP reprojection,
   or transition-band geometry, or front-sacrifice/rear-redistribution?

## Method (eval-parity reused from paper_s2_real_eval)
```
each model = own train preprocess (s1 aspect no-pad / stageB squash);
downstream metric IDENTICAL. raw & pnp both order-free Hungarian MEAN of
matched dists (same statistic -> directly comparable). raw/pnp/disp/front/
rear computed over the SAME pnp_ok frame subset to isolate the PnP effect.
  raw_corner  = mean hungarian(raw belief-peak pred8, gt8)   [PRE-PnP]
  pnp_corner  = mean hungarian(solve_pose reproj proj_all, gt8) [POST-PnP=honest8]
  raw-pnp_disp= median index-aligned ||pred8[i]-proj_all[i]|| [PnP pull]
  front/rear  = split_metrics front(0-3)/back(4-7) order-free median.
  dims=(1.1,1.3,0.11) order-free W/D. good<10.0 gross>20.0px per-corner.
  elevation = GT pose_transform. NO retrain (diagnosis only).
```

elev dist(deg): <3:46, 3-8:67, 8-15:10, 15+:0   (total with elev: 123)

## bin x model  (raw vs pnp split)
```
bin    model               n n_pnp  raw_cnr  pnp_cnr  raw-pnp  front   rear  honest8  gross%  pnp%  det%
--------------------------------------------------------------------------------------------------------
<3     paper_s1           46    39     17.2     16.3      3.0    9.3   16.7     16.3    30.0  85.0  85.0
<3     paper_s2_stageB    46    45     15.2     15.8      3.7    8.7   15.0     15.8    27.2  98.0  98.0

3-8    paper_s1           67    49     21.7     22.3      4.7    8.6   27.5     22.3    38.8  73.0  70.0
3-8    paper_s2_stageB    67    48     24.2     26.0      4.5    9.0   23.7     26.0    39.9  72.0  72.0

8-15   paper_s1           10     1     None    104.8      5.3   None   None    104.8    None  10.0   0.0
8-15   paper_s2_stageB    10     1     29.1     54.0      2.4    5.0   40.9     54.0    33.3  10.0  10.0

15+    paper_s1            0     0     None     None     None   None   None     None    None  None  None
15+    paper_s2_stageB     0     0     None     None     None   None   None     None    None  None  None

```

## 3-8deg PAIRED delta (StageB - s1), shared pnp_ok frames only
```
shared pnp_ok frames in 3-8deg: 42
  raw_corner             s1=   20.2  stageB=   23.2  d(median-of-med)=    3.0  paired-median-d=    1.0
  pnp_corner(honest8)    s1=   21.1  stageB=   24.2  d(median-of-med)=    3.1  paired-median-d=   1.71
  raw-pnp_disp           s1=    4.7  stageB=    4.3  d(median-of-med)=   -0.4  paired-median-d=  -0.74
  front                  s1=    7.9  stageB=    8.8  d(median-of-med)=    0.9  paired-median-d=   1.17
  rear                   s1=   26.0  stageB=   22.2  d(median-of-med)=   -3.8  paired-median-d=  -4.02
```

## VERDICT
```
3-8deg band (StageB - s1, +=worse for error metrics):
  raw_corner  d=2.5   pnp_corner d=3.7   raw-pnp_disp d=-0.2
  front d=0.4   rear d=-3.8

<3deg band (for redistribution check):
  raw_corner d=-2.0   front d=-0.6   rear d=-1.7

* RAW+PnP both regress -> data/appearance-driven (raw heatmap worse).
```

## DISAMBIGUATION (CORRECTED 2026-07-09) — what the numbers DO and DON'T isolate
```
[CORRECTION] An earlier draft labelled stageA as "lambda0 / NO DiffPnP" and used
it to claim "DiffPnP is NEUTRAL on raw, the driver is SQUASH". THAT IS WRONG.
Verified from weights/paper_s2/paper_s2_stageA/header.txt: stageA was diffpnp=True,
diffpnp_lambda=0.005 (same DiffPnP as stageB). So stageA is NOT a lambda0 baseline
and CANNOT isolate squash from DiffPnP.

  model             raw_corner(med)  honest8(med)  rear  front  gross%  det%
  paper_s1 (aspect,ft,no-DiffPnP) 12.1   22.3      27.5   8.6    38.8    70
  paper_s2_stageA(squash,scratch,DiffPnP l0.005) 17.9  22.8  24.2 10.3  41.6  45
  paper_s2_stageB(squash,scratch,DiffPnP l0.005,+maskFT/ArmB) 17.6 26.0 23.7 9.0 39.9 72

WHAT IS ESTABLISHED:
-> The 3-8deg regression is in the RAW heatmap (pre-PnP), NOT PnP re-projection
   (raw +1.0 paired; PnP adds only +0.7) and NOT a decode-PnP fight
   (raw-PnP disp -0.74, i.e. stageB pulls LESS). These are clean.
-> stageA vs stageB (BOTH DiffPnP l0.005, differ only in maskFT + ArmB data) have
   ~equal raw-median (17.9 vs 17.6) => maskFT/ArmB is NOT the raw driver.

WHAT IS NOT ESTABLISHED (do not claim):
-> "DiffPnP is neutral on raw" — NOT proven: there is NO lambda0 full-scale model;
   both A and B are l0.005. DiffPnP isolation exists ONLY in Q1 (synth, undertrain).
-> "The driver is SQUASH specifically" — NOT isolated: paper_s2 (A/B) differs from
   s1 as a BUNDLE {squash + scratch + data-recipe + DiffPnP} all at once. Squash is
   the leading HYPOTHESIS (anisotropic 640x480->400x400 plausibly distorts
   transition-band geometry) but is confounded with scratch and data-recipe.
```

## DECISION IMPLICATION (CORRECTED)
```
* The 3-8deg regression is a RAW-heatmap / INPUT-side problem, NOT a DiffPnP-loss
  or PnP-decode problem. So DiffPnP/PnP tuning will not fix it.
* lambda0-vs-lambda0.005 full ablation is DEPRIORITIZED (not "proven useless"):
  it would isolate DiffPnP's own effect, but (a) it does not address the raw 3-8deg
  regression (which is input-side), and (b) Stage B vs s1 confounds remain regardless.
  Run it ONLY if a paper reviewer demands the clean DiffPnP ablation.
* Next lever = DATA/PREPROCESS track (memory: rear = data/appearance):
  (a) squash vs aspect-preserving preprocess for transition-band geometry
      (needs a retrain to test the squash hypothesis cleanly), and/or
  (b) angle-binned (<3 / 3-8 / 8-15 / 15+) low-angle real-domain data.
* DiffPnP is RETAINED for its <3deg rear + det% gains; its small 3-8deg PnP-level
  cost (honest8 stageA 22.8 -> stageB 26.0) is a separate second-order knob.
```

CAVEAT: small samples per bin (3-8deg n=67, ~42-49 pnp_ok; domain-mixed low-angle).
ckpt = synthetic-val best (NOT real-selected). Diagnosis only, no retrain. Single
checkpoint per model (not multi-seed). Directional, not a significance claim.
