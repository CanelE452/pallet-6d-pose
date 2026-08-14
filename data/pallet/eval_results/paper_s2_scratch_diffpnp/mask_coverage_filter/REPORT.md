# PAPER_S2 mask-coverage filter validation

- no retraining: Stage B ep57 mask-aux + belief outputs from one forward
- calibration: filterval 123 (outside44/night43/manual36)
- fixed-threshold secondary check: handannot17 (cad11/noapril6); this set is
  development-touched, because the original mask hypothesis inspected one noapril frame
- score is GT-free; GT is used only for evaluation labels
- selected threshold: `score_effective >= 0.022702` flags under-coverage

## Metric

Primary `score_cover` is the soft mask probability mass outside the raw-keypoint
convex hull dilated by one 50x50 cell, divided by positive mask mass in the
low-threshold component(s) connected to that footprint inside a 0.75x-expanded
keypoint ROI. The per-frame border median is removed first.
Mask probability contrast must be >=0.10; lower-contrast frames
are UNAVAILABLE rather than rejected.

A secondary diagnostic uses relative mask thresholds r={0.02,0.05,0.10,0.20}.
It keeps components touching the keypoint box and records mean(E*O), where E is
directional extension and O is outside-box support. This extension score is saved
per frame but is not the selected filter score because calibration separation was
weaker than soft outside mass.

The main GT diagnostic label is computed from the final PnP `projected_all` 8-point
footprint, not the incomplete raw 6-8 point hull. It requires prediction center
inside GT, >=90% containment, side gaps >=-5%, and GT coverage<=75% or minimum
PCA-axis span ratio<=80%. Frames with GT visible fraction<90%, n_det<6, or PnP
failure are excluded from threshold fitting. Safety retention uses honest8<10px.

## Dataset / label counts

```
split/domain              N  det>=6  reliable  eligible  strict   main   loose  posegood
--------------------------------------------------------------------------------------
filterval/all           123      94        55        93       8     28      45      12
  outside                44      30        25        30       1     10      20       8
  night                  43      29        28        28       7     15      21       0
  manual                 36      35         2        35       0      3       4       4
handannot17/all          17       4         4         4       0      1       1       3
  cad                    11       0         0         0       0      0       0       0
  noapril                 6       4         4         4       0      1       1       3
```

## Selected threshold results

```
set             N  UC+  flag  TP  FP    Prec  Recall    Spec  pose lost
-------------------------------------------------------------------------------
cal/strict     93    8    11   3   8   0.273   0.375   0.906      0/12 
cal/main       93   28    11   6   5   0.545   0.214   0.923      0/12 
cal/loose      93   45    11   9   2   0.818   0.200   0.958      0/12 
secondary       4    1     3   1   2   0.333   1.000   0.333      2/3  
all/main       97   29    14   7   7   0.500   0.241   0.897      2/15 
```

## Flagged frames at selected threshold

```
split       domain   fid                     score  UC  pose   cmed     h8       old  f4tight
-----------------------------------------------------------------------------------------------
filterval   outside  1778651534387246080     0.609   Y     n   64.7   59.5      pass        n
filterval   outside  1778651518865604096     0.403   n     n   37.3   35.2      pass        n
handannot17 noapril  1775201447585014272     0.377   n     Y    4.4    9.9      pass        -
filterval   outside  1778653554639401216     0.292   n     n   18.4   16.6      pass        n
filterval   outside  1778651530557153024     0.251   n     n   13.5   34.1      pass        Y
handannot17 noapril  1775201432466607872     0.237   Y     n    5.2   39.2      pass        -
filterval   outside  1778653429693238272     0.054   n     n    6.6   16.9      pass        n
filterval   night    1779449196392532480     0.052   Y     n   12.6   16.4      pass        n
filterval   night    1779449254651112192     0.052   Y     n   46.6   47.7      pass        n
filterval   night    1779449198527179520     0.044   Y     n    9.9   44.4      pass        n
filterval   night    1779449318604401920     0.031   Y     n   13.7   39.5      pass        n
handannot17 noapril  1775201442780546816     0.029   n     Y    8.6    7.0      pass        -
filterval   outside  1778653337772735232     0.026   n     n    7.5   10.2      pass        Y
filterval   night    1779449247779662080     0.023   Y     n   37.4   51.0      pass        n
```

## Ranking metrics on calibration

```
label          AUPRC     AUROC
uc_strict      0.133     0.579
uc_main        0.377     0.574
uc_loose       0.614     0.621
```

## Existing filter impact on filterval

Here `good` means final PnP honest8 mean <10px (not partial raw-corner median).

- DEPLOY6: 86=10 good/76 bad (purity 0.116) -> 74=10 good/64 bad (purity 0.135); rejected 12
- DEPLOY6 + f4<=0.8: 30=8 good/22 bad (purity 0.267) -> 28=8 good/20 bad (purity 0.286); rejected 2

## Fixed selected threshold by calibration domain

```
domain          N  UC+  flag  TP  FP    Prec  Recall    Spec  pose lost
-------------------------------------------------------------------------------
outside        30   10     6   1   5   0.167   0.100   0.750      0/8  
night          28   15     5   5   0   1.000   0.333   1.000      0/0  
manual         35    3     0   0   0       -   0.000   1.000      0/4  
```

## Leave-one-domain-out stability

```
held domain          tau    N  UC+  flag  TP  FP    Prec  Recall
------------------------------------------------------------------------
outside          0.00100   30   10    11   2   9   0.182   0.200
night            0.60855   28   15     0   0   0       -   0.000
manual           0.60855   35    3     0   0   0       -   0.000
```

## Decision

The pooled calibration threshold meets the numeric constraints, but fixed-domain or secondary-check safety fails. Keep the score only as a diagnostic/manual-review ranking feature. Do not wire it into the pseudo-label hard-reject AND.

Initial sweep anchor tau=0.05: flag=7, TP=3, FP=4, precision=0.429, recall=0.107.

Caveats: mask head was trained as a weak synthetic auxiliary head; 50x50
quantization and small per-domain UC counts make threshold estimates unstable.
The old absolute mask>=0.5 area ratio is reported per frame for comparison but
is not used in the rule. The secondary set is not an untouched holdout.

## Conservative-feasible calibration sweep points

```
       tau  flag  TP  FP    Prec  Recall    Spec  goodloss
   0.02270    11   6   5   0.545   0.214   0.923     0.000
   0.01781    12   6   6   0.500   0.214   0.908     0.000
   0.03095     9   5   4   0.556   0.179   0.938     0.000
   0.02641    10   5   5   0.500   0.179   0.923     0.000
   0.04412     8   4   4   0.500   0.143   0.938     0.000
   0.60855     1   1   0   1.000   0.036   1.000     0.000
   0.40303     2   1   1   0.500   0.036   0.985     0.000
```

