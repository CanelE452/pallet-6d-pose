# Y0 vs Y1 — pose-quality-aware classification (G38 38,002, 30ep)

```
                               Y0         Y1          Δ
--------------------------------------------------------
ALL cbox                    0.828      0.828     +0.000
ALL median                 12.523     12.885     +0.362
ALL p90                    69.710     71.049     +1.339
ALL gross20                 0.334      0.344     +0.011
DAY cbox                    0.920      0.980     +0.060
DAY median                 11.861     12.503     +0.642
DAY p90                    59.701     77.638    +17.937
NIGHT median               23.674     16.852     -6.822
NIGHT p90                 151.213     38.517   -112.697

NIGHT candidate
any_cbox                    0.714      0.607     -0.107
top1_cbox                   0.500      0.286     -0.214
cand_per_frame              7.893      5.714     -2.179
wrong_present_frac          0.929      0.929     +0.000
margin_median               0.315     -0.014     -0.329

held-out negative
n_neg                   2689.0000  2689.0000    +0.0000
AP_AUPRC                   0.6460     0.5954    -0.0506
AUROC                      0.9374     0.8818    -0.0556
FPR@TPR95                  0.3020     0.5816    +0.2797
neg_p90                    0.3634     0.3965    +0.0331
neg_p95                    0.5542     0.5983    +0.0441
neg_p99                    0.8178     0.7617    -0.0561
FP/img@0.05                0.4425     0.4708    +0.0283
FP/img@0.1                 0.2986     0.3098    +0.0112
FP/img@0.25                0.1554     0.1610    +0.0056
FP/img@0.4                 0.0941     0.1045    +0.0104

synthetic (G38 val 1,998)
box_map50                  0.9938     0.9940    +0.0002
box_map                    0.9279     0.9292    +0.0013
pose_map50                 0.9400     0.9415    +0.0015
pose_map                   0.9059     0.9137    +0.0078
```

safety {'ALL_cbox': True, 'NIGHT_any_cbox': False, 'ALL_median': True, 'FP_img_040': False}
conditions {'A_NIGHT_top1': False, 'B_ALL_p90': False, 'C_NIGHT_p90': True}  (1/3, 필요 2)

**VERDICT = Y1_POSEAWARE_LOSS_NO_SIGNAL**

★ NIGHT n=28, seed 1, 30ep. real 은 loss/training 에 쓰이지 않았다.
★ Windows Y2 결과와 비교하기 전에는 60ep 을 시작하지 않는다.