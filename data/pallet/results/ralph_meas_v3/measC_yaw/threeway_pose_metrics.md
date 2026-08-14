============================================================================================
UNIFIED TABLE — self-domain pose error, median (N).  ALL LOWER=BETTER.
R0 vs each PL filter self-train.  improve = any filter median < R0 median.
============================================================================================
self-domainmetric                  R0          loo   reproj+flip     loo+flip   improved(<R0)
---------------------------------------------------------------------------------------------
outside    yaw deg          6.54(N73)   7.74(N102)     7.94(N97)    7.19(N99)   NONE
outside    centroid cm     27.56(N73)  30.65(N102)    41.56(N97)   25.10(N99)   loo+flip
outside    ADD m            0.36(N73)   0.39(N102)     0.45(N97)    0.30(N99)   loo+flip
---------------------------------------------------------------------------------------------
night      yaw deg          6.04(N28)    6.02(N32)     6.77(N34)    6.48(N32)   loo
night      centroid cm     23.61(N28)   21.19(N32)    31.49(N34)   17.54(N32)   loo,loo+flip
night      ADD m            0.27(N28)    0.22(N32)     0.45(N34)    0.19(N32)   loo,loo+flip
---------------------------------------------------------------------------------------------
noapril    yaw deg          0.89(N15)    0.76(N15)     1.11(N15)    0.73(N15)   loo,loo+flip
noapril    centroid cm      5.46(N15)    5.01(N15)     7.43(N15)    4.97(N15)   loo,loo+flip
noapril    ADD m            0.06(N15)    0.05(N15)     0.08(N15)    0.05(N15)   loo,loo+flip
---------------------------------------------------------------------------------------------

Floors (green dotted in fig):
  yaw deg   : outside 0.41 / night 0.69 / noapril 0.25   (B floor, task-given)
  cent/ADD  : measB translation 1med (lateral+depth, depth-dominant): outside~2.7 / night~2.8 / noapril~7.8 cm  (ADD floor read vs same pseudo-GT pose)

PSEUDO-GT / SAMPLE caveats:
  * GT pose = GT 8-corner 2D projection solved by the SAME PnP (auto_swap OFF,
    per-frame GT dims). It is a 2D->PnP pseudo-GT, NOT metrology; centroid/ADD
    floors are therefore nonzero (depth of a flat pallet is weakly constrained).
  * noapril N small (GT 18) -> read as COUNTS not %.

SANITY — R0 row yaw vs measC_perframe.csv (harness validation):
  outside  perfid_match=71/71 r0_yaw_med_mine=6.617
  night    perfid_match=28/28 r0_yaw_med_mine=6.044
  noapril  perfid_match=15/15 r0_yaw_med_mine=0.886
SANITY — filter self yaw median vs existing yaw matrix JSON:
  loo         outside  R2: mine=7.737(N102) matrix=7.737 match=True
  loo         night    R2: mine=6.023(N32) matrix=6.023 match=True
  loo         noapril  R2: mine=0.763(N15) matrix=0.763 match=True
  reproj+flip outside  R1: mine=7.935(N97) matrix=7.935 match=True
  reproj+flip night    R1: mine=6.766(N34) matrix=6.766 match=True
  reproj+flip noapril  R1: mine=1.109(N15) matrix=1.109 match=True
  loo+flip    outside  R2: mine=7.189(N99) matrix=7.189 match=True
  loo+flip    night    R2: mine=6.477(N32) matrix=6.477 match=True
  loo+flip    noapril  R2: mine=0.735(N15) matrix=0.735 match=True
============================================================================================
