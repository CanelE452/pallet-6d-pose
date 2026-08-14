================================================================================================
loo+flip SELF-TRAINING self-domain — R0 -> R1 -> R2 pose-error trajectory.
median (N).  ALL LOWER=BETTER.  MEASUREMENT ONLY (no prescription).
================================================================================================
self-domainmetric                   R0            R1            R2   verdict(R2 vs R1)
--------------------------------------------------------------------------------------
outside    yaw deg           6.54(N73)     9.65(N96)     7.19(N99)   R2 lower (2nd round improves)
outside    centroid cm      27.56(N73)    30.05(N96)    25.10(N99)   R2 lower (2nd round improves)
outside    ADD m             0.36(N73)     0.35(N96)     0.30(N99)   R2 lower (2nd round improves)
--------------------------------------------------------------------------------------
night      yaw deg           6.04(N28)     6.87(N31)     6.48(N32)   R2 lower (2nd round improves)
night      centroid cm      23.61(N28)    17.96(N31)    17.54(N32)   similar (R1~R2)
night      ADD m             0.27(N28)     0.21(N31)     0.19(N32)   R2 lower (2nd round improves)
--------------------------------------------------------------------------------------
noapril    yaw deg           0.89(N15)     0.74(N15)     0.73(N15)   similar (R1~R2)
noapril    centroid cm       5.46(N15)     4.79(N15)     4.97(N15)   similar (R1~R2)
noapril    ADD m             0.06(N15)     0.05(N15)     0.05(N15)   similar (R1~R2)
--------------------------------------------------------------------------------------

Floors (green dotted in fig):
  yaw deg  : outside 0.41 / night 0.69 / noapril 0.25   (B floor, task-given)
  cent/ADD : measB translation 1med (depth-dominant): outside~2.7 / night~2.8 / noapril~7.8 cm  (ADD floor read vs same pseudo-GT pose)

CAVEATS:
  * GT pose = GT 8-corner 2D projection solved by the SAME PnP (auto_swap OFF,
    per-frame GT dims) = a 2D->PnP pseudo-GT, NOT metrology; centroid/ADD floors
    are nonzero (flat-pallet depth weakly constrained).
  * unpaired N across rounds: each cell is an INDEPENDENT gate (a model may
    detect on a slightly different frame subset) -> N differs per round; medians
    are NOT strictly paired. Read verdicts as trend, not paired test.
  * noapril N small (GT 18, N<=15) -> read as COUNTS not %.

SANITY — this R2 self-domain vs existing threeway_pose_metrics.json 'loo+flip':
  outside  yaw      mine=7.189(N99) threeway=7.189(N99) match=True
  outside  centroid mine=25.098(N99) threeway=25.098(N99) match=True
  outside  ADD      mine=0.296(N99) threeway=0.296(N99) match=True
  night    yaw      mine=6.477(N32) threeway=6.477(N32) match=True
  night    centroid mine=17.539(N32) threeway=17.539(N32) match=True
  night    ADD      mine=0.189(N32) threeway=0.189(N32) match=True
  noapril  yaw      mine=0.735(N15) threeway=0.735(N15) match=True
  noapril  centroid mine=4.966(N15) threeway=4.966(N15) match=True
  noapril  ADD      mine=0.053(N15) threeway=0.053(N15) match=True
SANITY — R1/R2 yaw vs phase1style_yaw_matrix_looflip.json diagonal:
  outside  R1: mine=9.647 matrix=9.647 match=True
  outside  R2: mine=7.189 matrix=7.189 match=True
  night    R1: mine=6.866 matrix=6.866 match=True
  night    R2: mine=6.477 matrix=6.477 match=True
  noapril  R1: mine=0.736 matrix=0.736 match=True
  noapril  R2: mine=0.735 matrix=0.735 match=True
================================================================================================
