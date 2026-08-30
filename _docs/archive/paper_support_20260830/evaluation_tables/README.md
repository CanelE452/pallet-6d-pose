# Paper Evaluation Tables

These are blank paper-table templates. No DEV number may fill a FINAL cell.

## Population contract

```text
DEV_PLASTIC_POS140          140   diagnostic; compatibility alias DEV_POS140
COMMON_DEV_PLASTIC_POS128   128   controlled DEV; alias COMMON_DEV_POS128
DEV_WOOD_POS45               45   CROSS_SHAPE_DEV; previously evaluated
COMMON_DEV_MULTISHAPE_POS   173   plastic 128 + wood 45
DEV_NEG2689                2689   common pallet-absent DEV

FINAL_PLASTIC_POS          UNAVAILABLE
FINAL_WOOD_POS             UNAVAILABLE
FINAL_ALL_POS              UNAVAILABLE
FINAL_NEG                  UNAVAILABLE
```

Null FINAL membership is not a valid empty test. Existing DEV rows are never
copied into FINAL.

## Reporting rules

1. Main metrics are exactly Box AP50:95, Restricted ADD-S AUC,
   symmetry-aware rotation median, translation median, and symmetry-aware yaw
   median.
2. Every result family reports `ALL`, `PLASTIC`, and `WOOD` explicitly.
3. Every row binds Population ID, Object subset, N, pose-valid N/total,
   checkpoint SHA, geometry-registry SHA, symmetry-contract SHA mapping,
   DAY/NIGHT N, and session-cluster 95% CI.
4. Plastic and wood use their own registry geometry. Wood does not inherit the
   plastic `{I, Ry(180°)}` contract.
5. If either object is pose-blocked, ALL pose cells stay blank/null; a passing
   partial subset is never renamed ALL.
6. Current statuses are plastic selector FAIL, wood selector NOT_RUN, wood
   symmetry UNREVIEWED, and every FINAL membership UNAVAILABLE.
7. `5cm5deg` and `10cm10deg` are not paper-facing columns.
8. Controlled, native-reference, and architecture-only comparisons remain
   distinct.
