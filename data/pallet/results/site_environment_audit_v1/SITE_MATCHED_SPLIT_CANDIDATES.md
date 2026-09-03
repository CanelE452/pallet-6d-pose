# Site-matched split candidates

**Provisional.** No site identity is confirmed here. The grouping below comes
from background-masked SIFT matching between source recordings, which is a
candidate generator, not a decision. Nothing is FROZEN and no training ran.

`status` can reach at best `READY_PENDING_SITE_CONFIRMATION` because the
first READY condition — a human confirming the site — has not happened yet.

```text
Site             Adapt rec Eval rec  Adapt img  Eval img  rec ovl  SHA ovl  status
────────────────────────────────────────────────────────────────────────────────────────────────────────────
SITE_CAND_01             0        1          0     29028        0        0  NO_ADAPT_DATA
SITE_CAND_02            14        6       7163      7787        0       11  LEAKAGE_FAIL
SITE_CAND_03             0        1          0     13583        0        0  NO_ADAPT_DATA
SITE_CAND_04             1        0       5945         0        0        0  NO_EVAL_DATA
SITE_CAND_05             3        0       5181         0        0        0  NO_EVAL_DATA
SITE_CAND_06             3        0       3768         0        0        0  NO_EVAL_DATA
SITE_CAND_07             1        0       3729         0        0        0  NO_EVAL_DATA
SITE_CAND_08             1        0       2688         0        0        0  NO_EVAL_DATA
SITE_CAND_09             1        0       2362         0        0        0  NO_EVAL_DATA
SITE_CAND_10             1        0       1924         0        0        0  NO_EVAL_DATA
SITE_CAND_11             1        0       1452         0        0        0  NO_EVAL_DATA
SITE_CAND_12             1        0       1254         0        0        0  NO_EVAL_DATA
SITE_CAND_13             1        0        911         0        0        0  NO_EVAL_DATA
SITE_CAND_14             0        1          0       739        0        0  NO_ADAPT_DATA
SITE_CAND_15             1        0        136         0        0        0  NO_EVAL_DATA
```

## Per site

### SITE_CAND_01  —  NO_ADAPT_DATA

```text
lighting     day
adaptation   (none)
evaluation   REC_001
  REC_001  EVAL     29028  real_unlabeled_day_20260830
```

### SITE_CAND_02  —  LEAKAGE_FAIL

```text
lighting     day, night, unknown
adaptation   REC_011, REC_024, REC_028, REC_029, REC_031, REC_033, REC_034, REC_036, REC_040, REC_038, REC_030, REC_041, REC_043, REC_045
evaluation   REC_007, REC_012, REC_021, REC_022, REC_027, REC_044
  REC_007  EVAL      2773  capturepallet09
  REC_012  EVAL      1941  capturepallet08
  REC_011  adapt     1572  capturepallet11
  REC_021  EVAL      1179  capturepalletcad
  REC_022  EVAL      1059  capturenight09
  REC_024  adapt      782  capturenight02
  REC_027  EVAL       647  capturenight08
  REC_028  adapt      613  capturepallet10
  REC_029  adapt      591  capturenight07
  REC_031  adapt      571  capture03
  REC_033  adapt      540  capturenight05
  REC_034  adapt      493  capturenight06
  REC_036  adapt      440  capture0403middle
  REC_040  adapt      308  capturepallet03
  REC_038  adapt      298  capturepallet02
  REC_030  adapt      290  capture_20260902_kimjihoon
  REC_041  adapt      248  capturepallet05
  REC_043  adapt      225  20260618_132917
  REC_045  adapt      192  capturepallet04
  REC_044  EVAL       188  capture0403noapril
```

### SITE_CAND_03  —  NO_ADAPT_DATA

```text
lighting     night
adaptation   (none)
evaluation   REC_002
  REC_002  EVAL     13583  real_unlabeled_night_20260830
```

### SITE_CAND_04  —  NO_EVAL_DATA

```text
lighting     unknown
adaptation   REC_003
evaluation   (none)
  REC_003  adapt     5945  capture_20260902
```

### SITE_CAND_05  —  NO_EVAL_DATA

```text
lighting     unknown
adaptation   REC_008, REC_013, REC_026
evaluation   (none)
  REC_008  adapt     2501  forklift_v4_174342
  REC_013  adapt     1923  forklift_v4_174925
  REC_026  adapt      757  forklift_v4_174126
```

### SITE_CAND_06  —  NO_EVAL_DATA

```text
lighting     night
adaptation   REC_017, REC_020, REC_023
evaluation   (none)
  REC_017  adapt     1474  capturenight10
  REC_020  adapt     1219  capturenight03
  REC_023  adapt     1075  capturenight04
```

### SITE_CAND_07  —  NO_EVAL_DATA

```text
lighting     unknown
adaptation   REC_005
evaluation   (none)
  REC_005  adapt     3729  forklift_v4_173507
```

### SITE_CAND_08  —  NO_EVAL_DATA

```text
lighting     unknown
adaptation   REC_004
evaluation   (none)
  REC_004  adapt     2688  negative_real_20260823
```

### SITE_CAND_09  —  NO_EVAL_DATA

```text
lighting     unknown
adaptation   REC_009
evaluation   (none)
  REC_009  adapt     2362  vdoframes
```

### SITE_CAND_10  —  NO_EVAL_DATA

```text
lighting     unknown
adaptation   REC_014
evaluation   (none)
  REC_014  adapt     1924  real_data
```

### SITE_CAND_11  —  NO_EVAL_DATA

```text
lighting     unknown
adaptation   REC_018
evaluation   (none)
  REC_018  adapt     1452  capture02
```

### SITE_CAND_12  —  NO_EVAL_DATA

```text
lighting     night
adaptation   REC_019
evaluation   (none)
  REC_019  adapt     1254  capturenight01
```

### SITE_CAND_13  —  NO_EVAL_DATA

```text
lighting     day
adaptation   REC_015
evaluation   (none)
  REC_015  adapt      911  forklift_raw_20260528
```

### SITE_CAND_14  —  NO_ADAPT_DATA

```text
lighting     day
adaptation   (none)
evaluation   REC_025
  REC_025  EVAL       739  capturepallet07
```

### SITE_CAND_15  —  NO_EVAL_DATA

```text
lighting     unknown
adaptation   REC_039
evaluation   (none)
  REC_039  adapt      136  _annotate_pallet_20260618_183705
```

## What still has to happen

```text
1  a human confirms each site candidate from its contact sheets
2  SITE_GROUP_LOCK.json is written from those confirmations
3  only then can a site-matched split be called READY
```

`AUTO_GROUPING_IS_FINAL = NO`
`HUMAN_CONFIRMATION_REQUIRED = YES`
`TRAINING_STARTED = NO`

