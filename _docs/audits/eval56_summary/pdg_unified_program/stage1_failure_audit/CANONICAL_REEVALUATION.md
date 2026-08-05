# Canonical re-evaluation of the audits that used the deprecated set

Four audits from 08-05 judged on D13/C13, which come from
`data/_eval_sets/night_combined` and `outside_combined` -- the set CLAUDE.md
bars from evaluation, with `objects[0].split` absent on all 26 frames.  All of
them are re-run here on the canonical eval56 (56) and wood (45).  Three
verdicts flip.

## 1. The failure population barely exists in the canonical set

```
centroid raw <= 0.30 (A0)
  deprecated N87    13/87  = 14.9%     <- the D13 this whole programme targeted
  canonical eval56   2/56  =  3.6%
  canonical wood     0/45  =  0.0%
```

The two canonical frames:

```
centroid 0.0268   corner max 0.1058   corners > 0.30  0/8
centroid 0.1114   corner max 0.8426   corners > 0.30  3/8
```

Only the first is a global collapse.  The second has a strong corner response
and is a centroid-specific failure.  **The 13-frame no-response population is a
property of the deprecated set, not of the model on canonical data.**

## 2. Sigma sweep: CONFIG_ONLY_FAIL is overturned

```
        S00    S05    S10    S15    S20    S25    S30
eval56  54/56  54/56  54/56  53/56  51/56  41/56   0/56   centroid survival
        PnP 50    50     49     49     46     38      0
wood    45/45  45/45  45/45  45/45  45/45  43/45   0/45
        PnP 44    44     44     43     42     37      0
```

The deprecated N87 run capped at 74/87 = 85.1% even at sigma 0 and concluded
the ceiling was the model, not the blur.  On canonical data sigma 0 gives
**96.4% and 100%**, and the deployment decoder runs at full strength anywhere up
to sigma 1.0: eval56 54/56 centroid, 54 objects, 49 PnP; wood 45/45 and 44.

There is no ceiling.  **Lowering the deployment sigma alone makes the decoder
work**, which is the opposite of CONFIG_ONLY_FAIL.

## 3. Padding: helps on canonical data instead of costing

```
                    eval56                          wood
                    centroid  R4  PnP  reproj       centroid  R4  PnP  reproj
A0 original         54/56     47   50  11.5578      45/45     44   44   9.2839
A1 reflect          55/56     54   54  13.1741      45/45     44   44   9.0973
A3 constant127      56/56     55   55  12.1885      45/45     43   44   8.8474
```

On the deprecated set padding rescued detections but degraded the healthy
frames, so it was recorded as inadmissible.  On canonical data constant-grey
padding takes **centroid to 56/56, R4 47 to 55 and PnP 50 to 55** on eval56 and
improves wood reprojection to 8.8474.  Reprojection on eval56 rises 11.56 to
12.19, so it is a trade rather than a free win, but the earlier "harms healthy
frames" conclusion does not hold here.

## 4. Stage 1 arms on canonical data

```
        eval56                          wood
        PnP    cent   R4   reproj       PnP    cent   R4  reproj
A0      50/56  54/56  47  11.5578       44/45  45/45  44  9.2839
A1      52/56  55/56  50  10.1608       44/45  45/45  42  8.9569
A2      52/56  54/56  47  16.5769       44/45  45/45  44  8.7766
A2 D0V  50/56  54/56  44  16.1948       44/45  45/45  43  8.7766

paired, common success, bootstrap 10,000 seed 1
eval56  A0->A1  median -0.518px  improved 34 worsened 16  rescue 2  new failure 0
        A1->A2  median +1.551px  improved 14 worsened 38  P(improve) 0.005
        VAPA    median  0.000px                            new failure 2
```

## Verdicts that flip

```
LATE_FINETUNE_DRIFT (A1)     REJECTED  A1 improves eval56 reprojection 12.1%,
                                       R4 +3 with none lost, PnP +2 with none lost
VAPA_INACTIVE                REJECTED  becomes VAPA_OVER_SUPPRESSION: PnP 52->50,
                                       R4 47->44, 2 new failures on eval56
CONFIG_ONLY_FAIL             REJECTED  no ceiling on canonical data; sigma <= 1.0
                                       runs the deployment decoder at full strength
PADDING harms healthy frames REJECTED  constant127 raises eval56 PnP 50->55
```

## Verdicts that hold

```
A2 does not transfer         HELD and strengthened: eval56 reprojection +43.4%
                             against A1, P(improve) 0.005
FOUNDATION_STOP              HELD, for a different reason -- A2 is worse than A1,
                             not "A2 failed to rescue D13"
TOPOLOGY_COVERAGE_GAP        fact holds (GT geometry only) but loses its priority:
                             the population it addressed is 2 frames canonically
```

## Withdrawn for want of sample

`ENCODER_OBJECT_EVIDENCE_GAP` rested on F50 enrichment 0.866 and PRH alive 2/13
over the 26 deprecated frames.  Canonically there are 2 no-response frames, too
few to re-measure.  The claim is withdrawn rather than restated.

## What the canonical bottleneck actually is

```
eval56 A0   centroid detected 54/56
            corners >= 4 as well 47/56    <- 9 frames lost here
            PnP 50/56
            median corners detected 6/8, NaN corners 119/448, >50px 45, >100px 17
```

Seven of the nine R4 failures have a live centroid (0.41 to 0.89) and fewer than
four corners.  The canonical bottleneck is corner detection and localisation,
not object response.  TACA, PRH, KVH and VAPA all target the latter.
