# P0 -> P1: what the coordinate extractor does

Corners both paths accepted, so detection membership cannot flatter either one.

```
   set  arm  role    n  median delta  toward GT  P0 median  P1 median
─────────────────────────────────────────────────────────────────────
eval56   B0  near  133        +2.487      37.6%      4.631      6.649
eval56   B0   far  170        +3.814      30.0%     10.907     13.567
eval56   B0   ALL  303        +3.359      33.3%      6.967      9.954
eval56   E2  near  133        +2.487      37.6%      4.631      6.649
eval56   E2   far  168        +3.054      35.1%      9.154     11.037
eval56   E2   ALL  301        +2.791      36.2%      6.005      8.954
  wood   B0  near  123        +7.466      14.6%      6.415     13.386
  wood   B0   far  160        +1.418      45.6%     13.023     14.697
  wood   B0   ALL  283        +5.066      32.2%      8.552     13.761
  wood   E2  near  123        +7.466      14.6%      6.415     13.386
  wood   E2   far  161        +2.753      41.0%     11.070     13.876
  wood   E2   ALL  284        +5.516      29.6%      8.294     13.559
```

The Gaussian blur, the NMS and the 11x11 weighted average move the corner
**away** from GT about two times out of three.  eval56 median displacement
+3.36px, wood +5.07px.  The near face on wood is the worst case: 14.6% move
toward GT and the median error goes 6.42 -> 13.39px.

This is not a defect in the D2 code.  It is a mismatch: an 11x11 weighted
average over a 50x50 map is a wide window for a model whose belief blobs are
narrow, so the average is pulled by the surrounding background.  The 7x7
softargmax at temperature 0.1 that D0 uses is far more local.

The tail is the exception.  Exactly one corner per set crosses 50px in the
helpful direction (P0 > 50 and P1 <= 50) and none crosses the other way, and
the >50px count falls only because D2 declines to output those corners at all.

**For Q1 this matters:** the far-stage effect the arms were built on survives
the change of extractor.  E2 cuts far error 11.41 -> 9.64px on P0 (-15.5%) and
13.57 -> 11.04px on P1 (-18.6%); N2 gives -16.2% and -18.3%.  What does not
survive is anything that ran through detection counts -- see the verdict
matrix.
