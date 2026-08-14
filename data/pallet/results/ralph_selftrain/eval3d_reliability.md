# ralph self-training — usable-pose re-eval (value-aligned)

- ★PRIMARY usbl.10/usbl.15 = usable-pose rate = %(det AND ADD<0.10/0.15m) over
      ALL eval frames (miss/PnP-fail = not usable). rewards miss->roughly-right (recall).
- ADD(m) = same-index 8-corner 3D dist via pseudo-GT PnP; MEDIAN over det+PnP-ok
      frames (n3D). ★survivorship: coverage separate; pseudo-GT=PnP-from-2D not metrology.
- gross% / catas% = %(9kp-median px > 20 / 40), DETECTED frames as denom.
- relia = mean_i peak_i*grade(err_i); grade: <=10px +1, 10..40 +1->0, 40..80 0->-1,
      >80px -1 (lenient on small err, punishes only confident-huge-wrong). undet kp->0.
- squash-parity infer (all s2 lineage); same held-out eval GT as ralph_eval_all.
- self-domain = per-domain self-train (outside=h6,night=h3,noapril=h7); cad has none.
  ctrl=synthetic-only (PL=0) isolates PL-specific effect. combined=h4.

```
domain   model          N  det%  ADD(m)  n3D usbl.10 usbl.15  gross%  catas%   relia
------------------------------------------------------------------------------------
outside  R0           117  64.1   0.399   75     9.4    17.9    49.3    29.3   0.267
         ctrl         117  48.7   0.298   57    10.3    14.5    47.4    17.5   0.233
         self=st_outside 117  89.7   0.383  105     3.4    16.2    58.1    20.0   0.324
         combined     117  83.8   0.346   98     8.5    14.5    50.0    23.5   0.293
------------------------------------------------------------------------------------
night    R0            43  67.4   0.462   29     7.0     7.0    55.2    17.2   0.263
         ctrl          43  58.1   0.396   25     2.3     9.3    56.0    20.0   0.194
         self=st_night  43  76.7   0.274   33    11.6    16.3    39.4     9.1   0.331
         combined      43  76.7   0.261   33    14.0    16.3    42.4    18.2   0.339
------------------------------------------------------------------------------------
noapril  R0            18  83.3   0.071   15    55.6    66.7     6.7     0.0   0.715
         ctrl          18  83.3   0.070   15    61.1    66.7     6.7     0.0   0.663
         self=st_noapril  18  83.3   0.053   15    77.8    83.3     0.0     0.0   0.778
         combined      18  83.3   0.059   15    61.1    77.8     0.0     0.0   0.733
------------------------------------------------------------------------------------
cad      R0            44   4.5   0.043    2     4.5     4.5     0.0     0.0   0.200
         ctrl          44   6.8   0.088    3     4.5     4.5    33.3     0.0   0.181
cad      self(n/a)   (no cad-specific self-train model)
         combined      44   6.8   0.074    3     4.5     4.5    33.3     0.0   0.168
```

