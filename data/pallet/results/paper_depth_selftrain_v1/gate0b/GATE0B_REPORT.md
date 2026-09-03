# GATE 0B — depth contract resolution

Gate 0 ended PARTIAL with three gaps. This closes the coverage gap, settles what can
and cannot be established about how the depth was recorded, and — the substantive
part — measures inside the pallet rather than around it.

```text
FINAL = NOT_READY_FOR_GATE1_PILOT
```

## Coverage

```text
expected RGB-D pairs        8031
sensor frames audited       8031
R0 ROI frames               8031
  reused from cache         2727
  new inference             5304
frames with detection       7737
recipe parity               box 0.0e+00 px  kp 0.0e+00 px  conf 0.0e+00  -> PASS
```

Gate 0 audited regions of interest on 2,727 frames because that is what the caches
held. This one covers all 8,031.

## Sensor validity, whole population

```text
group                     N  strict valid  loose valid    zero  dtype max   p99 m
─────────────────────────────────────────────────────────────────────────────────
ALL                    8031         0.883        0.985   0.015      0.093    61.0
DAY                    2227         0.850        0.989   0.011      0.130    61.0
NIGHT                  5804         0.894        0.981   0.019      0.073    61.6
capturepallet01          42         0.910        0.997   0.003      0.086    50.9
capturepallet10         613         0.858        0.991   0.009      0.131    61.0
capturepallet11        1572         0.843        0.987   0.013      0.131    61.0
capturenight01         1254         0.915        0.957   0.043      0.042    56.0
capturenight02          782         0.949        0.965   0.035      0.016    32.4
capturenight03         1219         0.876        0.995   0.005      0.117    61.6
capturenight04         1075         0.878        0.993   0.007      0.113    61.6
capturenight10         1474         0.894        0.970   0.030      0.068    61.6
```

Both layers are reported and no far-clipping rule was invented.

## Pallet-local depth — the finding

```text
group                   det   face   rate    pts  face MAD cm  plane mm  ring-face cm  whole bbox cm
────────────────────────────────────────────────────────────────────────────────────────────────────
ALL                    7737   7672  0.992   2278         17.5      21.7          -2.5          152.3
DAY                    2122   2106  0.992   1478         64.0      20.0          -9.6          448.2
NIGHT                  5615   5566  0.991   2850         15.6      22.4          -1.2          124.4
capturepallet01          39     39  1.000    441        313.1      76.8        -574.4         1355.2
capturepallet10         572    571  0.998    844         91.4      13.1         -32.8          415.2
capturepallet11        1511   1496  0.990   2071         23.9      21.8          -7.1          471.3
capturenight01         1253   1253  1.000   2302         16.9      21.7          -1.6          120.2
capturenight02          762    755  0.991   6987         11.5      19.8           2.1           97.0
capturenight03         1218   1211  0.994   2451         15.2      20.3          -4.9           99.5
capturenight04         1075   1073  0.998   3546         20.7      25.3           1.5          969.0
capturenight10         1307   1274  0.975   1286         18.5      23.3          -1.0          156.4
```

The last two columns are the point. The whole box spans 152 cm of depth; inside the
projected faces the median absolute deviation is 17.5 cm and a least-squares plane
fits to 22 mm. Gate 0 measured the first number and concluded there was no structure.
There is structure; Gate 0 was looking at the container, not the object.

Nightside is markedly cleaner than day — 15.6 cm against 64.0 cm — which inverts the
usual expectation and is consistent with the daytime scenes being open outdoor plazas
where the background runs to tens of metres.

### What this does not settle

A low plane residual proves a plane, not a pallet. The ring around each face sits at
nearly the same depth as the face itself, which is exactly what an obliquely viewed
ground plane produces. `capturepallet10` shows it outright: a 91 cm face spread that
still fits a plane to 13 mm. Separating deck from ground needs the known pallet
dimensions and a fitted box, which is Gate 1 work, not a sensor question.

## Alignment

```text
recording             chamfer @0    best    gain  relative   modal shift share
──────────────────────────────────────────────────────────────────────────────
capturepallet01            22.08   19.27    2.81     12.7%                 33%
capturepallet10            17.69   16.98    0.70      4.0%                 17%
capturepallet11            14.28   13.65    0.63      4.4%                  8%
capturenight01             47.70   35.87   11.83     24.8%                 25%
capturenight02             23.32   17.73    5.58     23.9%                 17%
capturenight03             45.17   29.68   15.49     34.3%                 42%
capturenight04             25.51   21.31    4.20     16.5%                 17%
capturenight10             43.19   35.09    8.10     18.8%                 25%
```

Symmetric chamfer, both directions, so neither edge map is favoured. A genuine rigid
offset would show a consistent best shift and a large gain. The day recordings gain
about four percent from shifting, and the best shift agrees on a direction in only 8
to 42 percent of frames with many pinned to the search boundary — a metric escaping,
not an offset being found. The depth was never moved; this is verification only.

This rules out gross misalignment. It does not establish sub-pixel calibration.

## Verdict

```text
Acquisition provenance          UNRESOLVED
Full sensor validity            PASS
Pallet-local signal             PROMISING_WITH_ONE_UNRESOLVED_QUESTION
Alignment empirical             SUPPORTS_ALIGNMENT
FINAL                           NOT_READY_FOR_GATE1_PILOT
```

the local depth signal is real and much better than Gate 0 suggested, and alignment is not contradicted. What blocks a pilot is provenance: with every recording at C_COMPATIBLE_ONLY, a metric correction would rest on an alignment nobody recorded and, for the night half, on an intrinsic whose stream is unknown. The lock requires A_EXPLICIT or B_STRONG for CONTRACT_CONFIRMED and that is not met.

What would change it:

- an acquisition record, capture script or device log tying these eight sequences to a writer that states alignment and stream
- or a physical check that establishes alignment and the night K empirically, for example a target of known geometry captured with the same rig

Nothing here says depth correction would improve accuracy.

`NEXT_ACTION = USER_REVIEW_GATE0B`

