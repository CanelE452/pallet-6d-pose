# SITE_A pseudo-label preflight

Frozen R0 teacher, frozen filter lock, full SITE_A pool. No student was
trained, no threshold was chosen here, and no evaluation ground truth was
read — the question is quantity, diversity and exposure, not purity.

```text
stage                        A8 subset   Full SITE_A
────────────────────────────────────────────────────
Teacher input                      500          2227
Detected                             —          2122
>= 6 corners (F0)                    —          2114
Confidence (F1)                      —           574
Proposed (F4)                      120           563
```

## Per recording

```text
Recording              Input    Cand    Conf  Proposed  Retention   Share
─────────────────────────────────────────────────────────────────────────
capturepallet01           42      39       0         0      0.000   0.000
capturepallet10          613     572     118       118      0.192   0.210
capturepallet11         1572    1503     456       445      0.283   0.790
ALL                     2227    2114     574       563      0.253   1.000
```

## Exposure concentration

```text
pseudo exposures / epoch   1440  (existing contract, not chosen here)
A8   unique   120   repeat  12.00 x / epoch
Full unique   563   repeat   2.56 x / epoch
```

## Membership

```text
A8 accepted                120
Full accepted              563
A8 frames also in Full     120
A8 frames lost in Full     0
new in Full only           443
```

## Coverage (predictions only, no GT)

```text
quantity               A8 median            A8 p10-p90  Full median          Full p10-p90
─────────────────────────────────────────────────────────────────────────────────────────
bbox_area_frac            0.0188      [0.0105, 0.1119]       0.0183      [0.0094, 0.1195]
center_x_frac             0.5316      [0.2664, 0.8625]       0.5195      [0.2572, 0.8396]
center_y_frac             0.5785      [0.5694, 0.6437]       0.5777      [0.5691, 0.6451]
bbox_aspect               8.6229     [7.4481, 10.1992]       8.7913     [7.7476, 10.4697]
kp_spread_frac            0.2795      [0.2166, 0.5957]       0.2737      [0.2091, 0.6480]
box_conf                  0.9177      [0.8729, 0.9397]       0.9157      [0.8698, 0.9403]
s_remove                  0.0164      [0.0085, 0.0309]       0.0161      [0.0074, 0.0303]
s_flip                    0.0114      [0.0057, 0.0210]       0.0106      [0.0052, 0.0206]
valid_corners             8.0000      [8.0000, 8.0000]       8.0000      [8.0000, 8.0000]
```

## Temporal spacing of accepted frames

```text
Recording             accepted    pool   med gap   p10 gap    run    span s
───────────────────────────────────────────────────────────────────────────
gaps are in frames, using each recording's own median frame period
capturepallet10            118     613       1.0       1.0      9      17.2
capturepallet11            445    1572       1.0       1.0     16      63.2
```

No GO/STOP decision is made in this document and no threshold was invented.
It reports what the pool contains so a person can decide.
