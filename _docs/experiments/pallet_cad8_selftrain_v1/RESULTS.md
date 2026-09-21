# CAD8 pseudo fine-tuning → non-CAD occlusion

Training: CAD 2,6,7,8,9,10,11,12. Frozen pseudo labels only; no manual CAD targets. Entire CAD excluded from evaluation.
Two arms from R0; 512 synthetic replay +512 balanced CAD slots,5epochs,320updates,seed42. Standalone student at inference.
Primary excludes the Replay teacher training session (plastic_night_01). Condition tags locked before training. Reused DEV; not independent confirmation.

## PRIMARY_OCC96

| Arm | Median px | P90 px | PCK10 | PCK20 | Matched |
|---|---:|---:|---:|---:|---:|
| R0 | 11.387 | 64.137 | 41.79% | 62.82% | 88/96 |
| N3_PNP | 12.364 | 61.420 | 35.69% | 58.21% | 82/96 |
| REPLAY_PNP | 12.204 | 57.093 | 34.19% | 59.16% | 83/96 |

## ALL_OCC107

| Arm | Median px | P90 px | PCK10 | PCK20 | Matched |
|---|---:|---:|---:|---:|---:|
| R0 | 9.958 | 58.562 | 46.91% | 66.79% | 99/107 |
| N3_PNP | 10.771 | 57.526 | 41.09% | 62.67% | 93/107 |
| REPLAY_PNP | 10.876 | 55.153 | 39.64% | 63.52% | 94/107 |

## ALL_NONCAD176

| Arm | Median px | P90 px | PCK10 | PCK20 | Matched |
|---|---:|---:|---:|---:|---:|
| R0 | 6.792 | 49.239 | 61.95% | 76.76% | 168/176 |
| N3_PNP | 7.095 | 54.635 | 57.08% | 72.91% | 161/176 |
| REPLAY_PNP | 7.257 | 53.169 | 56.14% | 73.35% | 162/176 |

## CLEAN69

| Arm | Median px | P90 px | PCK10 | PCK20 | Matched |
|---|---:|---:|---:|---:|---:|
| R0 | 4.272 | 15.239 | 84.42% | 91.67% | 69/69 |
| N3_PNP | 4.236 | 24.110 | 80.98% | 88.22% | 68/69 |
| REPLAY_PNP | 4.401 | 24.577 | 80.80% | 88.04% | 68/69 |

