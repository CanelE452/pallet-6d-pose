# CAD8 pseudo fine-tuning → non-CAD occlusion

Training: CAD 2,6,7,8,9,10,11,12. Frozen pseudo labels only; no manual CAD targets. Entire CAD excluded from evaluation.
Two arms from R0; 512 synthetic replay +512 balanced CAD slots,5epochs,320updates,seed42. Standalone student at inference.
Primary excludes the Replay teacher training session (plastic_night_01). Condition tags locked before training. Reused DEV; not independent confirmation.

## PRIMARY_OCC96

| Arm | Median px | P90 px | PCK10 | PCK20 | Matched |
|---|---:|---:|---:|---:|---:|
| R0 | 11.387 | 64.137 | 41.79% | 62.82% | 88/96 |
| CLEAN | 11.687 | 64.192 | 40.43% | 62.96% | 88/96 |
| OCCLUDED | 11.520 | 64.235 | 40.98% | 63.64% | 88/96 |

## ALL_OCC107

| Arm | Median px | P90 px | PCK10 | PCK20 | Matched |
|---|---:|---:|---:|---:|---:|
| R0 | 9.958 | 58.562 | 46.91% | 66.79% | 99/107 |
| CLEAN | 10.195 | 58.268 | 45.70% | 66.91% | 99/107 |
| OCCLUDED | 10.124 | 58.442 | 46.30% | 67.52% | 99/107 |

## ALL_NONCAD176

| Arm | Median px | P90 px | PCK10 | PCK20 | Matched |
|---|---:|---:|---:|---:|---:|
| R0 | 6.792 | 49.239 | 61.95% | 76.76% | 168/176 |
| CLEAN | 6.879 | 47.821 | 61.58% | 76.91% | 168/176 |
| OCCLUDED | 6.830 | 48.174 | 62.02% | 77.20% | 168/176 |

## CLEAN69

| Arm | Median px | P90 px | PCK10 | PCK20 | Matched |
|---|---:|---:|---:|---:|---:|
| R0 | 4.272 | 15.239 | 84.42% | 91.67% | 69/69 |
| CLEAN | 4.254 | 15.137 | 85.33% | 91.85% | 69/69 |
| OCCLUDED | 4.260 | 14.846 | 85.51% | 91.67% | 69/69 |

