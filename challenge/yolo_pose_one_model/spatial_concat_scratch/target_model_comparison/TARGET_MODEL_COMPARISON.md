# New clean-start model versus historical targets

> DEV diagnostic only. These values cannot populate a FINAL paper cell or replace the current G38 paper baseline.

`FT_REFERENCE` is the repository-declared primary target, but it is a real-supervised upper bound. Its 12 known overlapping DAY frames are removed from the primary DEV128 comparison. `OLD_STAGE_A` is the secondary target-specific synthetic milestone.

## Leak-adjusted DEV128

| Comparison | Correct box | Pairwise both-correct | Corner median | Corner p90 | NIGHT top-1 |
|---|---:|---:|---:|---:|---:|
| New | 120/128 | 119 | 9.703px | 39.766px | 20/28 |
| FT_REFERENCE target | 126/128 | 119 | 6.443px | 26.945px | 27/28 |
| OLD_STAGE_A milestone | 124/128 | 118 | 9.614px | 41.276px | 26/28 |

Against FT_REFERENCE, the new base is 6 frames (4.688 pp) lower in correct-box coverage. On the 119 pairwise both-correct frames, median and p90 corner errors are 50.61% and 47.58% worse.

Against OLD_STAGE_A, correct-box coverage is 4 frames lower. On 118 both-correct frames, median is effectively tied (9.691px vs 9.614px), while p90 is 5.63% better.

## Detection ranking on leak-adjusted positive/negative DEV

| Model | AUROC | Positive recall @0.4 | Negative FPR @0.4 |
|---|---:|---:|---:|
| New | 0.983654 | 85.16% | 2.94% |
| OLD_STAGE_A | 0.988143 | 89.06% | 2.01% |
| FT_REFERENCE | 0.998922 | 98.44% | 0.63% |

The FT negative result is an upper-bound diagnostic, not a clean target-free comparison; its historical provenance explicitly marks the negative evaluation as non-honest.

## Fraction of the G38-to-FT gap recovered

- Correct-box coverage: 64.71%
- Fair common-frame corner median: 42.72%
- Fair common-frame corner p90: 68.35%
- NIGHT top-1: 41.67%

## S1 spatial head status

S1 cannot yet be compared honestly with FT_REFERENCE. It is a post-hoc parity head measured on mixed synthetic DEV, whereas the target values above are real YOLO detection/keypoint diagnostics. A real spatial-feature adapter, parity-to-PnP wiring, and a named untouched evaluation population are still required.
