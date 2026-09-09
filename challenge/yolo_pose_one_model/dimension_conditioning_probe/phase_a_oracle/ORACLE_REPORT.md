# Phase A — dimension/parity oracle diagnostic

Verdict: **ORACLE_PARITY_HEADROOM_PRESENT**

DEV diagnostic only. A1/A2/A3 are oracle arms, use GT, and are not
paper-eligible deployment methods. No YOLO parameter was updated.

## Plastic DEV140

| arm | parity acc | PnP solve | pose valid | restricted ADD-S AUC | R median ° | t median m | yaw median ° |
|---|---:|---:|---:|---:|---:|---:|---:|
| `A0_CURRENT_SELECTOR` | 0.592857 | 0.985714 | 138/140 | 0.264757 | 8.066685 | 0.132614 | 7.898054 |
| `A1_GT_PARITY_ORACLE` | 1.000000 | 0.985714 | 138/140 | 0.314657 | 3.670709 | 0.105495 | 3.008073 |
| `A2_BEST_OF_TWO_ORACLE` | 0.842857 | 0.985714 | 138/140 | 0.334961 | 3.475910 | 0.099125 | 2.909332 |
| `A3_GT_V2_KEYPOINT_ANNOTATIONS_REPLAY` | 1.000000 | 1.000000 | 140/140 | 0.999307 | 0.000000 | 0.000000 | 0.000000 |
| `A4_WRONG_OBJECT_DIMENSION_CONTROL` | 0.585714 | 0.985714 | 138/140 | 0.000000 | 81.151200 | 0.890285 | 81.104534 |

## A2 interpretation and static contract

- GT-v2 keypoint static mapping check: `PASS` (140/140)
- Predicted-keypoint pose-oracle agreement with GT parity: 118/138 comparable frames
- Predicted-keypoint disagreements are diagnostic evidence of noisy or permuted keypoint correspondences; they are not a static mapping failure.

## A3 replay disclosure

A3 uses `GT-v2 keypoint_annotations.xy`. Those annotations and the
canonical pose candidates share legacy annotation/PnP provenance, so A3
is a solver consistency replay, not an independent noise-floor estimate.

## Frozen oracle gate

- G1: `True`
- G2: `True` (4 conditions)
- G3: `True`
- G4: `True` (38 / 57 selector errors recovered)

G3 is implemented on the primary Plastic DEV140 `ALL` summary. The
user protocol does not explicitly resolve an every-subgroup reading;
under that strict sensitivity reading G3 fails because NIGHT median
translation worsens by more than 5%. See the final report.

## Safety and secondary population

A0/A1/A2 confidence, box and keypoint arrays are byte-identical inputs.
A4 executes only plastic images with wood dimensions. The reverse
wood-images/plastic-dimensions direction is blocked and was not evaluated.
All wood evaluation is blocked because symmetry is `UNREVIEWED` and
intrinsics are `SENSOR_PROFILE_SCALED`; the 45 wood predictions remain
in the cache only as immutable evidence for a later approved audit.
