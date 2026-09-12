# R0/C2 score–box–selection diagnostic

POSTHOC DEVELOPMENT DIAGNOSTIC. Training0; no new checkpoint, threshold selection, adapter or winner. Original C FAIL and original reports remain unchanged.

## Scope and integrity

Immutable R0 and all three final C2 seeds; four source combinations per seed; same DEV319 + NEG2689. R0 is repeated as a shared reference, not three independent R0 fits.
Before intervention, original full candidate lists are bit-exact to their saved caches on all3008 images. All image tensors, backbone/neck feature maps, feature-grid anchors/strides and dense decoded pose tensors are bit-exact between R0/C2.
Canonical fusion and one2one decoding/top-k/postprocessing are unchanged; unused one2many removal is the existing stock inference behavior. Both branches were checked in unfused checkpoint ownership. Hybrid replay uses the original postprocessing pipeline without a new neural forward.
Common-candidate identity means the pre-postprocess grid index, never IoU-based correspondence between final boxes. All jointly retained identical indices have bit-exact keypoints.

## Three-seed means

| Box / score source | AP50 | AP75 | AP50-95 | Det | AUROC | FPR95 | common kp median/P90 px | translation cm | yaw deg |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| R0 | 0.936282 | 0.884293 | 0.768767 | 0.974922 | 0.992131 | 0.041651 | 6.6285/38.6978 | 7.8969 | 1.2306 |
| C2 | 0.914874 | 0.859932 | 0.744155 | 0.981191 | 0.987231 | 0.067807 | 6.6373/38.7709 | 8.0003 | 1.2232 |
| R0box_C2score | 0.914729 | 0.868510 | 0.751695 | 0.981191 | 0.987231 | 0.067807 | 6.6373/38.7709 | 8.0003 | 1.2232 |
| C2box_R0score | 0.936282 | 0.880321 | 0.761184 | 0.974922 | 0.992131 | 0.041651 | 6.6285/38.6978 | 7.8969 | 1.2306 |

AP at every IoU0.50:0.05:0.95 and negative candidate counts are in RESULTS.json. Negative FP counts use only the existing0.001 inference floor, not a newly tuned operating point.
Geometry is pooled supervised0..8 raw-pixel error on the four-arm common matched frames within each seed. Pose uses the original MAIN selector and all319-frame denominator; failures/missing cases are not removed. Reference pose is geometry-reconstructed, not external sensor ground truth.

## Per seed

| Seed | Arm | AP50-95 | negative frames with candidates /2689 | pose coverage | translation cm | yaw deg |
|---|---|---:|---:|---:|---:|---:|
| 1 | R0 | 0.768767 | 1539 | 1.000000 | 7.8969 | 1.2306 |
| 1 | C2 | 0.743227 | 2321 | 1.000000 | 8.0536 | 1.1986 |
| 1 | R0box_C2score | 0.750964 | 2321 | 1.000000 | 8.0536 | 1.1986 |
| 1 | C2box_R0score | 0.759732 | 1539 | 1.000000 | 7.8969 | 1.2306 |
| 2 | R0 | 0.768767 | 1539 | 1.000000 | 7.8969 | 1.2306 |
| 2 | C2 | 0.747894 | 2359 | 1.000000 | 7.8936 | 1.2584 |
| 2 | R0box_C2score | 0.752924 | 2359 | 1.000000 | 7.8936 | 1.2584 |
| 2 | C2box_R0score | 0.765965 | 1539 | 1.000000 | 7.8969 | 1.2306 |
| 3 | R0 | 0.768767 | 1539 | 1.000000 | 7.8969 | 1.2306 |
| 3 | C2 | 0.741345 | 2402 | 1.000000 | 8.0536 | 1.2125 |
| 3 | R0box_C2score | 0.751199 | 2402 | 1.000000 | 8.0536 | 1.2125 |
| 3 | C2box_R0score | 0.757854 | 1539 | 1.000000 | 7.8969 | 1.2306 |

## Selection and common-candidate controls

| Seed | Four-arm matched frames | Same top-grid-index frames | C2 score top-index changes /positive both detected | Missing transitions /all3008 |
|---|---:|---:|---|---:|
| 1 | 311 | 266 | 48/319 | 784 |
| 2 | 311 | 276 | 37/319 | 820 |
| 3 | 310 | 282 | 32/319 | 863 |

All four arms have identical geometry statistics on the same-top-candidate subsets. For every image, changing only boxes preserves the selected index list; changing scores uses exactly the score-source index list. Thus dense invariance does not imply selected-output invariance.

## Diagnostic contrasts (AP50-95 percentage points)

- C2 minus R0: -2.4611 pp.
- R0box_C2score minus R0: -1.7071 pp.
- C2box_R0score minus R0: -0.7583 pp.
- R0box_C2score minus C2: +0.7540 pp.

## Interpretation limits

Score and box swaps identify output-path effects for these fixed checkpoints. They do not alone prove that frozen features are insufficient, nor isolate the training cause of head changes. Do not automatically add an adapter or adopt a hybrid as a novel method.
Box changes may change IoU matching/metric inclusion even when selected keypoints are identical. Compare common-frame and common-candidate statistics, not separately selected error pools.
Original C0 is replay-finetuned, not immutable R0. Original pose_safe allows up to10% translation/yaw regression and is not a no-regression guarantee.
The review also correctly identifies that D used19 scalar features without the requested explicit local appearance descriptor. Its FAIL covers that restricted feature set and risk rule, not full visual trust or student performance. D/B/E are not rerun here.
This repeated development population supplies no independent generalization, statistical significance or novelty claim. A remains frozen pending independent confirmation. No paper-final document is edited.
