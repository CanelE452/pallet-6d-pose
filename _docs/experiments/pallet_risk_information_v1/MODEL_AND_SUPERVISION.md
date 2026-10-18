# Frozen model and supervision contract

All new models select an archived action whose stored pose was produced by the fixed original pipeline. They do not create or refine poses. The original D1 action set, aliases, eligibility, F0 normalizer, and canonical action order remain fixed. Real evaluation references do not enter model fitting, checkpoint selection, or feature normalization.

| Group | Model input | Head | Trainable parameters |
|---|---|---|---:|
| R_CONT_F0 | Original 188-dimensional F0 | 188→64→64→2, ReLU hidden layers, softplus outputs | 16,386 |
| R_ORDER_F0 | Original F0 | 188→64→64→1 | 16,321 |
| R_ORDER_F1_ZERO / R_ORDER_F1 | F0 + 144 normalized additional scalars + 144 validity values | 476→64→64→1 | 34,753 each |
| V_ORDER_NULL / V_ORDER_RGB | F1 input + 320 local appearance values + 20 region validities | 816→64→64→1, plus shared local CNN | 57,977 each |

The F1 ZERO control neutralizes the entire additional 288-dimensional block inside the model. It retains the F1 architecture and initialization. F1 uses the identical extra feature file and train-only normalizer. No extra pooling changes the original F0 block.

The visual model queries eight projected corners and twelve projected edge centers for each actual action. Each query supplies a 16×16 patch with RGB and a pixel-validity mask. A shared 4→8→16 convolutional network with 3×3 kernels, ReLU, and global average pooling yields sixteen values per query. The NULL control zeros only RGB; it preserves pixel masks and the twenty region-validity values. Candidate patches are supplied in chunks of at most eight. The training provider receives a reference-free frame, actual actions, and action indices; it does not receive synthetic supervision.

Original RGB is bilinearly sampled with float64 coordinate arithmetic and float32 output, cached in float16, and restored to float32 for the model. Binary masks record usable samples inside the image and remain exact in the cache. Invalid geometry or nonpositive camera depth yields zero validity. Region validity is the valid-pixel fraction.

## Continuous error supervision

R_CONT_F0 predicts nonnegative values for `log1p(T_cm/5)` and `log1p(R_deg/5)`, without clipping targets. SmoothL1 with beta 1 first averages available axes within an action, then valid actions within a frame, then usable frames within a batch. Missing axes do not become zero-error targets. Selection minimizes the maximum of the two predicted log errors. This head is an error regression model, not a calibrated success probability.

## Partial-order supervision

Original joint success uses the fixed 5 cm and 5° definition. A successful action outranks an unsuccessful action. Between two unsuccessful actions, a preferred action must weakly improve both error axes and strictly improve at least one. Two successful actions, equal failures, and failure tradeoffs do not acquire an invented order or a 0.5 label.

The original C1/C2 metric expression and an independently stable expression are compared only to identify numerically ambiguous ordering. A conflicting ordering is excluded from the rank loss and recorded as `NUMERICAL_ORDER_AMBIGUOUS`. Original targets and success labels remain unchanged; no new numerical tolerance is introduced.

For a preferred pair `(i,j)`, the loss is `softplus(score_j − score_i)`. Success/failure pairs and failed-action Pareto pairs are averaged separately within a frame; present categories are then averaged. A batch averages frames with at least one valid relation. Frames without an ordering remain in all prediction and reporting denominators. Training uniformly samples at most 256 pairs per category per frame using the fixed seed, epoch, canonical training-frame number, and category. Validation evaluates every valid pair in chunks of 4,096.

The fixed original synthetic split has 768 training and 235 validation frames. Independent supervision reconstruction found 733,448 / 199,749 usable train / validation pairs, including 65,044 / 11,617 success/failure pairs and 668,404 / 188,132 failed-action Pareto pairs. There are 303 / 83 frames without ordering supervision. No numerical-order ambiguity or missing axis occurred in this fixed dataset. These are supervision counts, not real-data performance results.

## Fixed fits and selection

All new groups use seeds 20261018, 20261019, and 20261020, with 20261018 fixed as representative. AdamW uses learning rate 0.001, weight decay 0.0001, batches of 32 frames, at most 50 epochs, and patience 5. Each checkpoint is the first strict minimum of that group's own synthetic validation objective. Loss values from different objectives are not used to choose a method.

ZERO/F1 and NULL/RGB have identical initial weights and deterministic frame and pair schedules for each seed. Their stopping epochs can differ because each uses its own fixed validation loss. All best and last checkpoints, epoch diagnostics, attempts, and failures are retained. The campaign permits twelve basic new fits and at most six gated visual fits, with a global cap of eighteen optimizer attempts.

Selection preserves the original H1 order: N0 comes first, followed by physical numeric action order. Exact score ties use the first action in that order. Continuous selection uses the same rule for exact minimax ties. Scores are not interpreted as calibrated probabilities.

CPU fits use one Torch intra-operation thread and one inter-operation thread. Visual fits require a fresh process with `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic Torch algorithms and cuDNN, and disabled TF32. Reported forward/backward probes are fixed-subset measurements; they do not establish end-to-end detector or device FPS.

Implementation: [models and losses](../../../scripts/research/pallet_risk_information_v1/risk_models.py), [synthetic supervision](../../../scripts/research/pallet_risk_information_v1/supervision.py), [R/F training](../../../scripts/research/pallet_risk_information_v1/train.py), and [gated visual training](../../../scripts/research/pallet_risk_information_v1/train_visual.py).
