# Table 4. Detection preservation and computational cost

| Model | Added trainable params | Box AP50 | Box AP50-95 | Detection coverage | Full pipeline median (ms) | Refinement-only median (ms) |
|---|---|---|---|---|---|---|
| R0 | 0 | 0.936282 | 0.768767 | 319/319 | 11.408 | 0.000 |
| N3_DIM_SYM | 20259 | same as R0* | same as R0* | 319/319 | 15.350 | 3.204 |

Detection preservation and computational cost. *N3 box AP is unchanged from R0 by output-preservation contract, not separately measured. Canonical R0 AP uses historical DEV319 positives plus DEV_NEG2689 negatives; detection coverage here is detected/319. Exact candidate-field, ordering and selected-instance equality was checked on all DEV319 frames for all three N3 seeds. Class is implicitly single-class pallet. Timing: same-session GPU, historical 26 images, batch 1, 20 warmups per arm, five balanced repeats, 130 samples per arm, four Torch threads and one OpenCV thread; fixed runtime seed 1 only. Full pipeline includes RAM-image preprocessing, R0, optional refinement and prediction-only PnP. Refinement-only includes conditioning, serialization and preservation checks. R0 has no refinement stage (zero).

Device: NVIDIA GeForce RTX 3080, 580.178.04, 863 MiB, 10240 MiB, 53, 11 %
