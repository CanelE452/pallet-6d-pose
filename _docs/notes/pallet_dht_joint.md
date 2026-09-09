# Pallet DHT joint architecture — 2026-09-08

## Proposal before training

User correction: improve a single point-and-Deep-Hough architecture trained on synthetic data, without pseudo-label filtering or self-training. Previous frozen `pallet_line_pose_v1` was a local point-conditioned residual estimator and does not answer this full-network question. Preserve those results and use separate `scripts/research/pallet_dht_joint_v1` / `data/pallet/results/pallet_dht_joint_v1` folders.

Trainable path: YOLO26n backbone/neck → P4 image adapter → global feature Hough transform → Hough-space convolutions and 12 structural-line role maps → normalized transpose backprojection → residual fusion into P3/P4/P5 → original box and keypoint heads. The whole model receives gradients. The original YOLO point/box losses remain; the primary arm also supervises line maps from the transformed synthetic batch keypoints. No predicted-point initialization, GT crop, WLS, snapping, self-training or real-image labels enter model inference/training.

The HT/IHT feature-feedback idea follows [Deep Hough-Transform Line Priors, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670324.pdf); this is a task adaptation, not an exact reproduction or a new claim of inventing HT/IHT. Sparse voting reuses the already checked local implementation. P4 uses 16 channels, 90 normal-angle bins, signed rho −28..28 in 0.5 feature-cell increments. Rectangular inputs retain their actual feature geometry. Hough theta wrapping reverses rho. Zero-initialized residual output preserves R0 at initialization; line probability maps and latent Hough features both feed the point head.

Compare common-R0 supervised continuation `point_only`, Hough feedback without line auxiliary `hough_features`, and primary `hough_joint`. All original modules are trainable except stock fixed DFL. Match actual examples, microbatch, updates, optimizer, augmentations and seed schedule. Use all 55,980 baseline synthetic training rows; do not condition inclusion on old detections. Existing input PNGs already have reflection padding, so do not pad them twice. Main budgets and loss scale will be frozen after CPU geometry/gradient checks and a synthetic-only GPU memory/speed/plumbing probe, before real evaluation.

Targets use the already augmented `batch.keypoints`/`batch_idx`, all 12 cuboid roles, independent multi-peak sigmoid maps for multiple instances, and masks for unknown/outside-only lines. These are amodal structural lines, not labels of physical edge visibility. Multiple-role or instance overlaps must not be forced into a single softmax peak.

Evaluation uses actual new inference for positive319 and negative2689, original reflect100/640rect FP32 inference and canonical 2D/MAIN6D scoring. Both boxes and keypoints may change; copying old negatives is invalid. Report pooled9point median/P90, matching coverage, AP/negative false positives, rotation/translation/IoU3D/ADDsym AUC, paired session uncertainty, and full input-to-output runtime. The existing real319 set is reused DEV, not independent final test. Do not select checkpoints, weights or rules on real results.

Failure risks include synthetic-to-real transfer, amodal-line ambiguity, clutter/occlusion, keypoint identity symmetry, coarse line geometry and auxiliary-gradient conflict. More training alone must be separated from architecture benefit by the equally trained stock control. Functional code or falling synthetic loss alone is not a performance claim. Finish actual training, evaluation and reviewable HTML; auto-open and completion Discord notification are already user-authorized.

## Status

All nine full-network training cells and all nine real evaluations are complete. TRAIN_PROTOCOL.json remains frozen SHA `d24da92515f381c02a9300391787c292e06a53b97f44e3e5646d25a1fd0902af`: 3 arms × seeds1/2/3 × 2 full epochs, FP32 batch16, 55,980 training images/epoch, 6,998 actual optimizer updates/cell, AdamW1e−4 and final-epoch EMA only. All4,020 original validation images are diagnostics each epoch, not a new heldout set. Each model actually inferred all319 positive and2689 negative DEV images. No real checkpoint, seed or loss selection was performed.

The final feature operator uses effective votes A′=KρA with Kρ=[.25,.5,.25]. Both directions use this same matrix with their own mass normalization. This repairs exact-axis, between-cell semantic peaks that otherwise vanish on backprojection; it preserves the antipodal seam and constant inputs. The separately normalized reverse operator is not the mathematical adjoint of the normalized forward operator. Independent FP64 derivative/inner-product, actual YOLO point-location+RLE-only gradient, EMA and checkpoint round-trip checks pass.

AMP smoke exposed overflow at scales65536 and1024, and an aux-disabled FP16 zero-loss reduction bug. Failed runs are archived; the reduction is repaired and regression-tested. Common AMP16 smoke passed, but final main training uses FP32 for numerical stability. Full model B16 FP32 resource probe: ~193ms/step, 4,682MiB peak allocated on RTX3080 (data decoding excluded). Added parameters are15,052. The entire actual60,000 image+label source inventory was rehashed successfully; no old detection eligibility filter is applied.

Driver includes train→matched actual augmentation/state audit→all3008 actual forward per model→controlled runtime→statistics→HTML→open/authorized Discord, with source/protocol bindings, owned subprocess cleanup and preserved first-epoch restart evidence. Main launch and observed results are recorded in the handoff/history and results STATUS.

Before any new real-model predictions, secondary view cohorts were frozen using only existing GT. Reconstructed unsigned elevation bins <5/5–15/15–30/≥30 degrees contain 55/97/82/85 frames. A projected side/front face-area ratio gives 144/51/96 frames at ≤.1/(.1,.3]/>.3, plus28 unknown frames with insufficient supervised corners. This ratio is not yaw; reconstructed elevation is not a measured camera/world angle. The user-provided failure case remains frontness unknown because corner3 is unsupported. The standalone `view_diagnostics.py` completed after all9 evaluations and the main summary, without changing model selection or the primary verdict. Its protocol and source hashes are in `VIEW_DIAGNOSTIC_PROTOCOL.json` and `VIEW_STRATA.json`; results are in [VIEW_DIAGNOSIS.md](../../data/pallet/results/pallet_dht_joint_v1/VIEW_DIAGNOSIS.md).

On the fixed view cohorts, <5° point P90 decreases67.703→58.228px while translation worsens18.397→19.192cm. At≥30° P90 worsens55.530→66.794px with identical85-frame matching coverage. Near-frontal proxy≤.1 gives median7.563→7.431px, P9025.607→26.365px and translation7.670→7.911cm; on134 common matched frames its P90 contrast is only−.195px. Proxy>.3 improves descriptive point P90122.898→114.920px and pose translation9.074→8.612cm. These are secondary descriptions without a new subgroup superiority decision; they do not show that frontality or missing synthetic views alone caused failure.

An optional local gradient diagnostic was prepared while seed3 training was running, before new real predictions. It freezes16 synthetic validation images by ID SHA (G38 6/P0 5/TEX 5), actual deterministic train transforms, batch16 train-mode BN, and point-location+RLE versus weighted line gradients. The standalone `gradient_diagnostic.py` uses final raw training weights and final-epoch E2E weights, rather than the EMA weights used for real evaluation. Preparation checks pass, but execution was deferred by the predeclared14GiB host-memory guard (observed12.59GiB available), before checkpoint loading or neural forward/backward. No gradient conflict was measured or established. Frozen inputs and `DEFERRED_RESOURCE.json` are under `provenance/gradient_diagnostic_v1/`.

## Actual result

[Interactive report](../../data/pallet/results/pallet_dht_joint_v1/index.html), [source and reproduction](../../scripts/research/pallet_dht_joint_v1/README.md), [machine-readable verdict](../../data/pallet/results/pallet_dht_joint_v1/VERDICT.json).

Same-budget three-seed means (each seed's original evaluator statistic is calculated before averaging):

| Metric | Point only | Hough features, no line auxiliary | Hough + point joint |
|---|---:|---:|---:|
| Keypoint pooled median, px ↓ | 6.865 | 6.773 | 6.689 |
| Keypoint pooled P90, px ↓ | 40.395 | 36.320 | 38.690 |
| MAIN rotation median, deg ↓ | 2.376 | 2.434 | 2.319 |
| MAIN translation median, cm ↓ | 7.731 | 7.975 | 7.800 |
| MAIN IoU3D median ↑ | 0.59534 | 0.59676 | 0.59488 |
| MAIN ADDsym AUC ↑ | 0.42359 | 0.42760 | 0.42423 |
| Keypoint matched frames /319, seeds1/2/3 | 307/308/310 | 305/306/309 | 309/307/307 |
| Negative frames with detection at score≥.5 /2689, seeds1/2/3 | 105/236/219 | 94/180/249 | 199/158/151 |

The prespecified joint-vs-point decision is **overall_accuracy_improved=false**. All six paired session95% intervals include zero. The point-median contrast on302 common matched frames is −0.09993px, CI[−0.54625,+0.10951]; P90 contrast −3.24183px, CI[−22.61362,+3.15918]. These paired contrasts differ from subtracting full-population table means because the comparison uses the same available frames across all six models. Point matching coverage also decreases for seeds2/3; MAIN pose coverage is preserved. This supports a small descriptive average2D gain, not reliable overall superiority. Reused13-session DEV and two continuation epochs limit the conclusion; it is not evidence that every possible joint DHT architecture fails.

The requested raw case `eval_pallet07:1778652166837872128` retains the large corner-identity error in all nine models. Its official supervised-eight-point median, averaged over seeds, is258.042px for point-only and262.047px for joint. The previously fixed GT-only permutation lowers these to37.348/31.857px but is an oracle diagnostic, not automatic correction or reported accuracy. See [case diagnosis](../../data/pallet/results/pallet_dht_joint_v1/CASE_DIAGNOSIS.md). The visible leftmost height edge is role7 GT4–GT7; role3 GT0–GT3 is a different edge with unsupported GT3.

The result does not isolate a single cause. A global structural-line prior can carry geometry while still leaving corner identity and occlusion ambiguous; the retained identity error is directly observed. Synthetic-to-real transfer, auxiliary-loss competition and insufficient continuation remain hypotheses, not measured causal explanations. A next architecture experiment should test symmetry-aware joint point/edge correspondence and learned reliability of local line evidence inside the network, with synthetic-only selection and the same-budget point control. Those changes were not trained in this run.

## Runtime numerical limitation

All702 prescribed timing observations were collected, but the original `atol=1e-4, rtol=0` output comparison failed7/702 times. `RUNTIME.PASS` and `parity_PASS` remain **false**. Largest keypoint difference is0.006103515625px. Actual operator isolation reproduces variation in CUDA sparse HT/backprojection accumulation; exact fixed-input convolutions do not begin the variation. TF32 can amplify it. This is separate from the multi-pixel accuracy errors and does not establish their cause.

Three-seed means of measured per-seed median latency are8.891ms point-only,11.079ms Hough-features and10.997ms joint. These are reference observed timings with failed strict prediction parity, not a parity-validated speed comparison. Original failing logs and sources, the unchanged byte-exact comparison function, diagnostic evidence and the four-file reporting amendment are preserved under `provenance/runtime_parity_001/` and `provenance/runtime_parity_diagnostic/`. All29 frozen accuracy/protocol artifacts remained byte-identical after the repair. Completion of research/reporting does not mean every validation passed.

Actual offline HTML and six root-reviewed screenshots show the raw frame, original IDs, three joint seeds and runtime warning without missing panels. The physical role7 maps respond broadly near the left boundary and post/background, rather than isolating only the short edge. This is descriptive map evidence, not a causal attention explanation. `ROOT_VISUAL_REVIEW.json` binds the screenshots. Browser visibility and authorized Discord delivery were confirmed (HTTP204).
