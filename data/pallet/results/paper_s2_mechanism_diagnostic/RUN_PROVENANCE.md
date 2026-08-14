# RUN PROVENANCE — paper_s2_mechanism_diagnostic

- created: 2026-07-29T08:27:42.288150+00:00
- git HEAD: 97e1219a147e9e1681a4b0019cbd9a72fb2f95c5
- git status: dirty
- checkpoint: /home/minjae/Documents/github/pallet-pose/weights/paper_s2_stageB/net_epoch_0057.pth (SHA-256 c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896)
- python: 3.10.20
- torch: 2.1.1+cu118 (cuda 11.8)
- opencv: 4.9.0
- pandas: 2.3.3 / numpy: 1.26.4
- gpu: NVIDIA GeForce RTX 3080, 580.173.02
- cache key: 8dfcd6c238dab4fa70108b5dd4b3662760ac88471cc3faa272fea9fe11adf98c
- cache model forwards: 87
- manifest membership SHA: 8d086cd7f8a20cf6fd76af4b26fd5b2ed01ed2a1173e441ae777909202260914
- final-test open count: 0
- prohibited attempts: []

## Baseline reproduction gate
- GT-2D PnP: 87/87 (expected 87)
- predicted PnP: 70/87 (expected 70)
- yaw median: 6.0252° (expected 6.025 ±0.1)
- fixed-GT reproj median: 23.1616 px (expected 23.162 ±0.25)
- passed: True

## Row counts
- frames: 87
- keypoints: 783
- poses: 522
- interventions: 3367
- counterfactuals: 288
- figures: 9
- examples_failure_class_examples: 13
- examples_late_drift_examples: 3
- examples_error_propagation_examples: 12

## Reused prior artifacts (not recomputed)
- `data/pallet/results/paper_s2_scratch_diffpnp/diagnostic_audit/full_ep57_frozen_20260728/` — frozen frames/keypoints/yaw-ladder (baseline regression reference)
- `data/pallet/results/paper_s2_target_semantics_audit/` — target semantics / decoder parity / truncation population / DiffPnP funnel
- shared code: `paper_s2_frozen_diagnostic`, `paper_s2_decoder_parity_audit`, `filter_pr_camfacing.extract_keypoints_from_belief`, `annotate_pnp`

