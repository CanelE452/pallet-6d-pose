# Symmetry-aware reliability-gated DHT local fusion v1

## Scientific verdict

`NO_SYNTHETIC_LOCAL_FUSION_SIGNAL`

The frozen Point baseline remains the selected method. All three Hough seeds failed the predeclared H>P gate. No threshold, sigma, max-shift, grid, loss, seed, or epoch was added after observing the result, and real DEV319 was not opened.

## Confirmed execution

- Stock-R0 export/integrity and G1–G15: PASS.
- Direct and Hough: seeds 1/2/3, exactly 2,000 AdamW steps each, batch 8, last-step checkpoint only.
- CUDA: RTX 3080. A 580.173 kernel / 580.178 userspace mismatch was repaired without reboot or display restart by using a process-local copy of the still-loaded 580.173 libraries.
- Unit/contract tests: 11 passed.

## Synthetic metrics

| arm | seed | primary mean/diag | median px | P90 px | good-point damage | new catastrophic |
|---|---:|---:|---:|---:|---:|---:|
| Point | — | 0.009474144 | 2.0264 | 7.3339 | 0/3344 (0%) | 0 |
| Direct | 1 | 0.010263593 | 2.6731 | 8.9385 | 16/3344 (0.478%) | 0 |
| Direct | 2 | 0.012869815 | 5.1853 | 12.6200 | 329/3344 (9.839%) | 0 |
| Direct | 3 | 0.010167087 | 2.6434 | 8.6243 | 14/3344 (0.419%) | 0 |
| Hough | 1 | 0.009839839 | 2.2395 | 8.7296 | 40/3344 (1.196%) | 0 |
| Hough | 2 | 0.009956455 | 2.2965 | 8.8900 | 50/3344 (1.495%) | 0 |
| Hough | 3 | 0.009850239 | 2.2044 | 8.6316 | 43/3344 (1.286%) | 0 |

Every Hough seed had worse primary, median and P90 than Point, and damage exceeded 0.5%. All had zero new catastrophic frames. Hough was numerically better than Direct in 3/3 paired seeds (seed-average paired frame delta -0.001217987), but the precondition `LOCAL_LINE_FUSION_GO` failed, so `DHT_INCREMENTAL_GO` is false.

## Symmetry and integrity

- C1: 779; C2: 1781; C4: 0 (`C4_NOT_EVALUATED`).
- By split: train C1/C2/C4 = 578/1214/0; calibration = 87/169/0; synth_val = 114/398/0.
- `scene.usd` and `scene_1.usd`: confirmed C2. Unresolved GLB assets: C1. Center 8 is fixed by every explicit permutation.
- All 2560 source image byte hashes and all 2560 stock point rows were independently verified.
- GT-free predictions were serialized before evaluation. Missing/unmatched outputs were retained with penalty.

## Diagnostics

- Direct: correction mean px 1.836, 5.001, 1.680; reliability mean 0.225, 0.360, 0.146; P_nonnull mean 0.536, 0.749, 0.398; concentration mean 0.436, 0.445, 0.416.
- Direct parameters: 20,876; GPU scoring ms/frame: 7.078, 7.211, 7.228.
- Hough: correction mean px 2.329, 2.336, 2.297; reliability mean 0.729, 0.710, 0.719; P_nonnull mean 0.995, 0.996, 0.993; concentration mean 0.731, 0.711, 0.722.
- Hough parameters: 34,968; GPU scoring ms/frame: 7.197, 7.181, 7.346.
- Actual saved line-mode forces had maximum tangent component below 7.2e-15 (PASS).
- Reliability/null/concentration are availability proxies, not calibrated physical visibility probabilities.

## Code and artifacts

- Added a self-contained exporter/auditor, Direct and dense line-aligned DHT heads, reliability decoder, strict local WLS, whole-object symmetry loss, GT-free runner, frozen evaluator/gates, and regression tests.
- Point adds 0 trainable parameters; Direct adds 20,876 and Hough adds 34,968.
- Raw local root: `/home/minjae/Documents/github/pallet-pose/data/pallet/results/pallet_symmetry_dht_local_v1`. Reviewable raw artifact index SHA256: `c9d77433238aa08b7b8b979ba7e4c8806d08a819668f0bf2329226601880ad92`.

## Scope

No global pose solver, point-head/backbone training, real-label training, full online training, real DEV319, or FINAL data was used. The 1,792/256/512 split is a new line-head split, not an unseen-backbone claim. See `SYNTH_PER_FRAME.csv` for the complete 512-frame paired table.
