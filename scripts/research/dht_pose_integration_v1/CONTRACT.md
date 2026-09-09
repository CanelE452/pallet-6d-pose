# Frozen baseline + DHT integration verification

This experiment evaluates existing DOPE/YOLO corner predictions with the existing synthetic-trained DHT8 line model. It is inference-time regularized point-to-line fusion; it does not train a YOLO-native DHT head or claim end-to-end joint learning.

Results directory: `data/pallet/results/dht_pose_integration_v1/`.
`CONFIG.json` and `manifest.json` are immutable experiment inputs. Manifest records are flat, ordered, have `index`, original cache `source_index`, and the original record fields including `id`, `population`, `group`, `image`, `width`, `height`, `gt_points`, `gt_valid`. Population values are lists of indices. Only synth_val256, synth_test256, cross_v4128 and canonical real_dev52 are included (692frames).

## Baseline adapter interface

Each `dope_baseline.py` / `yolo_baseline.py` exposes `Baseline(config, device='cuda')` and `predict(bgr)` returning a dictionary with:

- `kps`:9×2 original-image pixel coordinates; keep raw finite coordinates even when invalid. Missing coordinates are null in JSON.
- `kp_conf`:9 confidence values, null when unavailable.
- `kp_valid`:9 booleans based only on the baseline's fixed decoder/confidence rule.
- `detected`:boolean; `n_instances`:integer or null if the decoder does not group instances.
- `box_xyxy`:4 original-image coordinates or null; `box_conf`:float or null.

`predict` performs preprocessing, inference, original-coordinate conversion and CPU return. It takes no GT or annotation input. Keep any single-pallet channelwise decoder limitation explicit; never introduce GT-based instance or role matching. For YOLO choose highest predicted box confidence.

CLI `--run-dir PATH` writes `BASELINE_DOPE.json` / `BASELINE_YOLO.json` with schema, config/manifest/weight/source SHA256, runtime and recipe, timing definition and ordered `records`. Each output record adds `id`, `population`, `group`, `width`, `height`, `inference_ms`. Perform fixed warmup5 on the first synthetic image; measure synchronized GPU wall time for `predict` (disk read and model load excluded), batch1. Prefer `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python` for a shared combined-pipeline benchmark environment; dependencies cv2,torch,scipy,PIL,matplotlib,ultralytics,simplejson,pyrr,skimage are installed.

DOPE uses exact synthetic backbone checkpoint and the established stage0 reflect100→resize-to-original→short-side400-aspect recipe; the canonical belief decoder returns a single set of channelwise points, not affinity-based instance grouping. YOLO uses existing synthetic R0, reflect100, imgsz640, box conf0.25, keypoint conf0.5, default predictor inversion then subtract pad100. Hyperparameters are fixed before real evaluation.

## DHT and fusion

Use saved original DHT seed1/2/3 final6000step checkpoints and their original square400/reflect100 uint8 feature recipe. Predict all8 side-role infinite supporting lines; do not use GT support or camera-facing labels to mask fusion inputs. Lines must be mapped to original pixels before fusion. DHT target roles0..7 correspond to edges(1,2),(3,0),(5,6),(7,4),(0,4),(1,5),(2,6),(3,7).

`fusion.py` provides `fuse_corners(points, valid, lines, lam, width, height)` in original pixels. Geometry agent defines return interface and sends it to root. Use positive baseline anchor, isotropic diagonal normalization, fixed incidence, no GT input. Missing corners stay missing; baseline center index8 stays unchanged.

Predeclared lambda grid:0,.25,.5,1,2,4. Select one lambda per baseline on synth_val256, averaging all3DHT seeds, minimizing mean per-frame capped diagonal-normalized corner error (missing=1; finite error capped at1). Ties choose smaller lambda. Freeze selection before test/cross/real evaluation. Primary is selected lambda; fixedlambda1 is a separate diagnostic, never selected by real performance.

Evaluation includes valid annotated top8 corners, conditional pixel errors, all-GT PCK@5/10/20px including missing failures, all8@10px frame success, prediction coverage,8derived side-line errors with GT support used only in metrics, and canonical real groups separately. No final-test data. Report actual combined runtime separately from component timing; do not label a sum of timings as a measured combined latency. Current52 real images are development data and synthetic backbone pretraining overlap is unverified.
