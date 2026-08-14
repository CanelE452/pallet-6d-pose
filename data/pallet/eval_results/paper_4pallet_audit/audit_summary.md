# paper_4pallet_mask_v1 — Pre-training Audit

Dataset: `data/pallet/training_data/paper_4pallet_mask_v1/` (10,000 frames, flat `{i:06d}.png/.json` + `mask/{i:06d}.png`)
Convention: `camera_dynamic_0123_v4` (front 0-3 near cam, rear 4-7 far, top {0,1,4,5}, bot {2,3,6,7}, centroid separate)
Audit date: 2026-07-06

## VERDICT: PASS (data integrity + convention) — with a DISTRIBUTION CAVEAT

All 12 integrity/convention items PASS. **BUT** the geometry distribution is a *clean full-view*
dataset (100% V=8, 0% truncation, elevation floor 15deg, depth 1.6-4.5m). If this dataset's
intended purpose was **truncation / low-angle / near boosting** (STAGE16 Phase A lever), it does
**NOT** provide it. As a clean paper-track base dataset it is training-ready.

## Schema reality (differs from generic validator assumptions)
- `camera_data` has **NO** camera rotation quaternion — only `location_worldframe`. The generic
  `paper_dataset_validator.py` `project()` (uses `quaternion_xyzw_worldframe`) would FAIL here; this
  audit uses the actual schema instead.
- `object.pose_transform` = 4x4 **object->camera, OpenCV** convention. `K @ t` reprojects to
  `projected_cuboid_centroid` to <1e-5 px. [confirmed]
- `object.cuboid` = 8x3 **world frame**. `object.projected_cuboid` = 8x2 px (no 9th row).
- `mask_rle` = COCO uncompressed toggling, column-major, area = sum(counts[1::2]).
- `mask/*.png` = **multi-instance/semantic viz** (values {0,1,2,254,255}, ~79% frame area) —
  **NOT the pallet mask. Training must use `mask_rle`.** [confirmed]

## 12-item results
```
#   item                         result   detail
────────────────────────────────────────────────────────────────────────────
1   presence (RGB/JSON/mask/ovl) PASS     10000 json / 10000 png / 10000 mask.png / 10000 overlay; 0 missing; 1 obj/frame
2   4-asset distribution         PASS     Pallet_0 49.8% / Pallet_2 16.9% / Pallet_3 16.7% / Pallet_1 16.6% (matches ~50/16.7)
3   scene.usd emissive/overexp   PASS     Pallet_0 mean-brightness 55.3 (LOWEST), 0% clipped px all assets. No glow.
4   mask_rle decode              PASS     10000/10000 area==json; unique {0,1}; 0 empty; 0 full-frame; bbox match; slat holes present
5   projected_cuboid overlay     PASS     60+ tiles eye-checked: cuboid wraps pallet edges tightly
6   keypoint convention + reproj PASS     PnP reproj p99-of-p99 ~0 (max ~1e-11); top/bot 100%, sep 100%, front-near-cam 100%
7   clamp check                  PASS     0 clamped, 0 sentinels; but ALSO 0 off-screen corners (see caveat)
8   per-corner status keys       ABSENT   inside_image/heatmap_valid/V_geom/missing_corner_bitmask NOT present (see below)
9   V distribution               PASS*    V=8 for 100% (10000/10000) — zero partial views
10  truncation distribution      NONE     0 off-screen corners, no L/R/T/B cuts — no truncation at all
11  geometry distribution        PASS     elev 15-60deg (p50 35), depth 1.6-4.5m (p50 3.3), proj-size p50 319px
12  montages                     DONE     overlay_montage/ (asset x4, low_angle, near, large_proj, high_angle, random)
```
`*` V=8 is internally consistent, but see distribution caveat.

## item8 — per-corner status keys (honest report)
Expected keys (`inside_image`, `in_front_of_camera`, `heatmap_valid`, `V_geom`,
`missing_corner_bitmask`) are **ABSENT**. Present per-object fields instead:
`visibility`, `raycast_visibility`, `front_facing_cos`, `facing_margin`, `front_rect_ok`,
`front_is_camera_near`, `visible_mask`(=path to mask png), `edge_connector_crossings`,
`edge_total_crossings`, `perm_v4`, `ratio_randomized/ratio_factors`, `dimensions_m`, `physical_audit`.
V is **computed** here from projected_cuboid in-frame count (= 8 for every frame). There is no
per-corner boolean array; not needed since no corner is ever off-screen.

## Distributions
```
Asset (name -> source_asset, 1:1 clean)      count    pct
──────────────────────────────────────────────────────────
Pallet_0  scene.usd                          4983    49.8%
Pallet_2  scene_2.usd                         1693    16.9%
Pallet_3  scene_3.usd                         1666    16.7%
Pallet_1  scene_1.usd                         1658    16.6%

V (num corners in-frame)     count
──────────────────────────────────
V=8                          10000   (100%)   <- NO truncation
V<8                              0

Geometry                     min    p10    p50    p90    max
──────────────────────────────────────────────────────────────
elevation_deg (GT pose)      15.0   18.8   34.6   51.6   60.4    <- floor 15deg, no edge-on(<15)
object depth cam-z (m)       1.59   2.28   3.27   4.17   4.49    <- mostly far/mid, few near
projected_size_px            177    243    319    462    720
frsep depth-edge px          46.6   90.7  136.5  200.1  337.6    <- healthy depth sep, not collapsed
raycast_visibility           0.30   0.30   0.40   0.60   0.80    <- moderate cargo occlusion
ratio_randomized             true 4040 (40%) / false 5960 (60%)  <- squash aug present (paper ratio-robust)
```

## Distribution-vs-purpose judgment (STAGE16 Phase A lever)
Current bottleneck = REAR/depth collapse on flat, truncated, near, low-angle real frames.
This dataset delivers **none** of those regimes:
- Truncation: **0%** (all 8 corners forced in-frame). No L/R side-cut frames.
- Low-angle: elevation **floor 15deg** — no true edge-on (<15deg) where depth squash occurs.
- Near: depth p10 2.28m — few very-near/large-fill frames (though proj-size reaches 720px).

=> As a **clean full-view 4-pallet base** (ratio-randomized, mid-angle, moderate occlusion) it is
clean and training-ready. It is **NOT** a truncation/low-angle/near add-on and will not, by itself,
address the rear/flat-view failure mode.

## Artifacts
- `audit_report.json` — full machine-readable report
- `emissive_check.json` — per-asset brightness/clipping
- `bad_frames.txt` — 0 entries
- `overlay_montage/` — asset_Pallet_0..3, low_angle_lt20, near_lt2.3m, large_projection, high_angle_gt50, random_spread
- `run_audit.py`, `make_montages.py` — reproducible audit scripts
```
