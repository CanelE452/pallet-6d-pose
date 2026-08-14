# DATA / GT / camera / PnP diagnostic audit

Audit date: 2026-07-28  
Repository: `/home/minjae/Documents/github/pallet-pose`  
Frozen checkpoint: `weights/paper_s2_stageB/net_epoch_0057.pth`  
Checkpoint SHA-256: `c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896`

## 0. Scope and sealing

This is a read-only diagnostic audit. It does not create, relabel, exclude, or
move any training/evaluation sample, and it does not change model code or
weights.

The populations are deliberately separated throughout:

| Name | Membership | Status |
|---|---:|---|
| strict filter-val | outside 44 + night 43 = **N87** | primary diagnostic population; split-lock compliant |
| manual | manual 36 = **N36** | exploratory only; all frames originate from `capturepallet11`, a PL-pool session |
| legacy combined | N87 + N36 = **N123** | historical filterval manifest only; must not be presented as one independent paper split |
| synthetic val | **N1500** (`V8=1363`) | historical Stage-B checkpoint-selection set |

Final-test sealing was maintained. No final-test image, GT JSON, annotation, or
cached prediction was opened or evaluated. The only final-test facts used are
the session names and aggregate counts already declared by
`data/pallet/eval_results/split_lock/split_assignment.json`:

- outside final-test: `capturepallet09`, `capturepallet07`, 3512 frames / 63 GT;
- night final-test: `capturenight09`, `capturenight08`, 1706 frames / 42 GT.

Those names are used only to detect historical script membership. There is no
final-test performance result in this audit.

## 1. Executive verdict

| Question | Verdict | Main evidence |
|---|---|---|
| Is ep57 the frozen historical best checkpoint? | **Yes.** | Full synthetic-val selector chose epoch 57; checkpoint hash is recorded above. |
| Are strict real GT intrinsics, units, and dimensions grossly wrong? | **No evidence of that.** | JSON K equals session `cam_K.txt`; dimensions are metre-scale and consistent; rotations are proper; pose-derived projections close exactly. |
| Is centroid keypoint 8 the arithmetic mean of eight 2D corners? | **No.** It is the projection of the 3D cuboid origin. | It agrees with pose-origin projection to sub-pixel precision, while differing from the arithmetic 2D mean by several pixels. |
| Are all stored real 2D points independent manual GT? | **No.** | JSON generation stores clicks where available, otherwise pose projections; historical JSON lacks an authoritative per-point provenance field. |
| Is there a sentinel bug? | **Yes, in multiple evaluation/PnP paths.** | Exact `(-1,-1)` occurs in 13 strict corner entries over 9 frames; unsanitized oracle error and pose drift explode. |
| Should every coordinate outside the image be discarded? | **No.** | There are 29 non-sentinel strict corners outside the image; the annotator explicitly defines only exact `(-1,-1)` as missing. |
| Does strict real GT follow the current camera-facing LR/TB convention? | **Yes.** | N87 has zero evaluable LR and TB pair violations. |
| Does synthetic val1500 follow that same per-channel convention? | **No.** | Only 690/1500 frames pass all LR pair tests; 810 violate at least one LR pair. |
| Does the synthetic convention issue invalidate ep57 entirely? | **No.** | Order-free corner/Hungarian and order-free 48-sym/PnP quantities remain usable. Current-semantic `front/rear` and individual-channel conclusions do not. |
| Is manual N36 an independent primary validation set? | **No.** | It comes from `capturepallet11`, classified as PL pool, and has materially noisier mixed-click GT. |
| Are later pseudo-label descendants independent of current filter-val? | **No.** | Exact RGB duplicates and same-session leakage are present in saved descendant training artifacts. |
| Can historical RALPH “all GT” tables select a paper checkpoint? | **No.** | Their source lists include `capturepallet07` and `capturepallet09`, which split-lock declares final-test. |

## 2. Provenance and reproducibility

### 2.1 Inputs and hashes

| Artifact | SHA-256 |
|---|---|
| `weights/paper_s2_stageB/net_epoch_0057.pth` | `c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896` |
| `data/pallet/eval_results/paper_s2_scratch_diffpnp/orthogonal_filters_exp.json` | `42fce4bfcdd22084f8385c0c95660ac907b4db90311777e3d23da835af237c2c` |
| `data/pallet/eval_results/split_lock/split_assignment.json` | `2ed92037ef1816d2adc9d514934de98bcdf790baf2ab5118c4d59028a761c67f` |
| `data/pallet/results/paper_s2_scratch_diffpnp/q1_split/val_list.json` | `5a88384f045faf22dda48465b440e69dba78bc94420f10a3db5217390befb56d` |
| `data/pallet/results/paper_s2_scratch_diffpnp/pnp_valid_3d_index/val.json` | `4942590d46dcf409a241dbfa66da9fd68fef14d87cdfd4a92a0c67c42df8544c` |

The canonical real manifest is rebuilt and count-checked in
`scripts/stage0/paper_s2_rgb1_eval.py:258-323`: outside 44, night 43,
manual 36, with duplicate `(domain,fid)` rejection. Split/session filtering is
shown independently in `scripts/stage0/stage25_paperbase_eval.py:93-115`.

### 2.2 Runtime

| Component | Value |
|---|---|
| Python | 3.10.20 |
| interpreter | `/home/minjae/anaconda3/envs/pallet-pose/bin/python` |
| NumPy | 1.26.4 |
| SciPy | 1.12.0 |
| OpenCV | 4.9.0 |
| PyTorch | 2.1.1+cu118 |
| GPU used for frozen inference | NVIDIA RTX 3080, 10 GB |
| NVIDIA driver | 580.173.02 |

### 2.3 Commands and calculation definitions

Repository-provided validation was executed as:

```bash
/home/minjae/anaconda3/envs/pallet-pose/bin/python \
  scripts/data_prep/validate/pnp_valid_3d_audit.py \
  --datasets val \
  --root data/pallet/training_data \
  --out /tmp/pallet_laneB_audit.A19iSS
```

Hashes are reproducible with:

```bash
sha256sum \
  weights/paper_s2_stageB/net_epoch_0057.pth \
  data/pallet/eval_results/paper_s2_scratch_diffpnp/orthogonal_filters_exp.json \
  data/pallet/eval_results/split_lock/split_assignment.json \
  data/pallet/results/paper_s2_scratch_diffpnp/q1_split/val_list.json \
  data/pallet/results/paper_s2_scratch_diffpnp/pnp_valid_3d_index/val.json
```

The remaining diagnostics were read-only Python calculations over the manifest
members. Their definitions are fixed as follows so the reported numbers are
not dependent on an undocumented visual judgment:

- exact duplicate: BLAKE2 digest equality of complete PNG or JSON file bytes;
- sentinel: both coordinates equal exactly `-1.0`;
- other off-image: not sentinel, but `x<0`, `x>=640`, `y<0`, or `y>=480`;
- pose-like stored point: Euclidean residual to stored-pose projection
  `<=0.25 px`; direct/manual-like: residual `>0.25 px`;
- PnP oracle: OpenCV 4.9 `SOLVEPNP_ITERATIVE`, initialized at stored
  `pose_transform`, using per-frame K and `(width, depth, height)`;
- frame reprojection: mean Euclidean error over the selected valid
  correspondences;
- rotation delta:
  `acos(clip((trace(R_est R_gt^T)-1)/2,-1,1))` in degrees;
- translation delta: `||t_est-t_gt||_2` in metres;
- LR convention: every evaluable pair `(0,1),(3,2),(4,5),(7,6)` has left
  keypoint x smaller than right keypoint x;
- TB convention: every evaluable pair `(0,3),(1,2),(4,7),(5,6)` has top
  keypoint y smaller than bottom keypoint y;
- front/rear screen check: polygon area of indices 0–3 is at least that of
  indices 4–7; this is a supporting screen test, not a full 3D proof;
- W/D swap screen: rerun the same local initialized PnP with width/depth
  exchanged; “ambiguous” means swapped error is within both `0.5 px` absolute
  or `5%` relative of the correct-dimension error.

The frozen ep57 pass used the repository's exact `load_model`, 640×480 to
400×400 squash, 50-grid decode, and threshold path in
`scripts/stage0/paper_s2_rgb1_eval.py`; weights and datasets were not modified.

## 3. Split, leakage, and duplicate audit

### 3.1 Strict split-lock membership

Strict N87 source counts:

| Domain/session | Count |
|---|---:|
| outside / `capturepallet02` | 5 |
| outside / `capturepallet03` | 8 |
| outside / `capturepallet04` | 6 |
| outside / `capturepallet05` | 5 |
| outside / `capturepallet08` | 20 |
| night / `capturenight05` | 12 |
| night / `capturenight06` | 15 |
| night / `capturenight07` | 16 |
| **strict total** | **87** |

Manual N36 all map to `capturepallet11`. Split-lock classifies that session as
outside `pl_pool`; therefore N36 is exploratory and cannot be pooled into a
primary independent validation claim.

### 3.2 Exact duplicate audit

Synthetic selection data:

- q1 list: 500 unique frame IDs/paths and a subset of val1500;
- val index: 1500 entries, no internal exact JSON or PNG duplicate;
- six Stage-B training roots contain 29,308 JSONs in total:
  `mixed_v8_train` 9000, `v4_split_base` 4000, `aug_squash_v2` 2212,
  `aug_trunc_v2` 2971, `aug_scale_v2` 1125, `paper_4pallet_mask_v1` 10000;
- val1500 has **zero exact JSON hits** and **zero exact PNG hits** against those
  29,308 Stage-B training examples.

Real legacy N123:

- 123 unique frame IDs;
- zero internal exact JSON duplicate groups;
- zero internal exact PNG duplicate groups.

This means the frozen ep57 Stage-B training/selection comparison passes the
exact-byte duplicate screen. It does not repair the synthetic labeling
convention issue discussed below.

### 3.3 Later descendant contamination

Saved descendant training artifacts contain current filter-val RGB frames:

| Training artifact | Exact current-filter RGB hits | Breakdown |
|---|---:|---|
| `paper_s2_fullpool_r1` | 15 | strict: outside 10 + night 5 |
| `paper_s2_fullpool_r2` | 7 | strict: outside 4 + night 3 |
| `paper_s2_full7_pl19_r1` | 18 | strict 15 + manual 3 |
| union across these descendants | 23 unique | strict 20 + manual 3 |

Their label JSON is pseudo-label output and therefore need not hash-identically
to GT; exact source-RGB equality is already direct membership contamination.
`scripts/stage0/build_s2_full7_pl19.py:24-25,639-642` explicitly records that
its 19 evaluation-derived frames cannot remain independent evaluation frames.

Other source pools avoid exact GT frame IDs but retain same-session dependence:

| Pool | Frames from strict validation sessions |
|---|---|
| `real_unlabeled_ralph_outside` (500) | cp02 59, cp03 59, cp04 38, cp05 48, cp08 296 |
| `real_unlabeled_ralph_night` (500) | n05 176, n06 160, n07 164 |
| `real_unlabeled_ralph1500` | cp02 97, cp03 100, cp04 62, cp05 80, cp08 416; n05 176, n06 160, n07 72 |

Exact filter-frame-ID overlap in those source pools is zero because
`scripts/stage0/ralph_build_pool.py:28-44` excludes known eval IDs. That guard
does not hold out whole sessions. The saved artifact state also takes
precedence over the intended exact-FID guard in
`scripts/stage0/s2_fullpool_build_pl.py:37-55,88-105`.

Consequences:

- frozen ep57 is synthetic-only and is not trained on these real descendants;
- metrics on a descendant model evaluated against any duplicated frame are
  invalid as independent evidence;
- metrics on same-session but different frames are session-dependent and need
  a group/session-aware interpretation;
- these later models must not be used retrospectively to select ep57.

### 3.4 Historical final-test contamination

`scripts/stage0/ralph_eval_all.py:32-38` includes outside manual-GT directories
for `capturepallet02` through `capturepallet09`. Consequently it includes both
split-lock final-test outside sessions, cp07 and cp09. The same broad source
membership is present in `ralph_matrix.py`, `ralph_phase1_matrix.py`,
`ralph_phase1_matrix_looflip.py`, and `s2_gt_accuracy.py`.

Any historical result produced on that broad outside “all GT” membership
(including the reported outside N117 family of results) is invalid for paper
checkpoint, threshold, filter, or loss selection. This conclusion uses code
membership plus split-lock metadata only; final-test samples were not opened.

## 4. Coordinate convention audit

The current intended convention is defined in
`challenge/scripts/convert_to_camera_facing_v4.py:3-34`: indices 0–3 are the
larger/near front face, 4–7 the rear face; indices form image-left/right and
top/bottom pairs. The annotator's 3D diagram is consistent with that topology
in `challenge/scripts/annotate_pnp.py:57-94`.

### 4.1 Real GT

| Population | LR all-pair pass | TB all-pair pass | front area ≥ rear area |
|---|---:|---:|---:|
| strict N87 | 87/87 frames; 335/335 evaluable pairs | 87/87; 339/339 pairs | 77/78 full-8 frames |
| manual N36 | 36/36; 144/144 pairs | 36/36; 144/144 pairs | 36/36 |
| legacy N123 | 123/123 | 123/123 | 113/114 full-8 frames |

The missing LR/TB pair evaluations in N87 come only from exact sentinels. The
one screen-area exception is not by itself evidence of a channel permutation;
the strict pair invariants still pass.

### 4.2 Stage-B training roots versus synthetic val1500

| Dataset | LR all-pair pass | TB all-pair pass | front area ≥ rear |
|---|---:|---:|---:|
| `mixed_v8_train` | 8997/9000 | 8994/9000 | 99.90% all checks |
| `v4_split_base` | 4000/4000 | 4000/4000 | 85.50% |
| `aug_squash_v2` | 99.95% | 99.95% | 94.08% |
| `aug_trunc_v2` | 99.97% | 99.97% | 95.59% |
| `aug_scale_v2` | 100% | 100% | 94.22% |
| `paper_4pallet_mask_v1` | 100% | 100% | 88.87% |
| **synthetic val1500** | **690/1500 (46.00%)** | **1500/1500** | **755/1500 (50.33%)** |

Therefore **810/1500 val frames violate at least one current LR pair**. Only
452/1500 pass LR, TB, and the front-area screen together.

The likely provenance is not a runtime resize error: val files predate the v4
conversion, while the converter's default roots explicitly include
`mixed_v8_train`, v1 and v2 but not val
(`challenge/scripts/convert_to_camera_facing_v4.py:242-248`).

### 4.3 What remains valid about ep57 selection

The synthetic JSON geometry is internally exact:

- all 1500 frames pass positive-depth / 48-sym projection audit;
- stored pose + K + 3D box reproduces stored 2D corners at zero error;
- `V8=1363`; visibility counts are V5=17, V6=97, V7=23, V8=1363;
- pose rotations have proper determinant/orthogonality;
- image size is 640×480 throughout.

`scripts/stage0/diffpnp3d_q1_eval.py:10-16` uses order-free Hungarian corner
matching and order-free PnP, so overall corner distance and `honest8` remain
meaningful geometry/order-free measures. But the labels “front”, “rear”, and
individual keypoint channel semantics on val1500 are legacy semantics, not the
current camera-facing semantics. `diffpnp3d_val_select.py:11-15` selected ep57
with a rank sum containing both `rear_med` and `honest8_med`.

Accordingly:

- **valid:** ep57 is the historical winner under the recorded legacy selector;
  order-free overall-corner and honest8 comparisons;
- **not valid without relabeled validation:** a claim that ep57 was selected
  for current-semantic rear-face quality, or that its synthetic per-channel
  kp0…kp7 ranking measures current camera-facing channels;
- **caveat:** because one primary selector term was legacy `rear_med`, ep57 is
  not yet a cleanly reselected current-convention optimum.

## 5. GT schema, cameras, units, and stored-point provenance

### 5.1 Cameras and units

All real N123 images/annotations declare 640×480. Intrinsics have two exact
patterns:

| Population | `(fx, fy, cx, cy)` |
|---|---|
| outside + manual | `(614.183898926, 614.313293457, 329.280395508, 234.528808594)` |
| night | `(605.906494141, 605.969787598, 317.596191406, 256.292297363)` |

For the allowed non-final source sessions, JSON K equals the corresponding
`cam_K.txt` with maximum absolute difference 0.

Real dimension patterns are:

| Population | `(width, depth, height)` count |
|---|---|
| strict N87 | `(1.3,1.1,0.11)` ×50; `(1.1,1.3,0.11)` ×37 |
| manual N36 | `(1.3,1.1,0.11)` ×35; `(1.1,1.3,0.11)` ×1 |

The annotator passes dimensions as `(width, depth, height)`
(`challenge/scripts/annotate_pnp.py:57-58`) and serializes them as named JSON
fields (`challenge/scripts/annotate_io.py:92-94`), so the apparent field order
is not a centimetre/metre or height/depth swap.

Translation-z ranges in metres:

| Population | min | median | p95 | max |
|---|---:|---:|---:|---:|
| strict N87 | 1.451 | 3.293 | 3.980 | 6.409 |
| manual N36 | 3.277 | 4.769 | 6.768 | 6.870 |

Rotation determinant and orthogonality residuals are at floating-point noise
(approximately `1e-15`). These are consistent with object-to-camera
OpenCV-z-forward poses, not a unit-scale collapse.

### 5.2 Mixed 2D GT is intentional

`challenge/scripts/annotate_io.py:46-76` documents and implements:

1. store a clicked keypoint if available;
2. otherwise store the PnP pose projection;
3. otherwise store exact `[-1,-1]`;
4. for centroid, prefer click, then pose-projected 3D origin, then a corner mean.

Old JSON does not preserve an authoritative per-corner source flag. The
following source classification is therefore an **inference**, not ground
truth: a stored point within 0.25 px of the stored-pose projection is called
pose-like; otherwise it is called direct/manual-like.

| Population | valid stored corners | direct/manual-like | pose-like | exact sentinel | other off-image |
|---|---:|---:|---:|---:|---:|
| strict N87 | 683 | 511 | 172 | 13 | 29 |
| manual N36 | 288 | 233 | 55 | 0 | 0 |

Inferred direct/manual-like corner count per strict frame:
`2×1, 3×1, 4×4, 5×7, 6×66, 7×5, 8×3`.  
For manual N36: `4×1, 6×23, 7×5, 8×7`.

This mixture explains why stored poses reproject some entries exactly but not
all clicks. It is not, by itself, evidence that K or 3D geometry is wrong.

## 6. Centroid keypoint 8: four-way comparison

Distances are in pixels.

| Population / comparison | median | p90 | p95 | max |
|---|---:|---:|---:|---:|
| strict N87: stored id8 ↔ stored-pose 3D origin | ~0.00000004 | 0.033 | 0.065 | 0.203 |
| strict: stored id8 ↔ corner-only PnP origin | ~0.00000005 | 0.064 | 0.182 | 1.533 |
| strict full-8 N78: stored id8 ↔ arithmetic stored-corner mean | 3.731 | 8.079 | 12.198 | 16.836 |
| strict: stored id8 ↔ arithmetic pose-projected-corner mean | 4.579 | 12.709 | 19.002 | 29.528 |
| manual N36: stored id8 ↔ stored-pose 3D origin | 0.015 | 0.149 | 0.156 | 0.158 |
| manual: stored id8 ↔ corner-only PnP origin | 0.006 | 0.207 | 0.307 | 0.481 |
| manual: stored id8 ↔ arithmetic stored-corner mean | 2.661 | 4.457 | 4.605 | 4.911 |
| synthetic N1500: id8 ↔ stored-pose 3D origin | 0 | 0 | 0 | 0 |
| synthetic: id8 ↔ corner-only PnP origin | 0 | 0 | 0 | 0 |
| synthetic: id8 ↔ arithmetic corner mean | 2.543 | — | 26.581 | 49.885 |

Verdict: keypoint 8 is the perspective projection of the 3D cuboid origin. In
general, perspective projection does not make it equal to the arithmetic mean
of the eight projected 2D corners. Replacing id8 with a corner average would
introduce a new definition bug.

## 7. GT-to-PnP oracle

The main oracle removes exact sentinels and retains other off-image
correspondences. It uses the stored pose only as an iterative initialization,
so it is a local consistency diagnostic, not a guarantee that every global PnP
branch behaves identically.

### 7.1 Strict N87

| Correspondences | solved | reproj median | p90 | p95 | max | rot Δ median / p95 / max | t Δ median / p95 / max |
|---|---:|---:|---:|---:|---:|---|---|
| mixed stored corners 0–7 | 87/87 | 2.152 | 4.616 | 6.843 | 9.577 | 0° / 0.312° / 10.08° | ~0 / 0.00166 / 0.246 m |
| mixed corners + stored id8 | 87/87 | 1.913 | 4.103 | 6.015 | 8.519 | 0° / 0.310° / 10.05° | ~0 / 0.00162 / 0.246 m |
| inferred direct/manual-like corners only | 85/87 | 2.864 | 5.777 | 8.363 | 12.700 | — | — |
| direct/manual-like corners + pose-derived id8 | 86/87 | 2.446 | 4.930 | 7.108 | 10.900 | — | — |
| pose-derived corners only | 87/87 | 0 | 0 | 0 | 0 | 0 | 0 |
| pose-derived corners + id8 | 87/87 | 0 | 0 | 0 | 0 | 0 | 0 |

### 7.2 Exploratory manual N36

| Correspondences | solved | reproj median | p90 | p95 | max |
|---|---:|---:|---:|---:|---:|
| mixed stored corners 0–7 | 36/36 | 6.312 | 10.770 | 11.637 | 13.305 |
| mixed corners + stored id8 | 36/36 | 5.633 | 9.593 | 10.350 | 11.846 |
| inferred direct/manual-like corners only | 36/36 | 7.666 | 12.870 | 14.750 | 16.540 |
| direct/manual-like corners + pose-derived id8 | 36/36 | 6.572 | 11.030 | 12.900 | 14.180 |
| pose-derived corners, id8 excluded/included | 36/36 | 0 | 0 | 0 | 0 |

Including id8 lowers the mean partly mechanically: id8 is almost exactly
pose-derived, so it contributes a near-zero residual rather than independent
manual evidence. N36 is visibly noisier numerically than strict N87 and remains
exploratory.

## 8. Sentinel versus legitimate off-image points

The annotator explicitly states that only exact `(-1,-1)` is its missing
sentinel, while other negative/out-of-frame projections are valid
(`challenge/scripts/annotate_pnp.py:145-151`).

Strict N87 contains 13 exact sentinel corners in 9 frames:

| Domain | Frame ID | Sentinel keypoint IDs |
|---|---|---|
| outside | `1778651614152548352` | 0, 3 |
| outside | `1778651616134528000` | 0, 3 |
| night | `1779449210935970816` | 3 |
| night | `1779449259354370816` | 7 |
| night | `1779449263958007040` | 4, 7 |
| night | `1779449338785268736` | 0, 3 |
| night | `1779449341220191744` | 3 |
| night | `1779449343555269120` | 3 |
| night | `1779449345790144768` | 3 |

Keypoint frequency: kp3 ×7, kp0 ×3, kp7 ×2, kp4 ×1. Manual N36 has no exact
sentinel.

Separately, strict N87 has 29 valid non-sentinel corners outside the 640×480
image. A blanket `x<0` or “inside-image only” sanitizer would incorrectly throw
away those usable correspondences.

### 8.1 Quantified sentinel failure

Using raw, unsanitized strict corners:

| Metric | sentinel removed | sentinel included |
|---|---:|---:|
| frame reproj median | 2.152 px | 2.266 px |
| frame reproj p90 | 4.616 px | 32.66 px |
| frame reproj p95 | 6.843 px | 82.04 px |
| frame reproj max | 9.577 px | 98.90 px |
| rotation Δ p95 | 0.312° | 20.17° |
| translation Δ p95 | 0.00166 m | 0.211 m |

The median hides the problem because only 9/87 frames contain sentinels.
Per-frame dirty versus clean PnP errors:

| Frame ID | clean px | unsanitized px | unsanitized rot Δ | unsanitized t Δ |
|---|---:|---:|---:|---:|
| `1778651614152548352` | 0.974 | 68.410 | 49.79° | 0.880 m |
| `1778651616134528000` | 0.776 | 67.293 | 49.26° | 0.903 m |
| `1779449210935970816` | 1.268 | 91.630 | 20.17° | 0.102 m |
| `1779449259354370816` | 3.023 | 85.227 | 20.17° | 0.136 m |
| `1779449263958007040` | 2.386 | 70.420 | 44.00° | 0.364 m |
| `1779449338785268736` | 0.692 | 74.618 | 38.38° | 0.243 m |
| `1779449341220191744` | 6.987 | 91.151 | 19.59° | 0.111 m |
| `1779449343555269120` | 4.078 | 91.366 | 16.83° | 0.093 m |
| `1779449345790144768` | 2.765 | 98.895 | 17.77° | 0.095 m |

### 8.2 Existing unsafe code paths

- `challenge/scripts/annotate_pnp.py:460-465` chooses `valid_idx` solely by
  `is not None`; a raw stored `[-1,-1]` entry is therefore treated as valid.
- `scripts/stage0/ralph_eval_3d_reliability.py:103-128` copies all eight raw
  GT corners, appends centroid, and sends them to `solve_pose` without removing
  exact sentinels.
- `scripts/stage0/paper_s2_rgb1_eval.py:442-450,485-507` filters predicted
  finite coordinates but not GT sentinels before Hungarian metrics.
- `scripts/stage0/paper_s2_rgb1_eval.py:462-470` sanitizes sentinels in the
  predicted pose projection, again not in raw GT.
- `scripts/stage0/ralph_eval_all.py:55-75` loads raw GT and evaluates it without
  a GT sentinel mask.
- `scripts/stage0/paper_s2_real_eval.py:88-113` follows the same predicted-only
  validity pattern through its shared evaluator.

Therefore existing cached filter-val metrics can be corrupted on these nine
strict frames even when the model prediction is reasonable.

## 9. Geometry versus intrinsics: separate perturbation tests

### 9.1 Intrinsics

Correct per-frame K reproduces pose-derived GT exactly. Holding the stored pose
fixed but substituting historical synthetic centered K creates:

| K substitution on strict N87 | direct projection shift median | p95 | max |
|---|---:|---:|---:|
| centered `f=615.111084` | 11.111 px | 15.592 | 16.009 |
| centered `f=554.256234` | 24.759 px | 36.126 | 39.980 |

If PnP is allowed to refit, wrong K can partly hide behind a good reprojection:

| K used for PnP | reproj median | rotation bias median | translation bias median |
|---|---:|---:|---:|
| correct per-frame K | 2.152 px | ~0° | ~0 m |
| centered 615 | 2.215 px | 1.232° | 0.065 m |
| centered 554 | 3.117 px | 2.382° | 0.272 m |

For centered 554, median estimated `tz / stored tz = 0.917`. Reprojection error
alone is therefore insufficient to validate intrinsics. Here, direct file
comparison resolves the ambiguity: real JSON K exactly matches session K.

### 9.2 Width/depth

Holding pose fixed and swapping W/D changes strict projections by median
18.345 px (p95 34.111, max 59.407). After local PnP refitting:

| Population | correct-dims reproj median | swapped-dims median | ambiguous by stated screen |
|---|---:|---:|---:|
| strict N87 | 2.152 px | 7.631 px | 0/87 |
| manual N36 | 6.312 px | 9.198 px | 0/36 |

The current stored dimensions are strongly preferred. This local initialized
screen does not claim to exhaust every global PnP branch, but it gives no
evidence of a systematic W/D swap in N123.

## 10. Resize, K scaling, and flip audit

The Stage-B DiffPnP path deliberately squashes 640×480 to 400×400, builds
targets in transformed image space, then decodes belief-grid coordinates back
to the original pixel space. `Deep_Object_Pose/common/utils_dataset.py:350-359,
542-580,736-767` controls the eligible resize and carries the original K.
`Deep_Object_Pose/train/diffpnp3d_loss.py:14-23,47-57,81-84` converts the
50-grid prediction by `x×12.8`, `y×9.6` before using original K. Evaluation
uses the same inverse in `scripts/stage0/diffpnp3d_q1_eval.py:4-8,51,64-75`.

Verdict: for the locked 640×480 eligible path, **K should not be independently
scaled to 400×400 inside the PnP loss**, because coordinates have already been
mapped back to 640×480 pixels.

Horizontal channel permutation is correctly defined as
`[1,0,3,2,5,4,7,6,8]` in
`Deep_Object_Pose/train/geo_loss_struct.py:21-46`. In its 50-grid coordinate
space, `(W-1)-x = 49-x` is correct. Filter evaluation also uses `(W-1)-x`
(`scripts/stage0/paper_s2_filterval_9filters.py:57-70`).

The self-training `_flip_score` path uses `image_size-x` rather than
`(image_size-1)-x`, a deterministic +1 px mismatch. It can affect later h8/PL
filtering. Stage-B ep57 has structural/flip loss disabled, so this does not
explain ep57.

## 11. Frozen ep57 diagnostics on non-final real data

These numbers are diagnostics, not checkpoint-selection evidence. Thresholding
and preprocessing are frozen; final-test is untouched.

### 11.1 Strict N87

| Quantity | Result |
|---|---:|
| ≥6 detected corners | 59/87 (67.82%) |
| detected centroid | 70/87 (80.46%) |
| all 8 corners detected | 47/87 |
| same-index frame median error | median 15.668 px; p90 62.357; p95 72.834; max 160.696 |
| order-free frame median error | median 15.406 px; p90 51.012; p95 62.605; max 84.336 |
| predicted id8 → stored id8 | median 16.431 px; p90 48.696; p95 59.258; max 68.485 |
| predicted id8 → stored-pose origin | numerically the same as stored id8 |
| id8 signed bias versus pose origin | median dx −4.424 px; mean dx −7.569; median dy +6.983; mean dy +4.942 |
| predicted id8 → mean of predicted corners, all-8 N47 | median 2.834 px; p95 9.010; max 13.249 |
| predicted id8 → predicted-corner PnP origin, N59 | median 3.360 px; p95 12.884; max 33.365 |

Per-channel same-index median error among detected points:

| kp | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| px | 5.688 | 10.847 | 12.501 | 5.545 | 21.496 | 22.879 | 30.373 | 20.955 |

### 11.2 Exploratory manual N36

| Quantity | Result |
|---|---:|
| ≥6 detected corners | 35/36 (97.22%) |
| detected centroid | 36/36 |
| all 8 corners detected | 30/36 |
| same-index frame median error | median 10.906 px |
| order-free frame median error | median 10.845 px |
| predicted id8 → stored id8 | median 8.853 px |
| predicted id8 → stored-pose origin | median 8.848 px |
| id8 signed bias versus pose origin | median dx +7.494 px; mean dx +10.359; median dy +4.900; mean dy +4.539 |
| predicted id8 → mean of predicted corners | median 2.151 px |
| predicted id8 → predicted-corner PnP origin | median 2.372 px |

Per-channel medians are kp0 10.779, kp1 5.895, kp2 6.665, kp3 11.193, kp4
36.453, kp5 11.121, kp6 13.051, kp7 17.610 px.

The model centroid follows its own predicted corner configuration/PnP origin
within roughly 2–3 px, while the whole predicted configuration is offset from
GT. Both sets show a downward y bias, but the x bias changes sign by domain.
There is no evidence for one universal fixed pixel correction.

## 12. GT review candidates

These are **review/relabel candidates**, not automatically invalid frames.
Visual review was intentionally not used to make exclusions.

Strict screening counts:

- mixed stored-vs-pose mean residual `>5 px`: 6/87;
- `>8 px`: 2/87;
- any point residual `>20 px`: 4/87;
- exact sentinel: 9/87.

Manual exploratory counts:

- mean residual `>5 px`: 23/36;
- `>8 px`: 5/36;
- any point residual `>20 px`: 15/36.

Highest residual candidates:

| Population/domain | Frame ID | mean px | max point px | Note |
|---|---|---:|---:|---|
| manual | `1778654553265639936` | 11.051 | 33.043 | exploratory |
| manual | `1778654552425688832` | 11.024 | 39.725 | exploratory |
| manual | `1778654552257856768` | 10.110 | 30.175 | exploratory |
| manual | `1778654552023006464` | 9.592 | 28.695 | exploratory |
| manual | `1778654550208514304` | 8.199 | 35.496 | exploratory |
| strict/outside | `1778653498432396288` | 8.540 | 22.670 | stored `reproj_error_px=12.810`; corner-PnP centroid difference 1.533 px |
| strict/outside | `1778653345465966336` | 8.500 | 24.417 | relabel review |
| strict/night | `1779449261622708480` | 6.509 | 19.728 | relabel review |
| strict/outside | `1778653453647432960` | 6.482 | 22.638 | stored field 0.528; mixed fallback likely |
| strict/night | `1779449341220191744` | 6.114 | 21.683 | also contains sentinel kp3 |

The nine sentinel frames in section 8 should be fixed or masked before any
GT-PnP oracle, Hungarian GT metric, or filter-threshold selection is rerun.

## 13. Required finding format

### Finding A — strict GT geometry and camera

**관찰:** N87 JSON K equals session K, dimensions are metre-scale, stored
rotations are proper, pose-derived projections close exactly, and LR/TB
camera-facing invariants have zero violations.

**원인 후보:** A gross intrinsic scaling error, metre/centimetre confusion,
pose direction reversal, or systematic W/D swap.

**지지 증거:** Wrong-K and W/D perturbations produce large projection/pose
biases, showing the tests are sensitive enough to detect those faults.

**반증 증거:** The actual K matches files exactly; correct geometry gives zero
pose-derived residual; no strict frame is W/D-swap ambiguous under the stated
local oracle.

**현재 판정:** Those gross camera/geometry faults are not present in current
N87. Residuals are primarily the expected mixed click/pose-projection schema,
plus sentinel misuse in consumers.

**남은 불확실성:** A local initialized PnP screen does not exhaust every global
solver branch; historical JSON lacks authoritative per-point source flags.

**권장 수정:** Preserve per-frame K and named dimensions; add an explicit
`keypoint_source[9]` or validity mask when GT is regenerated.

**수정 후 재검증 결과:** Not run—this task is audit-only and changed no data.

### Finding B — exact sentinel handling

**관찰:** 13 exact sentinel corners occur in 9 strict frames, while 29 other
off-image corners are legitimate. Existing consumers frequently keep
`(-1,-1)` in GT.

**원인 후보:** Conflating “outside image” with missing, or checking only
`None`/prediction finiteness.

**지지 증거:** Unsanitized p95 reprojection rises from 6.843 to 82.04 px,
rotation p95 from 0.312° to 20.17°, and translation p95 from 0.00166 to
0.211 m.

**반증 증거:** Removing only exact `(-1,-1)` restores stable oracle results;
removing every negative coordinate would wrongly discard valid annotations.

**현재 판정:** Confirmed GT-consumer bug, not a general off-image annotation
bug.

**남은 불확실성:** Historical cached metrics have not all been regenerated, so
the exact downstream ranking impact is unknown.

**권장 수정:** Centralize `valid_gt = finite & ~((x==-1)&(y==-1))`; keep other
off-image coordinates for PnP; apply it before Hungarian and all GT-PnP calls.

**수정 후 재검증 결과:** Not run—no code/cache mutation was authorized here.

### Finding C — synthetic val convention

**관찰:** 810/1500 synthetic validation frames violate current camera-facing LR
pairing, despite Stage-B training roots almost universally passing it.

**원인 후보:** Legacy val labels were not included in the later v4 camera-facing
conversion.

**지지 증거:** Converter defaults exclude val; exact 3D/pose/K projection still
closes, which distinguishes a permutation/semantic mismatch from corrupt
geometry.

**반증 증거:** TB passes 1500/1500 and order-free metrics remain internally
valid, so the dataset is not generally broken.

**현재 판정:** Legacy-channel convention mismatch. Historical ep57 selection is
reproducible, but its `rear_med` term is not a clean current-semantic rear
metric.

**남은 불확실성:** A current-convention relabel/reselection may or may not pick
the same epoch.

**권장 수정:** Build a versioned, camera-facing val index without touching the
locked original; rerun epochs 45/48/51/54/57 with order-free metrics plus
current-semantic per-face metrics.

**수정 후 재검증 결과:** Not run—new rendering/relabeling is outside this task.

### Finding D — split independence

**관찰:** N36 belongs to a PL-pool session; later descendants contain exact
filter-val RGBs and same-session samples; historical RALPH broad eval scripts
include cp07/cp09 final-test sessions.

**원인 후보:** Frame-ID-only leakage guards and legacy “all GT” directory
globbing predate the strict split lock.

**지지 증거:** Exact RGB hashes, session-count audit, explicit script source
lists, and split metadata agree.

**반증 증거:** Frozen ep57 itself is synthetic-only; val1500 has no exact
Stage-B train duplicate.

**현재 판정:** ep57 frozen diagnostics on N87 are usable as post-selection
diagnostics. N36 is exploratory. Descendant/current-filter results and legacy
RALPH all-GT results are not independent paper-selection evidence.

**남은 불확실성:** Near-duplicate video frames were not measured with a perceptual
or temporal similarity threshold; same-session dependence may extend beyond
exact hashes.

**권장 수정:** Enforce whole-session group splits, store immutable membership
hashes, and fail closed if any evaluation image hash appears in training.

**수정 후 재검증 결과:** Not run—saved descendants were audited, not rebuilt.

## 14. Minimal next rerun order

1. Add one shared exact-sentinel GT mask to PnP and metric consumers.
2. Rerun only strict N87 cached diagnostics; keep N36 in a separately labeled
   exploratory table.
3. Review/relabel the listed strict candidates, recording per-point source and
   visibility explicitly.
4. Create a versioned current-camera-facing synthetic validation membership,
   then re-evaluate the five Stage-B candidate epochs.
5. Keep final-test sealed until all model, epoch, threshold, filter, and
   exclusion decisions are frozen.

No step above was performed by this audit.
