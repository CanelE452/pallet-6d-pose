# Synthetic source data and coordinate contract

This module prepares the same **55,980 training / 4,020 validation images** used
by the frozen paper R0 checkpoint. It does not train or run a model. Source RGB,
labels, membership files, and previous experiment artifacts remain unchanged.
The generated `SOURCE_MANIFEST.json` and `SOURCE_DATA_AUDIT.json` are written in
`data/pallet/results/pallet_line_pose_v1/`; differing existing outputs are refused.

## Source and membership

The source is
`challenge/yolo_pose_one_model/datasets/g38_legacy_v1v2_p0_tex20k/data.yaml`.
It refers to `images/train` and `images/val` directories, rather than an image
list file. Images and labels are the existing merged links. Membership is
cross-checked against `spatial_concat_scratch/PROBE_METADATA_60K.jsonl`, its
SHA-bound audit, and the complete image/label directory inventories.

| Source | Train | Original validation |
|---|---:|---:|
| G38 | 38,002 | 1,998 |
| P0 | 8,989 | 1,011 |
| TEX | 8,989 | 1,011 |
| Total | 55,980 | 4,020 |

P0 and TEX have the same 10,000 scenario IDs. The existing SHA1(sample_id)
partition puts both renders in the same original split: 8,989 training and
1,011 validation pairs. The renders are related examples, not two independent
scenarios, and their geometry/texture can differ. G38 scenario IDs are singletons
under the available source metadata; this does not establish independence of
nearby renders or shared assets.

Training membership is retained in full. Only the original validation split is
subdivided. Namespace groups as `G38:<pair_group_id>` or
`P0_TEX:<pair_group_id>`, sort by SHA256 of
`pallet_line_pose_v1:synthetic_val_scenario:25-25-50:v1` + newline + group ID,
then use boundaries `floor(N/4)` and `floor(N/2)`. The 3,009 validation groups
therefore yield 752 calibration, 752 selection, and 1,505 heldout groups. Both
P0/TEX renders stay together. Image counts are recorded in the generated audit;
they need not have exact 25/25/50 ratios because group sizes differ.

The frozen build produced the following counts:

| Partition | G38 | P0 | TEX | Images | Scenarios |
|---|---:|---:|---:|---:|---:|
| Train | 38,002 | 8,989 | 8,989 | 55,980 | 46,991 |
| Calibration | 500 | 252 | 252 | 1,004 | 752 |
| Selection | 473 | 279 | 279 | 1,031 | 752 |
| Heldout | 1,025 | 480 | 480 | 1,985 | 1,505 |

Scenario crossings are zero. The source labels contain 535,066 supervised
keypoints and 4,934 unsupervised keypoints. Image and label content verification
completed for all 60,000 records; no model inference was executed.

"Heldout" is relative to the new line branch's training/selection. The frozen
R0 and previous exploratory probes already used the original validation
population. It is not a previously unseen independent final test. The audit
also records exact annotation-locator overlap with the earlier DHT manifests,
using metadata only. This does not prove absence of copied images or shared
render assets when locator intersection is zero.

The earlier `dht_padding_view_v1` manifest intersects 722 current G38 records:

| Previous population | Current train | Calibration | Selection | Heldout | Total |
|---|---:|---:|---:|---:|---:|
| low_train | 486 | 7 | 7 | 18 | 518 |
| low_val | 62 | 2 | 1 | 0 | 65 |
| low_test | 132 | 0 | 0 | 7 | 139 |

These are **previous manifest membership** intersections; they do not assert
that those previous main training runs executed. No exact annotation-locator
intersection was found with the `hough_attention_transfer_v1` or the live
`dht_pose_integration_v1` synthetic populations. Their image content was not
opened for this comparison. Previous train/val/test names belong to those
experiments and cannot be treated as untouched populations for this new model.

## RGB and label semantics

- Images already have **100 pixels of `cv2.BORDER_REFLECT_101` on all sides**.
  Do not pad these images again. The four raw shapes are 480×640, 480×720,
  540×960 and 560×560 (height×width).
- Each label contains exactly 32 values: class 0; normalized bounding-box
  `(cx,cy,w,h)`; nine `(x,y,v)` keypoints. Coordinates are normalized by the
  **prepared, already-padded canvas**. Order is the existing camera-facing eight
  corners followed by the centroid.
- `v=2` means a known coordinate lies within the padded canvas. It is not a
  physical visibility/occlusion label. A point outside the original image can
  therefore remain supervised inside its padding. `v=0` has stored coordinates
  `(0,0)` and supplies no coordinate target. No missing coordinate is recovered
  from original renderer labels.
- The bounding box is the source converter's axis-aligned envelope of valid
  keypoints, including the centroid; it is not a segmentation box.
- All 60,000 labels are parsed and checked; all prepared PNG headers and image
  and label SHA256 hashes are checked. Source paths are strictly allowlisted to
  the three synthetic sources. No real image or real GT enters this manifest.
- Dimension, parity, pose and source permutation values in the older probe
  metadata are discarded. Only the renderer annotation locator is retained as
  provenance; the loader does not open that annotation.

## Prediction and loss separation

`load_inference_sample(record)` returns only an image, ID and transform; it
returns no GT. `select_top1(scores, class_ids=...)` selects the highest predicted
pallet score, with the first index breaking a tie. It cannot access GT.

After prediction selection, `load_loss_targets(record, transform)` returns GT
boxes/keypoints in input pixels and validity masks. Then
`match_loss_target(top1_box, targets['boxes_xyxy'], minimum_iou=0.5)` determines
whether that already-selected prediction receives an instance loss. An unmatched
or missing prediction remains in the manifest and accounting denominator. GT
cannot select another detection, crop a feature, initialize a point, or rank a
line hypothesis. The module itself performs no predictions or target matching
over the dataset.

The eight side roles are `(1,2), (3,0), (5,6), (7,4), (0,4), (1,5), (2,6), (3,7)`.
An edge loss target is valid only if both endpoint coordinates are supervised.
Invalid coordinates are returned as NaN together with false masks, preventing
the source `(0,0,0)` sentinel from becoming a geometric border target.

## Actual YOLO input transform

Default preprocessing matches batch-one `YOLO.predict(imgsz=640, rect=True)`:
LetterBox with `auto=True`, `stride=32`, scaleup, centered padding value 114,
BGR→RGB, CHW float32 divided by 255. The four resulting tensor H×W shapes are
544×640, 480×640, 416×640 and 640×640. Mixed input shapes must not be stacked
before prediction; the feature cache can separately pad feature maps and retain
their actual input dimensions.

Use `tf = transform(prepared_canvas.shape[:2], actual_hooked_tensor.shape[-2:])`
when the forward hook supplies the actual shape. For canvas `(H,W)` and input
`(Hi,Wi)`, `r=min(Hi/H, Wi/W)`, resized shape is `(round(H*r),round(W*r))`.
Actual label padding is the integer left/top derived with `round(half_pad-0.1)`.

- GT uses `q_input = q_prepared * r + integer_left_top`, exactly like
  Ultralytics label updates. Original unpadded GT additionally adds 100 first.
- Installed `ops.scale_coords` uses half the total **rounded-resize** padding;
  this may differ from integer padding by 0.5 input pixel. The transform stores
  `scale_coords_pad_xy` separately and its prediction method reproduces that
  function, including optional boundary clipping.
- Installed `ops.scale_boxes` uses integer left/top, which is exposed separately.
- `raster_scale_xy` records actual rounded resize dimensions divided by canvas
  dimensions; `label_scale_xy` is the nominal `(r,r)` used for labels. No DHT
  half-cell correction is applied.
- Prefer raw-head points before `scale_coords` for branch conditioning. Lifting
  `Results.keypoints.xy` back to input pixels returns an interior mask because
  clipped border values cannot recover the raw prediction.

CPU validation passed on 14 shape/auto combinations: image tensors were
bit-exact with installed LetterBox; point and box inverse outputs were exactly
equal to installed `scale_coords`/`scale_boxes`; label affine roundtrip maximum
error was `1.1368683772161603e-13` pixels. Twelve actual source images spanning
three sources × four sizes reproduced every reflection-border pixel exactly.
Highest-score selection, class filtering, stable ties, empty detection and IoU
loss matching were checked without model inference. Machine-readable evidence
is saved as `SOURCE_TRANSFORM_QA.json` alongside the manifest.

## API and command

```python
import source_data as SD

# Frozen prediction/feature capture: no target in this dictionary.
sample = SD.load_inference_sample(record)
tf = SD.transform(sample['image_bgr'].shape[:2], actual_tensor.shape[-2:])
top1 = SD.select_top1(predicted_scores, class_ids=predicted_classes)

# Only after choosing the predicted instance; separate loss-target path.
if top1 is not None:
    targets = SD.load_loss_targets(record, tf)
    match = SD.match_loss_target(predicted_boxes_input[top1],
                                targets['boxes_xyxy'], minimum_iou=0.5)
```

`SourceDataset(manifest_path, partition='train', include_loss_targets=False)`
is compatible with a batch-one DataLoader or explicit list collation.

From the purpose-declared result directory:

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python \
  /home/minjae/Documents/github/pallet-pose/scripts/research/pallet_line_pose_v1/source_data.py \
  --run-dir /home/minjae/Documents/github/pallet-pose/data/pallet/results/pallet_line_pose_v1 \
  --workers 8
```

Frozen manifest SHA256:
`feaa24075d31c4227e3397b7450a19a0f1d3a73dc0203d12a8ca5b927eb59789`.
Frozen `source_data.py` SHA256:
`dec46a82d9db025736c1b390ff144f036a006ca7c0f0c5da21e3c7c803edee41`.
