---
license: agpl-3.0
tags:
  - object-detection
  - keypoint-detection
  - pose-estimation
  - 6dof
  - yolo
  - ultralytics
  - robotics
  - forklift
library_name: ultralytics
pipeline_tag: keypoint-detection
---

# pallet-pose-yolo26n-ft

Single-stage **pallet 6-DoF pose** model for forklift deployment. One RGB frame in,
one bounding box + **9 keypoints** out; a downstream PnP turns those keypoints into a
6-DoF pose (yaw / lateral offset / forward distance).

This is the **finetuned** checkpoint: a synthetic-only pretrain followed by a finetune on
real annotations **plus background frames from the deployment camera**. The finetune exists
to remove false positives, not to raise accuracy — see [Why the finetune](#why-the-finetune).

---

## ⚠️ Read this before running inference

Two contract details are not optional. Get either wrong and the model looks broken.

### 1. Reflect-pad the image by 100 px on all four sides

The model was trained on padded frames so that pallets clipped by the image border stay
learnable. Match that at inference:

```python
padded = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_REFLECT_101)
```

Subtract 100 from predicted coordinates before doing anything geometric (PnP, drawing).

Measured effect of skipping the padding — it costs **keypoint accuracy**, not detection rate:

| | detection | keypoint err (median) | keypoint err (p90) |
|---|---|---|---|
| padded (correct) | 157/161 | **7.38 px** | **26.75 px** |
| unpadded | 157/161 | 8.49 px | 28.42 px |

On the deployment sequence (911 frames) detection barely moves either (479 vs 476 at
conf 0.4). So an unpadded frame still detects — it just localises less precisely, which
matters because these keypoints feed PnP.

### 2. Keypoint order is **camera-facing**, not object-fixed

Index 0-3 is always the face pointing at the camera — it is *not* tied to a fixed side of
the physical pallet.

```
0 near_top_left      1 near_top_right      <- face toward the camera (fork pockets)
2 near_bottom_right  3 near_bottom_left
4 far_top_left       5 far_top_right       <- opposite face
6 far_bottom_right   7 far_bottom_left
8 centroid

top    = {0, 1, 4, 5}
bottom = {2, 3, 6, 7}
```

**Never enable horizontal flip augmentation** if you finetune this further. The ordering is
left-right asymmetric; `fliplr` silently corrupts it. Training used `fliplr=0.0`.

If you do enable it, the correct mapping is `flip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 8]`
(0↔1, 2↔3, 4↔5, 6↔7, centroid fixed) — the identity permutation is wrong.

---

## Usage

```python
import cv2, numpy as np
from ultralytics import YOLO

model = YOLO("pallet_yolo26n_pose_ft.pt")
PAD = 100

img = cv2.imread("frame.png")                     # 640x480 BGR
padded = cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)

r = model.predict(padded, imgsz=640, conf=0.4, verbose=False)[0]
if r.boxes is not None and len(r.boxes):
    i = int(np.argmax(r.boxes.conf.cpu().numpy()))     # highest-confidence instance
    kps = r.keypoints.xy.cpu().numpy()[i] - PAD        # (9, 2) in original image coords
    conf = float(r.boxes.conf.cpu().numpy()[i])
```

### Recovering 6-DoF pose

Feed the keypoints to PnP with the pallet's known 3-D corner model. In the reference
deployment: `SOLVEPNP_SQPNP` + `refineLM`, using only keypoints whose confidence ≥ 0.5.
Index 8 (centroid) is a real correspondence and may be included.

| setting | value |
|---|---|
| `imgsz` | 640 |
| `conf` | 0.4 (see [Tuning](#tuning-conf)) |
| padding | 100 px `BORDER_REFLECT_101` |
| input | 640×480 BGR (deployment camera) |

---

## Why the finetune

The synthetic-only pretrain had a structural blind spot: **all 73,916 training images
contained a pallet**, so the model had never seen a pallet-free scene. On deployment video
it confidently labelled the forklift's own forks and a metal fence as pallets — both are
horizontal-slat structures that genuinely resemble a pallet.

The fix was not more augmentation or class weighting. It was adding **background images**
(empty labels) from the deployment camera itself.

---

## Results

Measured on the deployment sequence (911 frames) and on a held-out real evaluation set of
161 manually annotated frames that were **excluded from training**.

### False positives — frames with no pallet (n=259)

Detection rate on pallet-free frames, swept over confidence threshold (lower is better):

| model | @0.05 | @0.10 | @0.25 | @0.40 |
|---|---|---|---|---|
| synthetic-only pretrain | 50.6% | 30.5% | 0.0% | 0.0% |
| **this model** | **0.0%** | **0.0%** | **0.0%** | **0.0%** |

> These 259 frames were part of the finetune set, so this figure is in-sample. The held-out
> numbers below are the honest ones.

### Held-out real evaluation set (n=161, never trained on)

| model | detection rate | keypoint err (median) | keypoint err (p90) |
|---|---|---|---|
| synthetic-only pretrain | 88.2% | 9.30 px | 28.41 px |
| **this model** | **97.5%** | **7.38 px** | 26.75 px |

Keypoint error is the median L2 distance over visible keypoints, in original-image pixels.

### Detection count drops — and that is the point

On the deployment sequence, detections fall from 558/911 to 479/911. Inspecting all 81
frames the pretrain caught and this model does not: almost all are fence, hedge, tarpaulin
or building — i.e. removed false positives. **Frames where the pretrain was confident
(conf ≥ 0.9, n=440) are retained at 100%.**

---

## Known limitations

- **Heavily clipped pallets can be missed.** One deployment frame with a pallet cut off at
  the bottom edge went 0.427 (pretrain) → 0.000 (this model). Training longer does not fix
  it; this comes from the background-image composition, not from undertraining.
- **Distant, small pallets** are weaker for the same reason. Roughly 5-8 of the 81 dropped
  frames look like genuine misses of a far pallet.
- **One pallet per frame.** Trained `single_cls=True` with one instance per image; the
  reference pipeline takes the highest-confidence box. Multi-pallet scenes are untested.
- **Fixed camera geometry.** Tuned for a 640×480 forklift-mounted camera. Other mounting
  heights and view angles are untested; real training data skews to low elevation angles.
- **Not a general pallet detector.** Plastic pallets dominate the real data. Wooden pallets
  were explicitly excluded.

### Tuning `conf`

`0.4` is the deployment default. Because the false-positive rate is 0% even at 0.05, you can
lower the threshold to **0.25** to recover some distant/clipped detections at little risk.

---

## Training

| | |
|---|---|
| base | `yolo26n-pose` (Ultralytics), then a synthetic-only pretrain (73,916 images, 60 epochs) |
| finetune data | 16,694 images = 3,140 real (157 unique ×20) + 1,554 background (259 unique ×6) + 12,000 synthetic |
| schedule | 40 epochs, SGD, `lr0=0.002`, `cos_lr`, `batch=32`, `nbs=64`, `imgsz=640`, `seed=42` |
| augmentation | `mosaic=0.15` (closed for last 10), `translate=0.10`, `scale=0.25`, HSV; **all flips and rotations disabled** |
| hardware | single RTX 3080, 2.6 h |

Synthetic images are kept in the mix so the finetune does not forget the keypoint structure
learned during pretraining — 157 real images alone are 0.2% of the pretrain set.

`training_results.csv` and `training_args.yaml` are included for reproducibility.

### A note on validation metrics

The bundled validation set is synthetic, and its mAP does **not** predict real-world quality.
Pushing synthetic pose mAP from 0.9448 to 0.9573 produced no measurable real improvement
(paired Wilcoxon over the held-out set, p = 0.83). Judge changes on real data.

---

## Files

| file | description |
|---|---|
| `pallet_yolo26n_pose_ft.pt` | the model (6.2 MB) |
| `inference_config.yaml` | the inference contract as machine-readable config |
| `training_results.csv` | per-epoch training log |
| `training_args.yaml` | full Ultralytics hyperparameters |
| `SHA256SUMS` | checksum |

## License

**AGPL-3.0**, inherited from the Ultralytics YOLO base weights. If you deploy this over a
network service, AGPL obligations apply to your application. For a commercial license,
contact Ultralytics.
