# PalletBinPoseEstimation Dataset

A real-world RGB-D dataset for 6-DoF pose estimation of Pallox pallet bins, captured across two environments, three camera heights, and multiple distances with yaw angle and occlusion variations.

---

## Citation

If you use this dataset, please cite:

```bibtex
@conference{icpram26,
author={Yanming Wu and Eric Demeester},
title={6D Pose Estimation of Pallet Bins for Autonomous Logistics Using a Synthetically Trained Object Detector and FoundationPose},
booktitle={Proceedings of the 15th International Conference on Pattern Recognition Applications and Methods - Volume 1: ICPRAM},
year={2026},
pages={468-475},
publisher={SciTePress},
organization={INSTICC},
doi={10.5220/0014455600004067},
isbn={978-989-758-797-9},
}
```

---

## Overview

| Property | Value |
|---|---|
| Object | Pallox pallet bin |
| Sensor | RGB-D camera (Intel RealSense L515) |
| Resolution | 640 × 480 |
| Depth encoding | 16-bit PNG, millimeters |
| Total frames | 284 (fully annotated) |
| Annotation type | 6-DoF pose (SE3), instance mask, visibility score |

---

## Dataset Structure

```
PalletBinPoseEstimation/
├── {height}_{env}/          # e.g. 0.55m_lab, 1.23m_hall, 1.92m_lab
│   ├── rgb/                 # 640×480 RGB images (.png)
│   ├── depth/               # 640×480 depth images, uint16, mm (.png)
│   ├── masks/               # Instance segmentation masks (.png)
│   ├── annotated_poses/     # 4×4 SE3 pose matrices (.txt, one per frame)
│   ├── cam_K.txt            # 3×3 camera intrinsic matrix
│   └── visibility_scores.csv
└── assets/
    └── mesh/
        ├── pallox.obj
        ├── pallox.obj.mtl
        └── pallox_tex.png
```

---

## Splits

| Split | Environment | Camera Height |
|---|---|---|
| `0.55m_hall` | Hallway | 55 cm |
| `0.55m_lab` | Lab | 55 cm |
| `1.23m_hall` | Hallway | 123 cm |
| `1.23m_lab` | Lab | 123 cm |
| `1.92m_hall` | Hallway | 192 cm |
| `1.92m_lab` | Lab | 192 cm |

Each split covers multiple distances from the pallet (1.5 m – 8 m depending on height and environment). At each distance, several frames are captured by varying the camera's **yaw angle** and the presence of an **occluder** placed in front of the pallet.

---

## Annotations

### Pose (`annotated_poses/XXXX.txt`)

Each file contains a **4×4 homogeneous transformation matrix** (camera-to-object, SE3) in row-major order:

```
R00 R01 R02 tx
R10 R11 R12 ty
R20 R21 R22 tz
0   0   0   1
```

Values are stored in scientific notation, one row per line. Translation is in **meters**.

### Instance Mask (`masks/XXXX_sam1_mask.png`)

Binary PNG mask. White pixels (255) indicate the pallet bin; black (0) is background. The filename format is `{frame_id}_sam1_mask.png` (e.g. `0000_sam1_mask.png`).

### Visibility Scores (`visibility_scores.csv`)

Per-frame visibility statistics computed from the projected 3D mesh:

| Column | Description |
|---|---|
| `image` | Frame index (matches filename) |
| `category` | Visibility category (see below) |
| `visible_percent` | % of mesh area that is unoccluded and in-frame |
| `truncated_percent` | % of mesh area outside the image boundary |
| `occluded_percent` | % of mesh area occluded by other objects |
| `visible_area` | Visible mesh area in pixels |
| `mesh_area` | Total projected mesh area in pixels |

Visibility categories:

| Category | Visible % |
|---|---|
| Fully Visible | > 75% |
| Mostly Visible | 50–75% |
| Partially Visible | 25–50% |
| Mostly Occluded | < 25% |

### Camera Intrinsics (`cam_K.txt`)

3×3 camera intrinsic matrix, stored as three space-separated rows:

```
fx  0  cx
0  fy  cy
0   0   1
```

Example (1.23m_lab):
```
605.821  0.000  322.850
0.000  606.242  237.404
0.000    0.000    1.000
```

---

## 3D Model

The textured mesh of the Pallox pallet bin is provided in `assets/mesh/`. The pose matrices in `annotated_poses/` express the object pose in the camera frame (`ob_in_cam`): a 3D point `p` in object coordinates transforms to camera coordinates as `p_cam = T @ p_obj`.

---

## Loading the Data

```python
import numpy as np
from PIL import Image

split = "1.23m_lab"
base = f"PalletBinPoseEstimation/{split}"
idx = "0000"

# RGB image
rgb = np.array(Image.open(f"{base}/rgb/{idx}.png"))                       # (480, 640, 3) uint8

# Depth image — values in millimeters
depth_mm = np.array(Image.open(f"{base}/depth/{idx}.png"))                # (480, 640) uint16
depth_m = depth_mm.astype(np.float32) / 1000.0

# Instance mask
mask = np.array(Image.open(f"{base}/masks/{idx}_sam1_mask.png"))          # (480, 640) uint8

# Pose matrix
pose = np.loadtxt(f"{base}/annotated_poses/{idx}.txt")                    # (4, 4) float64

# Camera intrinsics
K = np.loadtxt(f"{base}/cam_K.txt")                                       # (3, 3) float64
```
