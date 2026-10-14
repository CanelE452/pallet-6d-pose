"""Frozen spatial RGB head for observed-point usability and independent correction.

The classifier predicts DIRECT_VISIBLE and raw coordinate error <=8 pixels.
The coordinate head was trained on DIRECT_VISIBLE errors <=16 pixels. Neither
head consumes an initial-pose projection, a keypoint ID or a dataset record.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .appearance import infer_appearance

ROOT = Path(__file__).resolve().parents[3]
SCALARS = ('inward_material_fraction', 'outward_material_fraction', 'patch_material_fraction',
           'nearby_inward_material_fraction', 'soft_inward_color_support', 'soft_nearby_color_support',
           'corner_structure', 'edge_energy', 'patch_chroma_mean', 'patch_chroma_std')
VECTORS = ('patch_lab_mean', 'patch_lab_median', 'patch_lab_std', 'inward_lab_mean', 'inward_lab_std')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def legacy_rgb26(appearance):
    """Keep study v1's two absent color-support slots; never silently upgrade."""
    output = []
    radius = max(1, appearance['diagnostics']['patch_radius_px'])
    for row in appearance['diagnostics']['corners']:
        values = [float(row.get(key) or 0.) for key in SCALARS]
        for key in VECTORS:
            vector = row.get(key)
            values.extend([0., 0., 0.] if vector is None else vector)
        values.append(min(4, float(row.get('edge_distance_px') or 0.) / radius))
        output.append(values)
    return np.asarray(output, dtype=float)


def crop_context(image, point, width):
    """Exact study crop: floor origin, reflect101 padding, linear128 resize."""
    width = int(width)
    x0, y0 = np.floor(np.asarray(point) - width / 2).astype(int)
    x1, y1 = x0 + width, y0 + width
    height, image_width = image.shape[:2]
    tile = image[max(0, y0):min(height, y1), max(0, x0):min(image_width, x1)]
    tile = cv2.copyMakeBorder(tile, max(0, -y0), max(0, y1-height),
        max(0, -x0), max(0, x1-image_width), cv2.BORDER_REFLECT_101)
    if tile.shape[:2] != (width, width):
        raise ValueError('Invalid context crop geometry')
    return cv2.resize(tile, (128, 128), interpolation=cv2.INTER_LINEAR)


@dataclass
class JointObservationModel:
    descriptor: dict
    arrays: dict
    weights_path: Path
    device: str
    backbone: object = None

    def network(self):
        if self.backbone is None:
            import torch
            from torch import nn
            from torchvision.models import resnet18
            original = resnet18(weights=None)
            original.load_state_dict(torch.load(self.weights_path, map_location='cpu', weights_only=True))
            self.backbone = nn.Sequential(original.conv1, original.bn1, original.relu, original.maxpool,
                original.layer1, original.layer2).eval().requires_grad_(False).to(self.device)
        return self.backbone


@lru_cache(maxsize=4)
def _load_model(path, device):
    descriptor_path = Path(path)
    descriptor = json.loads(descriptor_path.read_text())
    if descriptor['schema'] != 'JOINT_OBSERVATION_FROZEN_V1' or descriptor['family'] != 'RGB26_F4F8_PCA32':
        raise ValueError('Unsupported joint observation model')
    arrays_path = descriptor_path.parent / descriptor['arrays_file']
    if sha256(arrays_path) != descriptor['arrays_sha256']:
        raise ValueError('Joint model parameter binding changed')
    weights_path = ROOT / descriptor['backbone']['weights_repo_relative']
    if sha256(weights_path) != descriptor['backbone']['weights_sha256']:
        raise ValueError('Frozen ImageNet backbone binding changed')
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    for array in arrays.values():
        array.setflags(write=False)
    return JointObservationModel(descriptor, arrays, weights_path, device)


def load_joint_model(model_path, device=None):
    """Load bound immutable JSON/NPZ; cache backbone for subsequent frames."""
    if device is None:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return _load_model(str(Path(model_path).resolve()), str(device))


def spatial_features(image_bgr, points, eligible_ids, radius, model):
    """Return frozen spatial2560 in the supplied native keypoint order."""
    import torch
    from torch.nn import functional as F
    if not len(eligible_ids):
        return np.empty((0, 2560), np.float32)
    patches = []
    for k in eligible_ids:
        patches.extend([crop_context(image_bgr, points[k], 4*radius),
                        crop_context(image_bgr, points[k], 8*radius)])
    bgr = np.stack(patches)
    tensor = torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1].transpose(0, 3, 1, 2))).float().div_(255)
    normalization = model.descriptor['backbone']['normalization']
    mean = torch.tensor(normalization['mean']).reshape(1, 3, 1, 1)
    std = torch.tensor(normalization['std']).reshape(1, 3, 1, 1)
    # Match extraction precision locally; restore the caller's CUDA settings.
    with torch.backends.cudnn.flags(benchmark=False, deterministic=True, allow_tf32=False), torch.inference_mode():
        spatial = model.network()(((tensor - mean) / std).to(model.device))
        if spatial.shape != (2*len(eligible_ids), 128, 16, 16):
            raise ValueError('Unexpected frozen backbone spatial shape')
        four = F.adaptive_avg_pool2d(spatial[::2], (4, 4)).flatten(1).cpu().numpy()
        eight = F.adaptive_avg_pool2d(spatial[1::2], (2, 2)).flatten(1).cpu().numpy()
    return np.column_stack([four, eight])


def predict_feature_rows(rgb26, frozen_spatial, radius, model):
    """Pure exported-parameter forward; separate class and ridge train bases."""
    a = model.arrays
    spatial = np.array(frozen_spatial, dtype=np.float32, copy=True)
    # sklearn StandardScaler preserves the input float32 dtype in this study.
    spatial -= a['spatial_mean']
    spatial /= a['spatial_scale']
    compressed = (spatial - a['pca_mean']) @ a['pca_components'].T
    x = np.column_stack([np.asarray(rgb26, float), compressed])
    class_z = (x - a['class_mean']) / a['class_scale']
    logits = class_z @ a['class_coefficients'].reshape(-1) + float(a['class_intercept'][0])
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -60, 60)))
    reg_z = (x - a['ridge_mean']) / a['ridge_scale']
    normalized_delta = reg_z @ a['ridge_coefficients'].T + a['ridge_intercept']
    radius = np.asarray(radius, float)
    if radius.ndim == 0:
        radius = np.full(len(x), radius)
    delta = np.clip(normalized_delta * radius[:, None], -16, 16)
    lengths = np.linalg.norm(delta, axis=1)
    delta *= np.minimum(1, 16 / np.maximum(lengths, 1e-12))[:, None]
    return probabilities, delta


def predict_joint_observation(image_bgr, q, H, initial_pose, K, model):
    """Observe current RGB; preserve supplied H, native order and center slot.

    ``initial_pose`` and ``K`` are accepted for the common postprocessing API;
    this selected image-only model does not consume either argument. The caller
    supplies the fixed initial self-occlusion mask ``H``. Returned points apply
    only independently predicted image deltas with usable probability >=.45.
    """
    if not isinstance(model, JointObservationModel):
        model = load_joint_model(model)
    image = np.asarray(image_bgr)
    points = np.array(q, dtype=float, copy=True)
    hidden = np.array(H, dtype=bool, copy=True)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('image_bgr must be uint8 HxWx3')
    if points.shape not in ((8, 2), (9, 2)) or hidden.shape != (8,):
        raise ValueError('q must have eight or nine native points and H eight booleans')
    appearance = infer_appearance(image, points, hidden)
    radius = int(appearance['diagnostics']['patch_radius_px'])
    rgb = legacy_rgb26(appearance)
    height, width = image.shape[:2]
    corners = points[:8]
    eligible = (np.isfinite(corners).all(1) & ~(corners == -1).all(1) & ~hidden &
        (corners[:, 0] >= 0) & (corners[:, 0] < width) & (corners[:, 1] >= 0) & (corners[:, 1] < height))
    ids = np.flatnonzero(eligible)
    probability, delta = np.zeros(8), np.zeros((8, 2))
    if len(ids):
        spatial = spatial_features(image, corners, ids, radius, model)
        probability[ids], delta[ids] = predict_feature_rows(rgb[ids], spatial, radius, model)
    threshold = float(model.descriptor['usable_threshold'])
    output = points.copy()
    refined, rejected = [], {}
    for k in range(8):
        if not eligible[k]:
            rejected[str(k)] = 'fixed_H_OOB_or_missing'
        elif probability[k] < threshold:
            rejected[str(k)] = 'below_joint_usable_cutoff'
        else:
            candidate = corners[k] + delta[k]
            if 0 <= candidate[0] < width and 0 <= candidate[1] < height:
                output[k] = candidate
                refined.append(k)
            else:
                rejected[str(k)] = 'RGB_corrected_point_outside_image'
    json_points = [[float(value) if np.isfinite(value) else None for value in point] for point in output]
    return dict(P_usable=probability.tolist(), points=json_points, RGB_delta_px=delta.tolist(),
        eligible=eligible.tolist(), observed_refined_ids=refined, threshold=threshold, raw_appearance=appearance,
        diagnostics=dict(model_family=model.descriptor['family'], model_arrays_sha256=model.descriptor['arrays_sha256'],
            backbone_weights_sha256=model.descriptor['backbone']['weights_sha256'], patch_radius_px=radius,
            coordinate_rejections=rejected, H_unchanged=True, native_order_unchanged=True, center_unchanged=True,
            coordinate_source='frozen_spatial_RGB_ridge', coordinate_bound_px=16.,
            geometry_features_used=False, projected_points_used=False, GT_input=False,
            backbone_trainable_parameters=0))
