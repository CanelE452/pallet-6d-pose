"""Full-image CPU input and heatmap contract for the fixed ResNet18 baseline.

No detector crop, YOLO import, model forward or reference pose is used here.
Source coordinates refer to the existing reflect100 prepared canvas; real
coordinates refer to the supplied native image. All transforms record actual
rounded resize dimensions. Gaussian/decoder grid index zero is network pixel
zero: no implicit +0.5, UDP transform or test-time flip is applied.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[3]
SOURCE_MANIFEST = ROOT / 'data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json'
INPUT_HW = (384, 512)
HEATMAP_HW = (96, 128)
STRIDE = 4
REFLECT_PAD = 100
LETTERBOX_VALUE = 128
SIGMA = 2.
TRUNCATE_SIGMAS = 3.
CONFIDENCE_THRESHOLD = .1
MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)


def contract():
    return dict(schema='resnet18_full_image_input_v1', input_hw=list(INPUT_HW),
        heatmap_hw=list(HEATMAP_HW), stride=STRIDE, reflect_pad_px=REFLECT_PAD,
        source_additional_padding=0, real_border='BORDER_REFLECT_101',
        resize='INTER_LINEAR; aspect fit, round actual width/height; exact per-axis inverse',
        letterbox='center, left/top floor, right/bottom remainder',
        letterbox_value=LETTERBOX_VALUE, color='RGB', normalization_dtype='float32',
        mean=MEAN.tolist(), std=STD.tolist(), sigma_heatmap=SIGMA,
        gaussian_centers='continuous network xy / 4, no half-pixel offset',
        gaussian_truncation='square abs(dx),abs(dy) <= 3*sigma',
        supervision='visibility>0, finite non-sentinel xy, and continuous center inside [0,W)x[0,H)',
        mse='mean_frames(sum9(mask * 0.5 * mean_pixels((prediction-target)^2)) / 9)',
        decoder='channel argmax; if both coordinates are interior 1..size-2, add .25*sign(right-left,down-up)',
        confidence_threshold=CONFIDENCE_THRESHOLD, threshold_inclusive=True,
        predicted_box='hull of >=3 finite corner channels0..7; each original-coordinate extent>1',
        corner_order='unchanged source channels0..7 plus center8',
        geometric_augmentation=False, GT_crops=False, YOLO_inputs=False,
        checkpoint_selection='fixed final epoch60; no real-based threshold selection')


def transform_points(points, affine):
    points = np.asarray(points, np.float64)
    affine = np.asarray(affine, np.float64)
    assert points.ndim == 2 and points.shape[1] == 2 and affine.shape == (3, 3)
    assert np.isfinite(affine).all()
    result = np.full(points.shape, np.nan, np.float64)
    valid = np.isfinite(points).all(-1)
    homogeneous = np.c_[points[valid], np.ones(int(valid.sum()))] @ affine.T
    assert np.all(homogeneous[:, 2] != 0)
    result[valid] = homogeneous[:, :2] / homogeneous[:, 2, None]
    return result


def prepare(image, source_pre_padded=False):
    """Return a normalized CPU tensor and input/native/canvas coordinate metadata."""
    image = np.asarray(image)
    assert image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8
    h, w = image.shape[:2]
    assert h > 0 and w > 0
    if source_pre_padded:
        assert min(h, w) > 2 * REFLECT_PAD
        native_hw = [h - 2 * REFLECT_PAD, w - 2 * REFLECT_PAD]
        canvas = image; added = 0
    else:
        native_hw = [h, w]
        canvas = cv2.copyMakeBorder(image, REFLECT_PAD, REFLECT_PAD,
                                   REFLECT_PAD, REFLECT_PAD, cv2.BORDER_REFLECT_101)
        added = REFLECT_PAD
    ch, cw = canvas.shape[:2]
    ih, iw = INPUT_HW
    ratio = min(iw / cw, ih / ch)
    resized_w = min(iw, max(1, int(round(cw * ratio))))
    resized_h = min(ih, max(1, int(round(ch * ratio))))
    left = (iw - resized_w) // 2; top = (ih - resized_h) // 2
    right = iw - resized_w - left; bottom = ih - resized_h - top
    raster = cv2.resize(canvas, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    raster = cv2.copyMakeBorder(raster, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=(LETTERBOX_VALUE,) * 3)
    rgb = cv2.cvtColor(raster, cv2.COLOR_BGR2RGB)
    normalized = (rgb.astype(np.float32) / np.float32(255.) - MEAN) / STD
    assert normalized.dtype == np.float32 and normalized.shape == (ih, iw, 3)
    tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))
    sx, sy = resized_w / cw, resized_h / ch
    affine = np.array([[sx, 0., left + added*sx],
                       [0., sy, top + added*sy], [0., 0., 1.]], np.float64)
    return dict(tensor=tensor, affine_input_to_net=affine,
        affine_net_to_input=np.linalg.inv(affine), input_shape=[3, ih, iw],
        input_hw=[h, w], native_hw=native_hw, canvas_hw=[ch, cw],
        resized_hw=[resized_h, resized_w], padding_ltrb=[left, top, right, bottom],
        additional_padding=added, source_pre_padded=bool(source_pre_padded),
        existing_source_padding=REFLECT_PAD if source_pre_padded else 0)


def gaussian_targets(points_net, eligible):
    """Continuous targets; zero heatmap for invalid/out-of-grid points."""
    points = np.asarray(points_net, np.float64)
    eligible = np.asarray(eligible, bool)
    assert points.shape == (9, 2) and eligible.shape == (9,)
    centers = points / STRIDE
    h, w = HEATMAP_HW
    finite = np.isfinite(centers).all(-1)
    inside = finite & (centers[:, 0] >= 0) & (centers[:, 0] < w)
    inside &= (centers[:, 1] >= 0) & (centers[:, 1] < h)
    valid = eligible & inside
    heatmaps = np.zeros((9, h, w), np.float32)
    yy = np.arange(h, dtype=np.float64)[:, None]
    xx = np.arange(w, dtype=np.float64)[None, :]
    radius = SIGMA * TRUNCATE_SIGMAS
    for k in np.flatnonzero(valid):
        dx = xx - centers[k, 0]; dy = yy - centers[k, 1]
        support = (np.abs(dx) <= radius) & (np.abs(dy) <= radius)
        value = np.exp(-(dx*dx + dy*dy) / (2*SIGMA*SIGMA))
        heatmaps[k] = np.where(support, value, 0.).astype(np.float32)
    return heatmaps, valid, centers, inside


def targets(record, prepared):
    """Transform existing normalized source labels; no candidate matching filter."""
    assert prepared['source_pre_padded'] is True
    assert len(record['targets']) == 1
    assert prepared['input_hw'] == list(record['prepared_shape_hw'])
    normalized = np.asarray(record['targets'][0]['keypoints_normalized'], np.float64)
    assert normalized.shape == (9, 3)
    visible = normalized[:, 2] > 0
    finite = np.isfinite(normalized[:, :2]).all(-1) & ~(normalized[:, :2] == -1).all(-1)
    original = normalized[:, :2] * np.array(record['prepared_shape_hw'][::-1], np.float64)
    original[~finite] = np.nan
    points = transform_points(original, prepared['affine_input_to_net'])
    heatmaps, valid, centers, inside = gaussian_targets(points, visible & finite)
    counts = dict(total_channels=9, visible_channels=int(visible.sum()),
        visible_finite_channels=int((visible & finite).sum()),
        outside_support_channels=int((visible & finite & ~inside).sum()),
        supervised_channels=int(valid.sum()),
        all_masked_frames=int(not valid.any()), denominator_channels=9)
    return dict(heatmaps=heatmaps, target_valid=valid, points_net=points,
        points_original=original, centers_grid=centers, visibility_mask=visible,
        finite_label_mask=finite, inside_heatmap=inside, counts=counts)


def masked_mse(prediction, target, target_valid):
    """Official factor0.5; full nine-channel and full batch denominators."""
    assert prediction.ndim == 4 and prediction.shape[1:] == (9, *HEATMAP_HW)
    assert target.shape == prediction.shape and target_valid.shape == prediction.shape[:2]
    mask = target_valid.to(dtype=torch.bool, device=prediction.device)
    squared = (prediction - target.to(prediction)).square().mean((-2, -1)) * .5
    return torch.where(mask, squared, torch.zeros_like(squared)).sum(-1).div(9).mean()


def predicted_box(points, valid):
    points = np.asarray(points, np.float64); valid = np.asarray(valid, bool)
    assert points.shape == (9, 2) and valid.shape == (9,)
    assert np.array_equal(np.isfinite(points).all(-1), valid)
    corners = points[:8][valid[:8]]
    if len(corners) < 3:
        return None
    box = np.r_[corners.min(0), corners.max(0)]
    return box if np.all(box[2:] - box[:2] > 1.) else None


def decode(heatmaps, prepared):
    """Fixed argmax/quarter-pixel decoder, followed by the exact recorded inverse."""
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.detach().cpu().numpy()
    maps = np.asarray(heatmaps, np.float32)
    assert maps.shape == (9, *HEATMAP_HW) and np.isfinite(maps).all()
    h, w = HEATMAP_HW
    index = maps.reshape(9, -1).argmax(-1)
    confidence = maps.reshape(9, -1)[np.arange(9), index].copy()
    valid = confidence >= np.float32(CONFIDENCE_THRESHOLD)
    grid = np.full((9, 2), np.nan, np.float64)
    for k in np.flatnonzero(valid):
        y, x = divmod(int(index[k]), w)
        grid[k] = [x, y]
        if 1 <= x <= w-2 and 1 <= y <= h-2:
            gradient = np.array([float(maps[k, y, x+1])-float(maps[k, y, x-1]),
                                 float(maps[k, y+1, x])-float(maps[k, y-1, x])])
            grid[k] += .25 * np.sign(gradient)
    points = grid * STRIDE
    original = transform_points(points, prepared['affine_net_to_input'])
    box_original = predicted_box(original, valid)
    box_net = None
    if box_original is not None:
        corners = points[:8][valid[:8]]
        box_net = np.r_[corners.min(0), corners.max(0)]
    return dict(points_net=points, points_original=original, valid=valid,
        confidence=confidence, score=float(confidence[:8].max()),
        bbox_original=box_original,
        bbox_net=box_net, grid_points=grid,
        affine_input_to_net=prepared['affine_input_to_net'],
        affine_net_to_input=prepared['affine_net_to_input'],
        input_shape=prepared['input_shape'], original_hw=prepared['input_hw'])


def load_image(record):
    path = Path(record['image'])
    if not path.is_absolute():
        path = ROOT / path
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == record['image_sha256'], record['id']
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert image is not None and list(image.shape[:2]) == record['prepared_shape_hw']
    return image


def load_records(manifest=SOURCE_MANIFEST):
    path = Path(manifest)
    payload = json.loads(path.read_text())
    records = payload['records']
    assert len(records) == 60000 and len({r['id'] for r in records}) == 60000
    assert payload['input_recipe']['prepared_reflect101_pad_px'] == REFLECT_PAD
    for index, record in enumerate(records):
        assert record['index'] == index and record['source_kind'] == 'synthetic'
        assert record['reflect_pad_px'] == REFLECT_PAD
        assert [int(v)+2*REFLECT_PAD for v in record['raw_shape_hw']] == record['prepared_shape_hw']
        assert len(record['targets']) == 1
    return records


class SourceDataset(Dataset):
    """No baseline-derived matching filter: every requested partition row remains."""
    def __init__(self, manifest=SOURCE_MANIFEST, partition='train'):
        self.records = load_records(manifest)
        allowed = {'train', 'calibration', 'selection', 'heldout', 'all'}
        assert partition in allowed
        self.partition = partition
        self.indices = np.array([i for i, row in enumerate(self.records)
            if partition == 'all' or row['partition'] == partition], np.int64)
        expected = dict(train=55980, calibration=1004, selection=1031, heldout=1985, all=60000)
        assert len(self.indices) == expected[partition]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, local_index):
        index = int(self.indices[local_index]); record = self.records[index]
        prepared = prepare(load_image(record), source_pre_padded=True)
        target = targets(record, prepared)
        return dict(image=prepared['tensor'], heatmaps=torch.from_numpy(target['heatmaps']),
            target_valid=torch.from_numpy(target['target_valid']), index=index, id=record['id'],
            points_net=torch.from_numpy(target['points_net']),
            points_original=torch.from_numpy(target['points_original']),
            affine_input_to_net=torch.from_numpy(prepared['affine_input_to_net']),
            affine_net_to_input=torch.from_numpy(prepared['affine_net_to_input']),
            original_hw=torch.tensor(prepared['input_hw'], dtype=torch.int64),
            counts=target['counts'])


def worker_init_fn(worker_id):
    """Prevent CPU thread multiplication; DataLoader owns seeded worker RNGs."""
    del worker_id
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
