"""Read-only synthetic RGB/point/GT adapter for the bounded PoseFix replay.

The historical feature cache is memory mapped; only requested images and rows
are materialized. Prediction inputs and synthetic supervision stay separate.
This module never writes an artifact or substitutes a GT box for a prediction.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from scripts.research.pallet_posefix_large_error_v1 import core as C


class SourceData:
    def __init__(self, data=None):
        self.data = data if data is not None else C.E.old('train').FeatureDataset(
            C.E.LINE, C.E.LINE / 'cache')
        d = self.data
        self.partitions = np.asarray(d.partitions)
        self.train_rows = np.array(d.train_rows, dtype=np.int64, copy=True)
        assert np.all(self.partitions[self.train_rows] == 'train')
        usable = np.asarray(d.arrays['matched'], dtype=bool) & np.asarray(
            d.arrays['gt_valid'][:, :8], dtype=bool).any(-1)
        self.held_rows = np.flatnonzero(
            (self.partitions == 'heldout') & usable).astype(np.int64)
        self._eligible = np.zeros(len(self.partitions), dtype=bool)
        self._eligible[self.train_rows] = True
        self._eligible[self.held_rows] = True
        assert len(self.train_rows) and len(self.held_rows)
        assert not np.intersect1d(self.train_rows, self.held_rows).size
        self._canvas_affine = C.E.old('features').canvas_affine

    def item(self, row):
        if not isinstance(row, (int, np.integer)):
            raise TypeError('A cache row must be an integer, not a record index')
        row = int(row)
        if row < 0 or row >= len(self._eligible) or not self._eligible[row]:
            raise ValueError('Only matched train/heldout source rows are allowed')
        d = self.data
        a = d.arrays
        record = d.source['records'][int(d.indices[row])]
        assert record['source_kind'] == 'synthetic'
        assert record['partition'] == self.partitions[row]
        assert record['partition'] in ('train', 'heldout')
        path = Path(record['image'])
        if not path.is_absolute():
            path = C.ROOT / path
        binding = C.E.bound(path)
        assert binding['sha256'] == record['image_sha256'], 'Source RGB changed'
        image = cv2.imread(str(path))
        assert image is not None and image.ndim == 3 and image.shape[2] == 3
        shape = np.asarray(record['prepared_shape_hw'], dtype=np.int64)
        assert shape.shape == (2,) and (shape > 0).all()
        assert list(image.shape[:2]) == shape.tolist(), 'Do not pad prepared RGB again'
        input_shape = np.asarray(a['input_shape'][row], dtype=np.int64)
        assert input_shape.shape == (2,) and (input_shape > 0).all()
        gain, offset = self._canvas_affine(shape, input_shape)
        offset = np.asarray(offset, dtype=np.float64)
        assert np.isfinite(gain) and gain > 0 and np.isfinite(offset).all()
        assert offset.shape == (2,)
        if 'gain' in a:
            assert np.isclose(float(a['gain'][row]), gain, rtol=1e-6, atol=1e-9), (
                'Cached/prepared-image gain mismatch', row)

        # The predicted R0 box alone determines crop position and scale.
        points = (a['points'][row] - offset) / gain
        box = ((a['boxes'][row].reshape(2, 2) - offset) / gain).reshape(4)
        matrix = C.axis_aligned_crop_matrix(box)
        assert np.isclose(matrix[0, 0], matrix[1, 1], rtol=0, atol=1e-12)
        crop = cv2.warpAffine(image, matrix[:2], (288, 384),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rgb = (crop[:, :, ::-1].astype(np.float32) - C.MEAN).transpose(2, 0, 1)
        valid = a['point_valid'][row].astype(bool) & np.isfinite(points).all(-1)
        input_points = C.transform_points(
            np.where(valid[:, None], points, 0), matrix).astype(np.float32)

        gt = (a['gt_points'][row] - offset) / gain
        gt_valid = a['gt_valid'][row].astype(bool) & np.isfinite(gt).all(-1)
        gt_valid[8] = False
        target = C.transform_points(
            np.where(gt_valid[:, None], gt, 0), matrix).astype(np.float32)
        support = gt_valid & (target >= 0).all(-1)
        support &= (target[:, 0] < 288) & (target[:, 1] < 384)
        assert rgb.shape == (3, 384, 288) and rgb.dtype == np.float32
        assert input_points.shape == target.shape == (9, 2)
        assert valid.shape == support.shape == (9,) and not support[8]
        return dict(rgb=rgb, points=input_points, valid=valid,
                    target=target, target_valid=support, matrix=matrix,
                    original_points=points, original_gt=gt,
                    original_gt_valid=gt_valid, box=box,
                    bbox_diagonal=float(np.linalg.norm(box[2:] - box[:2])),
                    id=record['id'], partition=record['partition'],
                    image_binding=binding, row=np.int64(row),
                    source_id=record['id'], source_partition=record['partition'])
