"""Subset only frozen F's OpenCV fit correspondences, preserving selector data.

Use serially around a single legacy F invocation. Its original q[9,2],
projection, full-nine-point scores, configuration and solver flags remain
unchanged. This context owns no models, references or experiment I/O.
"""
from __future__ import annotations

from contextlib import contextmanager
import threading

import cv2
import numpy as np


SIGNS = np.asarray([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
], dtype=np.float64)
_ACTIVE = False


class MaskContractError(RuntimeError):
    """An unexpected F call must stop rather than silently return no_pose."""


def _argument(args, kwargs, index, name):
    if len(args) > index:
        if name in kwargs:
            raise MaskContractError('Duplicate OpenCV argument: ' + name)
        return args[index]
    if name not in kwargs:
        raise MaskContractError('Missing OpenCV argument: ' + name)
    return kwargs[name]


def _replace(args, kwargs, index, name, value):
    if len(args) > index:
        args[index] = value
    else:
        kwargs[name] = value


@contextmanager
def correspondence_mask(points9, K, mask8):
    """Yield a mutable call audit; filter canonical eight-corner CV inputs.

    An ALL mask forwards the original args/kwargs untouched. For VIS, both
    SQPnP and LM receive the same retained canonical indices. Model order,
    original image points and K are checked before every call. projectPoints
    is never intercepted, so held-out corners still enter the unchanged
    full-nine-point hypothesis score. No sentinel/NaN masking is performed.
    """
    global _ACTIVE
    q = np.asarray(points9, dtype=np.float64)
    camera = np.asarray(K, dtype=np.float64)
    mask = np.asarray(mask8, dtype=bool).copy()
    if q.shape != (9, 2) or not np.isfinite(q).all():
        raise MaskContractError('Frozen selector needs finite points9[9,2]')
    if (q[:8] == -1).all(axis=1).any():
        raise MaskContractError('This adapter requires eight usable original corners')
    if camera.shape != (3, 3) or not np.isfinite(camera).all():
        raise MaskContractError('Expected finite K[3,3]')
    if mask.shape != (8,) or int(mask.sum()) < 4:
        raise MaskContractError('Apply the locked <4 ALL fallback before F')
    if _ACTIVE:
        raise MaskContractError('Nested/global correspondence contexts are forbidden')
    q = q.copy()
    camera = camera.copy()
    indices = np.flatnonzero(mask)
    original = {name: getattr(cv2, name)
                for name in ('solvePnP', 'solvePnPRefineLM')}
    projection = cv2.projectPoints
    owner = threading.get_ident()
    audit = dict(mask8=mask.tolist(), canonical_indices=indices.tolist(),
        original_correspondences=8, effective_correspondences=int(mask.sum()),
        solvePnP=0, solvePnPRefineLM=0,
        filtered_solvePnP=0, filtered_solvePnPRefineLM=0,
        calls=[], original_points_preserved=None, camera_preserved=None,
        projectPoints_unpatched=None, selector_input='original finite points9',
        selector_score='unchanged formula on original nine points')

    def wrapper(name):
        def fitted(*args, **kwargs):
            if threading.get_ident() != owner:
                raise MaskContractError('Foreign thread entered a serial F context')
            model = np.asarray(_argument(args, kwargs, 0, 'objectPoints'))
            image = np.asarray(_argument(args, kwargs, 1, 'imagePoints'))
            given_K = np.asarray(_argument(args, kwargs, 2, 'cameraMatrix'))
            if model.shape != (8, 3) or not np.isfinite(model).all():
                raise MaskContractError('Expected canonical model[8,3]')
            half = np.abs(model[0]).astype(np.float64)
            if not (half > 0).all() or not np.array_equal(model, SIGNS * half):
                raise MaskContractError('Unexpected 3-D canonical corner order')
            if image.shape != (8, 2) or not np.array_equal(image, q[:8]):
                raise MaskContractError('OpenCV image points differ from original q[:8]')
            if given_K.shape != (3, 3) or not np.array_equal(given_K, camera):
                raise MaskContractError('OpenCV camera differs from the bound K')
            if cv2.projectPoints is not projection:
                raise MaskContractError('Projection must remain the original function')
            audit[name] += 1
            filtered = not bool(mask.all())
            audit['filtered_' + name] += int(filtered)
            audit['calls'].append(dict(function=name,
                canonical_indices=indices.tolist(), input_correspondences=8,
                passed_correspondences=int(mask.sum()), filtered=filtered,
                camera_facing_extents_WH_D=(half * 2).tolist(),
                original_image_points_exact=True, original_camera_exact=True))
            if not filtered:
                return original[name](*args, **kwargs)
            positional, named = list(args), dict(kwargs)
            _replace(positional, named, 0, 'objectPoints',
                     np.ascontiguousarray(model[indices]))
            _replace(positional, named, 1, 'imagePoints',
                     np.ascontiguousarray(image[indices]))
            return original[name](*positional, **named)
        return fitted

    _ACTIVE = True
    try:
        for name in original:
            setattr(cv2, name, wrapper(name))
        yield audit
    finally:
        for name, function in original.items():
            setattr(cv2, name, function)
        _ACTIVE = False
        audit['original_points_preserved'] = bool(np.array_equal(np.asarray(points9), q))
        audit['camera_preserved'] = bool(np.array_equal(np.asarray(K), camera))
        audit['projectPoints_unpatched'] = cv2.projectPoints is projection
        if not all(audit[k] for k in ('original_points_preserved', 'camera_preserved',
                                      'projectPoints_unpatched')):
            raise MaskContractError('Original q/K/projection changed during F')
