"""GT-free control for the inherited DOPE floor-to-8 resize inverse.

Only the nominal-versus-actual resize ratio changes. The canonical belief peak
decoder, +0.4395 offset, padding and feature-coordinate convention are retained.
This is an exact inverse of the resize SIZES in that existing convention; it is
not a new claim of exact receptive-field/half-pixel alignment of the decoder.
"""
from __future__ import annotations

import numpy as np


def resize_control_parameters(width, height, input_shape_chw, pad=100,
                              shortest_side=400):
    """Validate the stored actual input shape and return the affine correction."""
    width, height = int(width), int(height)
    pad, shortest_side = int(pad), int(shortest_side)
    if min(width, height, shortest_side) <= 0 or pad < 0:
        raise ValueError("image/resize sizes must be positive and pad nonnegative")
    if len(input_shape_chw) != 3 or int(input_shape_chw[0]) != 3:
        raise ValueError("input_shape_chw must describe a three-channel tensor")
    channels, nh, nw = map(int, input_shape_chw)
    sc = float(shortest_side) / min(height, width)
    expected_nw = max(8, int(round(width * sc)) & ~7)
    expected_nh = max(8, int(round(height * sc)) & ~7)
    if (nw, nh) != (expected_nw, expected_nh):
        raise ValueError(f"Stored input shape {(nw, nh)} != stage0 expected "
                         f"{(expected_nw, expected_nh)}")
    scale = np.array([sc * width / nw, sc * height / nh], dtype=np.float64)
    translation = pad * (scale - 1)
    return {"scale_xy": scale, "translation_xy": translation,
            "nominal_scale": sc, "input_shape_chw": (channels, nh, nw),
            "pad": pad, "width": width, "height": height}


def correct_legacy_resize_coordinates(points, valid, width, height,
                                       input_shape_chw, pad=100, shortest_side=400):
    """Return (corrected points, effective valid), preserving semantic IDs.

    Inputs are stored ORIGINAL-pixel coordinates from the existing stage0 DOPE
    ``belief_to_orig_pad`` inverse and its stored CHW network tensor shape.
    Shapes are [8/9,2] points and [8] or [8/9] mask. All finite rows are transformed,
    including a center or a low-confidence finite point; the supplied mask is
    unchanged except nonfinite points cannot be valid. Missing rows are copied.
    No GT, predicted DHT line, confidence tuning or image pixels are consulted.
    """
    points = np.asarray(points, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if points.shape not in ((8, 2), (9, 2)):
        raise ValueError("points must have shape [8,2] or [9,2]")
    if valid.shape not in ((8,), (len(points),)):
        raise ValueError("valid must have 8 entries or one entry per point")
    params = resize_control_parameters(width, height, input_shape_chw,
                                        pad=pad, shortest_side=shortest_side)
    finite = np.isfinite(points).all(-1)
    result = points.copy()
    # Apply only axes that change so an unchanged axis remains bit-identical.
    for axis, factor in enumerate(params["scale_xy"]):
        if factor != 1:
            result[finite, axis] = ((points[finite, axis] + pad) * factor - pad)
    return result, valid.copy() & finite[:len(valid)]
