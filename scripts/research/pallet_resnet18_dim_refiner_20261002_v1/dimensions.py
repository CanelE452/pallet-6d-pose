"""Canonical fixed-axis W,D,H features shared with the frozen FULL estimator."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .common import DIMENSION_NORMALIZATION, DIMENSION_ARMS, HEAD_ARMS


FEATURE_NAMES = ("logW", "logD", "logH", "log(W/D)", "log(H/sqrt(WD))")


def dimension_features(dimensions) -> np.ndarray:
    values = np.asarray(dimensions, np.float64)
    if values.ndim < 1 or values.shape[-1] != 3:
        raise ValueError("Canonical dimensions must end in [W,D,H]")
    if not np.isfinite(values).all() or not (values > 0).all():
        raise ValueError("Canonical W,D,H must be finite positive metres")
    log = np.log(values)
    return np.stack(
        (log[..., 0], log[..., 1], log[..., 2], log[..., 0] - log[..., 1],
         log[..., 2] - 0.5 * (log[..., 0] + log[..., 1])), axis=-1)


def load_normalization(path: Path | str = DIMENSION_NORMALIZATION) -> dict:
    payload = json.loads(Path(path).read_text())
    if payload.get("input") != list(FEATURE_NAMES):
        raise ValueError("Dimension feature order differs from the frozen FULL estimator")
    if payload.get("rows") != 55980 or payload.get("calibration_selection_DEV_included") is not False:
        raise ValueError("Dimension normalization must use source TRAIN55980 only")
    mean = np.asarray(payload.get("mean"), np.float64)
    scale = np.asarray(payload.get("scale"), np.float64)
    if mean.shape != (5,) or scale.shape != (5,) or not np.isfinite(mean).all():
        raise ValueError("Invalid dimension normalization statistics")
    if not np.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("Dimension normalization scales must be finite and positive")
    return payload


def normalized_context(dimensions, normalization: dict) -> np.ndarray:
    mean = np.asarray(normalization["mean"], np.float64)
    scale = np.asarray(normalization["scale"], np.float64)
    result = (dimension_features(dimensions) - mean) / scale
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite normalized dimension context")
    return result.astype(np.float32)


def context_for_arm(context, arm: str):
    """Return the head input; CONSTANT is forced to zero inside this function."""
    if arm not in HEAD_ARMS:
        raise ValueError(f"Unknown head arm: {arm}")
    if arm not in DIMENSION_ARMS:
        return None
    if isinstance(context, torch.Tensor):
        if context.ndim != 2 or context.shape[1] != 5 or not torch.isfinite(context).all():
            raise ValueError("Head dimension context must be finite [B,5]")
        return torch.zeros_like(context) if arm == "P5_CONSTANT" else context
    value = np.asarray(context)
    if value.ndim != 2 or value.shape[1] != 5 or not np.isfinite(value).all():
        raise ValueError("Head dimension context must be finite [B,5]")
    return np.zeros_like(value) if arm == "P5_CONSTANT" else value

