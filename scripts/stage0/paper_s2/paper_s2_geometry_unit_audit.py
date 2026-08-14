#!/usr/bin/env python3
"""Numerical gates for the canonical PAPER_S2 geometry stack.

This is an audit-only harness.  It does not train a model or modify data/GT.
It exercises both geometry implementations that coexist in this repository:

* ``SpatialSoftArgmax2D`` + legacy ``BPnP`` from ``geo_loss_bpnp.py``.
* ``LocalSoftArgmax2D`` + unrolled-GN ``DiffPnP3DLoss`` used by PAPER_S2.

The JSON output is deliberately machine-readable so later frozen-inference
diagnostics can quote the exact gate results rather than prose summaries.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import scipy
import torch


ROOT = Path(__file__).resolve().parents[3]
TRAIN_DIR = ROOT / "Deep_Object_Pose" / "train"
sys.path.insert(0, str(TRAIN_DIR))

from diffpnp3d_loss import (  # noqa: E402
    DiffPnP3DLoss,
    LocalSoftArgmax2D,
    _jac_batch,
    _project_batch,
)
from geo_loss_bpnp import BPnP, SpatialSoftArgmax2D  # noqa: E402


DEFAULT_OUT = (
    ROOT
    / "data"
    / "pallet"
    / "results"
    / "paper_s2_scratch_diffpnp"
    / "diagnostic_audit"
    / "unit_audit.json"
)
CANONICAL_CKPT = ROOT / "weights" / "paper_s2_stageB" / "net_epoch_0057.pth"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _gaussian_value_map(
    height: int,
    width: int,
    mean_xy: tuple[float, float],
    covariance: np.ndarray,
    *,
    amplitude: float = 1.0,
    background: float = 0.0,
    log_density: bool = False,
) -> torch.Tensor:
    yy, xx = np.mgrid[:height, :width]
    delta = np.stack([xx - mean_xy[0], yy - mean_xy[1]], axis=-1)
    inv = np.linalg.inv(np.asarray(covariance, dtype=np.float64))
    exponent = -0.5 * np.einsum("...i,ij,...j->...", delta, inv, delta)
    if log_density:
        values = exponent + math.log(max(amplitude, 1.0e-12))
    else:
        values = background + amplitude * np.exp(exponent)
    return torch.from_numpy(values).to(torch.float64)


def _moment_from_logits(
    heatmap: torch.Tensor, temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean (x,y) and full 2x2 covariance from normalized logits."""
    height, width = heatmap.shape[-2:]
    weights = torch.softmax(
        heatmap.reshape(-1) / float(temperature), dim=0
    ).reshape(height, width)
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=heatmap.dtype, device=heatmap.device),
        torch.arange(width, dtype=heatmap.dtype, device=heatmap.device),
        indexing="ij",
    )
    mean = torch.stack([(weights * xx).sum(), (weights * yy).sum()])
    dx = xx - mean[0]
    dy = yy - mean[1]
    covariance = torch.stack(
        [
            torch.stack([(weights * dx * dx).sum(), (weights * dx * dy).sum()]),
            torch.stack([(weights * dx * dy).sum(), (weights * dy * dy).sum()]),
        ]
    )
    return mean, covariance


def _rotation_covariance(
    sigma_major: float, sigma_minor: float, angle_deg: float
) -> np.ndarray:
    theta = math.radians(angle_deg)
    rotation = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=np.float64,
    )
    return rotation @ np.diag([sigma_major**2, sigma_minor**2]) @ rotation.T


def _softargmax_cases() -> tuple[dict[str, Any], dict[str, Any]]:
    height = width = 50
    cases: list[dict[str, Any]] = []

    def add(
        name: str,
        heatmap: torch.Tensor,
        target_xy: tuple[float, float] | None,
        expected_covariance: np.ndarray | None = None,
        strict: bool = False,
        note: str = "",
    ) -> None:
        cases.append(
            {
                "name": name,
                "heatmap": heatmap,
                "target_xy": target_xy,
                "expected_covariance": expected_covariance,
                "strict": strict,
                "note": note,
            }
        )

    add(
        "single_gaussian_center",
        _gaussian_value_map(height, width, (25.0, 25.0), np.diag([2.0**2] * 2)),
        (25.0, 25.0),
    )
    add(
        "fractional_center",
        _gaussian_value_map(height, width, (20.35, 29.65), np.diag([1.5**2] * 2)),
        (20.35, 29.65),
    )
    add(
        "x_elongated",
        _gaussian_value_map(height, width, (24.0, 23.0), np.diag([4.0**2, 1.5**2])),
        (24.0, 23.0),
    )
    add(
        "y_elongated",
        _gaussian_value_map(height, width, (24.0, 23.0), np.diag([1.5**2, 4.0**2])),
        (24.0, 23.0),
    )
    rotated = _rotation_covariance(4.0, 1.5, 45.0)
    add(
        "rotated_45deg",
        _gaussian_value_map(height, width, (24.0, 23.0), rotated),
        (24.0, 23.0),
    )
    bimodal = _gaussian_value_map(
        height, width, (18.0, 25.0), np.diag([1.2**2] * 2)
    ) + _gaussian_value_map(
        height, width, (32.0, 25.0), np.diag([1.2**2] * 2)
    )
    add(
        "bimodal_equal",
        bimodal,
        (25.0, 25.0),
        note="global mean lies between modes; local decoder selects one mode",
    )
    add(
        "boundary_truncated",
        _gaussian_value_map(height, width, (0.5, 0.5), np.diag([1.5**2] * 2)),
        (0.5, 0.5),
        note="records unavoidable truncation/clamped-window boundary bias",
    )
    add(
        "low_peak_background",
        _gaussian_value_map(
            height,
            width,
            (25.0, 25.0),
            np.diag([2.0**2] * 2),
            amplitude=0.05,
            background=0.05,
        ),
        (25.0, 25.0),
        note="MSE-style value map interpreted as logits",
    )
    add(
        "uniform",
        torch.zeros((height, width), dtype=torch.float64),
        (24.5, 24.5),
        note="uniform logits have a defined global mean but no localized peak",
    )
    add(
        "negative_raw_output",
        _gaussian_value_map(height, width, (25.0, 25.0), np.diag([2.0**2] * 2))
        - 0.2,
        (25.0, 25.0),
        note="softmax should be invariant to a constant negative shift",
    )

    # A log-Gaussian is the mathematically strict covariance fixture because
    # both production decoders normalize *logits* with softmax.
    strict_cov = _rotation_covariance(1.2, 0.8, 32.0)
    add(
        "strict_log_gaussian_covariance",
        _gaussian_value_map(
            height,
            width,
            (23.4, 18.7),
            strict_cov,
            log_density=True,
        ),
        (23.4, 18.7),
        expected_covariance=strict_cov,
        strict=True,
        note="known Gaussian probability encoded as logits; temperature=1",
    )

    global_results: list[dict[str, Any]] = []
    local_results: list[dict[str, Any]] = []
    temperatures = [0.05, 0.1, 0.5, 1.0, 2.0]
    global_decoder = SpatialSoftArgmax2D(temperature=1.0).to(dtype=torch.float64)
    local_decoder = LocalSoftArgmax2D(
        window=7,
        temperature=1.0,
        orig_size=(640, 480),
        belief_size=(50, 50),
    ).to(dtype=torch.float64)

    for case in cases:
        source = case["heatmap"].clone().reshape(1, 1, height, width)
        source.requires_grad_(True)

        global_decoder.temperature = 1.0
        global_xy, global_sigma = global_decoder(source, return_sigma=True)
        global_mean, global_covariance = _moment_from_logits(source[0, 0], 1.0)
        global_grad = torch.autograd.grad(
            global_xy.sum(), source, retain_graph=False
        )[0]

        local_decoder.temperature = 1.0
        local_xy_orig, confidence = local_decoder(source)
        local_xy = local_xy_orig.clone()
        local_xy[..., 0] /= local_decoder.scale_x
        local_xy[..., 1] /= local_decoder.scale_y
        local_covariance = torch.stack(
            [
                torch.stack(
                    [confidence["var_x"][0, 0], confidence["cov_xy"][0, 0]]
                ),
                torch.stack(
                    [confidence["cov_xy"][0, 0], confidence["var_y"][0, 0]]
                ),
            ]
        )
        local_grad = torch.autograd.grad(local_xy_orig.sum(), source)[0]

        target = case["target_xy"]
        global_error = (
            float(torch.abs(global_mean - torch.tensor(target)).mean())
            if target is not None
            else None
        )
        local_error = (
            float(torch.abs(local_xy[0, 0] - torch.tensor(target)).mean())
            if target is not None
            else None
        )
        expected_covariance = case["expected_covariance"]

        def covariance_error(observed: torch.Tensor) -> float | None:
            if expected_covariance is None:
                return None
            expected = torch.from_numpy(expected_covariance).to(observed)
            return float(
                torch.linalg.norm(observed - expected)
                / torch.linalg.norm(expected).clamp_min(1.0e-12)
            )

        temperature_rows = []
        for temperature in temperatures:
            global_decoder.temperature = temperature
            gxy = global_decoder(source.detach())
            local_decoder.temperature = temperature
            lxy, _ = local_decoder(source.detach())
            lxy_belief = lxy.clone()
            lxy_belief[..., 0] /= local_decoder.scale_x
            lxy_belief[..., 1] /= local_decoder.scale_y
            temperature_rows.append(
                {
                    "temperature": temperature,
                    "global_xy_belief": gxy[0, 0],
                    "local_xy_belief": lxy_belief[0, 0],
                }
            )

        base = {
            "name": case["name"],
            "target_xy_belief": target,
            "strict": case["strict"],
            "note": case["note"],
            "temperatures": temperature_rows,
        }
        global_results.append(
            {
                **base,
                "mean_xy_belief": global_mean,
                "sigma_radial_belief": global_sigma[0, 0],
                "covariance_belief": global_covariance,
                "mean_abs_error_belief_px": global_error,
                "covariance_relative_error": covariance_error(global_covariance),
                "covariance_eigenvalues": torch.linalg.eigvalsh(global_covariance),
                "gradient_finite": bool(torch.isfinite(global_grad).all()),
                "gradient_nonzero": bool(global_grad.abs().max() > 0),
            }
        )
        local_results.append(
            {
                **base,
                "mean_xy_belief": local_xy[0, 0],
                "mean_xy_orig": local_xy_orig[0, 0],
                "covariance_belief": local_covariance,
                "mean_abs_error_belief_px": local_error,
                "covariance_relative_error": covariance_error(local_covariance),
                "covariance_eigenvalues": torch.linalg.eigvalsh(local_covariance),
                "gradient_finite": bool(torch.isfinite(local_grad).all()),
                "gradient_nonzero": bool(local_grad.abs().max() > 0),
                "covariance_requires_grad": bool(local_covariance.requires_grad),
            }
        )

    strict_global = next(row for row in global_results if row["strict"])
    strict_local = next(row for row in local_results if row["strict"])
    interior_unimodal = {
        "single_gaussian_center",
        "fractional_center",
        "x_elongated",
        "y_elongated",
        "rotated_45deg",
        "low_peak_background",
        "negative_raw_output",
    }

    def operational_errors(
        rows: list[dict[str, Any]], decoder_key: str, temperature: float
    ) -> dict[str, Any]:
        per_case: dict[str, float] = {}
        for row in rows:
            if row["name"] not in interior_unimodal:
                continue
            target = torch.tensor(row["target_xy_belief"], dtype=torch.float64)
            selected = next(
                item
                for item in row["temperatures"]
                if item["temperature"] == temperature
            )
            observed = torch.as_tensor(selected[decoder_key], dtype=torch.float64)
            per_case[row["name"]] = float(torch.abs(observed - target).mean())
        failures = sorted(name for name, error in per_case.items() if error > 0.1)
        return {
            "temperature": temperature,
            "representation": "MSE-style Gaussian value map used by DOPE targets",
            "interior_unimodal_mean_abs_error_belief_px": per_case,
            "threshold_belief_px": 0.1,
            "failures": failures,
            "pass": not failures,
        }

    global_operational = operational_errors(
        global_results, "global_xy_belief", 1.0
    )
    local_operational = operational_errors(
        local_results, "local_xy_belief", 0.1
    )
    global_gate = {
        "mean_mae_le_0p1": strict_global["mean_abs_error_belief_px"] <= 0.1,
        "covariance_rel_le_0p05": strict_global["covariance_relative_error"] <= 0.05,
        "psd": min(strict_global["covariance_eigenvalues"]) >= -1.0e-10,
        "finite_gradients_all_cases": all(
            row["gradient_finite"] for row in global_results
        ),
    }
    local_gate = {
        "mean_mae_le_0p1": strict_local["mean_abs_error_belief_px"] <= 0.1,
        "covariance_rel_le_0p05": strict_local["covariance_relative_error"] <= 0.05,
        "psd": min(strict_local["covariance_eigenvalues"]) >= -1.0e-10,
        "finite_gradients_all_cases": all(
            row["gradient_finite"] for row in local_results
        ),
    }

    # Verify the covariance coordinate transform explicitly: C_orig=S C_bel S^T.
    covariance_belief = torch.tensor(
        [[2.0, 0.4], [0.4, 3.0]], dtype=torch.float64
    )
    scale = torch.diag(
        torch.tensor(
            [local_decoder.scale_x, local_decoder.scale_y], dtype=torch.float64
        )
    )
    covariance_orig = scale @ covariance_belief @ scale.T
    expected_orig = torch.tensor(
        [
            [
                2.0 * local_decoder.scale_x**2,
                0.4 * local_decoder.scale_x * local_decoder.scale_y,
            ],
            [
                0.4 * local_decoder.scale_x * local_decoder.scale_y,
                3.0 * local_decoder.scale_y**2,
            ],
        ],
        dtype=torch.float64,
    )
    scaling_check = {
        "scale_x": local_decoder.scale_x,
        "scale_y": local_decoder.scale_y,
        "covariance_orig": covariance_orig,
        "expected_orig": expected_orig,
        "max_abs_error": float(torch.max(torch.abs(covariance_orig - expected_orig))),
        "pass": bool(torch.allclose(covariance_orig, expected_orig)),
    }

    return (
        {
            "implementation": "SpatialSoftArgmax2D (global softmax over raw logits)",
            "cases": global_results,
            "strict_gate": global_gate,
            "operational_gate": global_operational,
        },
        {
            "implementation": (
                "LocalSoftArgmax2D (argmax-selected 7x7 softmax; PAPER_S2)"
            ),
            "cases": local_results,
            "strict_gate": local_gate,
            "operational_gate": local_operational,
            "covariance_scale_check": scaling_check,
        },
    )


def _box_points(
    width: float = 1.1, depth: float = 1.3, height: float = 0.12
) -> np.ndarray:
    """Eight fixed correspondences; x/y/z are width/height/depth axes."""
    x = width / 2.0
    y = height / 2.0
    z = depth / 2.0
    return np.asarray(
        [
            [-x, -y, -z],
            [x, -y, -z],
            [x, y, -z],
            [-x, y, -z],
            [-x, -y, z],
            [x, -y, z],
            [x, y, z],
            [-x, y, z],
        ],
        dtype=np.float64,
    )


def _rotation_error_deg(rvec_a: np.ndarray, rvec_b: np.ndarray) -> float:
    rotation_a = cv2.Rodrigues(np.asarray(rvec_a, dtype=np.float64))[0]
    rotation_b = cv2.Rodrigues(np.asarray(rvec_b, dtype=np.float64))[0]
    relative = rotation_a @ rotation_b.T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def _yaw_deg(rvec: np.ndarray) -> float:
    rotation = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]
    # OpenCV camera-frame yaw about +Y.  Only paired differences matter here.
    return math.degrees(math.atan2(rotation[0, 2], rotation[2, 2]))


def _known_pose_fixture(
    *, height: float = 0.12
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = _box_points(height=height)
    intrinsics = np.asarray(
        [[615.0, 0.0, 320.0], [0.0, 614.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rvec = np.asarray([0.08, -0.18, 0.12], dtype=np.float64)
    tvec = np.asarray([0.04, -0.03, 4.0], dtype=np.float64)
    image_points = cv2.projectPoints(
        points, rvec, tvec, intrinsics, None
    )[0].reshape(-1, 2)
    return points, intrinsics, rvec, tvec, image_points


def _legacy_bpnp_audit() -> dict[str, Any]:
    points, intrinsics, rvec_gt, tvec_gt, image_points = _known_pose_fixture()
    kp2d = torch.tensor(image_points[None], dtype=torch.float32, requires_grad=True)
    kp3d = torch.tensor(points, dtype=torch.float32)
    camera = torch.tensor(intrinsics, dtype=torch.float32)
    pose, valid = BPnP.apply(kp2d, kp3d, camera, 1.0e-4)
    pose_np = pose.detach().cpu().numpy()[0].astype(np.float64)

    reprojection = cv2.projectPoints(
        points, pose_np[:3], pose_np[3:], intrinsics, None
    )[0].reshape(-1, 2)
    reprojection_error = np.linalg.norm(reprojection - image_points, axis=1)

    coefficient = torch.tensor(
        [[0.7, -0.3, 0.5, 0.2, -0.4, 0.6]], dtype=pose.dtype
    )
    scalar = (pose * coefficient).sum()
    scalar.backward()
    automatic = kp2d.grad.detach().cpu().numpy().reshape(-1)

    def objective(flat_points: np.ndarray) -> float:
        query = torch.tensor(
            flat_points.reshape(1, -1, 2), dtype=torch.float32
        )
        output, _ = BPnP.apply(query, kp3d, camera, 1.0e-4)
        return float((output * coefficient).sum())

    base = image_points.reshape(-1).copy()

    def numeric_gradient(epsilon: float) -> np.ndarray:
        result = np.zeros_like(automatic, dtype=np.float64)
        for index in range(base.size):
            plus = base.copy()
            minus = base.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            result[index] = (
                objective(plus) - objective(minus)
            ) / (2.0 * epsilon)
        return result

    finite_difference_sweep = []
    finite_difference = None
    relative_error = None
    epsilon = 1.0e-2
    for trial_epsilon in (1.0e-1, 5.0e-2, 1.0e-2, 5.0e-3, 1.0e-3):
        trial_gradient = numeric_gradient(trial_epsilon)
        denominator = max(
            float(np.linalg.norm(automatic)),
            float(np.linalg.norm(trial_gradient)),
            1.0e-12,
        )
        trial_error = float(
            np.linalg.norm(automatic - trial_gradient) / denominator
        )
        finite_difference_sweep.append(
            {
                "epsilon_px": trial_epsilon,
                "relative_l2_error": trial_error,
                "numeric_gradient_norm": float(np.linalg.norm(trial_gradient)),
            }
        )
        if trial_epsilon == epsilon:
            finite_difference = trial_gradient
            relative_error = trial_error
    assert finite_difference is not None and relative_error is not None
    best_relative_error = min(
        row["relative_l2_error"] for row in finite_difference_sweep
    )
    per_keypoint_gradient = np.linalg.norm(automatic.reshape(-1, 2), axis=1)

    return {
        "implementation": (
            "BPnP autograd.Function: OpenCV SOLVEPNP_EPNP forward + "
            "hand-written damped implicit backward"
        ),
        "valid": bool(valid.item() > 0.5),
        "pose_pred": pose_np,
        "pose_gt": np.concatenate([rvec_gt, tvec_gt]),
        "reprojection_median_px": float(np.median(reprojection_error)),
        "rotation_error_deg": _rotation_error_deg(pose_np[:3], rvec_gt),
        "yaw_error_deg": abs(_yaw_deg(pose_np[:3]) - _yaw_deg(rvec_gt)),
        "translation_error_m": float(np.linalg.norm(pose_np[3:] - tvec_gt)),
        "finite_difference": {
            "epsilon_px": epsilon,
            "automatic_gradient": automatic,
            "numeric_gradient": finite_difference,
            "relative_l2_error": relative_error,
            "epsilon_sweep": finite_difference_sweep,
            "best_relative_l2_error": best_relative_error,
            "per_keypoint_gradient_norm": per_keypoint_gradient,
        },
        "gate": {
            "oracle_reprojection_le_1px": float(np.median(reprojection_error)) <= 1.0,
            "oracle_yaw_le_1deg": abs(
                _yaw_deg(pose_np[:3]) - _yaw_deg(rvec_gt)
            )
            <= 1.0,
            "finite_difference_rel_le_1e_2": best_relative_error <= 1.0e-2,
            "finite": bool(
                np.isfinite(automatic).all() and np.isfinite(finite_difference).all()
            ),
        },
    }


def _unrolled_pose(
    pred_xy: torch.Tensor,
    points: torch.Tensor,
    camera: torch.Tensor,
    rotation_gt: torch.Tensor,
    translation_gt: torch.Tensor,
    *,
    n_gn: int = 4,
    damping: float = 1.0e-3,
    delta_clip: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The exact pose-update loop used by DiffPnP3DLoss, exposed for audit."""
    batch, count, _ = pred_xy.shape
    rvec = DiffPnP3DLoss.rvec_from_R(rotation_gt).clone()
    tvec = translation_gt.detach().clone()
    eye = torch.eye(
        6, dtype=pred_xy.dtype, device=pred_xy.device
    ).unsqueeze(0)
    condition = torch.full(
        (batch,), float("nan"), dtype=pred_xy.dtype, device=pred_xy.device
    )
    observations = pred_xy.reshape(batch, 2 * count, 1)
    for _ in range(n_gn):
        projected, _ = _project_batch(rvec, tvec, points, camera)
        residual = projected.reshape(batch, 2 * count, 1) - observations
        jacobian = _jac_batch(rvec, tvec, points, camera)
        normal = torch.bmm(jacobian.transpose(1, 2), jacobian) + damping * eye
        right = torch.bmm(jacobian.transpose(1, 2), residual)
        delta = torch.linalg.solve(normal, right).squeeze(-1)
        norm = delta.norm(dim=1, keepdim=True).clamp_min(1.0e-9)
        delta = delta * (delta_clip / norm).clamp(max=1.0)
        rvec = rvec - delta[:, :3]
        tvec = tvec - delta[:, 3:]
        eigenvalues = torch.linalg.eigvalsh(normal.detach())
        condition = eigenvalues[:, -1] / eigenvalues[:, 0].clamp_min(1.0e-12)
    return rvec, tvec, condition


def _diffpnp3d_audit() -> dict[str, Any]:
    points_np, camera_np, rvec_gt_np, tvec_gt_np, image_points_np = (
        _known_pose_fixture()
    )
    dtype = torch.float64
    points = torch.tensor(points_np[None], dtype=dtype)
    camera = torch.tensor(camera_np[None], dtype=dtype)
    rotation_gt = torch.tensor(
        cv2.Rodrigues(rvec_gt_np)[0][None], dtype=dtype
    )
    translation_gt = torch.tensor(tvec_gt_np[None], dtype=dtype)
    diagonal = torch.tensor(
        [float(np.linalg.norm(points_np.max(0) - points_np.min(0)))],
        dtype=dtype,
    )
    mask = torch.tensor([True])

    exact = torch.tensor(image_points_np[None], dtype=dtype)
    rvec_exact, tvec_exact, condition_exact = _unrolled_pose(
        exact, points, camera, rotation_gt, translation_gt
    )
    projected_exact, _ = _project_batch(
        rvec_exact, tvec_exact, points, camera
    )
    reprojection_exact = torch.linalg.norm(projected_exact - exact, dim=2)

    perturbation = torch.tensor(
        [
            [
                [0.20, -0.10],
                [-0.15, 0.12],
                [0.08, 0.16],
                [-0.18, -0.06],
                [0.11, -0.14],
                [-0.09, 0.07],
                [0.16, 0.05],
                [-0.12, -0.11],
            ]
        ],
        dtype=dtype,
    )
    perturbed = (exact + perturbation).requires_grad_(True)
    module = DiffPnP3DLoss(
        n_gn=4,
        gn_damping=1.0e-3,
        huber_delta=0.05,
        cond_max=1.0e12,
    ).to(dtype=dtype)
    loss, information = module(
        perturbed,
        points,
        camera,
        rotation_gt,
        translation_gt,
        diagonal,
        mask,
    )
    automatic = torch.autograd.grad(loss, perturbed)[0].detach().cpu().numpy()

    epsilon = 1.0e-4
    base = perturbed.detach().cpu().numpy().reshape(-1)
    finite_difference = np.zeros_like(base)

    def objective(flat_points: np.ndarray) -> float:
        query = torch.tensor(
            flat_points.reshape(1, 8, 2), dtype=dtype
        )
        value, _ = module(
            query,
            points,
            camera,
            rotation_gt,
            translation_gt,
            diagonal,
            mask,
        )
        return float(value)

    for index in range(base.size):
        plus = base.copy()
        minus = base.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        finite_difference[index] = (
            objective(plus) - objective(minus)
        ) / (2.0 * epsilon)
    automatic_flat = automatic.reshape(-1)
    denominator = max(
        float(np.linalg.norm(automatic_flat)),
        float(np.linalg.norm(finite_difference)),
        1.0e-12,
    )
    relative_error = float(
        np.linalg.norm(automatic_flat - finite_difference) / denominator
    )

    rvec_perturbed, tvec_perturbed, condition_perturbed = _unrolled_pose(
        perturbed.detach(), points, camera, rotation_gt, translation_gt
    )
    projected_perturbed, _ = _project_batch(
        rvec_perturbed, tvec_perturbed, points, camera
    )

    yaw_base = _yaw_deg(rvec_perturbed.detach().cpu().numpy()[0])
    yaw_sensitivity = np.zeros((8, 2), dtype=np.float64)
    yaw_epsilon = 1.0e-2
    observed = perturbed.detach().clone()
    for keypoint in range(8):
        for axis in range(2):
            plus = observed.clone()
            minus = observed.clone()
            plus[0, keypoint, axis] += yaw_epsilon
            minus[0, keypoint, axis] -= yaw_epsilon
            r_plus, _, _ = _unrolled_pose(
                plus, points, camera, rotation_gt, translation_gt
            )
            r_minus, _, _ = _unrolled_pose(
                minus, points, camera, rotation_gt, translation_gt
            )
            yaw_sensitivity[keypoint, axis] = (
                _yaw_deg(r_plus.detach().cpu().numpy()[0])
                - _yaw_deg(r_minus.detach().cpu().numpy()[0])
            ) / (2.0 * yaw_epsilon)

    damping_rows = []
    for damping in (1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2):
        rv, tv, cond = _unrolled_pose(
            perturbed.detach(),
            points,
            camera,
            rotation_gt,
            translation_gt,
            damping=damping,
        )
        projected, _ = _project_batch(rv, tv, points, camera)
        damping_rows.append(
            {
                "damping": damping,
                "condition": float(cond.item()),
                "reprojection_to_observation_median_px": float(
                    torch.linalg.norm(
                        projected - perturbed.detach(), dim=2
                    ).median()
                ),
                "yaw_delta_deg": _yaw_deg(rv.detach().cpu().numpy()[0])
                - _yaw_deg(rvec_gt_np),
                "translation_error_m": float(
                    torch.linalg.norm(tv - translation_gt)
                ),
            }
        )

    # Near-planar audit uses the same camera pose but a 1 mm box thickness.
    planar_points_np, _, _, _, planar_image_np = _known_pose_fixture(height=0.001)
    planar_points = torch.tensor(planar_points_np[None], dtype=dtype)
    planar_observed = torch.tensor(planar_image_np[None], dtype=dtype)
    _, _, planar_condition = _unrolled_pose(
        planar_observed,
        planar_points,
        camera,
        rotation_gt,
        translation_gt,
    )

    # Explicit invalid-mask behavior.
    invalid_loss, invalid_info = module(
        exact,
        points,
        camera,
        rotation_gt,
        translation_gt,
        diagonal,
        torch.tensor([False]),
    )

    # A finite scalar after torch.where masking is not sufficient: a NaN in a
    # rejected frame can still poison backward through a zero upstream
    # multiplier.  Keep one healthy frame in the same batch so this exercises
    # the production "skip one, train on the rest" path.
    guarded_observations = torch.cat(
        [exact.clone(), perturbed.detach().clone()], dim=0
    ).requires_grad_(True)
    with torch.no_grad():
        guarded_observations[0, 0, 0] = float("nan")
    guarded_loss, guarded_info = module(
        guarded_observations,
        points.repeat(2, 1, 1),
        camera.repeat(2, 1, 1),
        rotation_gt.repeat(2, 1, 1),
        translation_gt.repeat(2, 1),
        diagonal.repeat(2),
        torch.tensor([True, True]),
    )
    guarded_gradient = torch.autograd.grad(
        guarded_loss, guarded_observations
    )[0]
    guarded_gradient_finite_by_frame = (
        torch.isfinite(guarded_gradient)
        .reshape(2, -1)
        .all(dim=1)
        .detach()
        .cpu()
        .tolist()
    )

    exact_rvec_np = rvec_exact.detach().cpu().numpy()[0]
    exact_tvec_np = tvec_exact.detach().cpu().numpy()[0]
    exact_yaw_error = abs(_yaw_deg(exact_rvec_np) - _yaw_deg(rvec_gt_np))
    exact_reprojection_median = float(reprojection_exact.median())
    skip_fraction = 1.0 - float(information["n_valid"]) / 1.0
    return {
        "implementation": (
            "PAPER_S2 DiffPnP3DLoss: GT-pose initialized, four-step "
            "autograd-through-unrolled Gauss-Newton"
        ),
        "exact_oracle": {
            "rvec_pred": exact_rvec_np,
            "tvec_pred": exact_tvec_np,
            "reprojection_median_px": exact_reprojection_median,
            "rotation_error_deg": _rotation_error_deg(
                exact_rvec_np, rvec_gt_np
            ),
            "yaw_error_deg": exact_yaw_error,
            "translation_error_m": float(
                np.linalg.norm(exact_tvec_np - tvec_gt_np)
            ),
            "normal_condition": float(condition_exact.item()),
        },
        "perturbed": {
            "loss": float(loss),
            "reprojection_to_observation_median_px": float(
                torch.linalg.norm(
                    projected_perturbed - perturbed.detach(), dim=2
                ).median()
            ),
            "normal_condition": float(condition_perturbed.item()),
            "diagnostics": {
                key: value
                for key, value in information.items()
                if key != "per_frame_L"
            },
        },
        "finite_difference": {
            "epsilon_px": epsilon,
            "automatic_gradient": automatic,
            "numeric_gradient": finite_difference.reshape(1, 8, 2),
            "relative_l2_error": relative_error,
            "per_keypoint_gradient_norm": np.linalg.norm(automatic[0], axis=1),
        },
        "yaw_sensitivity_deg_per_px": yaw_sensitivity,
        "yaw_at_perturbed_pose_deg": yaw_base,
        "damping_sweep": damping_rows,
        "near_planar_condition": float(planar_condition.item()),
        "invalid_mask": {
            "loss": float(invalid_loss),
            "n_valid": invalid_info["n_valid"],
            "gated_out": invalid_info["gated_out"],
        },
        "nan_guard_backward": {
            "loss": float(guarded_loss),
            "n_valid": guarded_info["n_valid"],
            "skip_depth": guarded_info["skip_depth"],
            "skip_nan": guarded_info["skip_nan"],
            "skip_cond": guarded_info["skip_cond"],
            "gradient_finite_by_frame": guarded_gradient_finite_by_frame,
            "gradient_norm_by_frame": torch.linalg.vector_norm(
                guarded_gradient.reshape(2, -1), dim=1
            ),
        },
        "gate": {
            "oracle_reprojection_le_1px": exact_reprojection_median <= 1.0,
            "oracle_yaw_le_1deg": exact_yaw_error <= 1.0,
            "finite_difference_rel_le_1e_2": relative_error <= 1.0e-2,
            "finite": bool(
                np.isfinite(automatic).all()
                and np.isfinite(finite_difference).all()
                and math.isfinite(float(loss))
            ),
            "valid_skip_le_1pct": skip_fraction <= 0.01,
            "nan_guard_backward_finite": bool(
                all(guarded_gradient_finite_by_frame)
                and math.isfinite(float(guarded_loss))
            ),
        },
    }


def _write_markdown(output_path: Path, payload: dict[str, Any]) -> None:
    global_gate = payload["softargmax"]["global"]["strict_gate"]
    local_gate = payload["softargmax"]["local"]["strict_gate"]
    global_operational = payload["softargmax"]["global"]["operational_gate"]
    local_operational = payload["softargmax"]["local"]["operational_gate"]
    legacy = payload["pnp"]["legacy_bpnp"]
    current = payload["pnp"]["paper_s2_diffpnp3d"]
    rows = [
        "# PAPER_S2 geometry unit audit",
        "",
        f"- generated: `{payload['provenance']['ended_at']}`",
        f"- commit: `{payload['provenance']['git_commit']}`",
        f"- checkpoint: `{payload['provenance']['checkpoint']}`",
        "",
        "## Gate summary",
        "",
        "| gate | result |",
        "|---|---:|",
        f"| Global soft-argmax known Gaussian | {'PASS' if all(global_gate.values()) else 'FAIL'} |",
        f"| PAPER_S2 local soft-argmax/covariance | {'PASS' if all(local_gate.values()) else 'FAIL'} |",
        f"| Legacy global soft-argmax on DOPE value maps (T=1.0) | {'PASS' if global_operational['pass'] else 'FAIL'} |",
        f"| PAPER_S2 local soft-argmax on DOPE value maps (T=0.1) | {'PASS' if local_operational['pass'] else 'FAIL'} |",
        f"| Legacy BPnP oracle | {'PASS' if legacy['gate']['oracle_reprojection_le_1px'] and legacy['gate']['oracle_yaw_le_1deg'] else 'FAIL'} |",
        f"| Legacy BPnP finite difference | {'PASS' if legacy['gate']['finite_difference_rel_le_1e_2'] else 'FAIL'} ({legacy['finite_difference']['relative_l2_error']:.3e}) |",
        f"| PAPER_S2 DiffPnP3D oracle | {'PASS' if current['gate']['oracle_reprojection_le_1px'] and current['gate']['oracle_yaw_le_1deg'] else 'FAIL'} |",
        f"| PAPER_S2 DiffPnP3D finite difference | {'PASS' if current['gate']['finite_difference_rel_le_1e_2'] else 'FAIL'} ({current['finite_difference']['relative_l2_error']:.3e}) |",
        f"| PAPER_S2 NaN guard backward | {'PASS' if current['gate']['nan_guard_backward_finite'] else 'FAIL'} |",
        "",
        "The legacy BPnP result and PAPER_S2 DiffPnP3D result are intentionally",
        "reported separately: the canonical ep57 checkpoint used the latter.",
        f"Legacy operational failures: `{global_operational['failures']}`.",
        f"PAPER_S2 local operational failures: `{local_operational['failures']}`.",
        "",
    ]
    output_path.with_suffix(".md").write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    output_path = args.out.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc)

    global_softargmax, local_softargmax = _softargmax_cases()
    legacy_bpnp = _legacy_bpnp_audit()
    diffpnp3d = _diffpnp3d_audit()

    ended = dt.datetime.now(dt.timezone.utc)
    payload = {
        "provenance": {
            "started_at": started.isoformat(),
            "ended_at": ended.isoformat(),
            "duration_seconds": (ended - started).total_seconds(),
            "exact_command": " ".join(
                [sys.executable, str(Path(__file__).resolve()), "--out", str(output_path)]
            ),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "checkpoint": str(CANONICAL_CKPT),
            "checkpoint_sha256": _sha256(CANONICAL_CKPT),
            "seed": 0,
        },
        "softargmax": {
            "global": global_softargmax,
            "local": local_softargmax,
            "production_interpretation": {
                "paper_s2_decoder": "LocalSoftArgmax2D",
                "paper_s2_temperature": 0.1,
                "paper_s2_window": 7,
                "paper_s2_covariance_use": (
                    "local moments are returned under torch.no_grad as diagnostics; "
                    "ep57 has no covariance head and does not weight DiffPnP by them"
                ),
                "legacy_reliability_use": (
                    "SpatialSoftArgmax2D returns radial sigma; ReliabilityLoss "
                    "detaches it for scalar coordinate weighting; ep57 rel_loss=False"
                ),
            },
        },
        "pnp": {
            "legacy_bpnp": legacy_bpnp,
            "paper_s2_diffpnp3d": diffpnp3d,
        },
    }
    output_path.write_text(
        json.dumps(payload, indent=2, default=_jsonable), encoding="utf-8"
    )
    _write_markdown(output_path, payload)
    print(json.dumps(
        {
            "output": str(output_path),
            "global_softargmax_gate": global_softargmax["strict_gate"],
            "local_softargmax_gate": local_softargmax["strict_gate"],
            "legacy_bpnp_gate": legacy_bpnp["gate"],
            "paper_s2_diffpnp3d_gate": diffpnp3d["gate"],
        },
        indent=2,
        default=_jsonable,
    ))


if __name__ == "__main__":
    main()
