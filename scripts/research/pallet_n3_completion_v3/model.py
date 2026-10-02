"""Shared N3 head and symmetry-target contract for DOPE and ResNet-18.

The visual branch is the already audited 222-candidate local-distribution P
implementation.  N3 adds only the canonical five-value physical-dimension
branch and selects one whole-object ground-truth permutation from the frozen
base prediction before constructing the usual Gaussian candidate target.

This module deliberately owns no backbone, image decoder, PnP, affine inverse,
or real-data selection.  Adapters must convert original points to network
coordinates before calling the head, then convert the predicted network-space
delta with the full per-axis affine and apply the circular one-percent cap in
original-image coordinates.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn

from scripts.research.pallet_dope_refiner_20261001_v1.refiner import (
    GenericPointRefiner,
    decode,
    loss as point_distribution_loss,
)


LOCKED_STENCIL_FRACTION = 0.1310373991727829
BACKBONE_CONFIGS = {
    "dope": dict(c3=256, c4=128, stride3=4, stride4=8),
    "resnet18": dict(c3=128, c4=256, stride3=8, stride4=16),
}


def dimension_features(dimensions: torch.Tensor) -> torch.Tensor:
    """Return [logW,logD,logH,log(W/D),log(H/sqrt(WD))]."""
    if dimensions.ndim < 1 or dimensions.shape[-1] != 3:
        raise ValueError("Canonical dimensions must end in [W,D,H]")
    if not dimensions.is_floating_point():
        dimensions = dimensions.float()
    if not torch.isfinite(dimensions).all() or (dimensions <= 0).any():
        raise ValueError("Canonical [W,D,H] must be finite positive metres")
    value = dimensions.log()
    return torch.stack((
        value[..., 0],
        value[..., 1],
        value[..., 2],
        value[..., 0] - value[..., 1],
        value[..., 2] - 0.5 * (value[..., 0] + value[..., 1]),
    ), dim=-1)


def normalize_dimensions(
    dimensions: torch.Tensor,
    normalization: Mapping[str, object],
) -> torch.Tensor:
    """Apply the locked TRAIN55,980 normalization supplied by the caller."""
    if set(("mean", "scale")) - set(normalization):
        raise ValueError("Dimension normalization requires mean and scale")
    features = dimension_features(dimensions)
    mean = torch.as_tensor(normalization["mean"], dtype=features.dtype,
                           device=features.device)
    scale = torch.as_tensor(normalization["scale"], dtype=features.dtype,
                            device=features.device)
    if mean.shape != (5,) or scale.shape != (5,):
        raise ValueError("Dimension normalization must contain five values")
    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Dimension normalization is nonfinite or degenerate")
    return (features - mean) / scale


class DimensionSymmetryN3(GenericPointRefiner):
    """P visual head plus zero-initialized canonical-dimension logit residual."""

    def __init__(
        self,
        *,
        c3: int,
        c4: int,
        stride3: int,
        stride4: int,
        hidden: int = 16,
        encoded: int = 24,
        stencil_fraction: float = LOCKED_STENCIL_FRACTION,
    ):
        if float(stencil_fraction) != LOCKED_STENCIL_FRACTION:
            raise ValueError("N3 requires the locked stencil fraction")
        if (hidden, encoded) != (16, 24):
            raise ValueError("N3 requires the locked hidden/encoded widths")
        super().__init__(
            c3=c3,
            c4=c4,
            hidden=hidden,
            encoded=encoded,
            stencil_fraction=stencil_fraction,
            stride3=stride3,
            stride4=stride4,
        )
        self.metadata_encoder = nn.Sequential(
            nn.Linear(5, 16), nn.SiLU(), nn.Linear(16, 16), nn.SiLU())
        self.metadata_scorer = nn.Sequential(
            nn.Linear(27, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.metadata_scorer[-1].weight)
        nn.init.zeros_(self.metadata_scorer[-1].bias)

    def forward(
        self,
        p3: torch.Tensor,
        p4: torch.Tensor,
        points: torch.Tensor,
        boxes: torch.Tensor,
        point_valid: torch.Tensor,
        input_shape: torch.Tensor,
        *,
        dimension_context: torch.Tensor,
        lam: float = 1.0,
        temperature: float = 1.0,
        cap=None,
    ) -> dict:
        if (dimension_context is None
                or dimension_context.shape != (len(points), 5)
                or not torch.isfinite(dimension_context).all()):
            raise ValueError("N3 requires finite normalized dimension_context[B,5]")
        output = super().forward(
            p3, p4, points, boxes, point_valid, input_shape, lam=0.)
        batch = len(points)
        embedding = self.metadata_encoder(
            dimension_context.to(self.role_embedding.weight))
        role = self.role_embedding.weight[None, :, None].expand(batch, -1, 222, -1)
        displacement = (self.displacements / 0.08)[None, None].expand(
            batch, 8, -1, -1)
        null = torch.zeros(
            (batch, 8, 222, 1), device=points.device, dtype=embedding.dtype)
        null[:, :, -1] = 1
        metadata = torch.cat((
            embedding[:, None, None].expand(-1, 8, 222, -1),
            role,
            displacement,
            null,
        ), dim=-1)
        residual = self.metadata_scorer(metadata).squeeze(-1)
        output["base_logits"] = output["logits"]
        output["metadata_residual"] = residual
        output["logits"] = output["base_logits"] + residual
        output["points"] = decode(
            output, temperature=temperature, lam=lam, cap=cap)
        return output


def build_n3(backbone: str) -> DimensionSymmetryN3:
    """Construct a fresh N3 head for one supported frozen estimator."""
    try:
        config = BACKBONE_CONFIGS[backbone]
    except KeyError as error:
        raise ValueError(f"Unsupported N3 backbone: {backbone}") from error
    return DimensionSymmetryN3(**config)


def select_symmetric_target(
    initial_points: torch.Tensor,
    initial_valid: torch.Tensor,
    gt_points: torch.Tensor,
    gt_valid: torch.Tensor,
    permutations: torch.Tensor,
    group_valid: torch.Tensor,
    box_diagonal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select one approved whole-object GT branch from frozen initial points.

    The first valid branch wins an exact tie, so callers must store identity in
    column zero.  Missing initial predictions contribute normalized penalty 1.
    Selection is training/calibration supervision only and must never be called
    from an inference path.
    """
    if initial_points.ndim != 3 or initial_points.shape[1:] != (9, 2):
        raise ValueError("initial_points must be [B,9,2]")
    batch = len(initial_points)
    if (initial_valid.shape != (batch, 9)
            or gt_points.shape != (batch, 9, 2)
            or gt_valid.shape != (batch, 9)
            or permutations.ndim != 3
            or permutations.shape[0] != batch
            or permutations.shape[2] != 9
            or group_valid.shape != permutations.shape[:2]
            or box_diagonal.shape != (batch,)):
        raise ValueError("Symmetry target tensor shapes disagree")
    if permutations.dtype == torch.bool or permutations.is_floating_point():
        raise ValueError("Permutations must contain integer indices")
    identity = torch.arange(9, device=permutations.device)
    if not torch.equal(permutations[:, 0], identity[None].expand(batch, -1)):
        raise ValueError("Identity must be the first permutation")
    if not (permutations[..., 8] == 8).all():
        raise ValueError("Approved permutations must preserve center8")
    if not group_valid[:, 0].all():
        raise ValueError("Identity branch must be valid")
    if not torch.isfinite(box_diagonal).all() or (box_diagonal <= 0).any():
        raise ValueError("Positive finite predicted-box diagonals are required")

    row = torch.arange(batch, device=gt_points.device)[:, None, None]
    permuted_gt = gt_points[row, permutations]
    permuted_valid = gt_valid[row, permutations]
    permuted_valid = (permuted_valid
                      & torch.isfinite(permuted_gt).all(-1)
                      & ~(permuted_gt == -1).all(-1))
    prediction_valid = (initial_valid
                        & torch.isfinite(initial_points).all(-1)
                        & ~(initial_points == -1).all(-1))
    distance = torch.linalg.vector_norm(
        initial_points[:, None] - permuted_gt, dim=-1)
    distance = distance / box_diagonal[:, None, None]
    distance = torch.where(
        prediction_valid[:, None], distance, torch.ones_like(distance))
    distance = torch.where(
        permuted_valid, distance, torch.zeros_like(distance))
    count = permuted_valid[:, :, :8].sum(-1)
    cost = distance[:, :, :8].sum(-1) / count.clamp_min(1)
    cost = torch.where(
        group_valid & (count > 0), cost,
        torch.full_like(cost, float("inf")))
    branch = cost.argmin(-1)
    selected_row = torch.arange(batch, device=gt_points.device)
    return (
        permuted_gt[selected_row, branch],
        permuted_valid[selected_row, branch],
        branch,
        cost,
    )


def n3_loss(output: dict, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """N3 CE after frozen-initial whole-object symmetry selection."""
    required = {
        "gt_points", "gt_valid", "permutations", "group_valid",
    }
    missing = required - set(batch)
    if missing:
        raise ValueError(f"N3 supervision is missing: {sorted(missing)}")
    target, valid, _, _ = select_symmetric_target(
        output["points_raw"],
        output["point_valid"],
        batch["gt_points"],
        batch["gt_valid"],
        batch["permutations"],
        batch["group_valid"],
        output["box_diagonal"],
    )
    return point_distribution_loss(output, target, valid)


def trainable_parameter_count(backbone: str) -> int:
    return sum(parameter.numel() for parameter in build_n3(backbone).parameters())


__all__ = [
    "BACKBONE_CONFIGS",
    "LOCKED_STENCIL_FRACTION",
    "DimensionSymmetryN3",
    "build_n3",
    "dimension_features",
    "normalize_dimensions",
    "select_symmetric_target",
    "n3_loss",
    "trainable_parameter_count",
]
