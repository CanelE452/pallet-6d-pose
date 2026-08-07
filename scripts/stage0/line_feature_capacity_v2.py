"""Line refinement capacity, V2: confounds removed before judging any feature.

V1 concluded that no feature carries 1 degree / 0.5 cell precision.  It could not
have: its loss let the offset term dominate the angle term by two orders of
magnitude, its sampler followed a window anchored to the origin rather than the
line, and its strip was narrower than the offset jitter, so the evidence needed
to correct a line was often not in the input.  See
LINE_FEATURE_CAPACITY_V1_ADDENDUM.md.

V2 fixes all four and asks the identifiability question first: with perfect
evidence -- ground-truth lines rasterised -- can this refiner represent and learn
the correction at all?  Only if that passes does a real feature get judged.

No PnP, no dimensions, no validation512, no sealed set.
"""
from __future__ import annotations

import hashlib, importlib.util, json, math, pathlib, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _e in ("scripts/stage0", "Deep_Object_Pose/common", "Deep_Object_Pose/train",
           "challenge/scripts"):
    if str(ROOT / _e) not in sys.path:
        sys.path.insert(0, str(ROOT / _e))

GRID = 50                      # canonical belief grid; all errors defined here
ANGLE_BUDGET_DEG = 1.0
OFFSET_BUDGET_CELL = 0.5
JITTER_ANGLE_DEG = 8.0
JITTER_OFFSET_CELL = 4.0
TRANSVERSE_RADIUS_CELL = 6.0   # > jitter 4, so the true line is inside the strip
TRANSVERSE_SAMPLES = 13
LONGITUDINAL = 64
SEED, LR, WD, BATCH = 1, 1e-3, 1e-4, 12


def sha_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_rect_intersection(normal, rho, width, height):
    """Where the infinite line n.x = rho crosses the feature rectangle.

    Longitudinal sampling spans this visible chord instead of a fixed window at
    the foot of the perpendicular, which is what V1 did.
    """
    device = normal.device
    direction = torch.stack([-normal[..., 1], normal[..., 0]], -1)
    base = normal * rho[..., None]
    limits = []
    for axis, extent in ((0, width - 1.0), (1, height - 1.0)):
        d = direction[..., axis]
        b = base[..., axis]
        safe = torch.where(d.abs() < 1e-6, torch.full_like(d, 1e-6), d)
        t_low = (0.0 - b) / safe
        t_high = (extent - b) / safe
        limits.append((torch.minimum(t_low, t_high), torch.maximum(t_low, t_high)))
    t_enter = torch.maximum(limits[0][0], limits[1][0])
    t_exit = torch.minimum(limits[0][1], limits[1][1])
    valid = t_exit > t_enter
    return t_enter, t_exit, valid, direction, base


def sample_strip(feature, normal, rho, scale):
    """Feature strip along the line's visible chord, plus a validity mask.

    ``scale`` maps canonical 50-grid units to this feature's resolution, so every
    arm reads the same physical footprint.
    """
    batch, roles = normal.shape[0], normal.shape[1]
    height, width = feature.shape[-2:]
    rho_f = rho * scale
    t_enter, t_exit, valid, direction, base = line_rect_intersection(
        normal, rho_f, width, height)
    alpha = torch.linspace(0.0, 1.0, LONGITUDINAL, device=feature.device)
    t = t_enter[..., None] + (t_exit - t_enter)[..., None] * alpha
    radius = TRANSVERSE_RADIUS_CELL * scale
    s = torch.linspace(-radius, radius, TRANSVERSE_SAMPLES, device=feature.device)
    points = (base[:, :, None, None, :]
              + direction[:, :, None, None, :] * t[:, :, None, :, None]
              + normal[:, :, None, None, :] * s[None, None, :, None, None])
    norm_x = points[..., 0] / max(width - 1, 1) * 2 - 1
    norm_y = points[..., 1] / max(height - 1, 1) * 2 - 1
    inside = ((norm_x.abs() <= 1) & (norm_y.abs() <= 1)).float()
    flat = torch.stack([norm_x, norm_y], -1).reshape(
        batch, roles * TRANSVERSE_SAMPLES, LONGITUDINAL, 2)
    sampled = F.grid_sample(feature, flat, mode="bilinear",
                            padding_mode="zeros", align_corners=True)
    sampled = sampled.reshape(batch, -1, roles, TRANSVERSE_SAMPLES,
                              LONGITUDINAL).permute(0, 2, 1, 3, 4)
    return sampled, inside, valid


class Refiner(nn.Module):
    """Strip plus validity -> (delta theta, delta rho, two log sigmas)."""

    def __init__(self, in_channels, hidden=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels + 1, hidden, (TRANSVERSE_SAMPLES, 5), padding=(0, 2)),
            nn.GroupNorm(8, hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, (1, 5), padding=(0, 2)),
            nn.GroupNorm(8, hidden), nn.ReLU(inplace=True))
        self.head = nn.Linear(hidden * 2, 4)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, strip, inside):
        batch, roles = strip.shape[0], strip.shape[1]
        x = torch.cat([strip, inside[:, :, None]], dim=2)
        x = x.reshape(batch * roles, *x.shape[2:])
        x = self.body(x).squeeze(2)
        pooled = torch.cat([x.mean(-1), x.max(-1).values], -1)
        out = self.head(pooled).reshape(batch, roles, 4)
        return {"delta_theta_deg": out[..., 0], "delta_rho_cell": out[..., 1],
                "log_sigma_theta": out[..., 2].clamp(-6, 4),
                "log_sigma_rho": out[..., 3].clamp(-6, 4)}


def wrap_half_pi(angle):
    """Undirected lines: residual folded into (-pi/2, pi/2]."""
    return (angle + math.pi / 2) % math.pi - math.pi / 2


def budget_losses(theta_pred, rho_pred, theta_gt, rho_gt):
    """Each term normalised by its own budget, so error 1 is the gate boundary."""
    d_theta = wrap_half_pi(theta_pred - theta_gt)
    angle_deg = torch.rad2deg(d_theta).abs()
    offset = (rho_pred - rho_gt).abs()
    e_theta = torch.rad2deg(d_theta) / ANGLE_BUDGET_DEG
    e_rho = (rho_pred - rho_gt) / OFFSET_BUDGET_CELL
    return {"angle_deg": angle_deg, "offset_cell": offset,
            "L_theta": F.smooth_l1_loss(e_theta, torch.zeros_like(e_theta)),
            "L_rho": F.smooth_l1_loss(e_rho, torch.zeros_like(e_rho))}


def raster_lines(normal, rho, grid=GRID, sigma=1.0):
    """Anti-aliased line likelihood: the perfect-evidence oracle for O0."""
    device = normal.device
    axis = torch.arange(grid, device=device, dtype=normal.dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    distance = (normal[..., 0][..., None, None] * xx
                + normal[..., 1][..., None, None] * yy
                - rho[..., None, None]).abs()
    return torch.exp(-(distance ** 2) / (2 * sigma ** 2))
