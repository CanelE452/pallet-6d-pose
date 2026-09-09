"""Trainable, predicted-instance-conditioned pallet side-line voting.

All forward coordinates are letterboxed YOLO input pixels. GT is accepted only
by the separate target/loss functions at the end of this file. The YOLO model
and its detection selection live outside this module; this module reads the
same forward's spatial features and predicted points, not GT crops or points.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


SIDE_EDGES = ((1, 2), (3, 0), (5, 6), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7))
INCIDENT_ROLES = ((1, 4), (0, 5), (0, 6), (1, 7),
                  (3, 4), (2, 5), (2, 6), (3, 7))
THETA_BINS = 13
OFFSET_BINS = 17
NUM_CANDIDATES = THETA_BINS * OFFSET_BINS
NULL_INDEX = NUM_CANDIDATES


def pixel_to_grid(points, feature_height, feature_width, stride):
    """Explicit cell-center convention; not a receptive-field-center claim.

    grid_sample(..., align_corners=False) samples feature index x/stride-.5.
    Feature cell j therefore denotes input coordinate (j+.5)*stride. Padded
    cache dimensions, not each record's unpadded feature dimensions, go here.
    """
    scale = points.new_tensor([stride * feature_width, stride * feature_height])
    return 2 * points / scale - 1


def _finite_points(points, valid):
    return valid.bool() & torch.isfinite(points).all(-1) & ~(points == -1).all(-1)


def build_candidates(points, boxes, point_valid, theta_deg=None, offsets=None,
                     min_length=2.):
    """GT-free midpoint-pivoted lines: n=R(dt)n0, rho=n.m+dr*box_diagonal."""
    if points.ndim != 3 or points.shape[1:] != (9, 2):
        raise ValueError("points must be [B,9,2]")
    if boxes.shape != (len(points), 4) or point_valid.shape != points.shape[:2]:
        raise ValueError("boxes[B,4] and point_valid[B,9] are required")
    theta_deg = (torch.linspace(-12., 12., THETA_BINS, device=points.device, dtype=points.dtype)
                 if theta_deg is None else theta_deg.to(points))
    offsets = (torch.linspace(-.08, .08, OFFSET_BINS, device=points.device, dtype=points.dtype)
               if offsets is None else offsets.to(points))
    valid = _finite_points(points, point_valid)
    safe = torch.where(valid[..., None], points, torch.zeros_like(points))
    box_valid = torch.isfinite(boxes).all(-1) & (boxes[:, 2:] > boxes[:, :2]).all(-1)
    safe_boxes = torch.where(box_valid[:, None], boxes, boxes.new_tensor([0., 0., 1., 1.]))
    size = safe_boxes[:, 2:] - safe_boxes[:, :2]
    diagonal = size.norm(dim=-1).clamp_min(1.)
    center = (safe_boxes[:, :2] + safe_boxes[:, 2:]) * .5
    edges = torch.tensor(SIDE_EDGES, device=points.device)
    ends = safe[:, edges]
    delta = ends[..., 1, :] - ends[..., 0, :]
    length = delta.norm(dim=-1)
    line_valid = valid[:, edges].all(-1) & (length >= min_length) & box_valid[:, None]
    direction = delta / length.clamp_min(1e-6)[..., None]
    direction = torch.where(line_valid[..., None], direction, direction.new_tensor([1., 0.]))
    normal = torch.stack([-direction[..., 1], direction[..., 0]], -1)
    midpoint = ends.mean(-2)
    dt, dr = torch.meshgrid(torch.deg2rad(theta_deg), offsets, indexing="ij")
    dt, dr = dt.flatten(), dr.flatten()
    nx = normal[..., 0, None] * dt.cos() - normal[..., 1, None] * dt.sin()
    ny = normal[..., 0, None] * dt.sin() + normal[..., 1, None] * dt.cos()
    normals = torch.stack([nx, ny], -1)
    rho = (normals * midpoint[:, :, None]).sum(-1) + dr * diagonal[:, None, None]
    h = torch.cat([normals, -rho[..., None]], -1)
    return dict(candidate_h=h, normal_initial=normal, midpoint=midpoint,
                length=length, box_diagonal=diagonal, box_center=center,
                box_size=size, line_valid=line_valid, point_valid=valid,
                theta_deg=theta_deg, offsets=offsets, delta_theta=dt,
                delta_offset=dr, anchor_std=(.025 * diagonal).clamp_min(1.))


def summarize_distribution(logits, candidate_h):
    """Conditional non-null mean line and its full-distribution spread.

    Normals share the initial line's sign (local angular range +/-12 degrees).
    h_mean is E[h] divided by the length of its normal. M is
    E[(h_candidate-h_mean)(h_candidate-h_mean)^T], including the normalization
    difference, rather than silently calling it covariance about E[h]. zMz is
    the expected squared residual difference from the returned mean line.
    """
    if logits.shape[-1] != candidate_h.shape[-2] + 1:
        raise ValueError("logits need one null class after all line candidates")
    conditional = logits[..., :-1].softmax(-1)
    null_probability = logits.softmax(-1)[..., -1]
    raw_mean = (conditional[..., None] * candidate_h).sum(-2)
    h_mean = raw_mean / raw_mean[..., :2].norm(dim=-1, keepdim=True).clamp_min(1e-6)
    delta = candidate_h - h_mean[..., None, :]
    moment = torch.einsum("brq,brqi,brqj->brij", conditional, delta, delta)
    entropy = -(conditional * conditional.clamp_min(1e-30).log()).sum(-1) / math.log(conditional.shape[-1])
    return dict(conditional_probability=conditional, null_probability=null_probability,
                h_mean=h_mean, moment=moment, entropy_normalized=entropy)


def fuse_corners(points, point_valid, h_mean, moment, null_probability,
                 line_valid, anchor_std, lam=1.):
    """Differentiable anchored 2x2 solve; only predicted quantities are inputs.

    Pixel point variance is anchor_std**2. Incident line variance is max(1,zMz)
    in pixels squared, and null probability multiplies its precision by 1-pnull.
    No line intersection or hard predicted-corner confidence gate is used.
    """
    if not math.isfinite(float(lam)) or float(lam) < 0:
        raise ValueError("lam must be finite and nonnegative")
    valid = _finite_points(points, point_valid)
    safe = torch.where(valid[..., None], points, torch.zeros_like(points))
    incident = torch.tensor(INCIDENT_ROLES, device=points.device)
    h = h_mean[:, incident]
    m = moment[:, incident]
    z = torch.cat([safe[:, :8], torch.ones_like(safe[:, :8, :1])], -1)
    raw_variance = torch.einsum("bki,bkrij,bkj->bkr", z, m, z).clamp_min(0.)
    variance = raw_variance.clamp_min(1.)
    precision = ((1 - null_probability[:, incident]).clamp(0., 1.)
                 * line_valid[:, incident].to(points.dtype) / variance)
    precision = precision * valid[:, :8, None].to(points.dtype)
    # This explicit early return preserves all original bits, including NaNs.
    if float(lam) == 0:
        return points.clone(), variance, raw_variance
    n = h[..., :2]
    residual = (h * z[:, :, None]).sum(-1)
    eye = torch.eye(2, device=points.device, dtype=points.dtype)
    anchor_precision = anchor_std.clamp_min(1.).square().reciprocal()
    system = (anchor_precision[:, None, None, None] * eye
              + float(lam) * torch.einsum("bkr,bkri,bkrj->bkij", precision, n, n))
    rhs = -float(lam) * (precision[..., None] * n * residual[..., None]).sum(-2)
    displacement = torch.linalg.solve(system, rhs[..., None]).squeeze(-1)
    # Zero contribution also copies exactly, rather than introducing solve noise.
    movable = valid[:, :8] & (precision.sum(-1) > 0)
    corners = torch.where(movable[..., None], safe[:, :8] + displacement, points[:, :8])
    return torch.cat([corners, points[:, 8:]], 1), variance, raw_variance


class PalletLinePoseHead(nn.Module):
    """Two spatial adapters, line sampling, candidate scoring, and corner solve."""

    def __init__(self, c3, c4, hidden=16, along_samples=32):
        super().__init__()
        if hidden < 1 or along_samples < 2:
            raise ValueError("positive hidden channels and >=2 samples required")
        self.c3, self.c4 = int(c3), int(c4)
        self.hidden, self.along_samples = int(hidden), int(along_samples)
        self.adapt3 = nn.Sequential(nn.Conv2d(c3, hidden, 1), nn.SiLU())
        self.adapt4 = nn.Sequential(nn.Conv2d(c4, hidden, 1), nn.SiLU())
        self.line_body = nn.Sequential(nn.Conv1d(2 * hidden, 2 * hidden, 3, padding=1),
                                       nn.SiLU(), nn.Conv1d(2 * hidden, 2 * hidden, 3, padding=1), nn.SiLU())
        self.role_embedding = nn.Embedding(8, 8)
        # image mean/max (4h), base geometry (6), role (8), candidate (3).
        self.scorer = nn.Sequential(nn.Linear(4 * hidden + 17, 64), nn.SiLU(), nn.Linear(64, 1))
        self.null_scorer = nn.Sequential(nn.Linear(4 * hidden + 14, 64), nn.SiLU(), nn.Linear(64, 1))
        self.register_buffer("theta_deg", torch.linspace(-12., 12., THETA_BINS))
        self.register_buffer("offsets", torch.linspace(-.08, .08, OFFSET_BINS))
        self.register_buffer("along", torch.linspace(-.5, .5, along_samples))

    @staticmethod
    def _sample(feature, positions, input_shape, stride):
        batch, channels, height, width = feature.shape
        # Zero adapter biases in cache padding before interpolation.
        ys = (torch.arange(height, device=feature.device, dtype=feature.dtype) + .5) * stride
        xs = (torch.arange(width, device=feature.device, dtype=feature.dtype) + .5) * stride
        support = ((ys[None, :, None] < input_shape[:, 0, None, None])
                   & (xs[None, None, :] < input_shape[:, 1, None, None]))
        feature = feature * support[:, None]
        grid = pixel_to_grid(positions, height, width, stride)
        sampled = F.grid_sample(feature, grid.reshape(batch, -1, positions.shape[-2], 2),
                                mode="bilinear", padding_mode="zeros", align_corners=False)
        return sampled.reshape(batch, channels, *positions.shape[1:-1]).permute(0, 2, 3, 1, 4)

    def forward(self, p3, p4, points, boxes, point_valid, input_shape, lam=1., geometry_only=False):
        if p3.ndim != 4 or p4.ndim != 4 or p3.shape[:2] != (len(points), self.c3) or p4.shape[:2] != (len(points), self.c4):
            raise ValueError("P3/P4 batches or constructor channel counts disagree")
        if input_shape.shape != (len(points), 2) or not torch.isfinite(input_shape).all() or (input_shape <= 0).any():
            raise ValueError("input_shape must be positive finite [B,2] in H,W order")
        if ((input_shape[:, 0] > min(p3.shape[-2] * 8, p4.shape[-2] * 16)).any()
                or (input_shape[:, 1] > min(p3.shape[-1] * 8, p4.shape[-1] * 16)).any()):
            raise ValueError("feature cache must cover the actual input rectangle")
        # Cache may be float16; line geometry/2x2 solves intentionally stay float32
        # unless the entire module is explicitly converted to float64 for tests.
        dtype = self.role_embedding.weight.dtype
        p3, p4 = p3.to(dtype), p4.to(dtype)
        points, boxes, input_shape = points.to(dtype), boxes.to(dtype), input_shape.to(dtype)
        out = build_candidates(points, boxes, point_valid, self.theta_deg, self.offsets)
        h = out["candidate_h"]
        normal, rho = h[..., :2], -h[..., 2]
        tangent = torch.stack([-normal[..., 1], normal[..., 0]], -1)
        midpoint = out["midpoint"][:, :, None]
        # Shift the predicted midpoint onto each candidate supporting line.
        base = midpoint + (rho - (normal * midpoint).sum(-1))[..., None] * normal
        positions = (base[..., None, :] + tangent[..., None, :]
                     * out["length"][:, :, None, None, None] * self.along[None, None, None, :, None])
        in_frame = ((positions[..., 0] >= 0) & (positions[..., 1] >= 0)
                    & (positions[..., 0] < input_shape[:, None, None, None, 1])
                    & (positions[..., 1] < input_shape[:, None, None, None, 0]))
        a3, a4 = self.adapt3(p3), self.adapt4(p4)
        if geometry_only:
            a3, a4 = a3 * 0, a4 * 0
        features = torch.cat([self._sample(a3, positions, input_shape, 8),
                              self._sample(a4, positions, input_shape, 16)], -2)
        features = features * in_frame[..., None, :]
        batch, roles, candidates, channels, samples = features.shape
        encoded = self.line_body(features.reshape(-1, channels, samples))
        pooled = torch.cat([encoded.mean(-1), encoded.amax(-1)], -1).reshape(batch, roles, candidates, -1)
        diagonal = out["box_diagonal"]
        base_geometry = torch.cat([out["normal_initial"],
            (out["midpoint"] - out["box_center"][:, None]) / diagonal[:, None, None],
            (out["length"] / diagonal[:, None])[..., None],
            (out["box_size"][:, 0] / out["box_size"][:, 1].clamp_min(1e-6)).log()[:, None, None].expand(-1, 8, 1)], -1)
        role = self.role_embedding.weight[None].expand(batch, -1, -1)
        context = torch.cat([base_geometry, role], -1)
        candidate_geometry = torch.stack([out["delta_theta"] / math.radians(12.),
                                         out["delta_offset"] / .08], -1)
        candidate_geometry = candidate_geometry[None, None].expand(batch, 8, -1, -1)
        coverage = in_frame.to(dtype).mean(-1)
        score_input = torch.cat([pooled, context[:, :, None].expand(-1, -1, candidates, -1),
                                 candidate_geometry, coverage[..., None]], -1)
        score = self.scorer(score_input).squeeze(-1)
        null = self.null_scorer(torch.cat([pooled.mean(-2), context], -1))
        logits = torch.cat([score, null], -1)
        # A line with no sampled image support at any candidate cannot move points.
        out["line_valid"] = out["line_valid"] & in_frame.any(-1).any(-1)
        out.update(summarize_distribution(logits, h))
        refined, variance, raw_variance = fuse_corners(points, out["point_valid"], out["h_mean"],
            out["moment"], out["null_probability"], out["line_valid"], out["anchor_std"], lam)
        out.update(points=refined, points_raw=points, logits=logits,
                   line_variance=variance, raw_line_variance=raw_variance,
                   sample_coverage=coverage, boxes=boxes, input_shape=input_shape)
        return out


@torch.no_grad()
def make_targets(output, gt_points, gt_valid, gt_support=None):
    """Training-only bilinear local-grid targets, with an explicit null class.

    Both annotated GT endpoints and length>=2 input pixels are required. Optional
    gt_support[B,8] further restricts supervision (e.g. original-image segment
    support). Neither label availability nor support means physical visibility.
    Invalid predicted initial lines are also masked: their coordinate chart is
    undefined and inference preserves those baseline points.
    """
    points = output["points_raw"]
    gt = gt_points.to(points)
    if gt.shape != points.shape or gt_valid.shape != points.shape[:2]:
        raise ValueError("GT points/valid must match predicted [B,9,2]/[B,9]")
    valid = _finite_points(gt, gt_valid)
    safe = torch.where(valid[..., None], gt, torch.zeros_like(gt))
    edge = torch.tensor(SIDE_EDGES, device=points.device)
    ends = safe[:, edge]
    direction = ends[..., 1, :] - ends[..., 0, :]
    length = direction.norm(dim=-1)
    normal = torch.stack([-direction[..., 1], direction[..., 0]], -1) / length.clamp_min(1e-6)[..., None]
    n0 = output["normal_initial"]
    sign = torch.where((normal * n0).sum(-1) >= 0, 1., -1.)
    normal = normal * sign[..., None]
    cosine = (normal * n0).sum(-1)
    sine = n0[..., 0] * normal[..., 1] - n0[..., 1] * normal[..., 0]
    theta = torch.rad2deg(torch.atan2(sine, cosine))
    rho = (normal * ends.mean(-2)).sum(-1)
    offset = (rho - (normal * output["midpoint"]).sum(-1)) / output["box_diagonal"][:, None]
    support = valid[:, edge].all(-1) & (length >= 2.) & output["line_valid"]
    if gt_support is not None:
        if gt_support.shape != support.shape:
            raise ValueError("gt_support must be [B,8]")
        support &= gt_support.bool()
    tgrid, rgrid = output["theta_deg"], output["offsets"]
    in_range = ((theta >= tgrid[0] - 1e-5) & (theta <= tgrid[-1] + 1e-5)
                & (offset >= rgrid[0] - 1e-7) & (offset <= rgrid[-1] + 1e-7))
    tx = ((theta - tgrid[0]) / (tgrid[1] - tgrid[0])).clamp(0, len(tgrid) - 1)
    rx = ((offset - rgrid[0]) / (rgrid[1] - rgrid[0])).clamp(0, len(rgrid) - 1)
    ti, ri = tx.floor().long(), rx.floor().long()
    tf, rf = tx - ti, rx - ri
    target = torch.zeros_like(output["logits"])
    for td, tw in ((0, 1 - tf), (1, tf)):
        for rd, rw in ((0, 1 - rf), (1, rf)):
            index = (ti + td).clamp_max(len(tgrid) - 1) * len(rgrid) + (ri + rd).clamp_max(len(rgrid) - 1)
            target.scatter_add_(-1, index[..., None], (tw * rw * in_range)[..., None])
    target[..., -1] = (~in_range).to(target.dtype)
    target *= support[..., None]
    return dict(distribution=target, support=support, in_range=in_range & support,
                null_target=(~in_range) & support, delta_theta_deg=theta,
                delta_offset=offset, gt_valid=valid)


def compute_loss(output, gt_points, gt_valid, corner_weight=1., gt_support=None):
    """CE + corner_weight*SmoothL1((q-GT)/anchor_std, beta=1).

    Corner and line means are first calculated per image, then averaged across
    images with at least one supported item. Center8 is never a refinement loss.
    corner_weight=0 is image-line-only; geometry_only=True is a forward ablation.
    """
    if not math.isfinite(float(corner_weight)) or float(corner_weight) < 0:
        raise ValueError("corner_weight must be finite and nonnegative")
    targets = make_targets(output, gt_points, gt_valid, gt_support)
    per_line = -(targets["distribution"] * output["logits"].log_softmax(-1)).sum(-1)
    line_count = targets["support"].sum(-1)
    per_image_line = per_line.sum(-1) / line_count.clamp_min(1)
    line_loss = per_image_line.sum() / (line_count > 0).sum().clamp_min(1)
    valid = targets["gt_valid"][:, :8] & output["point_valid"][:, :8]
    safe_q = torch.where(valid[..., None], output["points"][:, :8], torch.zeros_like(output["points"][:, :8]))
    gt = gt_points.to(safe_q)[:, :8]
    safe_gt = torch.where(valid[..., None], gt, torch.zeros_like(gt))
    normalized = (safe_q - safe_gt) / output["anchor_std"][:, None, None]
    per_corner = F.smooth_l1_loss(normalized, torch.zeros_like(normalized), beta=1., reduction="none").mean(-1)
    corner_count = valid.sum(-1)
    per_image_corner = (per_corner * valid).sum(-1) / corner_count.clamp_min(1)
    corner_loss = per_image_corner.sum() / (corner_count > 0).sum().clamp_min(1)
    return dict(loss=line_loss + float(corner_weight) * corner_loss,
                line_loss=line_loss, corner_loss=corner_loss,
                line_count=line_count.sum(), corner_count=corner_count.sum(),
                null_count=targets["null_target"].sum())
