"""Local image-evidence line refinement with a matched direct-line control.

All distances/line offsets inside the model use canonical input pixels. Input
pixel coordinate (.5,.5) denotes its first pixel centre. The supplied affine maps
raw point coordinates to this system; only the inverse linear part maps a
predicted displacement back. No annotation or physical-visibility mask is read.
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

ARMS = ('no_hough', 'hough')
INPUT_KEYS = ('image_gray', 'input_content_mask', 'baseline_points',
              'raw_to_input_affine', 'point_valid')
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
         (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
DEFAULT_CONFIG = dict(along=32, normal=33, halfwidth=16., angle_degrees=8.,
    angle_bins=17, minimum_length=16., minimum_coverage=.75,
    anchor_precision=1., maximum_shift=4., contrast_floor=1./255.,
    temperature=.05, confidence_floor=.05)


def raw_to_input(points, affine):
    return torch.einsum('bij,bnj->bni', affine[..., :2], points)+affine[..., 2][:, None]


def input_delta_to_raw(delta, affine):
    return torch.linalg.solve(affine[..., :2], delta.transpose(-1, -2)).transpose(-1, -2)


def edge_geometry(points_input, point_valid, *, minimum_length=16., along=32):
    edges = torch.tensor(EDGES, device=points_input.device)
    finite = torch.isfinite(points_input).all(-1)
    p = torch.where(finite[..., None], points_input, torch.zeros_like(points_input))
    first, second = p[:, edges[:, 0]], p[:, edges[:, 1]]
    displacement = second-first
    length = torch.linalg.vector_norm(displacement, dim=-1)
    tangent = displacement/length.clamp_min(1e-8)[..., None]
    normal = torch.stack((-tangent[..., 1], tangent[..., 0]), -1)
    valid = point_valid & finite
    valid = valid[:, edges[:, 0]] & valid[:, edges[:, 1]] & (length >= minimum_length)
    fractions = (torch.arange(along, device=p.device, dtype=p.dtype)+.5)/along-.5
    return dict(midpoint=(first+second)*.5, tangent=tangent, normal=normal,
        length=length, along_coordinates=length[..., None]*fractions,
        edge_valid=valid, point_valid=point_valid & finite)


def extract_corridors(image_gray, input_content_mask, points_raw,
                      raw_to_input_affine, point_valid, *, config=None):
    """Sample luma and directional Sobel evidence on the same 12 corridors.

    Eroding content validity by one pixel ensures every Sobel neighbour belongs
    to original image content. Reflected/letterbox/bottom-padding edges cannot
    supply support. Low contrast is an image condition, not GT visibility.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    b, _, height, width = image_gray.shape
    points = raw_to_input(points_raw, raw_to_input_affine)
    geometry = edge_geometry(points, point_valid, minimum_length=cfg['minimum_length'], along=cfg['along'])
    kernel_x = image_gray.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])/8.
    kernels = torch.stack((kernel_x, kernel_x.T))[:, None]
    derivatives = F.conv2d(F.pad(image_gray, (1, 1, 1, 1), mode='replicate'), kernels)
    support = (F.conv2d(input_content_mask.to(image_gray.dtype),
        image_gray.new_ones((1, 1, 3, 3)), padding=1) >= 9.).to(image_gray.dtype)
    normal_offsets = torch.linspace(-cfg['halfwidth'], cfg['halfwidth'], cfg['normal'],
        device=image_gray.device, dtype=image_gray.dtype)
    xy = (geometry['midpoint'][:, :, None, None]
        + geometry['along_coordinates'][:, :, :, None, None]*geometry['tangent'][:, :, None, None]
        + normal_offsets[None, None, None, :, None]*geometry['normal'][:, :, None, None])
    # Same boundary-coordinate convention as the stride16 cache: pixel centres
    # are .5,1.5,... and a stride16 cell centre is8,24,... in these units.
    grid = 2.*xy/image_gray.new_tensor([width, height])-1.
    sampled = F.grid_sample(torch.cat((image_gray, derivatives, support), 1),
        grid.reshape(b, len(EDGES)*cfg['along'], cfg['normal'], 2),
        mode='bilinear', padding_mode='zeros', align_corners=False)
    sampled = sampled.reshape(b, 4, len(EDGES), cfg['along'], cfg['normal']).permute(0, 2, 1, 3, 4)
    valid = (sampled[:, :, 3] >= 1.-1e-6) & geometry['edge_valid'][..., None, None]
    gradient_xy = sampled[:, :, 1:3].permute(0, 1, 3, 4, 2)
    gn = (gradient_xy*geometry['normal'][:, :, None, None]).sum(-1).abs()
    gt = (gradient_xy*geometry['tangent'][:, :, None, None]).sum(-1).abs()
    magnitude = torch.linalg.vector_norm(gradient_xy, dim=-1)*valid
    contrast_max = magnitude.flatten(-2).amax(-1)
    scale = contrast_max.clamp_min(cfg['contrast_floor'])
    strength = ((magnitude-cfg['contrast_floor']).clamp_min(0.)/
                (contrast_max-cfg['contrast_floor']).clamp_min(1e-8)[..., None, None]).clamp_max(1.)
    strength = strength*valid
    corridors = torch.stack((sampled[:, :, 0]*valid, gn/scale[..., None, None]*valid,
        gt/scale[..., None, None]*valid, valid.to(image_gray.dtype)), 2)
    coverage = valid.any(-1).to(image_gray.dtype).mean(-1)
    geometry.update(contrast_max=contrast_max, contrast_present=contrast_max > cfg['contrast_floor'],
        corridor_coverage=coverage, edge_valid=geometry['edge_valid'] & (coverage >= cfg['minimum_coverage']),
        points_input=points, normal_offsets=normal_offsets, sample_xy=xy)
    return corridors, valid, strength, geometry


def local_hough(weights, sample_valid, geometry, *, config=None):
    """Bilinear aggregation along local candidate lines, then a soft posterior.

    In local tangent/normal coordinates (s,u), a candidate satisfies
    -s*sin(alpha)+u*cos(alpha)=rho. Mean score normalizes observed line length.
    This task-adapted differentiable local Hough is not a paper reproduction.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    b, e, along, normal = weights.shape
    alpha = torch.linspace(-math.radians(cfg['angle_degrees']), math.radians(cfg['angle_degrees']),
        cfg['angle_bins'], device=weights.device, dtype=weights.dtype)
    rho = torch.linspace(-cfg['halfwidth'], cfg['halfwidth'], cfg['normal'],
        device=weights.device, dtype=weights.dtype)
    u = ((rho[None, None, None, :, None]
        + geometry['along_coordinates'][:, :, None, None]*alpha.sin()[None, None, :, None, None])
        / alpha.cos()[None, None, :, None, None])
    y = torch.linspace(-1., 1., along, device=weights.device, dtype=weights.dtype)
    grid = torch.stack((u/cfg['halfwidth'], y.expand_as(u)), -1)
    sampled = F.grid_sample(torch.stack((weights, sample_valid.to(weights.dtype)), 2).reshape(b*e, 2, along, normal),
        grid.reshape(b*e, cfg['angle_bins']*cfg['normal'], along, 2),
        align_corners=True, padding_mode='zeros', mode='bilinear')
    sampled = sampled.reshape(b, e, 2, cfg['angle_bins'], cfg['normal'], along)
    # Weights already include local sample validity; fractional boundary support
    # normalizes partial bilinear samples without multiplying the mask twice.
    mass = sampled[:, :, 1].sum(-1)
    coverage = mass/along
    scores = sampled[:, :, 0].sum(-1)/mass.clamp_min(1e-8)
    candidate_valid = (coverage >= cfg['minimum_coverage']) & geometry['edge_valid'][..., None, None]
    active = candidate_valid.any(-1).any(-1) & geometry['contrast_present']
    # v3b: fixed GT-free point-conditioned prior BEFORE posterior selection.
    # rho uses input pixels; alpha and its edge-length-dependent sigma use radians.
    sigma_alpha = torch.atan(weights.new_tensor(2.*math.sqrt(2.))/geometry['length'].clamp_min(1e-8))
    prior = (rho[None, None, None].square()/(2.*2.**2)
        + alpha[None, None, :, None].square()/(2.*sigma_alpha[..., None, None].square()))
    logits = (scores/cfg['temperature']-prior).masked_fill(~candidate_valid, -1e4)
    probability = logits.flatten(-2).softmax(-1).reshape_as(logits)
    probability = probability*candidate_valid*active[..., None, None]
    probability = probability/probability.sum((-2, -1), keepdim=True).clamp_min(1e-8)
    rho_mean = (probability*rho[None, None, None]).sum((-2, -1))
    cos_mean = (probability*alpha.cos()[None, None, :, None]).sum((-2, -1))
    sin_mean = (probability*alpha.sin()[None, None, :, None]).sum((-2, -1))
    mean_normal = cos_mean[..., None]*geometry['normal']-sin_mean[..., None]*geometry['tangent']
    line_c = -(mean_normal*geometry['midpoint']).sum(-1)-rho_mean
    norm = torch.linalg.vector_norm(mean_normal, dim=-1)
    line = torch.cat((mean_normal, line_c[..., None]), -1)/norm.clamp_min(1e-8)[..., None]
    active = active & (norm > 1e-8)
    entropy = -(probability*probability.clamp_min(1e-12).log()).sum((-2, -1))
    maximum_entropy = candidate_valid.sum((-2, -1)).clamp_min(2).to(weights.dtype).log()
    normalized_entropy = (entropy/maximum_entropy).clamp(0., 1.)
    concentration = (1.-normalized_entropy).clamp_min(cfg['confidence_floor'])
    confidence = concentration*active
    return dict(line_h_input=line, confidence=confidence, edge_valid=active,
        rho_input_px=rho_mean, angle_offset_radians=torch.atan2(sin_mean, cos_mean.clamp_min(1e-8)),
        normalized_entropy=normalized_entropy, probability=probability,
        candidate_scores=scores, candidate_valid=candidate_valid, candidate_coverage=coverage)


def regularized_refine(points_input, line_h_input, confidence, gate_logits,
                       point_valid, *, anchor_precision=1., maximum_shift=4.):
    """Stable displacement-form WLS; its smallest eigenvalue is >= anchor.

    gate_logits are signed tanh gates. At zero, output is exactly the original
    input without a coordinate round trip. Private branch gradients open after
    the first gate update; zero initialization does not claim otherwise.
    """
    if anchor_precision <= 0 or maximum_shift <= 0:
        raise ValueError('Positive anchor precision and displacement cap required')
    if bool((confidence < 0).any()):
        raise ValueError('Nonnegative line confidence required for SPD WLS')
    incidence = points_input.new_zeros((8, len(EDGES)))
    for edge, (u, v) in enumerate(EDGES):
        incidence[u, edge] = incidence[v, edge] = 1.
    finite = torch.isfinite(points_input).all(-1)
    safe = torch.where(finite[..., None], points_input, torch.zeros_like(points_input))
    n, c = line_h_input[..., :2], line_h_input[..., 2]
    line_finite = torch.isfinite(line_h_input).all(-1) & torch.isfinite(confidence)
    n = torch.where(line_finite[..., None], n, torch.zeros_like(n))
    c = torch.where(line_finite, c, torch.zeros_like(c))
    safe_confidence = torch.where(line_finite, confidence, torch.zeros_like(confidence))
    weights = safe_confidence[:, None]*incidence[None]
    weights = torch.where((point_valid & finite)[:, :8, None], weights, torch.zeros_like(weights))
    matrix = (torch.eye(2, device=safe.device, dtype=safe.dtype)[None, None]*anchor_precision
        + torch.einsum('bke,bei,bej->bkij', weights, n, n))
    residual = torch.einsum('bki,bei->bke', safe[:, :8], n)+c[:, None]
    rhs = -torch.einsum('bke,bke,bei->bki', weights, residual, n)
    delta = torch.linalg.solve(matrix, rhs[..., None])[..., 0]
    norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    bounded = delta*(maximum_shift/norm.clamp_min(1e-8)).clamp_max(1.)
    gates = gate_logits.tanh()
    if gates.ndim == 1:
        gates = gates[None].expand(len(safe), -1)
    if gates.shape != (len(safe), 8):
        raise ValueError('Eight shared or per-image corner gate logits required')
    movement = bounded*gates[..., None]
    all_delta = torch.cat((movement, torch.zeros_like(safe[:, 8:9])), 1)
    return points_input+all_delta, dict(point_delta_input=all_delta, wls_delta_input=delta,
        bounded_delta_input=bounded, signed_gate=gates, wls_matrix=matrix)


class LocalHoughPointRefiner(nn.Module):
    def __init__(self, arm='hough', **config):
        super().__init__()
        if arm not in ARMS or set(config)-set(DEFAULT_CONFIG):
            raise ValueError('Unknown arm or local geometry configuration')
        self.arm, self.config = arm, {**DEFAULT_CONFIG, **config}
        self.model_config = {'arm': arm, **self.config}
        self.cnn = nn.Sequential(nn.Conv2d(4, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.GELU())
        self.weight_head = nn.Conv2d(16, 1, 1)
        self.direct_head = nn.Sequential(nn.Linear(16, 16), nn.GELU(), nn.Linear(16, 2))
        self.gate_logits = nn.Parameter(torch.zeros(8))

    @property
    def inactive_parameter_prefixes(self):
        return ('direct_head.',) if self.arm == 'hough' else ('weight_head.',)

    def forward(self, batch):
        forbidden = [k for k in batch if k in ('targets', 'target', 'labels', 'gt', 'gt_points', 'loss_valid', 'matched') or k.startswith('gt_')]
        if forbidden:
            raise ValueError(f'Annotation/target fields are forbidden model inputs: {forbidden}')
        if any(k not in batch for k in INPUT_KEYS):
            raise ValueError('Missing local-refinement model inputs')
        image, content = batch['image_gray'], batch['input_content_mask']
        points, affine, valid = batch['baseline_points'], batch['raw_to_input_affine'], batch['point_valid'].bool()
        b = len(image)
        if image.shape != (b, 1, 640, 640) or content.shape != image.shape or image.dtype != torch.float32:
            raise ValueError('FP32 grayscale and content mask [B,1,640,640] required')
        if not bool(torch.isfinite(image).all()) or bool((image < 0).any()) or bool((image > 1).any()):
            raise ValueError('Image intensity must be finite and in [0,1]')
        if points.shape != (b, 9, 2) or valid.shape != (b, 9) or affine.shape != (b, 2, 3):
            raise ValueError('Expected nine raw points, validity and 2x3 affine')
        affine = affine.to(device=image.device, dtype=image.dtype)
        points = points.to(device=image.device, dtype=image.dtype)
        if not bool(torch.isfinite(affine).all()) or bool((torch.linalg.det(affine[..., :2]).abs() < 1e-8).any()):
            raise ValueError('Finite invertible raw-to-input affine required')
        cfg = self.config
        corridors, sample_valid, strength, geometry = extract_corridors(image, content, points, affine, valid, config=cfg)
        features = self.cnn(corridors.reshape(b*len(EDGES), 4, cfg['along'], cfg['normal']))
        if self.arm == 'hough':
            learned = self.weight_head(features).sigmoid().reshape(b, len(EDGES), cfg['along'], cfg['normal'])
            evidence = local_hough(learned*strength, sample_valid, geometry, config=cfg)
        else:
            hidden = features.reshape(b, len(EDGES), 16, cfg['along'], cfg['normal'])
            pooled = (hidden*sample_valid[:, :, None]).sum((-2, -1))/sample_valid.sum((-2, -1)).clamp_min(1)[..., None]
            direct = self.direct_head(pooled).tanh()
            rho = direct[..., 0]*cfg['halfwidth']
            alpha = direct[..., 1]*math.radians(cfg['angle_degrees'])
            normal = alpha.cos()[..., None]*geometry['normal']-alpha.sin()[..., None]*geometry['tangent']
            line_c = -(normal*geometry['midpoint']).sum(-1)-rho
            active = geometry['edge_valid'] & geometry['contrast_present']
            evidence = dict(line_h_input=torch.cat((normal, line_c[..., None]), -1),
                confidence=active.to(image.dtype), edge_valid=active,
                rho_input_px=rho, angle_offset_radians=alpha,
                normalized_entropy=torch.zeros_like(rho), probability=None,
                candidate_scores=None, candidate_valid=None, candidate_coverage=None)
        # A flat/low-contrast corridor has zero confidence even if the CNN bias
        # produces nonzero weights. Entropy is uncertainty, not correctness.
        contrast = ((geometry['contrast_max']-cfg['contrast_floor'])/
            (geometry['contrast_max']+cfg['contrast_floor']).clamp_min(1e-8)).clamp(0., 1.)
        confidence = torch.where(evidence['edge_valid'] & geometry['contrast_present'],
            (evidence['confidence']*contrast).clamp_min(cfg['confidence_floor']), torch.zeros_like(contrast))
        _, fusion = regularized_refine(geometry['points_input'], evidence['line_h_input'], confidence,
            self.gate_logits, geometry['point_valid'], anchor_precision=cfg['anchor_precision'], maximum_shift=cfg['maximum_shift'])
        raw_delta = input_delta_to_raw(fusion['point_delta_input'], affine)
        result = points+raw_delta
        result = torch.cat((result[:, :8], points[:, 8:9]), 1)
        diagnostics = dict(arm=self.arm, **evidence, **fusion, edge_confidence=confidence,
            edge_length_input=geometry['length'], corridor_coverage=geometry['corridor_coverage'],
            contrast_max=geometry['contrast_max'], point_valid=geometry['point_valid'],
            point_delta_raw=raw_delta, line_weights_are_calibrated_correctness=False,
            physical_visibility_supervision=False, inactive_parameter_prefixes=self.inactive_parameter_prefixes,
            uncertainty_semantics='Hough entropy describes spread of local candidates, not calibrated correctness; control has no posterior entropy.',
            first_step_gradient_semantics='Zero gate gives exact identity; branch gradients open after a nonzero gate update.')
        return result, diagnostics
