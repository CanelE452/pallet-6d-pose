"""Learn one quality score for an entire predicted eight-corner layout.

All three arms have identical trainable modules. Their only difference is which
predicted image/line evidence enters those modules. No target or GT mask is read.
Semantic corner IDs remain meaningful; the ordering of layout candidates does not.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

ARMS = ('point_only', 'point_segment', 'point_segment_hough')
INPUT_KEYS = ('p4', 'baseline_points', 'layouts', 'point_conf', 'diagonal',
              'raw_to_input_affine', 'input_shape_hw', 'line_h', 'peak_logits', 'peak_valid')
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0),
         (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7))
FEATURE_STRIDE = 16.


def raw_to_input(raw_xy, affine):
    """Map any B,...,2 original-pixel query array with the saved content affine."""
    if raw_xy.ndim < 3 or raw_xy.shape[-1] != 2 or affine.shape != (raw_xy.shape[0], 2, 3):
        raise ValueError('Expected queries[B,...,2] and affine[B,2,3]')
    flat = raw_xy.reshape(raw_xy.shape[0], -1, 2)
    result = torch.bmm(flat, affine[:, :, :2].transpose(1, 2)) + affine[:, None, :, 2]
    return result.reshape_as(raw_xy)


def sample_p4(feature, input_xy, input_shape_hw):
    """Sample zero-padded P4 using input-pixel centres and the actual footprint.

    align_corners=False implements feature index=input_pixel/16 - 0.5.
    The normalizer is the padded feature canvas, not the shorter actual image.
    Queries outside the actual input footprint are explicitly zero and marked.
    """
    b, c, h, w = feature.shape
    if input_xy.shape[0] != b or input_xy.shape[-1] != 2 or input_shape_hw.shape != (b, 2):
        raise ValueError('Query or actual-input shape differs from feature batch')
    query_shape = input_xy.shape[1:-1]
    xy = input_xy.reshape(b, -1, 2)
    actual_wh = input_shape_hw[:, [1, 0]].to(xy.dtype)
    in_frame = ((xy >= 0.) & (xy < actual_wh[:, None])).all(-1)
    normalizer = xy.new_tensor([w*FEATURE_STRIDE, h*FEATURE_STRIDE])
    grid = 2.*xy/normalizer-1.
    values = F.grid_sample(feature, grid[:, None], mode='bilinear',
        padding_mode='zeros', align_corners=False)[:, :, 0].transpose(1, 2)
    values = values * in_frame[..., None].to(values.dtype)
    return values.reshape(b, *query_shape, c), in_frame.reshape(b, *query_shape)


class LayoutVerifier(nn.Module):
    def __init__(self, arm='point_segment_hough', *, visual_channels=16, width=64, n_layers=2):
        super().__init__()
        if arm not in ARMS or visual_channels <= 0 or width % 4 or n_layers <= 0:
            raise ValueError('Invalid verifier arm or dimensions')
        self.arm = arm
        self.model_config = dict(arm=arm, visual_channels=visual_channels, width=width, n_layers=n_layers)
        self.visual_channels = visual_channels
        self.width = width
        self.visual_projection = nn.Sequential(nn.Conv2d(128, visual_channels, 1), nn.GELU())
        self.corner_encoder = nn.Sequential(nn.Linear(9*visual_channels+9+7, width),
            nn.LayerNorm(width), nn.GELU(), nn.Linear(width, width))
        self.line_encoder = nn.Sequential(nn.Linear(4*4, 32), nn.GELU(), nn.Linear(32, 16))
        self.edge_encoder = nn.Sequential(nn.Linear(8*3*visual_channels+8*3+9+16, width),
            nn.LayerNorm(width), nn.GELU(), nn.Linear(width, width))
        self.semantic_embedding = nn.Embedding(20, width)
        # Context uses sampled endpoint appearance and predictions. A full-map
        # mean would leak altered edge interiors into the point-only control.
        self.context_encoder = nn.Sequential(nn.Linear(visual_channels+18+9, width), nn.GELU(), nn.Linear(width, width))
        layer = nn.TransformerEncoderLayer(d_model=width, nhead=4, dim_feedforward=2*width,
            dropout=0., activation='gelu', batch_first=True, norm_first=True)
        self.interaction = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(width)
        self.quality = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        self.corner_error = nn.Sequential(nn.Linear(width, 32), nn.GELU(), nn.Linear(32, 1))
        offsets = [(x*FEATURE_STRIDE, y*FEATURE_STRIDE) for y in (-1., 0., 1.) for x in (-1., 0., 1.)]
        self.register_buffer('corner_patch_offsets', torch.tensor(offsets), persistent=True)
        self.register_buffer('edge_t', (torch.arange(8, dtype=torch.float32)+.5)/8., persistent=True)
        self.register_buffer('normal_offsets', torch.tensor([-FEATURE_STRIDE, 0., FEATURE_STRIDE]), persistent=True)
        self.register_buffer('edge_indices', torch.tensor(EDGES, dtype=torch.long), persistent=True)

    def _validate_shapes(self, batch):
        missing = [key for key in INPUT_KEYS if key not in batch]
        if missing:
            raise ValueError(f'Missing model inputs: {missing}')
        forbidden = [key for key in batch if 'gt' in key.lower() or key.lower() in ('targets', 'labels', 'loss_valid', 'loss_matched')]
        if forbidden:
            raise ValueError(f'Target/GT fields must stay outside model inputs: {forbidden}')
        b = batch['p4'].shape[0]
        if batch['p4'].shape != (b, 128, 40, 40):
            raise ValueError('P4 must be bottom-padded FP32[B,128,40,40]')
        q = batch['layouts']
        if q.ndim != 4 or q.shape[0] != b or q.shape[2:] != (9, 2) or q.shape[1] < 1:
            raise ValueError('Expected layouts[B,L,9,2] with semantic IDs0..8')
        expected = dict(baseline_points=(b, 9, 2), point_conf=(b, 9), raw_to_input_affine=(b, 2, 3),
            input_shape_hw=(b, 2), line_h=(b, 12, 4, 3), peak_logits=(b, 12, 4), peak_valid=(b, 12, 4))
        for key, shape in expected.items():
            if batch[key].shape != shape:
                raise ValueError(f'{key} has {tuple(batch[key].shape)}, expected {shape}')
        if batch['diagonal'].numel() != b:
            raise ValueError('One original-image diagonal is required per frame')

    def _project_visual(self, p4, input_shape_hw):
        b, _, h, w = p4.shape
        # Mask padded cells BEFORE projection and AFTER its learned bias.
        yy = (torch.arange(h, device=p4.device, dtype=p4.dtype)+.5)*FEATURE_STRIDE
        xx = (torch.arange(w, device=p4.device, dtype=p4.dtype)+.5)*FEATURE_STRIDE
        mask = (yy[None, :, None] < input_shape_hw[:, 0, None, None]) & (xx[None, None, :] < input_shape_hw[:, 1, None, None])
        projected = self.visual_projection(p4*mask[:, None].to(p4.dtype))
        projected = projected*mask[:, None].to(projected.dtype)
        return projected, mask

    def _line_cues(self, batch, layouts):
        peak_valid = batch['peak_valid'].bool()
        h = torch.where(peak_valid[..., None], batch['line_h'], torch.zeros_like(batch['line_h']))
        logits = torch.where(peak_valid, batch['peak_logits'], torch.zeros_like(batch['peak_logits']))
        q = layouts[:, :, :8]
        first = q[:, :, self.edge_indices[:, 0]]
        second = q[:, :, self.edge_indices[:, 1]]
        sigma = (.01*batch['diagonal'].reshape(-1, 1, 1, 1)).clamp_min(1.)
        d1 = ((first[..., None, :]*h[:, None, ..., :2]).sum(-1)+h[:, None, ..., 2])/sigma
        d2 = ((second[..., None, :]*h[:, None, ..., :2]).sum(-1)+h[:, None, ..., 2])/sigma
        valid = peak_valid[:, None].expand(-1, layouts.shape[1], -1, -1)
        mass = logits.sigmoid()[:, None].expand_as(d1)
        # Preserve absolute sigmoid masses; never renormalize them to sum to1.
        cues = torch.stack((d1, d2, mass, valid.to(mass.dtype)), dim=-1)
        cues = torch.where(valid[..., None], cues, torch.zeros_like(cues))
        if self.arm != 'point_segment_hough':
            cues = torch.zeros_like(cues)
        return cues.reshape(*cues.shape[:-2], 16)

    def forward(self, batch, *, return_diagnostics=False, token_order=None):
        self._validate_shapes(batch)
        p4 = batch['p4']
        # Cache affines/line equations may retain float64 for provenance. The
        # learned verifier consistently uses the projected-feature precision.
        batch = {k: (v.to(dtype=p4.dtype) if torch.is_tensor(v) and v.is_floating_point() else v)
                 for k, v in batch.items()}
        layouts = batch['layouts']
        b, n = layouts.shape[:2]
        base = batch['baseline_points']
        diagonal = batch['diagonal'].reshape(b, 1, 1).clamp_min(1.)
        visual, feature_valid = self._project_visual(p4, batch['input_shape_hw'])
        input_corners = raw_to_input(layouts[:, :, :8], batch['raw_to_input_affine'])
        patch_xy = input_corners[..., None, :]+self.corner_patch_offsets
        patch, patch_valid = sample_p4(visual, patch_xy, batch['input_shape_hw'])
        centre = base[:, 8:9]
        position = (layouts[:, :, :8]-centre[:, None])/diagonal[:, None]
        baseline_position = ((base[:, :8]-centre)/diagonal)[:, None].expand(-1, n, -1, -1)
        displacement = (layouts[:, :, :8]-base[:, None, :8])/diagonal[:, None]
        confidence = batch['point_conf'][:, None, :8, None].expand(-1, n, -1, -1)
        corner_input = torch.cat((patch.flatten(-2), patch_valid.to(patch.dtype),
            position, baseline_position, displacement, confidence), dim=-1)
        corner_tokens = self.corner_encoder(corner_input)

        u, v = self.edge_indices[:, 0], self.edge_indices[:, 1]
        start, end = input_corners[:, :, u], input_corners[:, :, v]
        t = self.edge_t[None, None, None, :, None]
        if self.arm == 'point_only':
            # Same encoder/dimensions, but no new edge-interior image query.
            endpoint_feature = patch[..., 4, :]
            endpoint_mask = patch_valid[..., 4].to(patch.dtype)
            interpolation = endpoint_feature[:, :, u, None]*(1.-t)+endpoint_feature[:, :, v, None]*t
            edge_visual = interpolation[..., None, :].expand(-1, -1, -1, -1, 3, -1)
            interpolation_mask = endpoint_mask[:, :, u, None]*(1.-self.edge_t)+endpoint_mask[:, :, v, None]*self.edge_t
            edge_valid = interpolation_mask[..., None].expand(-1, -1, -1, -1, 3)
        else:
            tangent = end-start
            length = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp_min(1e-6)
            normal = torch.stack((-tangent[..., 1], tangent[..., 0]), dim=-1)/length
            centres = start[..., None, :]*(1.-t)+end[..., None, :]*t
            roi_xy = centres[..., None, :]+normal[..., None, None, :]*self.normal_offsets[None, None, None, None, :, None]
            edge_visual, edge_valid = sample_p4(visual, roi_xy, batch['input_shape_hw'])
            edge_valid = edge_valid.to(edge_visual.dtype)
        edge_direction = position[:, :, v]-position[:, :, u]
        edge_length = torch.linalg.vector_norm(edge_direction, dim=-1, keepdim=True)
        edge_geometry = torch.cat((position[:, :, u], position[:, :, v], edge_direction,
            edge_length, confidence[:, :, u], confidence[:, :, v]), dim=-1)
        line_cues = self._line_cues(batch, layouts)
        line_embedding = self.line_encoder(line_cues)
        edge_tokens = self.edge_encoder(torch.cat((edge_visual.flatten(-3), edge_valid.flatten(-2),
            edge_geometry, line_embedding), dim=-1))
        tokens = torch.cat((corner_tokens, edge_tokens), dim=2)
        tokens = tokens+self.semantic_embedding.weight[None, None]
        image_context = patch[..., 4, :].mean(2)
        base_context = ((base-centre)/diagonal).flatten(1)
        context = self.context_encoder(torch.cat((image_context, base_context[:, None].expand(-1, n, -1),
            batch['point_conf'][:, None].expand(-1, n, -1)), dim=-1))
        tokens = tokens+context[:, :, None]
        flat = tokens.reshape(b*n, 20, self.width)
        inverse = None
        if token_order is not None:
            order = torch.as_tensor(token_order, dtype=torch.long, device=flat.device)
            if order.shape != (20,) or not torch.equal(torch.sort(order).values, torch.arange(20, device=flat.device)):
                raise ValueError('token_order must permute all20 tokens and their already-attached semantic IDs')
            inverse = torch.argsort(order)
            flat = flat[:, order]
        output = self.output_norm(self.interaction(flat))
        if inverse is not None:
            output = output[:, inverse]
        output = output.reshape(b, n, 20, self.width)
        cost = self.quality(output.mean(2)).squeeze(-1)
        if not return_diagnostics:
            return cost
        return cost, dict(percorner_error=self.corner_error(output[:, :, :8]).squeeze(-1),
            corner_input_coverage=patch_valid.to(cost.dtype).mean(-1),
            edge_input_coverage=edge_valid.mean((-2, -1)),
            edge_interior_image_evidence_used=self.arm != 'point_only',
            hough_evidence_used=self.arm == 'point_segment_hough',
            line_cues=line_cues, line_mass_semantics='Absolute sigmoid(logit), not a normalized probability of correctness.',
            point_confidence_semantics='Frozen predicted confidence input, not calibrated pixel accuracy.',
            score_semantics='Learned unconstrained whole-layout cost; lower is intended to mean lower supervised coordinate error.')
