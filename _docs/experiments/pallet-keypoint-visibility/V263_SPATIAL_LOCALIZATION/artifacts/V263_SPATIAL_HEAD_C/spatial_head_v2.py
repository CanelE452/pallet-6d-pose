"""Small spatial query head. No file I/O, backbone load, PnP or data discovery.

Official DINOv2 intermediate block8, norm=True, patch grid16x16x384.
FP16 storage is converted to FP32 at head input. No pooled CLS representation.
"""
import torch
from torch import nn
from torch.nn import functional as F

GRID = 16
LAYER_INDEX = 8
LOC_MATCH, LOC_NO_MATCH, LOC_OUT_OF_SEARCH = 0, 1, 2


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


class SpatialPointHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(392, 32, 1), nn.ReLU(),
                                  nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.heatmap = nn.Conv2d(32, 1, 1)
        self.location_state = nn.Linear(32, 3)
        self.visibility = nn.Linear(32, 1)
        for module in (self.heatmap, self.location_state, self.visibility):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, patch_tokens, token_support, native_slot):
        require(patch_tokens.ndim == 3 and tuple(patch_tokens.shape[1:]) == (256, 384), 'N256x384 spatial tokens')
        n = len(patch_tokens)
        require(token_support.shape == (n, 16, 16) and token_support.dtype == torch.bool, 'Input-only bool token support')
        require(torch.isfinite(patch_tokens).all(), 'Finite frozen tokens')
        x = patch_tokens.float().transpose(1, 2).reshape(n, 384, 16, 16)
        require(native_slot.shape == (n,) and native_slot.dtype == torch.int64 and ((native_slot >= 0) & (native_slot < 8)).all(), 'Immutable native0..7, no GT remap')
        identity = F.one_hot(native_slot, 8).float()[:, :, None, None].expand(-1,-1,16,16)
        x = torch.cat((x,identity),1)*token_support[:, None]
        hidden = self.stem(x)
        raw = self.heatmap(hidden)[:, 0]
        logits = raw.masked_fill(~token_support, -1e4)
        p = logits.flatten(1).softmax(1).reshape(n, 16, 16)*token_support
        p = p/p.sum((1, 2), keepdim=True).clamp_min(1e-12)
        denominator = token_support.sum((1, 2)).clamp_min(1).float()
        pooled = (hidden*token_support[:, None]).sum((2, 3))/denominator[:, None]
        state = self.location_state(pooled)
        visibility = self.visibility(pooled)[:, 0]  # E logit; fixed threshold0
        yy, xx = torch.meshgrid(torch.arange(16, device=x.device), torch.arange(16, device=x.device), indexing='ij')
        xy = (torch.stack((xx, yy), -1).float()+.5)/16
        uv = (p[..., None]*xy).sum((1, 2))  # heatmap soft-argmax; no learned offset head
        return {'heatmap_logits': logits, 'probability': p, 'uv_subpixel': uv,
                'location_logits': state, 'visibility_E_logit': visibility,
                'input_supported': token_support.any(2).any(1)}



def fixed_loc_mask(input_available, direct_cue, identity_correct, reference_xy, q0, crop_side):
    """GT mask only; unknown direct cue/identity must be encoded False, not promoted.

    Root/A supplies separate authority fields and knownness counts. No predicted
    visibility, confidence, heatmap or inferred mesh support enters this mask.
    """
    require(q0.shape == reference_xy.shape and q0.shape[1:] == (2,), 'Reference and original q shapes')
    require(crop_side.shape == (len(q0),), 'Per-row input-only side')
    valid_geometry = torch.isfinite(q0).all(1) & torch.isfinite(crop_side) & (crop_side > 0)
    safe_side = torch.where(valid_geometry, crop_side, torch.ones_like(crop_side))
    uv = (reference_xy-q0)/safe_side[:, None]+.5
    in_range = torch.isfinite(uv).all(1) & (uv >= 0).all(1) & (uv < 1).all(1)
    mask = input_available & direct_cue & identity_correct & valid_geometry & in_range
    return mask, uv, in_range


def losses(output, reference_uv, loc_valid, visibility_target, location_target, mode):
    """Fixed masks, D0/E1 and location-state -1 unknown. LOC never gated by predicted VIS."""
    require(mode in ('LOC', 'VIS', 'JOINT'), 'Three fixed objectives')
    n = len(reference_uv)
    require(reference_uv.shape == (n, 2) and loc_valid.shape == (n,) and loc_valid.dtype == torch.bool, 'Fixed LOC sidecar')
    require(visibility_target.shape == location_target.shape == (n,), 'Separate visibility/state targets')
    require(((visibility_target >= -1) & (visibility_target <= 1)).all(), 'Visibility -1/D0/E1 only')
    require(((location_target >= -1) & (location_target <= 2)).all(), 'LOCATION -1unknown/0match/1no-match/2out-of-search; not visibility')
    require(torch.isfinite(reference_uv[loc_valid]).all() and
            ((reference_uv[loc_valid] >= 0) & (reference_uv[loc_valid] < 1)).all(), 'No outside target clamped onto border')
    zero = output['visibility_E_logit'].sum()*0 if mode == 'VIS' else output['heatmap_logits'].sum()*0
    loc_heat, state_loss, vis_loss = zero, zero, zero
    if mode in ('LOC', 'JOINT') and bool(loc_valid.any()):
        uv = reference_uv[loc_valid]
        yy, xx = torch.meshgrid(torch.arange(16, device=uv.device), torch.arange(16, device=uv.device), indexing='ij')
        centres = torch.stack((xx, yy), -1).float()+.5
        distance = (centres[None]-uv[:, None, None]*16).square().sum(-1)
        target = torch.exp(-distance/(2*.7**2))
        support = output['heatmap_logits'][loc_valid] > -9999
        target = target*support
        require((target.sum((1, 2)) > 0).all(), 'Location target has input support')
        target = target/target.sum((1, 2), keepdim=True)
        loc_heat = -(target.flatten(1)*F.log_softmax(output['heatmap_logits'][loc_valid].flatten(1), 1)).sum(1).mean()
    if mode in ('LOC', 'JOINT'):
        known = location_target >= 0
        if bool(known.any()):
            state_loss = F.cross_entropy(output['location_logits'][known], location_target[known].long())
    if mode in ('VIS', 'JOINT'):
        known = visibility_target >= 0
        if bool(known.any()):
            vis_loss = F.binary_cross_entropy_with_logits(output['visibility_E_logit'][known], visibility_target[known].float())
    total = loc_heat+state_loss+vis_loss
    return total, {'LOC_heatmap': loc_heat, 'LOC_state': state_loss, 'VIS': vis_loss}


def diagnostics(output):
    """Uncalibrated distribution diagnostics; never human/native truth."""
    p = output['probability']
    n = len(p)
    yy, xx = torch.meshgrid(torch.arange(16, device=p.device), torch.arange(16, device=p.device), indexing='ij')
    xy = (torch.stack((xx, yy), -1).float()+.5)/16
    mean = (p[..., None]*xy).sum((1, 2))
    d = xy[None]-mean[:, None, None]
    covariance = (p[..., None, None]*d[..., :, None]*d[..., None, :]).sum((1, 2))
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    index = p.flatten(1).argmax(1)
    row, col = index//16, index % 16
    far = ((xx[None]-col[:, None, None]).abs() > 1) | ((yy[None]-row[:, None, None]).abs() > 1)
    maxp = p.flatten(1).max(1).values
    second = (p*far).flatten(1).max(1).values
    border = torch.zeros((16, 16), dtype=torch.bool, device=p.device)
    border[0] = True; border[-1] = True; border[:, 0] = True; border[:, -1] = True
    return {'max_probability': maxp, 'normalized_entropy': -(p*p.clamp_min(1e-12).log()).sum((1, 2))/torch.log(p.new_tensor(256.)),
            'nonlocal_peak_ratio': maxp/second.clamp_min(1e-12),
            'border_mass': p[:, border].sum(1), 'std_max_normalized': eigenvalues[:, 1].sqrt(),
            'anisotropy': eigenvalues[:, 1]/eigenvalues[:, 0].clamp_min(1e-8),
            'location_match_probability': output['location_logits'].softmax(1)[:, LOC_MATCH],
            'location_state': output['location_logits'].argmax(1), 'heatmap_subpixel_uv': mean}


def acceptance(output, rule):
    """Root-predeclared GT-free rule; SYN DEV may select from preregistered rules only."""
    d = diagnostics(output)
    accepted = (output['input_supported'] & (d['location_state'] == LOC_MATCH) &
                (d['location_match_probability'] >= rule['min_location_match_probability']) &
                (d['max_probability'] >= rule['min_peak_probability']) &
                (d['normalized_entropy'] <= rule['max_normalized_entropy']) &
                (d['nonlocal_peak_ratio'] >= rule['min_nonlocal_peak_ratio']) &
                (d['border_mass'] <= rule['max_border_mass']) &
                (d['std_max_normalized'] <= rule['max_std_normalized']) &
                (d['anisotropy'] <= rule['max_anisotropy']))
    return accepted, d


def image_points(output, q0, crop_side, accepted):
    proposed = q0+(output['uv_subpixel']-.5)*crop_side[:, None]
    return torch.where(accepted[:, None], proposed, q0)  # immutable original fallback


def simple_crop_side(predicted_2d_scale, factor):
    require(factor > 0, 'Root fixed simple scale factor')
    valid = torch.isfinite(predicted_2d_scale) & (predicted_2d_scale > 0)
    return predicted_2d_scale*factor, valid


def geometric_crop_side(initial_T, K, object_dimensions, object_fraction):
    """Initial pose/K/dimensions only. No GT pose, 2D-span alias or pose optimization."""
    require(initial_T.shape[1:] == (4, 4) and K.shape[1:] == (3, 3) and object_dimensions.shape[1:] == (3,), 'Explicit initial geometry shapes')
    require(object_fraction > 0, 'Root fixed object-to-crop fraction')
    signs = initial_T.new_tensor([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                 [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
    xyz = signs[None]*object_dimensions[:, None]/2
    camera = xyz@initial_T[:, :3, :3].transpose(1, 2)+initial_T[:, None, :3, 3]
    homogeneous = camera@K.transpose(1, 2)
    finite = (torch.isfinite(initial_T).flatten(1).all(1) & torch.isfinite(K).flatten(1).all(1) &
              torch.isfinite(object_dimensions).all(1) & (object_dimensions > 0).all(1) &
              (camera[:, :, 2] > 1e-8).all(1))
    uv = homogeneous[:, :, :2]/homogeneous[:, :, 2:].clamp_min(1e-8)
    spans = uv.amax(1)-uv.amin(1)
    side = spans.amax(1)*object_fraction
    valid = finite & torch.isfinite(side) & (side > 0)
    return torch.where(valid, side, torch.full_like(side, float('nan'))), valid


def semantic_visibility(output):
    """D0/E1 at fixed logit0; unavailable input is U-1, independent of LOC."""
    predicted = (output['visibility_E_logit'] > 0).to(torch.int8)
    return torch.where(output['input_supported'], predicted, torch.full_like(predicted, -1))


def choose_crop_side(simple_side, simple_valid, geometry_side, geometry_valid, variant):
    """Geometry failure has explicit SIMPLE fallback, never GT or row deletion."""
    require(variant in ('simple2D', 'initial_geometry'), 'Only Root two fixed crop variants')
    valid_simple = simple_valid & torch.isfinite(simple_side) & (simple_side > 0)
    valid_geometry = geometry_valid & torch.isfinite(geometry_side) & (geometry_side > 0)
    fallback = (~valid_geometry) & valid_simple if variant == 'initial_geometry' else torch.zeros_like(valid_simple)
    use_geometry = valid_geometry if variant == 'initial_geometry' else torch.zeros_like(valid_simple)
    side = torch.where(use_geometry, geometry_side, simple_side)
    valid = use_geometry | valid_simple
    return torch.where(valid, side, torch.full_like(side, float('nan'))), valid, fallback
