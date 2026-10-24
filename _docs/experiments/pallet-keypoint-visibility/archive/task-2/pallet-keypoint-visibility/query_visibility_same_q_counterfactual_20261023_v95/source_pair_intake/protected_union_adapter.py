"""Explicit V95 support-only adapter. Pinned V3/V2/V1 math remains unchanged."""
from pathlib import Path
import hashlib
import importlib.util
import numpy as np

V3_PATH = Path(__file__).resolve().parents[2] / 'query_visibility_matched_query_scalar_20261023_v94/paired_source_preparation/v3_integrated_same_q_builder/integrated_same_q_v3.py'
V3_SHA = 'ce40397c0bfd695aee4ee07dde425711c92e0c0f37fb854f8ffe02fd396517fb'
assert hashlib.sha256(V3_PATH.read_bytes()).hexdigest() == V3_SHA
_spec = importlib.util.spec_from_file_location('v95_pinned_v3', V3_PATH)
v3 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v3)


def closed_range_cells(box, hw):
    """Original closed physical-cue uncertainty -> all inclusive nearest cells."""
    if box is None or np.asarray(box).shape != (4,) or not np.isfinite(box).all():
        return None
    x0, y0, x1, y1 = map(float, box); h, w = hw
    if x0 > x1 or y0 > y1 or x0 < 0 or y0 < 0 or x1 > w - 1 or y1 > h - 1:
        return None
    lo = np.floor(np.array([x0, y0]) + .5).astype(int)
    hi = np.floor(np.array([x1, y1]) + .5).astype(int)
    out = np.zeros((h, w), bool); out[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1] = True
    return out


def protected_union(row, decision):
    v3.checked_row(row); v3.decision_identity(row, decision)
    if decision['before_aux_state'] != 'D' or not decision['paired_construction_eligible']:
        return None, 'ROOT_NOT_ELIGIBLE_D_BEFORE_U'
    if decision['source'] == 'LOCO':
        return v3.controlled_protected_support(row, decision)
    if decision['source'] != 'PALLOX':
        raise ValueError('Unfrozen source')
    regions = decision['protected_regions_GT_only']
    if not regions:
        return None, 'PALLOX_NO_RETAINED_FULL_D_RANGE_U'
    union = np.zeros(tuple(row['raw_hw']), bool)
    for region in regions:
        assert region['kind'] == 'PALLOX_CLOSED_NATIVE_UNCERTAINTY'
        part = closed_range_cells(region['bbox'], row['raw_hw'])
        if part is None:
            return None, 'PALLOX_UNUSABLE_OR_OUT_OF_IMAGE_RETAINED_RANGE_U'
        union |= part
    return union, 'PALLOX_ALL_RETAINED_FULL_D_RANGES_ACTUAL_UNION'


def edit_geometry(native_rgb, row, decision, donor_rgb, donor_visible, donor_meta):
    """Return only controlled pair+cutout+certificate; no features/model/P3."""
    normalized = v3.checked_row(row)
    assert native_rgb.dtype == np.uint8 and native_rgb.shape == (*row['raw_hw'], 3)
    protected, reason = protected_union(row, decision)
    facts = dict(pair_ready=False, reason=reason, donor_count=0 if donor_meta is None else 1,
                 protected_pixels=0 if protected is None else int(protected.sum()),
                 protected_sha256=None if protected is None else hashlib.sha256(protected.tobytes()).hexdigest(),
                 protected_region_count=len(decision['protected_regions_GT_only']),
                 no_query_or_geometry_change=True, null_unlocated_limit_count=decision.get('null_unlocated_limit_count'))
    if protected is None:
        return None, None, protected, facts
    donor_ok = (donor_meta is not None and donor_meta.get('foreign_opaque_cutout_admitted') is True
                and donor_meta.get('split') == normalized['split']
                and donor_meta.get('image_sha256') != row['image_sha256']
                and v3.v1.donor_family_allowed(donor_meta.get('object_id'), normalized['split']))
    if not donor_ok:
        facts['reason'] = 'ASSIGNED_DONOR_UNUSABLE_OR_PARTITION_NOT_ADMITTED_U'
        return None, None, protected, facts
    cutout, scale = v3.v2.support_scaled_cutout(donor_rgb, donor_visible, protected, row['L'])
    facts['scale'] = scale
    if cutout is None:
        facts['reason'] = scale['reason']; return None, None, protected, facts
    pair, mass = v3.v2.core_aligned_pair(normalized, cutout, protected)
    facts['mass'] = mass
    if pair is None:
        facts['reason'] = mass['reason']; return None, cutout, protected, facts
    (cxy, ca), (sxy, sa) = pair
    cover = v3.v1.compose(native_rgb, cutout, cxy, ca)
    sham = v3.v1.compose(native_rgb, cutout, sxy, sa)
    assert ca[protected].all() and not sa[protected].any()
    assert np.array_equal(sham[protected], native_rgb[protected])
    assert ca.sum() == sa.sum() == cutout['alpha'].sum()
    passed, replay_mass = v3.v2.mass_certificate(normalized, ca, sa); assert passed
    facts.update(pair_ready=True, reason='CONTROLLED_PAIR_READY', cover_xy=list(map(int, cxy)), sham_xy=list(map(int, sxy)),
                 same_RGBA_transform=True, protected_all_alpha1_cover=True, protected_all_alpha0_sham=True,
                 same_global_alpha_cells=int(ca.sum()), mass_recheck=replay_mass,
                 donor_transform=dict(source_bbox_xyxy=cutout['donor_native_bbox_xyxy'], resized_hw=cutout['resized_hw'],
                                      scale=cutout['scale'], opaque_core_xyxy=cutout['opaque_core_xyxy']))
    return (cover, sham), cutout, protected, facts
