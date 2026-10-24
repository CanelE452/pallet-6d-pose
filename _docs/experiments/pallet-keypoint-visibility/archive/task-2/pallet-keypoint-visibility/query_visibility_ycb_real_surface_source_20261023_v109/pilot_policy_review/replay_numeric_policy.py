"""Independent policy replay of saved numeric JSON only; no source-array IO."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
POLICY = ROOT / 'ROOT_SOURCE_POLICY_V2_CORRECTED_INVENTORY_BEFORE_BULK.json'
NUMERIC = PARENT / 'query_visibility_ycb_surface_cues_20261023_v106/numeric_pilot32/FIXED32_NUMERIC_EVIDENCE.json'


def binding(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def ray_at(ray, index):
    sensitivity = ray.get('sensitivity', [])
    return sensitivity[index] if len(sensitivity) > index else {}


def unique_owner(sample, owner):
    return (sample.get('in_bounds', False)
            and sample.get('all_visible_owner_indices') == [owner]
            and not sample.get('own_other_overlap', False))


def classify(obj, row):
    own = obj['pilot_pose']['instance_id']
    samples = row['all9_native_samples']
    centre = next(s for s in samples if s['offset_xy'] == [0, 0])
    a = ray_at(row['target_alone_fixed_point_ray'], 1)
    b = ray_at(row['target_alone_fixed_point_ray'], 2)
    ar, br = a.get('first_hit_range_mm'), b.get('first_hit_range_mm')
    stable = finite(ar) and finite(br) and abs(ar - br) <= .001
    gap = row['camera_range_mm'] - ar if stable else None
    geometry = ('S' if gap > 2 else 'SELF_VISIBLE' if abs(gap) <= 2 else 'UNKNOWN') if stable else 'UNKNOWN'
    ref_supported = bool(row['continuous_point_in_image'] and centre.get('in_bounds', False))
    positive = [s for s in samples if s.get('depth_valid', False) and finite(s.get('depth_mm')) and s['depth_mm'] > 0]
    intended_count = sum(unique_owner(s, own) for s in samples)
    centre_positive = centre in positive
    centre_own = unique_owner(centre, own) and centre.get('own_visible', False)
    integer_ray = ray_at(centre.get('target_alone_integer_pixel_ray', {}), 1)
    integer_z = integer_ray.get('first_hit_camera_Z_mm')
    centre_delta = centre['depth_mm'] - integer_z if centre_positive and finite(integer_z) else None
    d_checks = {
        'centre_unique_intended_owner': centre_own,
        'centre_positive_depth': centre_positive,
        'centre_integer_ray_64eps_finite': finite(integer_z),
        'centre_abs_Z_delta_le15mm': finite(centre_delta) and abs(centre_delta) <= 15,
        'unique_intended_owner_ge5': intended_count >= 5,
        'positive_depth_ge5': len(positive) >= 5,
    }
    foreign_ids = centre.get('all_visible_owner_indices') or []
    foreign_id = foreign_ids[0] if len(foreign_ids) == 1 and foreign_ids[0] != own else None
    all_same_foreign = foreign_id is not None and all(unique_owner(s, foreign_id) and not s.get('own_visible', False) for s in samples)
    e_checks = {
        'all9_in_bounds': len(samples) == 9 and all(s.get('in_bounds', False) for s in samples),
        'all9_unique_same_foreign_no_own': all_same_foreign,
        'centre_positive_depth': centre_positive,
        'centre_Z_nearer_by_gt15mm': centre_positive and centre['depth_mm'] < row['camera_Z_mm'] - 15,
        'positive_depth_ge7': len(positive) >= 7,
        'all_positive_Z_nearer_by_gt15mm': bool(positive) and all(s['depth_mm'] < row['camera_Z_mm'] - 15 for s in positive),
        'own_full_support_ge1': any(s.get('own_full', False) for s in samples),
    }
    if not ref_supported:
        state, reason = 'U', 'REFERENCE_OFF_IMAGE_OR_NO_SUPPORTED_REFERENCE_PIXEL'
    elif geometry == 'S':
        state, reason = 'S', 'STABLE_GEOMETRIC_SELF_PRIORITY_BEFORE_SCENE_DEPTH'
    elif geometry != 'SELF_VISIBLE':
        state, reason = 'U', 'REFERENCE_RAY_UNSTABLE_NO_HIT_OR_NEGATIVE_GAP'
    elif all(d_checks.values()):
        state, reason = 'D', 'SELF_VISIBLE_UNIQUE_INTENDED_OWNER_CENTRE_DEPTH_AGREEMENT_AND_LITERAL_SUPPORT'
    elif all(e_checks.values()):
        state, reason = 'E', 'SELF_VISIBLE_SAME_FOREIGN9_POSITIVE_NEARER_CENTRE_AND_SUPPORT'
    else:
        state, reason = 'U', 'INSUFFICIENT_OR_CONTRADICTORY_SCENE_EVIDENCE'
    return {
        'aux_surface_id': row['aux_surface_id'],
        'object_id': obj['object_id'],
        'scene_id': obj['pilot_pose']['scene_id'],
        'frame_id': obj['pilot_pose']['frame_id'],
        'intended_instance_id': own,
        'fixed_projected_r_xy': row['fixed_projected_r_xy'],
        'reference_supported': ref_supported,
        'geometry_state': geometry,
        'reference_ray_stable': stable,
        'point_range_minus_first_hit_range_mm': gap,
        'expected_aux_surface_state_before_q_guard': state,
        'reason': reason,
        'centre_depth_raw_uint16': centre.get('depth_raw_uint16'),
        'centre_depth_mm': centre.get('depth_mm'),
        'centre_visible_owner_indices': foreign_ids,
        'centre_integer_ray_Z_mm': integer_z,
        'centre_observed_minus_integer_ray_Z_mm': centre_delta,
        'positive_depth_count': len(positive),
        'unique_intended_owner_count': intended_count,
        'saved_counts': row['counts'],
        'D_checks': d_checks,
        'E_checks': e_checks,
    }


def main():
    policy = json.loads(POLICY.read_text())
    numeric = json.loads(NUMERIC.read_text())
    assert binding(POLICY)['sha256'] == '2817c6233634bc0238b211dc6ceca4efe9d9d0f7c24b32e5c90929b4a5c0e8fd'
    assert binding(NUMERIC)['sha256'] == '49125554c7ee4d812583d18adfee5c79cca3508a0a145addc629e6cc5228bce4'
    rows = [classify(obj, row) for obj in numeric['all4_instances'] for row in obj['all8']]
    counts = dict(Counter(row['expected_aux_surface_state_before_q_guard'] for row in rows))
    assert len(rows) == 32 and counts == {'S': 5, 'E': 1, 'U': 21, 'D': 5}, counts
    result = {
        'completed': True, 'PASS': True,
        'scope': 'Independent replay of existing fixed32 numeric evidence, not a source-label build or admission.',
        'policy': binding(POLICY), 'numeric_evidence': binding(NUMERIC), 'review_code': binding(Path(__file__).resolve()),
        'counts_before_q_guard': {k: counts[k] for k in ['D', 'E', 'S', 'U']},
        'q_emulation_final_guard': 'NOT_EVALUATED; no emulated q/L or bulk source arrays read/generated by this review.',
        'interpretation': [
            'Supported fixed-reference known geometric S precedes missing measured depth; YCB07_SURFACE4 is S despite zero centre depth.',
            'D unique-intended-owner>=5 and positive-depth>=5 are separate counts; mixed other cells remain diagnostics.',
            'E requires positive nearer centre, all9 same unique foreign owner/no own overlap, >=7 positive and every positive Z nearer by >15mm.',
            'Geometry compares Euclidean ranges; measured scene evidence compares camera Z to camera Z, not unconverted range.',
            'D centre agreement uses finite 64eps target-alone integer-pixel-ray Z; a fixed-point ray does not rescue a missing integer-pixel hit.',
            'Original1100/8800 arithmetic failure preserved; corrected prospective inventory1000/8000, no pilot32 or prior C8/C9 source counts changed.',
        ],
        'remaining_ambiguities': [],
        'limits': {'human_gold': False, 'native_mapping': 'UNKNOWN', 'physical_junction_gold': False, 'training_admitted': False,
                   'RGB_depth_mask_arrays_read': 0, 'new_model_forward': 0, 'fit': 0, 'current223': 0, 'PnP': 0},
        'corrected_source_inventory': {k: policy[k] for k in ['instances', 'slots_per_instance', 'all_slots']},
        'rows': rows,
    }
    out = Path(__file__).with_name('RESULT.json')
    assert not out.exists(), 'Preserve existing review output.'
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'result': binding(out), 'counts': result['counts_before_q_guard']}))


if __name__ == '__main__':
    main()
