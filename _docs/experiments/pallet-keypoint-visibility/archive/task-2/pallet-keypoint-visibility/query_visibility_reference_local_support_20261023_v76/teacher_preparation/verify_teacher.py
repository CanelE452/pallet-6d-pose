"""Generated critical cases plus the already fixed one-public-instance check."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from reference_local_support_teacher import reference_local_support, VISIBLE, COVERED, UNKNOWN

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
OLD = BASE / 'query_visibility_loco_native_context_20261023_v74/source_contract/public_ycb_native_proxy_feasibility'
YCB = BASE / 'query_visibility_public_real_20261022_v63/ycb'


def bound(path):
    p = Path(path)
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}


def clean(value):
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)): return value.item()
    if isinstance(value, (float, np.floating)) and not np.isfinite(value): return None
    if isinstance(value, np.floating): return float(value)
    return value


def write(name, data):
    with (HERE / name).open('x') as f:
        json.dump(clean(data), f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def generated():
    refs = np.asarray([[64, 64], [448, 64], [448, 448], [64, 448],
                       [128, 128], [384, 128], [384, 384], [128, 384]], float)
    full = np.ones((512, 512), bool); empty = np.zeros_like(full)
    saved_refs = refs.copy(); records = []
    def check(name, a, b, r, expected, rule=None):
        result = reference_local_support(a, b, r)
        assert result['label'][0] == expected, (name, result)
        if rule is not None: assert result['rule'][0] == rule, (name, result)
        records.append({'case': name, 'label0': int(result['label'][0]), 'rule0': str(result['rule'][0]),
            'ratio0': result['visible_over_expected_ratio'][0],
            'A_pixels0': int(result['expected_support_pixels_A'][0]),
            'A_mass0': float(result['weighted_expected_mass_A'][0]),
            'near_boundary0': bool(result['near_boundary'][0]),
            'GTref_valid0': bool(result['GTref_valid_xy'][0])})
        return result
    check('all_visible_strong', full, full, refs, VISIBLE, 'STRONG_3X3_VISIBLE')
    check('all_covered_strong', full, empty, refs, COVERED, 'STRONG_3X3_COVERED')
    check('all_expected_zero_unknown', empty, empty, refs, UNKNOWN, 'U_INSUFFICIENT_EXPECTED_SUPPORT')
    hole = full.copy(); hole[61:68, 61:68] = False
    result = check('fully_visible_center_hole_fallback_visible', hole, hole, refs, VISIBLE, 'FOOTPRINT_VISIBLE')
    assert result['center_M0'][0] == 0 and result['center_M4'][0] == 0
    large_hole = full.copy(); large_hole[40:89, 40:89] = False
    check('fully_visible_hole_no_expected_support_U', large_hole, large_hole, refs, UNKNOWN, 'U_INSUFFICIENT_EXPECTED_SUPPORT')
    visible_neighbors = full.copy(); visible_neighbors[63:66, 63:66] = False
    result = check('covered_3x3_overrides_broad_visible', full, visible_neighbors, refs, COVERED, 'STRONG_3X3_COVERED')
    assert result['visible_over_expected_ratio'][0] >= .9
    visible_center = empty.copy(); visible_center[63:66, 63:66] = True
    result = check('visible_3x3_overrides_broad_covered', full, visible_center, refs, VISIBLE, 'STRONG_3X3_VISIBLE')
    assert result['visible_over_expected_ratio'][0] <= .1
    yy, xx = np.indices(full.shape)
    partial = (xx + yy) % 2 == 0
    result = check('partial_mixed_support_U', full, partial, refs, UNKNOWN, 'U_MIXED_FOOTPRINT_SUPPORT')
    assert .1 < result['visible_over_expected_ratio'][0] < .9
    edge = refs.copy(); edge[0] = [.1, .1]
    result = check('valid_frame_edge_clipped_fallback', full, full, edge, VISIBLE, 'FOOTPRINT_VISIBLE')
    assert result['near_boundary'][0] and not result['strong_3x3_window_valid'][0]
    assert 0 < result['footprint_valid_fraction'][0] < 1
    outside = refs.copy(); outside[0] = [-.1, 64]
    check('continuous_outside_reference_U_no_clipping', full, full, outside, UNKNOWN, 'U_REFERENCE_INVALID')
    rounded_outside = refs.copy(); rounded_outside[0] = [511.8, 64]
    check('continuous_inside_nearest_outside_U', full, full, rounded_outside, UNKNOWN, 'U_REFERENCE_INVALID')
    missing = refs.copy(); missing[0] = np.nan
    result = check('missing_reference_layout_U_no_guessed_span', full, full, missing, UNKNOWN, 'U_INVALID_REFERENCE_LAYOUT')
    assert (result['label'] == UNKNOWN).all()
    q = refs + [100, -20]; q_copy = q.copy()
    with_self = reference_local_support(full, visible_neighbors, refs, self_proxy8=np.zeros(8, bool), q8=q, local_side8=32.)
    without_self = reference_local_support(full, visible_neighbors, refs, self_proxy8=np.ones(8, bool))
    np.testing.assert_array_equal(with_self['label'], without_self['label'])
    np.testing.assert_array_equal(q, q_copy); np.testing.assert_array_equal(refs, saved_refs)
    assert not with_self['query_diagnostics_separate']['reference_in_local_square'].any()
    encoded = reference_local_support(full.astype(np.uint8) * 255, visible_neighbors.astype(np.uint8) * 255, refs)
    np.testing.assert_array_equal(encoded['label'], with_self['label'])
    rejected = []
    for name, a, b in [('M4_outside_M0', empty, full), ('soft_mask_not_binary', full.astype(float) * .5, empty)]:
        try: reference_local_support(a, b, refs)
        except ValueError: rejected.append(name)
        else: raise AssertionError(name)
    tree = ast.parse((HERE / 'reference_local_support_teacher.py').read_text())
    imports = [n.names[0].name for n in tree.body if isinstance(n, ast.Import)]
    assert imports == ['numpy']
    result = {'completed': True, 'PASS': True, 'critical_cases': records,
        'self_and_query_label_invariance': True, 'input_q_and_r_unmodified': True,
        'binary_encodings_equal': True, 'invalid_masks_rejected': rejected,
        'pure_helper_imports': imports, 'model_forward_fit_current_PnP': 0,
        'bindings': [bound(HERE / x) for x in ['reference_local_support_teacher.py', 'TARGET_CONTRACT_PROPOSAL_V2.json', 'verify_teacher.py']]}
    write('GENERATED_CRITICAL_CASES.json', result)


def fixed_actual():
    prior_result_path = OLD / 'SUBFOOTPRINT_COMPARISON/SUBFOOTPRINT_COMPARISON.json'
    prior = json.loads(prior_result_path.read_text())
    refs_path = OLD / 'ONE_INSTANCE_V2/ONE_INSTANCE_NATIVE_REFERENCES_GT_ONLY.npz'
    with np.load(refs_path, allow_pickle=False) as f:
        refs = f['reference_projected8']; self_proxy = f['convex_self_visible_proxy8']
    manifest = YCB / 'train_instances.jsonl'
    row = next(r for r in map(json.loads, manifest.read_text().splitlines())
               if (r['scene_id'], r['frame_id'], r['instance_id']) == (0, 1, 0))
    M0 = np.asarray(Image.open(row['mask_full'])) > 0
    M4 = np.asarray(Image.open(row['mask_vis'])) > 0
    before_r = refs.copy()
    teacher = reference_local_support(M0, M4, refs, self_proxy8=self_proxy)
    np.testing.assert_array_equal(refs, before_r)
    assert np.array_equal(M0, M4)
    for i, old in enumerate(prior['per_reference']):
        assert teacher['expected_support_pixels_A'][i] == old['expected_support_pixel_count_A']
        np.testing.assert_allclose(teacher['weighted_expected_mass_A'][i],
                                  old['weighted_expected_mass_A_pixel_equivalents'], atol=1e-10, rtol=0)
        np.testing.assert_allclose(teacher['weighted_visible_mass_V'][i],
                                  old['weighted_visible_mass_V_pixel_equivalents'], atol=1e-10, rtol=0)
    assert (teacher['label'] == VISIBLE).sum() == 3
    assert (teacher['label'] == COVERED).sum() == 0
    assert (teacher['label'] == UNKNOWN).sum() == 5
    assert teacher['label'][6] == UNKNOWN and teacher['weighted_expected_mass_A'][6] == 0
    result = {'completed': True, 'PASS': True, 'instance': [0, 1, 0, 8],
        'known_prior_fixed_instance_only': True,
        'counts': {'VISIBLE': int((teacher['label'] == VISIBLE).sum()),
                   'COVERED': int((teacher['label'] == COVERED).sum()),
                   'U': int((teacher['label'] == UNKNOWN).sum())},
        'teacher': teacher, 'reference_unchanged': True,
        'same_footprint_masses_as_prior_V74': True,
        'ray_E_slot6_remains_support_U': True,
        'self_proxy_separate_unchanged': True,
        'no_q_producer_manufactured_for_actual_case': True,
        'semantic_human_visibility8': [-1] * 8,
        'scope': 'Strong3x3/footprint sourceGTteacher check, not nativehumansemanticgold/corpus/modelvalidation',
        'bindings': [bound(x) for x in [HERE / 'reference_local_support_teacher.py',
            HERE / 'TARGET_CONTRACT_PROPOSAL_V2.json', refs_path, prior_result_path,
            manifest, row['mask_full'], row['mask_vis']]],
        'RGB_reads': 0, 'model_forward_fit_current_PnP': 0,
        'fullcorpus_labels_created': 0,
    }
    write('FIXED_YCB_ONE_INSTANCE.json', result)


if __name__ == '__main__':
    generated()
    fixed_actual()
    write('TEACHER_PREPARATION_SEAL.json', {'completed': True, 'PASS': True,
        'bindings': [bound(HERE / name) for name in ['reference_local_support_teacher.py',
            'TARGET_CONTRACT_PROPOSAL_V2.json', 'verify_teacher.py', 'GENERATED_CRITICAL_CASES.json', 'FIXED_YCB_ONE_INSTANCE.json']],
        'fullcorpus_labeling_fit_current_PnP': 0})
    print('PURE_TEACHER_AND_FIXED_CASE_PASS', flush=True)
