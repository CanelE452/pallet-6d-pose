"""Focused reach/preservation/quality contracts, without real labels."""
from copy import deepcopy

import numpy as np
import pytest

from . import visible_bi_corner_reach_v4_7 as NEW
from . import visible_bi_boundary_recovery_v4_3b as OLD
from .visible_bi_correction_v2 import DEFAULT_CONFIG, correct_visible_bi
from .test_visible_bi_boundary_recovery_v4_3 import displaced_scene


def run(a, base=None, **cfg):
    image, mask, q, K, X, pose, states, _, original = a
    return NEW.recover_visible_boundary(image,q,K,pose,states,pallet_mask=mask,
        native_xyz=X,base_correction=original if base is None else base,config=cfg)


def test_radius_floor_uses_only_original_extent_and_context_and_does_not_reduce_larger_reach():
    config = deepcopy(DEFAULT_CONFIG)
    working,floor = NEW._working_configuration(config)
    assert floor == working['normal_search_radius_px'] == 43.
    assert {k:v for k,v in working.items() if k != 'normal_search_radius_px'} == {
        k:v for k,v in config.items() if k != 'normal_search_radius_px'}
    assert config == DEFAULT_CONFIG
    config['normal_search_radius_px'] = 80.
    assert NEW._working_configuration(config)[0]['normal_search_radius_px'] == 80.


def test_synthetic_success_has_original_config_quality_guards_and_finite_support():
    a = displaced_scene()
    saved = deepcopy(a[8])
    result = run(a)
    assert result['recovered_native_ids']
    assert result['configuration'] == saved['configuration'] and a[8] == saved
    audit = result['corner_reach_audit']
    assert audit['working_normal_search_radius_px'] == 43.
    assert audit['observed_support_extrapolation_px'] == 11.
    for row in result['boundary_recovery_records']:
        if not row['recovered']:
            continue
        for ray in row['selected_rays']:
            assert ray['angle_change_deg'] <= 12.
            assert ray['inlier_rms_px'] <= 1.25
            assert ray['inlier_count'] >= 6 and ray['inlier_fraction'] >= .6
            assert ray['inlier_span_fraction'] >= .55
            assert ray['rgb_supported_sample_count'] >= 6
        assert row['boundary_recovery_support_guard']['passed']
    assert result['measured_points'][8] == a[2][8].tolist()


def test_supplied_recovered_B_and_all_I_are_exact_and_never_requeried():
    a = displaced_scene()
    image,mask,q,K,X,pose,states,_,base = a
    previous = OLD.recover_visible_boundary(image,q,K,pose,states,pallet_mask=mask,
        base_correction=base,native_xyz=X)
    saved = deepcopy(previous)
    result = run(a,previous)
    assert previous == saved
    for k,s in enumerate(states):
        if s == 'I' or s == 'B' and previous['records'][k]['corrected']:
            assert result['records'][k] == previous['records'][k]
            assert result['measured_points'][k] == previous['measured_points'][k]
            assert not result['boundary_recovery_records'][k]['pixel_queried']
            assert next(m for m in result['visible_measurements'] if m['native_corner']==k) == next(
                m for m in previous['visible_measurements'] if m['native_corner']==k)


def test_excluded_state_coordinate_perturbation_cannot_change_recovery_measurements():
    a = list(displaced_scene())
    states = a[6].copy()
    states[3],states[4],states[5] = 'O','T','U'
    a[6] = states
    a[8] = correct_visible_bi(a[0],a[2],a[3],a[5],states,pallet_mask=a[1],native_xyz=a[4])
    first = run(a)
    a[2] = a[2].copy()
    for k,s in enumerate(states):
        if s in {'S','O','T','U'}:
            a[2][k] = [1e7*(k+1),-1e8]
    a[8] = correct_visible_bi(a[0],a[2],a[3],a[5],states,pallet_mask=a[1],native_xyz=a[4])
    second = run(a)
    assert first['visible_measurements'] == second['visible_measurements']
    assert first['boundary_recovery_records'] == second['boundary_recovery_records']
    assert all(not r['pixel_queried'] for r in second['boundary_recovery_records'] if r['state'] in {'S','O','T','U'})


def test_empty_mask_and_blank_RGB_cannot_make_a_corner_from_wider_search():
    a = list(displaced_scene())
    a[1] = np.zeros_like(a[1])
    a[0] = np.full_like(a[0],128)
    a[8] = correct_visible_bi(a[0],a[2],a[3],a[5],a[6],pallet_mask=a[1],native_xyz=a[4])
    result = run(a)
    assert not result['recovered_native_ids'] and result['visible_measurements'] == a[8]['visible_measurements']


def test_actual_RGB_guard_cannot_be_disabled():
    with pytest.raises(ValueError,match='RGB support'):
        run(displaced_scene(),require_rgb_support=False)


def pair(point, sine, start, stop):
    point = np.asarray(point, float)
    first = dict(normal=[0.,1.],offset=-point[1],supported=True,
        observed_points=[[start,point[1]],[stop,point[1]]],inlier_mask=[True,True],native_neighbor=4)
    normal = np.array([sine,-np.sqrt(1-sine**2)])
    tangent = np.array([-normal[1],normal[0]])
    second = dict(normal=normal.tolist(),offset=-float(normal@point),supported=True,
        observed_points=(point+np.array([4.,12.])[:,None]*tangent).tolist(),
        inlier_mask=[True,True],native_neighbor=6)
    return OLD.ORIGINAL._intersection(first,second,.25)


def test_better_conditioned_unsupported_intersection_does_not_discard_supported_alternative():
    unsupported = pair([21.,0.],.995,0.,4.)
    supported = pair([21.5,.01],.98,14.,30.)
    chosen,audit = NEW._choose_supported_pair([unsupported,supported],11.)
    assert chosen is supported and audit['selected_candidate_index'] == 1
    assert audit['candidate_records'][0]['support_guard']['rays'][0]['extrapolation_px'] == 17.
    assert not audit['candidate_records'][0]['admissible']
    assert audit['candidate_records'][1]['admissible']


def test_all_unsupported_intersections_remain_unresolved_and_inputs_unchanged():
    unsupported = pair([21.,0.],.995,0.,4.)
    before = deepcopy(unsupported)
    chosen,audit = NEW._choose_supported_pair([unsupported],11.)
    assert chosen is None and audit['support_admissible_count'] == 0
    np.testing.assert_array_equal(unsupported['point'],before['point'])
    assert unsupported['first'] == before['first']
