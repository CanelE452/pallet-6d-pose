"""Small, label-free checks of recorded-stage coverage semantics."""
from copy import deepcopy
import pytest
from .candidates import extract_family, packet, pose_record


def family():
    return dict(name='ORIGINAL_WD', native_xyz=[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
        [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], cf_extents=[2,2,2],
        physical_registration_Q=[[0,0,1],[0,1,0],[-1,0,0]], coarse_pose={},
        solution=dict(schema='archived_outline', accepted=True, candidates=[
            dict(initialization='rejected', joint_feasible=False, objective=2),
            dict(initialization='retained', joint_feasible=True, objective=1)],
            feasible_pose_samples=[dict(R=[[1,0,0],[0,1,0],[0,0,1]],t=[0,0,5],objective=1)]))


def test_missing_outcome_poses_stay_missing_even_when_feasible_sample_exists():
    source = family()
    before = deepcopy(source)
    out = extract_family(source)
    assert source == before
    assert len(out['feasible_samples']) == 1
    assert out['counts']['outcome_pose_missing'] == 2
    assert all(r['pose'] is None for r in out['original_outcomes'])
    assert out['coverage']['generated_oracle_status'] == 'CANDIDATE_RECORD_INCOMPLETE'


def test_native_origin_and_registration_are_applied_to_same_pose():
    f = family()
    f['native_xyz'] = [[a+3,b,c] for a,b,c in f['native_xyz']]
    p = pose_record(dict(R=[[1,0,0],[0,1,0],[0,0,1]],t=[0,0,5]), f)
    assert p['centroid'] == [3,0,5]
    assert p['R_physical'] == f['physical_registration_Q']


def test_no_truncation_of_all_retained_feasible_samples():
    f = family()
    f['solution']['feasible_pose_samples'] *= 39
    out = extract_family(f)
    assert len(out['feasible_samples']) == 39


def test_original_e_frozen_family_mismatch_fails_before_extraction():
    a = dict(id='frame', image_sha256='rgb', states=['B']*8, selected_family='ORIGINAL_WD')
    b = dict(id='frame', image_sha256='rgb', states=['B']*8)
    e = dict(states=['B']*8, WD_family='SWAPPED_WD')
    with pytest.raises(ValueError, match='frozen G10 family'):
        packet(a, b, e, {})
