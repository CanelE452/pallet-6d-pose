"""Pose transport and diagnosis categories; no real data or reference input."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from . import diagnose_frozen_e_pose_under_v24_v4_25 as D


def test_physical_Q_transport_preserves_the_same_camera_body_projection():
    physical_xyz = D.C.cuboid(.8, .4, .6)
    Q = Rotation.from_euler('y', 90., degrees=True).as_matrix()
    current_xyz = physical_xyz @ Q.T
    Rphysical = Rotation.from_rotvec([.2, -.5, .1]).as_matrix()
    t = np.array([.1, -.03, 3.])
    family = dict(native_xyz=current_xyz.tolist(), physical_registration_Q=Q.tolist())
    pose = dict(R_physical=Rphysical.tolist(), centroid=t.tolist())
    Rcf, transported_t, audit = D.transport_physical_pose_to_family(pose, family)
    K = np.array([[500., 3., 320.], [0., 520., 240.], [0., 0., 1.]])
    assert np.allclose(D.C.project(current_xyz, Rcf, transported_t, K),
        D.C.project(physical_xyz, Rphysical, t, K), atol=1e-10, rtol=0)
    assert np.array_equal(transported_t, t)
    assert audit['physical_rotation_max_absolute_difference'] < 1e-12
    original = dict(native_xyz=physical_xyz.tolist(), physical_registration_Q=np.eye(3).tolist())
    assert D._physical_body_gap(original, family) < 1e-12
    with pytest.raises(ValueError):
        D.transport_physical_pose_to_family({**pose, 'R_physical': np.zeros((3, 3))}, family)


def test_hard_exclusion_objective_preference_and_search_gap_are_distinct():
    assert D.classify_comparison(False, True, 1., 2.) == 'diagnostic_e_excluded_by_current_hard_or_numeric_guard'
    assert D.classify_comparison(True, True, 1., 2.) == 'feasible_diagnostic_e_has_lower_energy_bounded_search_coverage_gap'
    assert D.classify_comparison(True, True, 3., 2.) == 'current_objective_prefers_selected_over_diagnostic_e'
    assert D.classify_comparison(True, False, 1., None) == 'selected_is_own_fallback_not_a_solved_pose'
    assert D.classify_comparison(True, True, 2., 2. + 1e-10) == 'same_current_energy_within_reporting_tolerance'


def test_factor_decomposition_retains_vertex_identity_and_each_cost_once():
    bykind, byfactor = D._costs([
        dict(kind='COMPLETE_BI', native_corner=0, weighted_robust_cost=2.),
        dict(kind='COMPLETE_BI', native_corner=1, weighted_robust_cost=3.),
        dict(kind='PROJECTED_NATIVE_EDGE', native_corner=0, native_neighbor=1, weighted_robust_cost=4.)])
    assert bykind == {'COMPLETE_BI': 5., 'PROJECTED_NATIVE_EDGE': 4.}
    assert len(byfactor) == 3 and sum(byfactor.values()) == 9.
