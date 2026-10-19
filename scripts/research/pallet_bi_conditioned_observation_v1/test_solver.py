"""Analytic known-pose and contract fixtures; no dataset references or fitting."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from . import solver as S


def fixture():
    K = np.array([[420., 0, 320], [0, 420., 240], [0, 0, 1.]])
    R = Rotation.from_euler('xyz', [12, -24, 5], degrees=True).as_matrix()
    Q = Rotation.from_euler('y', 90, degrees=True).as_matrix()
    xyz = S.RT.C.cuboid(1.2, .25, .8)
    t = np.array([.04, -.03, 3.2])
    R0 = Rotation.from_euler('xyz', [.7, -.5, .4], degrees=True).as_matrix() @ R
    t0 = t + [.008, -.004, .02]
    initial = dict(available=True, R_cf=R0.tolist(), R_physical=(R0 @ Q).tolist(),
        centroid=t0.tolist(), cf_extents=[1.2, .25, .8])
    geometry = S.RT.registered_geometry((480, 640), K, xyz, initial)
    truth_xy, _ = S.V3.project_native(xyz, R, t, K)
    raw = np.vstack([truth_xy + [2.5, -.8], [320., 240.]])
    states = ['S' if hidden else role for hidden, role in
        zip(geometry['initial_self_hidden'], geometry['initial_roles'])]
    frame = dict(id='CONSTRUCTED_KNOWN_POSE', K=K.tolist(), native_xyz=xyz.tolist(),
        raw_hw=[480, 640], initial_pose=initial, initial_points=raw.tolist())
    full = []
    for k, state in enumerate(states):
        if state not in ('B', 'I'):
            continue
        full.append(dict(native_corner=k, state=state, xy=truth_xy[k].tolist(),
            covariance_px2=[[1., 0.], [0., 1.]], protection_px=2., source='ANALYTIC_POINT',
            physical_source_ids=['fixture_point:' + str(k)]))
    return frame, dict(states=states, full_points=full, partial_lines=[]), R, t, truth_xy


def partial_selection(frame, selection, truth_xy, *, all_partial=False):
    result = deepcopy(selection)
    for record in result['full_points']:
        k = record['native_corner']
        if not all_partial and k != result['full_points'][0]['native_corner']:
            continue
        # Archived original q0 is deliberately not the truth measurement.
        record['xy'] = deepcopy(frame['initial_points'][k])
        record['fit_covariance_px2'] = deepcopy(record['covariance_px2'])
        record['fit_reliability_weight'] = 1.
        normal = np.array([np.cos(.83*k + .2), np.sin(.83*k + .2)])
        line = dict(normal_unit_xy=normal.tolist(), offset_px=-float(normal @ truth_xy[k]),
            normal_variance_px2=1., support_tangent_interval_px=[1e5, 1e5 + 1],
            support_extrapolation_px=0., physical_source_id='fixture_line:' + str(k))
        provenance = dict(original_measurement_source=record['source'],
            original_observed_xy=deepcopy(record['xy']),
            original_observed_covariance_px2=deepcopy(record['covariance_px2']),
            original_effective_fit_covariance_px2=deepcopy(record['fit_covariance_px2']),
            original_fit_reliability_weight=1., original_raw_xy=deepcopy(record['xy']),
            display_coordinate_is_full_2D_measurement=False,
            revocation_reason='no_unique_face_supported_junction')
        result['partial_lines'].append(dict(native_corner=k, state=record['state'], lines=[line],
            image_binding=dict(sha256='0'*64, image_hw=frame['raw_hw']), fit_reliability_weight=1.,
            replaces_complete_BI_likelihood=True, explicit_complete_BI_likelihood_revocation=True,
            association_provenance=provenance))
    return result


def test_known_pose_points_and_native_Q_two_WD_contract():
    frame, selection, R, t, _ = fixture()
    before = deepcopy((frame, selection))
    result = S.solve_frame(frame, selection)
    assert result['accepted'], result
    pose = result['actual_pose']
    assert np.linalg.norm(np.asarray(pose['centroid']) - t) < 1e-5
    Q = np.asarray(frame['initial_pose']['R_cf']).T @ np.asarray(frame['initial_pose']['R_physical'])
    assert S.WD._physical_rotation_difference(pose['R_physical'], R @ Q) < .001
    assert result['optimizer_calls'] == 2
    assert result['solution']['families'][0]['solution']['initial_N0_R'] == frame['initial_pose']['R_cf']
    assert max(f['initial_rendered_vertex_set_gap_m'] for f in result['solution']['families']) < 1e-9
    assert before == (frame, selection)


def test_actual_applied_coordinates_and_own_covariance_enter_objective():
    frame, selection, _, _, _ = fixture()
    selected = deepcopy(selection)
    selected['full_points'][0]['xy'][0] += 4.
    selected['full_points'][0]['covariance_px2'] = [[3., .2], [.2, 1.5]]
    result = S.solve_frame(frame, selected)
    assert result['accepted']
    first = result['solution']['solution']['visible_audit'][0]
    assert first['measurement_xy'] == selected['full_points'][0]['xy']
    assert first['observed_covariance_px2'] == selected['full_points'][0]['covariance_px2']
    assert result['actual_pose']['centroid'] != S.solve_frame(frame, selection)['actual_pose']['centroid']


def test_partial_normal_revokes_full_likelihood_and_ignores_uncertified_tangent():
    frame, selection, _, t, truth = fixture()
    selected = partial_selection(frame, selection, truth)
    result = S.solve_frame(frame, selected)
    assert result['accepted'], result
    solution = result['solution']['solution']
    assert solution['objective_components']['complete_BI_likelihood_count'] == len(selection['full_points']) - 1
    assert solution['objective_components']['removed_complete_BI_2D_likelihood_ids'] == [selected['partial_lines'][0]['native_corner']]
    assert np.linalg.norm(np.asarray(result['actual_pose']['centroid']) - t) < 1e-5
    assert result['physical_endpoint_constraints'] == 0
    # An arbitrarily shifted *archived* point cannot become a hidden seed or
    # tangent cost.  The actual N0 pose, normals and complete points are fixed.
    altered = deepcopy(selected)
    k = altered['partial_lines'][0]['native_corner']
    record = next(m for m in altered['full_points'] if m['native_corner'] == k)
    record['xy'][0] += 25.
    altered['partial_lines'][0]['association_provenance']['original_observed_xy'] = deepcopy(record['xy'])
    actual = S.solve_frame(frame, altered)
    assert actual['accepted']
    np.testing.assert_array_equal(actual['actual_pose']['R_physical'], result['actual_pose']['R_physical'])
    np.testing.assert_array_equal(actual['actual_pose']['centroid'], result['actual_pose']['centroid'])


def test_partial_only_information_rank_and_known_pose():
    frame, selection, _, t, truth = fixture()
    partial = partial_selection(frame, selection, truth, all_partial=True)
    result = S.solve_frame(frame, partial)
    assert result['accepted'], result
    solution = result['solution']['solution']
    assert solution['complete_point_count'] == 0
    assert solution['rank']['image']['rank'] == 6
    assert solution['objective_components']['complete_BI_likelihood_count'] == 0
    assert np.linalg.norm(np.asarray(result['actual_pose']['centroid']) - t) < 1e-4


def test_duplicate_physical_source_not_counted_at_two_corners():
    frame, selection, _, _, truth = fixture()
    partial = partial_selection(frame, selection, truth, all_partial=True)
    first, second = partial['partial_lines'][:2]
    second['lines'][0]['physical_source_id'] = first['lines'][0]['physical_source_id']
    result = S.solve_frame(frame, partial)
    suppressed = result['source_deduplication']['suppressed']
    assert len(suppressed) == 1
    assert suppressed[0]['native_corner'] == second['native_corner']
    assert len(result['effective_partial_lines']) == len(partial['partial_lines']) - 1
    assert result['source_deduplication']['independent_reuse_of_duplicate_source'] is False


def test_rank_deficiency_and_empty_are_explicit_N0_fallbacks():
    frame, selection, _, _, _ = fixture()
    selection['full_points'] = selection['full_points'][:1]
    result = S.solve_frame(frame, selection)
    assert result['status'] == 'rank_deficient'
    assert result['fallback'] and not result['accepted'] and result['optimizer_calls'] == 0
    assert result['actual_pose']['centroid'] == frame['initial_pose']['centroid']
    selection['full_points'] = []
    result = S.solve_frame(frame, selection)
    assert result['status'] == 'empty_observation'
    assert result['fallback'] and not result['accepted']


@pytest.mark.parametrize('covariance', [[[1., 2.], [2., 1.]], [[float('nan'), 0.], [0., 1.]], [[1., .1], [0., 1.]]])
def test_invalid_covariance_is_explicit_failed_row(covariance):
    frame, selection, _, _, _ = fixture()
    selection['full_points'][0]['covariance_px2'] = covariance
    result = S.solve_frame(frame, selection)
    assert result['status'] == 'invalid_observation'
    assert result['fallback'] and result['optimizer_calls'] == 0


def test_excluded_states_rejected_before_poisoned_partial_line_fields():
    frame, selection, _, _, _ = fixture()
    hidden = selection['states'].index('S')
    selection['partial_lines'] = [dict(native_corner=hidden, lines=object())]
    result = S.solve_frame(frame, selection)
    assert result['status'] == 'invalid_observation'
    assert result['fallback']


def test_numerical_failure_optimizer_nonconvergence_and_guard_rejection_distinct(monkeypatch):
    frame, selection, _, _, _ = fixture()
    def crash(*args, **kwargs):
        raise FloatingPointError('constructed optimizer error')
    monkeypatch.setattr(S, 'minimize', crash)
    assert S.solve_frame(frame, selection)['status'] == 'numerical_failure'
    def failed(fun, x0, **kwargs):
        return SimpleNamespace(x=x0, success=False, message='iteration limit', status=9, nit=250)
    monkeypatch.setattr(S, 'minimize', failed)
    assert S.solve_frame(frame, selection)['status'] == 'optimizer_failed'
    def infeasible(fun, x0, **kwargs):
        x = x0.copy(); x[-1] = -10.
        return SimpleNamespace(x=x, success=True, message='constructed infeasible', status=0, nit=1)
    monkeypatch.setattr(S, 'minimize', infeasible)
    assert S.solve_frame(frame, selection)['status'] == 'algorithm_rejected'


def test_identical_measurements_all_arms_share_exact_pose_backend_and_digest():
    frame, selection, _, _, _ = fixture()
    outputs = []
    for arm in ('RAW_OBSERVATION_MATCHED', 'LEGACY_RULE_MATCHED', 'OBS_COMMON', 'OBS_BI'):
        packet = deepcopy(selection); packet['arm_ignored_by_backend'] = arm
        outputs.append(S.solve_frame(frame, packet))
    for actual in outputs[1:]:
        assert actual['actual_pose'] == outputs[0]['actual_pose']
        assert actual['solution'] == outputs[0]['solution']
        assert actual['observation_digest'] == outputs[0]['observation_digest']
    assert S.SOLVER_CONFIG['prior_weight'] == 0.
    assert S.SOLVER_CONFIG['max_p3p_triples'] == 0
    assert S.SOLVER_CONFIG['global_rotation_starts'] == 0


def test_candidate_adapter_uses_applied_not_raw_intersection_and_no_covariance_transplant():
    frame, selection, _, _, _ = fixture()
    corners = []
    choices = {}
    for point in selection['full_points']:
        k = point['native_corner']; choices[k] = 'candidate:' + str(k)
        corners.append(dict(native_corner=k, partial_line=None, candidates=[dict(
            candidate_id=choices[k], kind='intersection', q0=frame['initial_points'][k],
            raw_intersection_xy=[123., 234.], blend_alpha=.25, applied_xy=point['xy'],
            covariance_px2=point['covariance_px2'], source_line_ids=['directed:' + str(k)],
            physical_source_ids=['actual_pixels:' + str(k)])]))
    selected = S.selection_from_candidates(dict(states=selection['states'], corners=corners), choices)
    assert [m['xy'] for m in selected['full_points']] == [m['xy'] for m in selection['full_points']]
    assert all(m['xy'] != [123., 234.] for m in selected['full_points'])
    assert S.solve_frame(frame, selected)['accepted']
