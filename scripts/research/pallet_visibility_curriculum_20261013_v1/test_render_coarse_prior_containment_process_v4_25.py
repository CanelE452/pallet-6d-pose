"""Renderer contracts without new experiment rows, metrics, pixels or GT."""
from copy import deepcopy

import numpy as np
import pytest

from . import render_coarse_prior_containment_process_v4_25 as R


def arm(containment, mode='TRANSLATION'):
    solution = dict(native_observation_objective=10.,
        regularizer_costs=[dict(kind=kind, weighted_cost=value)
            for kind, value in zip(R.EXP.P.REGULARIZER_KINDS, (2., 3., 4.))],
        total_regularized_objective=19., regularizer_objective=9., CAL_coarse_pose_prior_objective=4.,
        coarse_pose_prior_count=1, objective=25. if containment else 19.,
        CAL_coarse_pose_prior_audit=dict(mode=mode, mean_bias_subtracted_once=True,
            translation_covariance_is_marginal_block=mode == 'TRANSLATION'))
    if containment:
        solution.update(global_containment_cost=4., global_containment_weight=1.5,
                        weighted_containment_objective=6., containment_count_in_optimizer=1)
    return dict(accepted=True, solution=solution, candidate_selection=dict(selected_name='one',
        old_full_outline_factor_or_tie_break_used=False, candidates=[dict(name='one',
        global_containment_cost=4. if containment else 0., total_energy=25. if containment else 19.)]))


def test_fixed_seven_methods_four_cases_fourteen_stages_and_explicit_memory_gate(monkeypatch):
    assert R.METHODS == tuple(R.EXP.ARMS[i] for i in (0, 1, 3, 4, 5, 7, 8))
    assert R.CASES == R.PREVIOUS.CASES and len(R.CASES) == 4
    assert len(R.TITLES) == len(R.CAPTIONS) == 14
    assert R.AUDIT_SUBDIR == 'independent_root_audit_v1'
    def forbidden(*args, **kwargs):
        raise AssertionError('Renderer read data before explicit memory GO')
    monkeypatch.setattr(R.C, 'read', forbidden)
    monkeypatch.setattr(R.C, 'rows', forbidden)
    with pytest.raises(ValueError, match='explicit root'):
        R.render()


def test_three_priors_and_C_once_no_old_mask_or_e_energy_reuse():
    accepted = arm(True)
    saved = deepcopy(accepted)
    energy = R._energies(accepted, R.TP_C)
    assert accepted == saved
    assert (energy['native'], energy['O'], energy['S'], energy['P'], energy['weighted_C'], energy['total']) == (10., 2., 3., 4., 6., 25.)
    assert energy['coarse_prior_count'] == energy['containment_count_in_optimizer'] == 1
    native = R._energies(arm(False, 'FULL'), R.FULL_NATIVE)
    assert native['total'] == 19. and native['C'] == native['weighted_C'] == 0.
    assert native['coarse_mode'] == 'FULL'
    assert R._energies({}, R.BASELINE)['applied'] is False
    assert R._energies(dict(accepted=False), R.TP_LOCK)['failed_joint']
    bad = arm(True)
    bad['solution']['objective'] += 6.
    with pytest.raises(ValueError, match='Same optimizer/selector'):
        R._energies(bad, R.TP_LOCK)
    bad = arm(False)
    bad['candidate_selection']['candidates'][0]['global_containment_cost'] = 5.
    with pytest.raises(ValueError, match='Native-only arm'):
        R._energies(bad, R.TP_NATIVE)
    bad = arm(True)
    bad['solution']['CAL_coarse_pose_prior_audit']['translation_covariance_is_marginal_block'] = False
    with pytest.raises(ValueError, match='marginal covariance'):
        R._energies(bad, R.TP_C)


def test_global_display_uses_full_unclipped_model_and_only_saved_boundary_samples():
    class Poison:
        def __array__(self, *args, **kwargs):
            raise AssertionError('A state/raw/canvas-clipped substitute was read')
    projected = np.array([[-20., -20.], [120., -20.], [120., 120.], [-20., 120.]])
    observation = dict(supported=True, observed_boundary_samples_xy=[[5., 5.], [50., 50.], [125., 50.]],
        image_hw=[100, 100], original_raw_points=Poison(), states=Poison(), polygon_xy=Poison())
    display = R._containment_display_geometry(projected, observation)
    assert display['inside_or_boundary_indices'] == [0, 1]
    assert display['outside_indices'] == [2]
    assert display['model_hull_canvas_clipped'] is False and display['model_to_observed_term'] == 0.
    assert display['new_RGB_queries'] == display['new_factor_or_pose_evaluations'] == 0
    assert R._containment_display_geometry(Poison(), dict(supported=False))['available'] is False


def test_first_pose_display_uses_saved_original_N3_and_identical_mode_target():
    from .common import cuboid
    xyz = cuboid(1.3, .11, 1.1) + [.02, -.01, .03]
    R0, t0 = np.eye(3), np.array([.1, .3, 2.4])
    prepared = dict(original_native_xyz=xyz.tolist(), original_R_cf=R0.tolist(), original_t=t0.tolist(),
        original_first_R_physical=R0.tolist(), original_first_center_camera_xyz=(R0 @ xyz.mean(0) + t0).tolist(),
        prior_covariance6=np.diag(np.arange(1., 7.)).tolist(), prior_bias6=[.01] * 6,
        factor_weight=1., bias_strength=1.)
    inputs = dict(TRANSLATION=deepcopy(prepared), FULL=deepcopy(prepared))
    K = np.array([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]])
    display = R._original_coarse_geometry(inputs, K)
    np.testing.assert_array_equal(display['original_first_model_projection_xy'], R.C.project(xyz, R0, t0, K))
    np.testing.assert_array_equal(display['TP_marginal_covariance3'], np.diag([4., 5., 6.]))
    assert display['e_numeric_pose_used'] is False and display['target_wire_is_original_before_bias']
    inputs['FULL']['original_t'][0] += 1.
    with pytest.raises(ValueError, match='identical original first N3'):
        R._original_coarse_geometry(inputs, K)
