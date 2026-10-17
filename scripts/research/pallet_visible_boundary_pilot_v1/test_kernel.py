"""Small physical/numerical invariants of the predeclared one-pass score."""
from copy import deepcopy
import math

import numpy as np
import pytest

from scripts.research.pallet_visible_boundary_pilot_v1.kernel import (
    VARIANTS, evaluate_candidate, finite_distances, robust_losses,
    sample_visible, validate_polylines,
)


def fixture():
    K = [[100., 0., 100.], [0., 100., 100.], [0., 0., 1.]]
    observed = [[100., 60.], [160., 100.]]
    rho = 1.25**2*(math.sqrt(1+(10/1.25)**2)-1)
    outline = dict(supported=True, image_hw=[480, 640],
        configuration=dict(observed_sample_spacing_px=1., outline_factor_weight=2.,
                           normal_sigma_px=1., huber_delta=1.25),
        observed_boundary_samples_xy=observed)
    rows = []
    for direction in ('observed_to_model', 'model_to_observed'):
        rows.append(dict(loss_family='outline', direction=direction, normalizer=2.,
            arc_or_sample_weights=[1., 1.], weighted_sample_contributions=[rho/2, rho/2],
            weighted_contribution=rho, factor_weight=2., direction_weight=.5,
            sample_xy=observed, residual_xy_px=[[0., 10.], [10., 0.]], robust_losses=[rho, rho]))
    total = 10.+2*rho
    candidate = dict(candidate_id='saved', stage='G13', family='ORIGINAL_WD',
        native_xyz=[[-1., -1., 0.], [1., -1., 0.], [1., 1., 0.], [-1., 1., 0.],
                    [-1., -1., 1.], [1., -1., 1.], [1., 1., 1.], [-1., 1., 1.]],
        R=np.eye(3).tolist(), t=[0., 0., 2.], original_record=dict(objective=total),
        dense=dict(objective=total, weighted_components=dict(full_BI=7., partial_normal=3.,
                   prior=0., outline=2*rho), outline_active=True, per_observation=rows))
    return candidate, outline, K, rho


def test_finite_distance_clamps_endpoints():
    distances, _, _ = finite_distances([[2., 1.]], [[[0., 0.], [1., 0.]]])
    assert distances[0] == math.sqrt(2.)


def test_disconnected_midpoints_do_not_bridge_gaps_and_preserve_mass():
    lines = [dict(line_id='one', points_xy=[[0., 5.], [2., 5.]]),
             dict(line_id='two', points_xy=[[10., 5.], [11.5, 5.]])]
    sampled = sample_visible(lines, [480, 640], 1.)
    assert sampled['sample_xy'] == [[.5, 5.], [1.5, 5.], [10.375, 5.], [11.125, 5.]]
    assert sampled['sample_masses'] == [1., 1., .75, .75]
    assert sampled['visible_mass_Z_E'] == 3.5
    assert not sampled['gaps_joined']


@pytest.mark.parametrize('lines', [
    [dict(line_id='a', points_xy=[[1., 1.], [1., 1.]])],
    [dict(line_id='a', points_xy=[[1., 1.], [2., 1.], [1., 1.]])],
    [dict(line_id='a', points_xy=[[1., 1.], [3., 1.]]), dict(line_id='b', points_xy=[[3., 1.], [1., 1.]])],
    [dict(line_id='a', points_xy=[[1., 1.], [3., 1.]]), dict(line_id='b', points_xy=[[2., 1.], [4., 1.]])],
    [dict(line_id='a', points_xy=[[639., 1.], [640., 1.]])],
])
def test_invalid_or_inflated_raw_support_rejected(lines):
    with pytest.raises(ValueError):
        validate_polylines(lines, [480, 640])


def test_no_input_all_new_arms_are_bit_exact_original_scalar():
    candidate, outline, K, _ = fixture()
    result = evaluate_candidate(candidate, outline, K)
    assert result['scores'] == {variant: candidate['original_record']['objective'] for variant in VARIANTS}
    assert result['identity_shortcut'] and not result['treatment_active']


def test_common_mass_and_half_weight_apply_once_preserving_C():
    candidate, outline, K, rho = fixture()
    support = sample_visible([dict(line_id='actual', points_xy=[[100., 55.], [101., 55.]])], [480, 640], 1.)
    result = evaluate_candidate(candidate, outline, K, support, 'AVAILABLE')
    rho5 = 1.25**2*(math.sqrt(1+(5/1.25)**2)-1)
    assert result['r'] == .5
    assert result['scores']['HULL_ONEWAY'] == pytest.approx(10.+rho, abs=1e-12)
    assert result['scores']['HULL_ONEWAY_MASS'] == pytest.approx(10.+.5*rho, abs=1e-12)
    assert result['scores']['VISIBLE_SUPPORT'] == pytest.approx(10.+.5*rho5, abs=1e-12)
    assert result['original_weighted_components']['full_BI'] == 7.
    assert result['original_weighted_components']['partial_normal'] == 3.
    assert result['non_outline_cost'] == 10.
    assert result['visible_evidence']['reverse_direction_present'] is False


def test_original_inactive_outline_has_exact_identity_despite_new_input():
    candidate, outline, K, _ = fixture()
    candidate = deepcopy(candidate)
    candidate['dense'].update(outline_active=False, per_observation=[], objective=10.)
    candidate['dense']['weighted_components']['outline'] = 0.
    candidate['original_record']['objective'] = 10.
    visible = sample_visible([dict(line_id='visible', points_xy=[[30., 40.], [31., 40.]])], [480, 640], 1.)
    result = evaluate_candidate(candidate, outline, K, visible, 'AVAILABLE')
    assert result['scores'] == {variant: 10. for variant in VARIANTS}
    assert not result['treatment_active']


def test_more_input_length_never_increases_ratio_above_one():
    candidate, outline, K, _ = fixture()
    support = sample_visible([dict(line_id='actual', points_xy=[[100., 50.], [110., 50.]])], [480, 640], 1.)
    result = evaluate_candidate(candidate, outline, K, support, 'AVAILABLE')
    assert result['r'] == 1.
    assert result['visible_mass_Z_E'] == 10.
    assert result['scores']['VISIBLE_SUPPORT'] == 10.


def test_outline_robust_is_not_full_point_double_scaled():
    radius, losses = robust_losses([0., 1.25], 1., 1.25)
    assert radius.tolist() == [0., 1.25]
    assert losses[1] == 1.25**2*(math.sqrt(2)-1)


def test_scientific_artifact_ban_survives_temporary_directory_allowance(tmp_path):
    from scripts.research.pallet_visible_boundary_pilot_v1.replay import ReferenceFreeIO
    guard = ReferenceFreeIO(tmp_path/'NEUTRAL_PACKAGE.json', None, tmp_path/'new_output')
    guard.active = True
    with pytest.raises(PermissionError, match='forbidden_scientific_artifact'):
        guard._audit('open', (str(tmp_path/'FIXED_REFERENCES.json'), 'r', 0))
    guard.active = False
    assert guard.denied == ['forbidden_scientific_artifact:fixed_references.json']


def test_qhull_bytes_temporary_path_keeps_same_scientific_checks(tmp_path):
    from scripts.research.pallet_visible_boundary_pilot_v1.replay import ReferenceFreeIO
    guard = ReferenceFreeIO(tmp_path/'NEUTRAL_PACKAGE.json', None, tmp_path/'new_output')
    guard.active = True
    guard._audit('open', (bytes(tmp_path/'scipy-legitimate-runtime'), None, 655554))
    with pytest.raises(PermissionError, match='forbidden_scientific_artifact'):
        guard._audit('open', (bytes(tmp_path/'FIXED_REFERENCES.json'), 'r', 0))
    guard.active = False
