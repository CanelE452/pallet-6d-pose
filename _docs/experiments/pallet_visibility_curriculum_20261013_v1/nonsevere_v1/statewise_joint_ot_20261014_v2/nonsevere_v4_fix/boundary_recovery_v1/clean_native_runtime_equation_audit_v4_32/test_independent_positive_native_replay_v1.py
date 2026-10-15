"""Synthetic real-core checks of the independent saved-native replay."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.research.pallet_visibility_curriculum_20261013_v1.test_clean_fresh_BI_input_projection_v4_32 import P, project, attached

path = Path(__file__).with_name('independent_positive_native_replay_v1.py')
spec = importlib.util.spec_from_file_location('independent_positive_native_replay', path)
AUDIT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AUDIT)


def fixture(policy='ALL_BI_REPLACE', name='ORIGINAL_WD'):
    _, row, projected = project(partial=True)
    packet = P.assemble_worker_input(projected, **attached(projected, row))
    clean = P.CLEAN.prepare_clean_fresh_BI_task(**packet['clean_input'])
    family = clean['families'][name]
    base = P.FACTORY.native_baseline(family=family, K=packet['clean_input']['K'],
        states=clean['states'], visible_measurements=packet['clean_input']['visible_measurements'],
        **packet['physics'], initial_source_binding=clean['initial_source_binding'])
    core = P.G.GeometryGuidedEdgeProblem(base, packet['typed_bundles'][policy][name],
        source_policy=policy, tangent_gamma=packet['source_contract']['tangent_parameters'][policy]['gamma'])
    R, t = np.asarray(family['coarse_pose']['R']), np.asarray(family['coarse_pose']['t'])
    expected = core.evaluate(R, t)
    p = np.r_[Rotation.from_matrix(R).as_rotvec(), t/core.scale]
    _, gradient, _ = core.objective(p)
    return packet, family, core.plan, R, t, expected, gradient


@pytest.mark.parametrize('policy', ('ALL_BI_REPLACE', 'RGB_NONFALLBACK_PRESERVE'))
@pytest.mark.parametrize('name', ('ORIGINAL_WD', 'SWAPPED_WD'))
def test_real_G32_core_energy_identity_guards_and_independent_gradient(policy, name):
    packet, family, plan, R, t, expected, gradient = fixture(policy, name)
    result = AUDIT.replay(packet, family, plan, R, t, expected=expected, expected_gradient=gradient)
    assert result['failure_count'] == 0, result['checks']
    assert result['scalar_absolute_difference'] < 1e-8
    assert result['V32_consumer_planner_or_objective_called_by_replay'] is False


def test_energy_and_gradient_corruption_detected_without_changing_saved_packet():
    packet, family, plan, R, t, expected, gradient = fixture()
    expected['objective'] += 3.
    result = AUDIT.replay(packet, family, plan, R, t, expected=expected, expected_gradient=gradient+7.)
    failed = {c['reason'] for c in result['checks'] if not c['passed']}
    assert 'independent native scalar energy' in failed
    assert 'independent scalar central-difference gradient' in failed


def test_old_profile_cannot_be_relabelled_as_current_native_audit():
    packet, family, plan, R, t, _, _ = fixture()
    plan['selection_profile'] = 'OLD_V31'
    with pytest.raises(ValueError, match='Current initial-only'):
        AUDIT.replay(packet, family, plan, R, t)


def test_wrong_physical_family_center_rejected_before_energy():
    packet, family, plan, R, t, _, _ = fixture(name='SWAPPED_WD')
    family['coarse_pose']['t'][0] += .1
    with pytest.raises(ValueError, match='initial physical body'):
        AUDIT.replay(packet, family, plan, R, t)


def test_no_raw_archive_or_baseline_argument_in_positive_boundary():
    with pytest.raises(ValueError, match='Exact current positive'):
        AUDIT.positive_task(dict(schema=AUDIT.WORKER_SCHEMA, baseline=object()))
