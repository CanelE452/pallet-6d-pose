"""Pure v25 profile, original-coarse provenance and pre-reference seal gates."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from . import evaluate_coarse_prior_containment_v4_25 as NEW
from .common import cuboid


def test_exact_nine_arm_mapping_and_native_before_same_mode_containment():
    assert len(NEW.ARMS) == len(set(NEW.ARMS)) == 9
    assert NEW.ARMS[0] == 'SAME_FROZEN_E'
    assert len(NEW.PROFILES) == 4
    flattened = []
    seen_native = set()
    combinations = set()
    for name, mode, containment, arms in NEW.PROFILES:
        assert mode in ('TRANSLATION', 'FULL')
        assert (mode, containment) not in combinations
        combinations.add((mode, containment))
        assert len(arms) == 2 and not arms[0].endswith('_E_FAMILY') and arms[1] == arms[0] + '_E_FAMILY'
        assert NEW.PROFILE_ARMS[name] == arms
        if containment:
            assert mode in seen_native
        else:
            seen_native.add(mode)
        flattened.extend(arms)
    assert flattened == list(NEW.ARMS[1:])
    assert combinations == {(m, c) for m in ('TRANSLATION', 'FULL') for c in (False, True)}


@pytest.mark.parametrize('mode', ['TRANSLATION', 'FULL'])
def test_prepare_pose_prior_uses_original_first_coarse_not_e_numeric_pose_or_trial_family(mode):
    xyz = cuboid(1.3, .11, 1.1) + [.02, -.01, .03]
    R = Rotation.from_rotvec([.12, -.4, .03]).as_matrix()
    Q = Rotation.from_euler('y', 90, degrees=True).as_matrix()
    t = np.array([.1, .3, 2.4])
    class Poison:
        def __array__(self, *args, **kwargs):
            raise AssertionError('An e-pose or other trial family set the original coarse target')
    task = dict(families={
        'ORIGINAL_WD': dict(native_xyz=xyz, physical_registration_Q=Q, coarse_pose=dict(R=R, t=t)),
        'SWAPPED_WD': Poison()}, baseline=dict(R=Poison(), t=Poison(), actual_pose=Poison()))
    covariance = np.diag([.03, .04, .05, .02, .025, .06]) ** 2
    bias = [.01, -.02, .003, .002, -.001, .012]
    models = dict(COARSE=dict(prior_covariance6=covariance, prior_fit=dict(median_bias6=bias)),
                  COARSE_BINDING=dict(sha256='a' * 64))
    prior = NEW._prepare_pose_prior(task, models, mode)
    assert prior['mode'] == mode
    np.testing.assert_array_equal(prior['original_R_cf'], R)
    np.testing.assert_array_equal(prior['original_t'], t)
    np.testing.assert_allclose(prior['original_first_R_physical'], R @ Q, rtol=0, atol=1e-15)
    np.testing.assert_allclose(prior['original_first_center_camera_xyz'], R @ xyz.mean(0) + t, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(prior['prior_covariance6'], covariance)
    np.testing.assert_array_equal(prior['prior_bias6'], bias)
    assert prior['factor_weight'] == prior['bias_strength'] == 1.


def sealed_metadata():
    ids = ['synthetic_' + str(k) for k in range(238)]
    seal = dict(all9_arms_complete=True, all_declared_arms_complete=True,
        arms=list(NEW.ARMS), arm_count=9, rows=238, ordered_ids=ids,
        predictions_sha256='prediction-bytes', protocol_sha256='protocol-bytes', input_seal_sha256='input-seal-bytes')
    protocol = dict(ids=ids, source_sha256={}, input_sha256={}, protected_user_sha256={})
    return seal, protocol


def install_metadata(monkeypatch, seal, protocol, *, audit=None):
    reads, verifications = [], []
    def read(path):
        path = Path(path)
        reads.append(path.name)
        if path.name == 'PREDICTION_SEAL.json':
            return seal
        if path.name == 'PROTOCOL.json':
            return protocol
        if path.name == 'AUDIT.json' and audit is not None:
            return audit
        if path.name == 'AUDIT_SEAL.json' and audit is not None:
            return dict(receipt_sha256='audit-bytes')
        raise AssertionError('A numeric reference or unsealed prediction was read: ' + str(path))
    hashes = {'PREDICTIONS.jsonl.gz': 'prediction-bytes', 'PROTOCOL.json': 'protocol-bytes',
              'INPUT_SEAL.json': 'input-seal-bytes', 'PREDICTION_SEAL.json': 'prediction-seal-bytes',
              'AUDIT.json': 'audit-bytes'}
    monkeypatch.setattr(NEW.C, 'read', read)
    monkeypatch.setattr(NEW.C, 'sha', lambda path: hashes[Path(path).name])
    monkeypatch.setattr(NEW.E, '_verify_sources', lambda value: verifications.append(value))
    return reads, verifications


@pytest.mark.parametrize('violation', ['incomplete', 'arm_missing', 'wrong_count', 'prediction_changed', 'input_seal_changed', 'id_order_changed'])
def test_score_rejects_incomplete_or_changed_seal_before_reference_access(tmp_path, monkeypatch, violation):
    seal, protocol = sealed_metadata()
    if violation == 'incomplete':
        seal.pop('all9_arms_complete')
    elif violation == 'arm_missing':
        seal['arms'].pop()
    elif violation == 'wrong_count':
        seal['rows'] = 237
    elif violation == 'prediction_changed':
        seal['predictions_sha256'] = 'different-bytes'
    elif violation == 'input_seal_changed':
        seal['input_seal_sha256'] = 'different-bytes'
    else:
        seal['ordered_ids'] = list(reversed(seal['ordered_ids']))
    reads, verifications = install_metadata(monkeypatch, seal, protocol)
    with pytest.raises(ValueError, match='Full nine-arm seal required before numeric references'):
        NEW.score(tmp_path / 'nonexistent_run')
    assert reads == ['PREDICTION_SEAL.json', 'PROTOCOL.json']
    assert verifications == []


def test_score_requires_current_independent_full_population_audit_before_numeric_reference(tmp_path, monkeypatch):
    seal, protocol = sealed_metadata()
    audit = dict(rows=238, failure_count=1, prediction_seal=dict(sha256='prediction-seal-bytes'))
    reads, verifications = install_metadata(monkeypatch, seal, protocol, audit=audit)
    with pytest.raises(ValueError, match='independent all238 zero-failure pose audit'):
        NEW.score(tmp_path / 'nonexistent_run')
    assert reads == ['PREDICTION_SEAL.json', 'PROTOCOL.json', 'AUDIT.json', 'AUDIT_SEAL.json']
    assert len(verifications) == 3


@pytest.mark.parametrize('violation', ['missing_arm', 'wrong_order'])
def test_score_stream_validates_all_arms_and_order_before_numeric_reference(tmp_path, monkeypatch, violation):
    seal, protocol = sealed_metadata()
    audit = dict(rows=238, failure_count=0, failures=[], prediction_seal=dict(sha256='prediction-seal-bytes'))
    reads, _ = install_metadata(monkeypatch, seal, protocol, audit=audit)
    row = dict(id='unlisted_id', partition='CAL', fixed_family={}, tangent_control_audit={},
        arms={arm: dict(actual_pose=object(), accepted=True) for arm in NEW.ARMS})
    if violation == 'missing_arm':
        row['arms'].pop(NEW.ARMS[-1])
    monkeypatch.setattr(NEW.C, 'rows', lambda path: iter([row]))
    with pytest.raises(ValueError, match='Every streamed row|ordered238 before reference access'):
        NEW.score(tmp_path / 'nonexistent_run')
    assert reads == ['PREDICTION_SEAL.json', 'PROTOCOL.json', 'AUDIT.json', 'AUDIT_SEAL.json']


def test_prepare_rejects_missing_declared_arm_before_streaming_inputs_or_writing(tmp_path, monkeypatch):
    output = tmp_path / 'new_output'
    declaration = dict(arms=list(NEW.ARMS[:-1]))
    monkeypatch.setattr(NEW, '_verified_models', lambda: {})
    def read(path):
        name = Path(path).name
        if name == 'PROTOCOL.json':
            return {}
        if name == 'INPUT_SEAL.json':
            return dict(rows=238, input_sha256='input-bytes')
        if Path(path) == NEW.PLAN:
            return declaration
        raise AssertionError('Unexpected additional preparation read')
    monkeypatch.setattr(NEW.C, 'read', read)
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid declaration streamed inputs or wrote output')
    monkeypatch.setattr(NEW.C, 'rows', forbidden)
    monkeypatch.setattr(NEW.C, 'write', forbidden)
    with pytest.raises(ValueError, match='Exact nine-arm declaration and frozen parent238 required'):
        NEW.prepare(output)
    assert not output.exists()
