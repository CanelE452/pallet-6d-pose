"""Synthetic controller tests only; no actual CAL metadata or labels read."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from . import calibrate_geometry_guided_edge_noise_v4_31 as D
from . import test_calibrate_geometry_guided_edge_noise_v4_30 as PREP_FIXTURE
from .test_geometry_guided_edge_noise_calibration_v4_31 import records

M = D.M


def prepared():
    row = PREP_FIXTURE.prepared()
    row.update(observation_generation=M.OBSERVATION_FAMILY,
        raw_observation_generation=M.RAW_OBSERVATION_GENERATION,
        noise_population=M.POPULATION, selection_profile=M.SELECTION_PROFILE)
    for profile in row['profiles'].values():
        profile['selection_profile'] = M.SELECTION_PROFILE
    return row


def central():
    return dict(observation_family=M.OBSERVATION_FAMILY, population=M.POPULATION,
        selection_profile=M.SELECTION_PROFILE, raw_observation_generation=M.RAW_OBSERVATION_GENERATION,
        source_policies=list(M.SOURCE_POLICIES),
        partitions=dict(CAL=46, TEST=49, REMAINDER=143, PRIMARY=192, severe=0),
        source_sha256={'source.py': 'source-sha'}, input_sha256={'input.json': 'input-sha'},
        protected_user_sha256={'protected.json': 'protected-sha'})


def test_new_paths_and_no_central_SHA_literal_or_default():
    assert not hasattr(D, 'CENTRAL_PLAN_SHA256')
    assert D.CENTRAL_PLAN.parent.name == 'observed_anchor_native_ray_method_plan_v4_31'
    assert D.PREPARED.name == 'geometry_guided_native_evidence_v4_31_checks'
    assert D.OUTPUT.name == 'CAL_geometry_guided_edge_noise_v4_31'
    assert D.PLAN_OUTPUT.name == 'CAL_geometry_guided_edge_noise_plan_v4_31'
    assert D.M.CORE.OBSERVATION_FAMILY == M.OBSERVATION_FAMILY


def test_exact_explicit_central_SHA_gates_before_any_json(monkeypatch):
    monkeypatch.setattr(D.C, 'read', lambda *args: pytest.fail('JSON decoded before exact SHA'))
    for value in (None, '', 'x', 'z'*64, 'A'*64):
        with pytest.raises(ValueError, match='no default'):
            D._central_declaration(value)
    monkeypatch.setattr(D.C, 'sha', lambda *args: 'b'*64)
    with pytest.raises(ValueError, match='no default'):
        D._central_declaration('a'*64)


def test_central_three_maps_and_typed_population_are_verified(monkeypatch):
    value = central(); calls = []
    def fake_sha(path):
        calls.append(path)
        return 'a'*64 if path == D.CENTRAL_PLAN else {
            D.C.ROOT/'source.py': 'source-sha', D.C.ROOT/'input.json': 'input-sha',
            D.C.ROOT/'protected.json': 'protected-sha'}[path]
    monkeypatch.setattr(D.C, 'sha', fake_sha)
    monkeypatch.setattr(D.C, 'read', lambda path: deepcopy(value))
    assert D._central_declaration('a'*64) == value
    assert set(calls) == {D.CENTRAL_PLAN, D.C.ROOT/'source.py', D.C.ROOT/'input.json', D.C.ROOT/'protected.json'}
    value['selection_profile'] = 'old'
    with pytest.raises(ValueError, match='New V31'):
        D._central_declaration('a'*64)


def test_cross_class_central_conflict_rejected(monkeypatch):
    value = central();value['input_sha256']['source.py'] = 'different'
    monkeypatch.setattr(D.C, 'sha', lambda *args: 'a'*64)
    monkeypatch.setattr(D.C, 'read', lambda *args: deepcopy(value))
    monkeypatch.setattr(D, '_verify', lambda mapping: None)
    with pytest.raises(ValueError, match='ancestry conflict'):
        D._central_declaration('a'*64)


def test_separate_prepare_and_fit_GO_before_any_IO(monkeypatch):
    monkeypatch.setattr(D.C, 'read', lambda *args: pytest.fail('IO before root GO'))
    monkeypatch.setattr(D.C, 'sha', lambda *args: pytest.fail('Hash IO before root GO'))
    for action in (D.prepare, D.fit):
        with pytest.raises(PermissionError):
            action(expected_central_plan_sha256='a'*64)


def test_frozen_numerical_code_objects_and_parent_namespaces_are_unchanged():
    before = {name: dict(getattr(D.PARENT, name).__globals__) for name in
        ('fit', 'prepare', '_verify_preparation', 'validate_prepared_row', '_fit_or_status', '_heldout_description')}
    facade = D._facade('a'*64)
    for name in ('fit', 'prepare', 'validate_prepared_row', '_fit_or_status', '_heldout_description'):
        fn = getattr(facade, name)
        assert fn.__code__ is getattr(D.PARENT, name).__code__
        assert fn.__globals__ is not getattr(D.PARENT, name).__globals__
        assert fn.__globals__['M'] is M
        assert fn.__globals__['CENTRAL_PLAN_SHA256'] == 'a'*64
    assert all(dict(getattr(D.PARENT, name).__globals__) == value for name, value in before.items())


def test_row_profile_axis_covariance_floor_and_archive_exact():
    row = prepared();before = deepcopy(row)
    result = D.validate_prepared_row(row)
    assert row == before
    assert result['selection_profile'] == M.SELECTION_PROFILE
    for policy in M.SOURCE_POLICIES:
        profile = result['profiles'][policy]
        assert profile['selection_profile'] == M.SELECTION_PROFILE
        assert profile['tangents'][0]['original_variance_axis_px2'] == 9.
        assert profile['tangents'][0]['selected_line_tangent_unit_xy'] == [0., 1.]


@pytest.mark.parametrize('field', ['observation_generation', 'raw_observation_generation',
    'selection_profile', 'noise_population', 'policy_profile'])
def test_old_or_mixed_population_rejected_before_fit(field):
    row = prepared()
    if field == 'policy_profile':
        row['profiles']['ALL_BI_REPLACE']['selection_profile'] = 'old'
    else:
        row[field] = 'old'
    with pytest.raises(ValueError):
        D.validate_prepared_row(row)


def test_status_and_LOGO_fit_use_new_M31_tags_no_old_model_substitution():
    rows = records()
    line = D._fit_or_status(rows, 'ALL_BI_REPLACE', 'line', {}, 'b')
    tangent = D._fit_or_status(rows, 'ALL_BI_REPLACE', 'tangent', {}, 'b')
    report = D._heldout_description(rows, 'b', line, tangent)
    assert line['model']['fit_ids'] == ['0', '1']
    assert line['model']['selection_profile'] == M.SELECTION_PROFILE
    assert np.isfinite(report['heldout_line_NLL']) and np.isfinite(report['heldout_tangent_NLL'])
    for row in rows:
        row['groups'] = [];row['tangents'] = []
    result = D._fit_or_status(rows, 'ALL_BI_REPLACE', 'line', {})
    assert result['model'] is None and result['old_model_substituted'] is False


def test_runtime_preparation_checks_same_explicit_central_SHA(monkeypatch):
    sha = 'a'*64
    protocol = dict(source_sha256={}, protected_user_sha256={},
        input_sha256={str(D.CENTRAL_PLAN.relative_to(D.C.ROOT)): sha})
    monkeypatch.setattr(D.PARENT, '_verify_preparation', lambda output: deepcopy(protocol))
    # This stand-in retains the parent function signature while only exercising
    # the NEW post-verification central pin, not reading an actual preparation.
    monkeypatch.setattr(D, '_central_declaration', lambda expected: {
        'source_sha256': {}, 'input_sha256': {}, 'protected_user_sha256': {}})
    assert D._facade(sha)._verify_preparation(Path('synthetic')) == protocol
    protocol['input_sha256'] = {str(D.CENTRAL_PLAN.relative_to(D.C.ROOT)): 'b'*64}
    with pytest.raises(ValueError, match='same caller-supplied'):
        D._facade(sha)._verify_preparation(Path('synthetic'))


def test_source_union_includes_new_driver_test_and_M31_with_frozen_ancestors():
    sources = D._sources()
    for module in (D, M, M.PARENT, M.CORE, M.CORE.ASSOCIATION, D.PARENT):
        path = Path(module.__file__).resolve()
        assert sources[str(path.relative_to(D.C.ROOT))] == D.C.sha(path)
    assert sources[str(Path(__file__).resolve().relative_to(D.C.ROOT))] == D.C.sha(__file__)
