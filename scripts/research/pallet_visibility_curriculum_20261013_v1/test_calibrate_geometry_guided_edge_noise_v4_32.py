"""Synthetic controller tests only; no actual CAL metadata or labels read."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from . import calibrate_geometry_guided_edge_noise_v4_32 as D
from . import test_calibrate_geometry_guided_edge_noise_v4_30 as PREP_FIXTURE
from .test_geometry_guided_edge_noise_calibration_v4_32 import records

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
    assert D.CENTRAL_PLAN.parent.name == 'initial_only_observed_ray_method_plan_v4_32'
    assert D.PREPARED.name == 'geometry_guided_native_evidence_v4_32_checks'
    assert D.OUTPUT.name == 'CAL_geometry_guided_edge_noise_v4_32'
    assert D.PLAN_OUTPUT.name == 'CAL_geometry_guided_edge_noise_plan_v4_32'
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
    with pytest.raises(ValueError, match='New V32'):
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
    for action in (D.plan, D.prepare, D.fit):
        with pytest.raises(PermissionError):
            action(expected_central_plan_sha256='a'*64)


def test_every_authorized_stage_rejects_missing_central_before_source_hash_IO(monkeypatch):
    monkeypatch.setattr(D.C,'read',lambda *args:pytest.fail('No JSON before explicit SHA'))
    monkeypatch.setattr(D.C,'sha',lambda *args:pytest.fail('No source/hash IO before explicit SHA'))
    for action,flag in ((D.plan,'root_metadata_GO'),(D.prepare,'root_metadata_GO'),(D.fit,'root_CAL_fit_GO')):
        for value in (None,'','A'*64):
            with pytest.raises(ValueError,match='no default'):
                action(**{flag:True,'expected_central_plan_sha256':value})


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


def test_status_and_LOGO_fit_use_new_M32_tags_no_old_model_substitution():
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
    monkeypatch.setattr(D, '_verify_metadata_ancestry', lambda *args: None)
    assert D._facade(sha)._verify_preparation(Path('synthetic')) == protocol
    protocol['input_sha256'] = {str(D.CENTRAL_PLAN.relative_to(D.C.ROOT)): 'b'*64}
    with pytest.raises(ValueError, match='same caller-supplied'):
        D._facade(sha)._verify_preparation(Path('synthetic'))


def test_source_union_includes_new_driver_test_and_M32_with_frozen_ancestors():
    sources = D._sources()
    for module in (D, M, M.PARENT, M.CORE, M.CORE.ASSOCIATION, D.PARENT):
        path = Path(module.__file__).resolve()
        assert sources[str(path.relative_to(D.C.ROOT))] == D.C.sha(path)
    assert sources[str(Path(__file__).resolve().relative_to(D.C.ROOT))] == D.C.sha(__file__)


def metadata_documents(monkeypatch):
    folder = D.PREPARED
    immutable = {f'synthetic/CAL/file{i}':f'{i:064x}' for i in range(53)}
    source = {str(Path(D.METADATA.__file__).resolve().relative_to(D.C.ROOT)):D.METADATA_SOURCE_SHA256,
        str(Path(D.METADATA.M.__file__).resolve().relative_to(D.C.ROOT)):D.FACTORY_SOURCE_SHA256}
    tags = dict(observation_generation=M.OBSERVATION_FAMILY,raw_observation_generation=M.RAW_OBSERVATION_GENERATION,
        selection_profile=M.SELECTION_PROFILE,noise_population=M.POPULATION)
    prep = dict(schema='ZERO_COMPONENT_CAL46_V32_POSITIVE_NATIVE_METADATA_PREPARATION',
        ids=[f'CAL:{i}'for i in range(46)],central_declaration_sha256='a'*64,CAL_labels_read=False,
        original_CAL_file_sha256=immutable,source_sha256=source,input_sha256=deepcopy(immutable),
        protected_user_sha256={'synthetic/user':'u'},initial_source_binding=dict(
        sha256=M.CORE.INITIAL_SOURCE_CLOSURE_SHA256,verified_by_caller=True,
        independent_initial_geometry_and_BI_q=True),raw_query_source_audit_sha256=D.METADATA.RAW_QUERY_AUDIT_SHA256,**tags)
    report = dict(rows=46,successful_metadata_rows=46,error_count=0,errors=[],CAL_reference_decodes=0,
        primary_rows_decoded=0,fresh_PnP_calls=0,pose_objective_or_fit_calls=0,**tags)
    seal = dict(preparation_sha256='p',report_sha256='r',prepared_groups_sha256='g')
    docs = {folder/'PREPARATION_SEAL.json':prep,folder/'CAL_PREFLIGHT_V1.json':report,folder/'CAL_PREFLIGHT_SEAL_V1.json':seal}
    hashes = {folder/'PREPARATION_SEAL.json':'p',folder/'CAL_PREFLIGHT_V1.json':'r',folder/'CAL_PREPARED_GROUPS.jsonl.gz':'g'}
    monkeypatch.setattr(D.C,'read',lambda path:deepcopy(docs[path]))
    monkeypatch.setattr(D.C,'sha',lambda path:hashes[Path(path)])
    return prep,report,seal,hashes


def test_new_metadata_preflight_typed46_CAL53_and_pinned_positive_sources(monkeypatch):
    prep,_,_,_=metadata_documents(monkeypatch)
    actual,immutable=D._metadata_documents(D.PREPARED,'a'*64)
    assert actual==prep and immutable==prep['original_CAL_file_sha256'] and len(immutable)==53


@pytest.mark.parametrize('change',['53missing','53notbound','labels','population','factory','errors','PnP','proof','central'])
def test_bad_new_metadata_cannot_reach_numeric_preparation(change,monkeypatch):
    prep,report,seal,hashes=metadata_documents(monkeypatch)
    if change=='53missing':prep['original_CAL_file_sha256'].popitem()
    elif change=='53notbound':prep['input_sha256'].popitem()
    elif change=='labels':prep['CAL_labels_read']=True
    elif change=='population':report['noise_population']='old31'
    elif change=='factory':prep['source_sha256'][str(Path(D.METADATA.M.__file__).resolve().relative_to(D.C.ROOT))]='old'
    elif change=='errors':report['error_count']=1
    elif change=='PnP':report['fresh_PnP_calls']=1
    elif change=='proof':prep['initial_source_binding']['verified_by_caller']=False
    else:prep['central_declaration_sha256']='b'*64
    with pytest.raises(ValueError):D._metadata_documents(D.PREPARED,'a'*64)


def test_full_metadata_prepare_closes_actual_operator_CAL53_and_initial_source_proofs(monkeypatch):
    prep,_,_,_=metadata_documents(monkeypatch);verified=[];calls=[]
    monkeypatch.setattr(D.METADATA,'_initial_source_closure',lambda:calls.append('source'))
    monkeypatch.setattr(D.METADATA,'_observation_closure',lambda sha:(None,None,prep['ids'],None,prep['original_CAL_file_sha256']))
    monkeypatch.setattr(D,'_verify',lambda mapping:verified.append(mapping))
    assert D._metadata_closure(D.PREPARED,'a'*64)==prep
    assert calls==['source'] and verified[-1]==prep['original_CAL_file_sha256']
    monkeypatch.setattr(D.METADATA,'_observation_closure',lambda sha:(None,None,prep['ids'],None,{}))
    with pytest.raises(ValueError,match='same original'):D._metadata_closure(D.PREPARED,'a'*64)


def test_fitted_protocol_keeps_new_metadata_whole_ancestry_and_immutable53(monkeypatch):
    prep,_,_,_=metadata_documents(monkeypatch)
    path=str((D.PREPARED/'PREPARATION_SEAL.json').relative_to(D.C.ROOT))
    protocol={k:deepcopy(prep[k])for k in ('source_sha256','input_sha256','protected_user_sha256')}
    protocol['input_sha256'][path]='p'
    D._verify_metadata_ancestry(protocol,'a'*64)
    protocol['input_sha256'].pop(next(iter(prep['original_CAL_file_sha256'])))
    with pytest.raises(ValueError,match='ancestry omitted|CAL53'):D._verify_metadata_ancestry(protocol,'a'*64)


def test_positive_factory_operator_pins_fail_before_any_actual_data_json(monkeypatch):
    monkeypatch.setattr(D.C,'sha',lambda path:'changed')
    monkeypatch.setattr(D.C,'read',lambda *args:pytest.fail('Source pin failure before data IO'))
    with pytest.raises(ValueError,match='positive factory'):D._pins()
