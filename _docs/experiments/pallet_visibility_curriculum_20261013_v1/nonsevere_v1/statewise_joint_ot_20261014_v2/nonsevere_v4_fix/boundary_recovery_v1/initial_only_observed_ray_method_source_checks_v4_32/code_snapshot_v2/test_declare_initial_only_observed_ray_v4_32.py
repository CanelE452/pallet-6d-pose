"""Synthetic declaration plus frozen small source-export contract tests.

No real declaration, observation rows, model values, labels or RGB are read.
"""
import ast
from copy import deepcopy
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import declare_initial_only_observed_ray_v4_32 as D


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    D.C.write(path, value)
    return path


def _no_io(*args, **kwargs):
    raise AssertionError('IO forbidden before explicit GO / valid final bindings')


def _args():
    return dict(root_metadata_GO=True, expected_fullsolver_sha256='1'*64,
        fullsolver_peer_binding=dict(path='solver_peer.json', sha256='2'*64),
        operator_peer_binding=dict(path='operator_peer.json', sha256='3'*64))


def test_go_and_required_final_bindings_precede_any_io(monkeypatch):
    monkeypatch.setattr(D.C, 'sha', _no_io)
    monkeypatch.setattr(D.C, 'read', _no_io)
    monkeypatch.setattr(Path, 'exists', _no_io)
    with pytest.raises(PermissionError):
        D.declare()
    for key in ('expected_fullsolver_sha256', 'fullsolver_peer_binding', 'operator_peer_binding'):
        arguments = _args();arguments[key] = None
        with pytest.raises(ValueError, match='before IO'):
            D.declare(**arguments)


def test_pure_current_family_profile_and_five_arm_contract(monkeypatch):
    D._contracts()
    assert len(D.ARMS) == 5 and D.ARMS[0] == 'SAME_FROZEN_E'
    assert tuple(D.ARMS[1:]) == D.SOLVER.ARMS
    monkeypatch.setattr(D, 'ARMS', D.ARMS[:-1])
    with pytest.raises(ValueError, match='five|four uniform'):
        D._contracts()


def test_profile_mismatch_is_rejected(monkeypatch):
    monkeypatch.setattr(D.NOISE, 'POPULATION', 'old population')
    with pytest.raises(ValueError, match='typed32'):
        D._contracts()


def test_nested_source_binding_collection_and_conflicts():
    value = dict(source_sha256={'a.py': 'a'*64}, children=[dict(path='b.py', sha256='b'*64),
        dict(deeper=dict(source_sha256={'c.py': 'c'*64}))])
    assert D._source_bindings(value) == {'a.py': 'a'*64, 'b.py': 'b'*64, 'c.py': 'c'*64}
    value['children'].append(dict(path='a.py', sha256='d'*64))
    with pytest.raises(ValueError):
        D._source_bindings(value)


def test_peer_sha_is_checked_before_json_decode(tmp_path, monkeypatch):
    monkeypatch.setattr(D.C, 'ROOT', tmp_path)
    path = tmp_path/'peer.json';path.write_text('not JSON', encoding='utf-8')
    monkeypatch.setattr(D.C, 'read', _no_io)
    with pytest.raises(ValueError, match='before metadata decode'):
        D._peer(dict(path='peer.json', sha256='0'*64), ())


@pytest.mark.parametrize('verdict_key', ['status', 'verdict', 'essential_source_review', 'essential_review'])
def test_peer_exact_source_and_pass_contract(tmp_path, monkeypatch, verdict_key):
    monkeypatch.setattr(D.C, 'ROOT', tmp_path)
    source = tmp_path/'core.py';source.write_text('source bytes', encoding='utf-8')
    module = SimpleNamespace(__file__=str(source))
    review = {verdict_key: 'PASS', 'blockers': [], 'failures': 0,
        'source_sha256': {'core.py': D.C.sha(source)}}
    path = _write(tmp_path/'peer.json', review)
    def checked(value):
        _write(path, value)
        return D._peer(dict(path='peer.json', sha256=D.C.sha(path)), (module,))
    assert checked(review) == path
    for patch in (dict(blockers=['issue']), {verdict_key: 'FAIL'},
                  dict(source_sha256={'core.py': '0'*64})):
        with pytest.raises(ValueError):
            checked({**review, **patch})


def test_final_solver_bytes_checked_before_peer_read(tmp_path, monkeypatch):
    source = tmp_path/'solver.py';source.write_text('frozen solver', encoding='utf-8')
    monkeypatch.setattr(D.SOLVER, '__file__', str(source))
    monkeypatch.setattr(D, '_peer', _no_io)
    with pytest.raises(ValueError, match='final independently reviewed'):
        D._critical_bindings('0'*64, {}, {})


def _raw_fixture(tmp_path, monkeypatch):
    raw = tmp_path/'raw';raw.mkdir()
    monkeypatch.setattr(D, 'RAW', raw)
    ids = [f'id{i:03}' for i in range(238)]
    protected = {f'protected{i}': str(i)*64 for i in range(6)}
    parent = dict(ids=ids, protected_user_sha256=protected)
    _write(raw/'PROTOCOL.json', parent)
    _write(raw/'PREPARATION_SEAL.json', dict(prepared=True))
    (raw/'OBSERVATIONS.jsonl.gz').write_bytes(b'opaque synthetic journal; never decoded')
    partition_seals = {}
    for part, selected, offset in (('CAL', ids[:46], 0), ('REST', ids[46:], 46)):
        entries = [dict(id=ident, partition='CAL' if index < 46 else 'TEST' if index < 95 else 'REMAINDER')
            for index, ident in enumerate(selected, offset)]
        indexpath = _write(raw/(part+'_ROW_INDEX.json'), dict(entries=entries))
        sealpath = _write(raw/(part+'_OBSERVATION_SEAL.json'), dict(rows=len(entries),
            ordered_ids=selected, row_index_sha256=D.C.sha(indexpath)))
        partition_seals[part] = D.C.sha(sealpath)
    _write(raw/'OBSERVATION_SEAL.json', dict(rows=238, ordered_ids=ids,
        observations_sha256=D.C.sha(raw/'OBSERVATIONS.jsonl.gz'),
        protocol_sha256=D.C.sha(raw/'PROTOCOL.json'), partition_seals=partition_seals))
    closure = dict(ordered_ids=ids, protected_user_sha256=protected)
    monkeypatch.setattr(D.OPERATOR.OP, 'verify_operational_closure', lambda: closure)
    monkeypatch.setattr(D.C, 'rows', _no_io)
    return raw


def test_raw_parent_238_partition_seals_without_row_decode(tmp_path, monkeypatch):
    raw = _raw_fixture(tmp_path, monkeypatch)
    parent, closure, paths = D._raw_parent_metadata()
    assert len(parent['ids']) == 238 and closure['ordered_ids'] == parent['ids']
    assert len(paths) == 8
    indexpath = raw/'REST_ROW_INDEX.json';index = D.C.read(indexpath)
    index['entries'][0]['partition'] = 'REMAINDER';_write(indexpath, index)
    partpath = raw/'REST_OBSERVATION_SEAL.json';part = D.C.read(partpath)
    part['row_index_sha256'] = D.C.sha(indexpath);_write(partpath, part)
    fullpath = raw/'OBSERVATION_SEAL.json';full = D.C.read(fullpath)
    full['partition_seals']['REST'] = D.C.sha(partpath);_write(fullpath, full)
    with pytest.raises(ValueError, match='TEST49'):
        D._raw_parent_metadata()


def test_raw_parent_mismatched_order_or_bytes_rejected(tmp_path, monkeypatch):
    raw = _raw_fixture(tmp_path, monkeypatch)
    path = raw/'OBSERVATION_SEAL.json';value = D.C.read(path)
    value['ordered_ids'] = value['ordered_ids'][::-1];_write(path, value)
    with pytest.raises(ValueError, match='raw238'):
        D._raw_parent_metadata()


def test_fixed_cal3_metadata_and_independent333_check_only(tmp_path, monkeypatch):
    cal = tmp_path/'cases/RERANK_SEAL.json'
    plan = _write(tmp_path/'initial_only_observed_ray_CAL_plan_v4_32/PREDECLARATION.json', dict(fixed=True))
    _write(plan.parent/'PLAN_SEAL.json', dict(plan_sha256=D.C.sha(plan)))
    value = dict(rows=3, ordered_ids=list(D.FIXED_CAL3_IDS), selection_profile=D.G.SELECTION_PROFILE,
        plan_sha256=D.C.sha(plan), external_pose_guides_consumed=0, source_sha256={})
    _write(cal, value)
    source = tmp_path/'audit_selected_query_equations_v1.py'
    source.write_text('standalone source bytes', encoding='utf-8')
    audit = _write(tmp_path/'audit.json', dict(checks=333, failures=[], execution_seal_sha256=D.C.sha(cal),
        plan_sha256=D.C.sha(plan), source_sha256=D.C.sha(source)))
    monkeypatch.setattr(D, 'CAL3', cal);monkeypatch.setattr(D, 'CAL3_SHA256', D.C.sha(cal))
    monkeypatch.setattr(D, 'CAL3_AUDIT', audit);monkeypatch.setattr(D, 'CAL3_AUDIT_SHA256', D.C.sha(audit))
    assert D._prior_cal_metadata()[:2] == [cal, audit]
    wrong = D.C.read(audit);wrong['checks'] = 332;_write(audit, wrong)
    monkeypatch.setattr(D, 'CAL3_AUDIT_SHA256', D.C.sha(audit))
    with pytest.raises(ValueError, match='CAL3 metadata'):
        D._prior_cal_metadata()


def test_actual_small333_audit_single_source_digest_contract():
    # Whitelist source metadata only; no cases/equations/coordinates are used.
    stored = D.C.read(D.CAL3_AUDIT)
    audit = {key: stored[key] for key in ('checks', 'failures', 'source_sha256')}
    assert audit['checks'] == 333 and not audit['failures']
    assert isinstance(audit['source_sha256'], str)
    source = D._independent_cal_audit_source(audit, D.CAL3_AUDIT)
    assert source.name == 'audit_selected_query_equations_v1.py'
    assert D.C.sha(source) == audit['source_sha256']


def test_standalone333_source_digest_mismatch_is_rejected(tmp_path):
    source = tmp_path/'audit_selected_query_equations_v1.py'
    source.write_text('source', encoding='utf-8')
    for value in ('0'*64, {'not': 'a source map'}):
        with pytest.raises(ValueError, match='standalone audit source'):
            D._independent_cal_audit_source(dict(source_sha256=value), tmp_path/'audit.json')


def test_actual_prior_export_filenames_exist_without_model_value_decode(monkeypatch):
    monkeypatch.setattr(D.C, 'read', _no_io)
    paths = D._prior_model_paths()
    assert len(paths) == 10 and all(path.is_file() for path in paths)
    assert {path.name for path in paths} == {'PROTOCOL.json','PREPARATION_SEAL.json',
        'MODEL.json','MODEL_SEAL.json','CAL_LOGO_REPORT.json'}


def test_operational_closure_source_exports_consumed_keys_without_execution():
    tree = ast.parse(inspect.getsource(D.OPERATOR.OP.verify_operational_closure))
    direct = [node for node in ast.walk(tree) if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
        and node.value.func.id == 'dict']
    assert len(direct) == 1
    keys = {keyword.arg for keyword in direct[0].value.keywords}
    assert {'ordered_ids','source_sha256','input_sha256','protected_user_sha256',
        'original_immutable_file_sha256'} <= keys


def test_fake_declaration_five_arms_allrows_no_bank_and_appendonly(tmp_path, monkeypatch):
    monkeypatch.setattr(D.C, 'ROOT', tmp_path)
    monkeypatch.setattr(D, 'OUTPUT', tmp_path/'declaration')
    monkeypatch.setattr(D, 'PLAN', D.OUTPUT/'PREDECLARATION.json')
    source = tmp_path/'solver.py';source.write_text('frozen numerical source', encoding='utf-8')
    monkeypatch.setattr(D.SOLVER, '__file__', str(source))
    inputpath = _write(tmp_path/'parent.json', dict(opaque=True))
    sources = {'solver.py': D.C.sha(source)}
    protected = {}
    for number in range(6):
        path = tmp_path/f'protected{number}';path.write_text(str(number), encoding='utf-8')
        protected[path.name] = D.C.sha(path)
    parent = dict(ids=[f'id{i}' for i in range(238)], protected_user_sha256=protected)
    closure = dict(source_sha256=sources, input_sha256={'parent.json': D.C.sha(inputpath)},
        original_immutable_file_sha256={'parent.json': D.C.sha(inputpath)})
    monkeypatch.setattr(D, '_critical_bindings', lambda *args: (sources, []))
    monkeypatch.setattr(D, '_raw_parent_metadata', lambda: (parent, closure, [inputpath]))
    monkeypatch.setattr(D, '_prior_cal_metadata', lambda: [])
    monkeypatch.setattr(D.OPERATOR, '_pins', lambda: None)
    for attr in ('SOURCE_CLOSURE', 'ANCESTRY'):
        monkeypatch.setattr(D.OPERATOR, attr, inputpath)
    monkeypatch.setattr(D, 'CAL3', inputpath);monkeypatch.setattr(D, 'CAL3_AUDIT', inputpath)
    for number, module in enumerate((D.SCAL, D.OCAL)):
        folder = tmp_path/f'model{number}';monkeypatch.setattr(module, 'OUTPUT', folder)
        for name in ('PROTOCOL.json','PREPARATION_SEAL.json','MODEL.json','MODEL_SEAL.json','CAL_LOGO_REPORT.json'):
            _write(folder/name, dict(opaque_frozen_model=True))
    monkeypatch.setattr(D.C, 'rows', _no_io)
    answer = D.declare(**_args())
    plan = D.C.read(D.PLAN);seal = D.C.read(D.OUTPUT/'PLAN_SEAL.json')
    assert answer['rows'] == 238 and len(answer['arms']) == 5 and answer['new32_numeric_execution'] == 0
    assert len(plan['ordered_ids']) == 238 and len(plan['protected_user_sha256']) == 6
    assert plan['new_CAL_model_count'] == 20 and plan['new_CAL_optimizer_starts'] == 60
    assert plan['cached_old_method_pose_prior_start_candidate_familylock_or_fallback_used'] is False
    assert plan['numeric_e_pose_prior_start_candidate_or_fallback_added'] is False
    assert plan['current_other_instance_mask_consumer_connection'] is False
    assert plan['S_only_model_prior_and_PnP_projection'] is True
    assert plan['actual_new32_CAL_selection_labels_fits_RGB_or_pose_evaluation_in_declare'] == 0
    assert seal['plan_sha256'] == D.C.sha(D.PLAN)
    assert (D.OUTPUT/'code_snapshot/solver.py').read_bytes() == source.read_bytes()
    before = D.PLAN.read_bytes()
    with pytest.raises(FileExistsError):
        D.declare(**_args())
    assert D.PLAN.read_bytes() == before
