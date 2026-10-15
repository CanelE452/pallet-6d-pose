"""Synthetic central33 contracts only; no actual masks/packets/GT are read."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import declare_largest_component_global_outline_v4_33 as D


def _no_io(*args, **kwargs):
    raise AssertionError('Actual IO forbidden')


def _arguments():
    return dict(root_metadata_GO=True, operator_binding=dict(path='operator.py', sha256='1'*64),
        operator_peer_binding=dict(path='operator_peer.json', sha256='2'*64),
        controller_binding=dict(path='controller.py', sha256='3'*64),
        controller_peer_binding=dict(path='controller_peer.json', sha256='4'*64))


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    D.C.write(path, value)
    return path


def test_root_go_and_required_bindings_before_any_io(monkeypatch):
    monkeypatch.setattr(D.C, 'read', _no_io)
    monkeypatch.setattr(D.C, 'sha', _no_io)
    monkeypatch.setattr(Path, 'exists', _no_io)
    with pytest.raises(PermissionError):
        D.declare()
    for key in _arguments():
        if key == 'root_metadata_GO':
            continue
        args = _arguments(); args[key] = None
        with pytest.raises(ValueError, match='before IO'):
            D.declare(**args)


@pytest.mark.parametrize('value', [None, {}, {'path': '/absolute.py', 'sha256': 'a'*64},
    {'path': '../outside.py', 'sha256': 'a'*64}, {'path': 'x.py', 'sha256': 'A'*64}])
def test_relative_final_binding_fail_closed(value):
    with pytest.raises(ValueError):
        D._validate_binding(value)


def test_union_conflict_and_nested_review_source_bindings():
    value = dict(source_sha256={'one.py': '1'*64}, children=[dict(path='two.py', sha256='2'*64)])
    assert D._source_bindings(value) == {'one.py': '1'*64, 'two.py': '2'*64}
    with pytest.raises(ValueError, match='Conflicting'):
        D._union({'one.py': '1'*64}, {'one.py': '2'*64})


def test_peer_hash_before_json(tmp_path, monkeypatch):
    monkeypatch.setattr(D.C, 'ROOT', tmp_path)
    path = tmp_path/'peer.json'; path.write_text('invalid JSON')
    monkeypatch.setattr(D.C, 'read', _no_io)
    with pytest.raises(ValueError, match='before metadata decode'):
        D._peer(dict(path='peer.json', sha256='0'*64), ())


@pytest.mark.parametrize('key', ['status', 'verdict', 'essential_review', 'essential_source_review'])
def test_peer_exact_source_and_blocker_contract(tmp_path, monkeypatch, key):
    monkeypatch.setattr(D.C, 'ROOT', tmp_path)
    source = tmp_path/'source.py'; source.write_text('frozen source')
    peer = tmp_path/'peer.json'
    base = {key: 'PASS', 'source_sha256': {'source.py': D.C.sha(source)}, 'essential_blocker_count': 0}
    def call(value):
        _write(peer, value)
        return D._peer(dict(path='peer.json', sha256=D.C.sha(peer)), (source,))
    assert call(base) == peer
    for patch in ({key: 'FAIL'}, {'essential_blocker_count': 1}, {'failures': ['bad']},
            {'source_sha256': {'source.py': '0'*64}}):
        with pytest.raises(ValueError):
            call({**base, **patch})


def test_components_final_hash_before_import_or_peer(tmp_path, monkeypatch):
    monkeypatch.setattr(D.C, 'ROOT', tmp_path)
    monkeypatch.setattr(D, 'S', tmp_path)
    (tmp_path/(D.OPERATOR_NAME+'.py')).write_text('source')
    monkeypatch.setattr(D, '_peer', _no_io)
    monkeypatch.setattr(D.importlib, 'import_module', _no_io)
    with pytest.raises(ValueError, match='before import'):
        D._components(dict(path=D.OPERATOR_NAME+'.py', sha256='0'*64), {}, {}, {})


def _metadata_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(D, 'PARENT', tmp_path/'parent')
    monkeypatch.setattr(D, 'CENTRAL32', tmp_path/'central.json')
    monkeypatch.setattr(D, 'MODEL32', tmp_path/'modelseal.json')
    monkeypatch.setattr(D, 'PINNED', {D.CENTRAL32: 'c'*64, D.MODEL32: 'd'*64})
    protocolpath = _write(D.PARENT/'PROTOCOL.json', {'opaque_protocol_fixture': True})
    inputpath = _write(D.PARENT/'INPUT_SEAL.json', {'opaque_inputseal_fixture': True})
    ids = [f'case:{n:03}' for n in range(238)]
    records = [dict(id=ident, partition='CAL' if n < 46 else 'TEST' if n < 95 else 'REMAINDER',
        ready=n != 17, positive_packet_sha256=f'{n:064x}') for n,ident in enumerate(ids)]
    protocol = dict(ordered_ids=ids, records=records, arms=list(D.PARENT_ARMS),
        partitions=dict(CAL=46, TEST=49, REMAINDER=143),
        protected_user_sha256={f'p{i}': str(i)*64 for i in range(6)},
        original_CAL_file_sha256={f'cal{i}': f'{i:064x}' for i in range(53)},
        parent_e_comparison_separate_from_numeric_inputs=True, old_pose_bank_start_or_fallback_inputs=0,
        central_plan_sha256='c'*64, new_model_seal_sha256='d'*64)
    seal = dict(rows=238, ordered_ids=ids, arms=list(D.PARENT_ARMS), arm_count=5,
        protocol_sha256=D.C.sha(protocolpath))
    prediction = dict(seal, input_seal_sha256=D.C.sha(inputpath), all_declared_arms_complete=True)
    return protocol, seal, prediction, dict(rows=238, ordered_ids=ids), dict(ordered_ids=ids, partitions=D.PARTITIONS)


def test_parent_metadata_238_allpartitions_unavailable_retained_no_rows(tmp_path, monkeypatch):
    values = _metadata_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(D.C, 'rows', _no_io)
    ids, records = D._validate_parent_metadata(*values)
    assert len(ids) == len(records) == 238 and records[17]['ready'] is False
    assert [record['id'] for record in records] == ids


def test_parent_sealed_positive_and_E_payload_bindings_no_decode(tmp_path, monkeypatch):
    monkeypatch.setattr(D.C, 'ROOT', tmp_path)
    monkeypatch.setattr(D, 'PARENT', tmp_path/'parent')
    monkeypatch.setattr(D.C, 'rows', _no_io)
    monkeypatch.setattr(D.C, 'read', _no_io)
    protocol = dict(e_comparison_sha256='e'*64)
    inputseal = dict(input_sha256='a'*64, e_comparison_sha256='e'*64)
    predseal = dict(separate_e_comparison_sha256='e'*64)
    assert D._parent_payload_bindings(protocol,inputseal,predseal) == {
        'parent/INPUTS.jsonl.gz':'a'*64,'parent/E_COMPARISON.jsonl.gz':'e'*64}
    for patch in (dict(input_sha256=None),dict(e_comparison_sha256='f'*64)):
        with pytest.raises(ValueError,match='sealed positive'):
            D._parent_payload_bindings(protocol,{**inputseal,**patch},predseal)
    with pytest.raises(ValueError,match='sealed positive'):
        D._parent_payload_bindings(protocol,inputseal,dict(separate_e_comparison_sha256='f'*64))


@pytest.mark.parametrize('change', ['order', 'partition', 'drop', 'CAL53', 'e_numeric', 'seal', 'model'])
def test_parent_metadata_mismatch_rejected(tmp_path, monkeypatch, change):
    values = deepcopy(_metadata_fixture(tmp_path, monkeypatch))
    protocol, inputseal, pred, obs, central = values
    if change == 'order': protocol['records'] = list(reversed(protocol['records']))
    elif change == 'partition': protocol['records'][46]['partition'] = 'REMAINDER'
    elif change == 'drop': protocol['records'].pop()
    elif change == 'CAL53': protocol['original_CAL_file_sha256'].pop('cal0')
    elif change == 'e_numeric': protocol['parent_e_comparison_separate_from_numeric_inputs'] = False
    elif change == 'seal': pred['all_declared_arms_complete'] = False
    elif change == 'model': protocol['new_model_seal_sha256'] = 'f'*64
    with pytest.raises(ValueError):
        D._validate_parent_metadata(*values)


def test_make_plan_uniform_LCC_only_same32_noise_and_native_math(tmp_path, monkeypatch):
    monkeypatch.setattr(D.C, 'ROOT', tmp_path)
    values = _metadata_fixture(tmp_path, monkeypatch)
    protocol = values[0]
    monkeypatch.setattr(D.C, 'read', _no_io)
    monkeypatch.setattr(D.C, 'rows', _no_io)
    arms = ('SAME_FROZEN_E','LCC_ALL_N','LCC_ALL_C','LCC_SOURCE_N','LCC_SOURCE_C')
    plan = D._make_plan(protocol, protocol['ordered_ids'], protocol['records'], arms,
        {'helper.py':'a'*64}, {'opaque_parent.json':'b'*64}, *([{}]*4))
    assert plan['selection_profile'] == D.LCC.SELECTION_PROFILE
    assert plan['mask_source_packet_records'] == protocol['records']
    assert plan['partitions'] == D.PARTITIONS and plan['compatible_CAL_inventory'] == 44
    assert plan['new_noise_fit_or_distillation'] is False
    assert plan['numeric_C_or_noise_or_line_population_or_guard_changes'] is False
    assert plan['S_image_raw_2D_or_region_queries'] == plan['U_precise_point_or_local_queries'] == 0
    assert plan['numeric_e_prior_seed_start_candidate_familylock_or_fallback'] == 0
    assert plan['inherited_input_map_target_bytes_rehashed_here'] is False
    assert plan['native_arms_outline_independent_exact_parity_required'] is True
    assert 'regularizers.outline_observation ONLY' in plan['changed_numerical_input']


def test_fake_declare_append_only_maps_snapshots_and_no_numeric_IO(tmp_path, monkeypatch):
    monkeypatch.setattr(D.C, 'ROOT', tmp_path)
    monkeypatch.setattr(D, 'OUTPUT', tmp_path/'declaration')
    monkeypatch.setattr(D, 'PLAN', D.OUTPUT/'PREDECLARATION.json')
    monkeypatch.setattr(D, 'S', tmp_path)
    helper = tmp_path/'largest_component_global_outline_v4_33.py'; helper.write_text('pure LCC source')
    (tmp_path/'test_largest_component_global_outline_v4_33.py').write_text('pure test')
    own = tmp_path/'declare_largest_component_global_outline_v4_33.py'; own.write_text('pure declaration')
    (tmp_path/'test_declare_largest_component_global_outline_v4_33.py').write_text('pure test')
    common = tmp_path/'common.py'; common.write_text('shared finite JSON storage')
    monkeypatch.setattr(D, '__file__', str(own)); monkeypatch.setattr(D.C, '__file__', str(common))
    monkeypatch.setattr(D.LCC, '__file__', str(helper)); monkeypatch.setattr(D, 'LCC_SHA256', D.C.sha(helper))
    monkeypatch.setattr(D, 'FIXED_REVIEWS', ())
    protocol, *_ = _metadata_fixture(tmp_path, monkeypatch)
    protected = {}
    for n in range(6):
        path=tmp_path/f'protected{n}';path.write_text(f'protected {n}');protected[path.name]=D.C.sha(path)
    cal = {}
    for n in range(53):
        path=tmp_path/f'cal{n}.gz';path.write_bytes(b'opaque synthetic CAL bytes');cal[path.name]=D.C.sha(path)
    protocol.update(source_sha256={'common.py':D.C.sha(common)}, input_sha256={'opaque_reference_not_opened.json':'a'*64},
        protected_user_sha256=protected, original_CAL_file_sha256=cal)
    payload = tmp_path/'opaque-positive.gz';payload.write_bytes(b'opaque synthetic positive journal')
    payloads={payload.name:D.C.sha(payload)}
    monkeypatch.setattr(D, '_parent_metadata', lambda:(protocol,protocol['ordered_ids'],protocol['records'],payloads))
    arms=('SAME_FROZEN_E','LCC_ALL_N','LCC_ALL_C','LCC_SOURCE_N','LCC_SOURCE_C')
    monkeypatch.setattr(D,'_components',lambda *args:(arms,{},[]))
    # PINNED files are only recorded, not newly decoded by _make_plan.
    for path in D.PINNED: path.write_text('opaque metadata')
    monkeypatch.setattr(D.C,'rows',_no_io)
    result=D.declare(**_arguments())
    plan=D.C.read(D.PLAN);seal=D.C.read(D.OUTPUT/'PLAN_SEAL.json')
    assert result['rows']==238 and len(result['arms'])==5 and result['actual_new_mask_or_pose_execution']==0
    assert plan['input_sha256']['opaque_reference_not_opened.json']=='a'*64
    assert plan['input_sha256'][payload.name]==D.C.sha(payload)
    assert plan['opaque_parent_payload_sha256']==payloads
    assert len(plan['original_CAL_file_sha256'])==53
    assert seal['plan_sha256']==D.C.sha(D.PLAN) and seal['actual_numeric_decodes']==0
    for relative,digest in plan['source_sha256'].items():
        assert D.C.sha(D.OUTPUT/'code_snapshot'/relative)==digest
    with pytest.raises(FileExistsError):D.declare(**_arguments())
    assert D.C.sha(D.PLAN)==seal['plan_sha256']
