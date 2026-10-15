"""Toy masks/positive packets and mocked metadata only; no saved cohort IO."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from multiprocessing import get_context
from pathlib import Path
import inspect
import tempfile

import numpy as np
import pytest

from . import evaluate_largest_component_global_outline_poses_v4_33 as V
from .test_evaluate_clean_fresh_BI_poses_v4_32_r2 import packet


def forbidden(*args, **kwargs):
    raise AssertionError('Actual input/reference/mask/model or legacy path reached')


def toy(*, empty=False, partition='CAL'):
    _, _, _, value = packet()
    value = V.C.finite(value)
    value['identifiers']['partition'] = partition
    value['clean_input']['identifiers']['partition'] = partition
    value['physics']['config']['max_joint_iterations'] = 1
    h, w = value['physics']['image_hw']
    mask = np.zeros((h, w), np.float32)
    if not empty:
        mask[100:250, 200:400] = 1.
        mask[20:25, 30:35] = 1.
    binding = dict(image_sha256=value['identifiers']['image_sha256'], mask_file_sha256='b'*64,
                   mask_npz='SYNTHETIC.npz')
    pair = V.LCC.observe_union_and_largest_component(mask, (h, w), image_binding=binding,
        configuration={'observed_sample_spacing_px': 15.})
    old = {key: item for key, item in pair['original_union'].items() if key != 'selection_profile'}
    value['regularizers']['outline_observation'] = deepcopy(old)
    value['regularizers']['outline_configuration'] = deepcopy(old['configuration'])
    row = dict(id=value['identifiers']['id'], partition=partition,
        selection_profile=V.SELECTION_PROFILE, baseline_profile=V.LCC.BASELINE_PROFILE,
        positive_packet_sha256=V._digest(value), target_mask_binding={'sha256':'b'*64},
        union_exact_match=True, original_union=pair['original_union'], largest_component=pair['largest_component'])
    return value, row


def invalid_packet():
    _, _, _, value = packet()
    value = V.C.finite(value)
    value['clean_input']['initial_coarse_pose'] = dict(source=V.M.FACTORY.FRESH.COARSE_SOURCE,
        independent_of_cached_pose_bank=True)
    value['clean_input']['visible_measurements'] = []
    value['source_contract']['ready'] = False
    value['physics'] = value['typed_bundles'] = value['regularizers'] = None
    return value


def test_all_stage_GOs_and_explicit_SHA_checks_precede_IO(monkeypatch):
    monkeypatch.setattr(V.C, 'read', forbidden)
    monkeypatch.setattr(V.C, 'sha', forbidden)
    monkeypatch.setattr(V.M, '_load_prepared', forbidden)
    for function in (V.plan, V.prepare, V.predict):
        with pytest.raises(PermissionError):
            function()
    with pytest.raises(ValueError, match='before IO'):
        V.plan(root_metadata_GO=True)
    with pytest.raises(ValueError, match='before IO'):
        V._central(None)
    with pytest.raises(ValueError, match='1..3'):
        V.predict(root_prediction_GO=True, workers=4)


def test_frozen_worker_pool_merge_and_fresh_solver_are_exact_direct_function_aliases():
    assert V.solve_worker_packet is V.M.solve_worker_packet
    assert V._bounded_predict is V.M._bounded_predict
    assert V._merged_rows is V.M._merged_rows
    assert V.ARMS == V.M.ARMS and len(V.ARMS) == 5 and V.ARMS[0] == 'SAME_FROZEN_E'
    assert len(V.NATIVE_ARMS) == 2
    assert V.OUTPUT != V.PARENT and V.PLAN_OUTPUT != V.M.PLAN_OUTPUT
    assert V.SELECTION_PROFILE == V.LCC.SELECTION_PROFILE != V.M.G.SELECTION_PROFILE
    assert V.C.sha(V.M.__file__) == V.PARENT_CONTROLLER_SHA256
    assert V.C.sha(V.LCC.__file__) == V.LCC_SHA256
    assert V.C.sha(V.M.SOLVER.__file__) == V.M.SOLVER_SHA256


def test_one_field_replacement_keeps_q_cov_states_guards_models_and_all_other_values_exact():
    value, row = toy()
    before = deepcopy(value)
    changed = V.replace_outline_observation(value, row)
    assert value == before and changed is not value
    assert set(changed) == V.M.PROJECT.WORKER_KEYS
    for key in changed.keys() - {'regularizers'}:
        assert changed[key] is value[key]
        assert V._digest(changed[key]) == V._digest(value[key])
    for key in value['regularizers'].keys() - {'outline_observation'}:
        assert changed['regularizers'][key] is value['regularizers'][key]
    assert changed['regularizers']['outline_observation'] == row['largest_component']
    assert changed['regularizers']['outline_observation'] is not row['largest_component']
    assert changed['source_contract'] == before['source_contract']
    assert V._digest(changed['typed_bundles']) == V._digest(before['typed_bundles'])


@pytest.mark.parametrize('change', ['packet', 'id', 'partition', 'profile', 'union', 'mask', 'image', 'canvas', 'configuration', 'recipe'])
def test_mismatched_positive_or_outline_provenance_rejected_without_repair(change):
    value, row = toy()
    if change == 'packet': value['clean_input']['visible_measurements'][0]['xy'][0] += 1.
    elif change == 'id': row['id'] = 'OTHER'
    elif change == 'partition': row['partition'] = 'TEST'
    elif change == 'profile': row['selection_profile'] = 'OLD'
    elif change == 'union': row['original_union']['polygon_xy'][0][0] += 1.
    elif change == 'mask': row['target_mask_binding']['sha256'] = 'c'*64
    elif change == 'image': row['largest_component']['image_binding']['image_sha256'] = 'd'*64
    elif change == 'canvas': row['largest_component']['image_hw'] = [1, 1]
    elif change == 'configuration': row['largest_component']['configuration']['normal_sigma_px'] += 1.
    else: row['largest_component']['foreground_component_selection']['connectivity'] = 4
    with pytest.raises(ValueError):
        V.replace_outline_observation(value, row)


def test_empty_unsupported_outline_retained_without_q_or_model_changes():
    value, row = toy(empty=True)
    changed = V.replace_outline_observation(value, row)
    assert not changed['regularizers']['outline_observation']['supported']
    assert changed['regularizers']['outline_observation']['foreground_component_selection']['selected_label'] is None
    assert changed['clean_input'] == value['clean_input'] and changed['typed_bundles'] == value['typed_bundles']


def test_invalid_coarse_retains_all_four_no_pose_arms_without_other_payload_access(monkeypatch):
    value = invalid_packet()
    class Poison:
        def __getitem__(self, key): raise AssertionError('Invalid coarse inspected payload')
        def __iter__(self): raise AssertionError('Invalid coarse iterated payload')
    value['physics'] = value['typed_bundles'] = value['regularizers'] = Poison()
    monkeypatch.setattr(V.M, 'build_native_cores', forbidden)
    result = V.solve_worker_packet(value)
    assert result['row_retained'] and result['invalid_initial_coarse']
    assert len(result['arms']) == 4
    assert all(not arm['pose_available'] and arm['actual_pose'] is None for arm in result['arms'].values())


def test_only_original_seven_keys_accepted_and_E_or_bank_never_worker_inputs(monkeypatch):
    value, row = toy()
    for key in ('baseline', 'proposals', 'initializations', 'pose_guides', 'e', 'task'):
        wrong = dict(value, **{key:object()})
        with pytest.raises(ValueError, match='seven-key'):
            V.replace_outline_observation(wrong, row)
    source = inspect.getsource(V)
    for forbidden_call in ('M.prepare(', 'M.predict(', 'M._prepare_packet(', 'M._comparison_row(', 'M._models(', 'M._stage_inputs('):
        assert forbidden_call not in source
    assert inspect.signature(V.solve_worker_packet) == inspect.signature(V.M.solve_worker_packet)


def test_genuine_four_arm_solver_native_exact_C_only_control_and_input_immutable(monkeypatch):
    value, row = toy()
    before = deepcopy(value)
    changed = V.replace_outline_observation(value, row)
    monkeypatch.setattr(V.M.SOLVER.SCALAR, '_run', forbidden)
    monkeypatch.setattr(V.M.SOLVER.SCALAR, '_REFINER', forbidden, raising=False)
    original = V.solve_worker_packet(value)
    actual = V.solve_worker_packet(changed)
    assert value == before
    for arm in V.NATIVE_ARMS:
        assert V._digest(original['arms'][arm]) == V._digest(actual['arms'][arm])
    assert set(actual['arms']) == set(V.ARMS[1:])
    assert all(arm['fallback_audit']['cached_pose_inputs'] == 0 for arm in actual['arms'].values())
    old = dict(id='toy', partition='CAL', arms={'SAME_FROZEN_E':{'exact':[1.,-0.,3.]}, **original['arms']})
    new = dict(id='toy', partition='CAL', arms={'SAME_FROZEN_E':{'exact':[1.,-0.,3.]}, **actual['arms']})
    proof = V.verify_native_and_e_parity([old], [new], ['toy'])
    assert proof['native_arm_payload_equalities'] == 2 and proof['e_payload_equalities'] == 1
    for arm in actual['arms'].values():
        pose = arm['actual_pose']
        assert pose['native_t'] == arm['t']
        assert pose['WD_family'] == arm['WD_family']
        assert np.isfinite(pose['centroid']).all()


def test_native_or_E_differences_fail_closed_but_C_and_row_elapsed_can_differ():
    old = dict(id='a', partition='TEST', elapsed_seconds=1., arms={name:{'accepted':False,'pose_available':False} for name in V.ARMS})
    new = deepcopy(old);new['elapsed_seconds']=999.;new['arms'][V.ARMS[2]]={'C_diagnostic':12}
    assert V.verify_native_and_e_parity([old], [new], ['a'])['failure_count'] == 0
    for arm in (*V.NATIVE_ARMS, 'SAME_FROZEN_E'):
        wrong=deepcopy(new);wrong['arms'][arm]['R']=[[0.]]
        with pytest.raises(ValueError, match='invariant failed'):
            V.verify_native_and_e_parity([old], [wrong], ['a'])
    with pytest.raises(ValueError, match='join'):
        V.verify_native_and_e_parity([old], [], ['a'])


def test_streaming_all238_cohorts_order_and_zero_rows(monkeypatch):
    value = invalid_packet()
    ids=[f'SYNTHETIC:{i}' for i in range(238)]
    packets=[];rows=[];original=[];entries=[]
    for index, identifier in enumerate(ids):
        part='CAL' if index<46 else 'TEST' if index<95 else 'REMAINDER'
        p=deepcopy(value);p['identifiers']['id']=p['clean_input']['identifiers']['id']=identifier
        p['identifiers']['partition']=p['clean_input']['identifiers']['partition']=part
        row=dict(id=identifier,partition=part,selection_profile=V.SELECTION_PROFILE,
            baseline_profile=V.LCC.BASELINE_PROFILE,positive_packet_sha256=V._digest(p),union_exact_match=True)
        packets.append(p);rows.append(row)
        original.append(dict(id=identifier,partition=part,positive_packet_sha256=V._digest(p)))
        entries.append(dict(id=identifier,partition=part,canonical_row_sha256=V._digest(row)))
    records=[]
    result=list(V._prepared_rows(iter(packets),iter(rows),original,entries,ids,records))
    assert len(result)==len(records)==238
    assert all(p['regularizers'] is None for p in result)
    assert [record['id'] for record in records]==ids and all(record['changed_field'] is None for record in records)
    wrong=deepcopy(entries);wrong[95]['partition']='CAL'
    with pytest.raises(ValueError, match='join'):
        list(V._prepared_rows(iter(packets),iter(rows),original,wrong,ids,[]))


def outline_metadata(monkeypatch):
    ids=[f'SYNTHETIC:{i}' for i in range(238)]
    entries=[dict(id=identifier,partition='CAL' if i<46 else 'TEST' if i<95 else 'REMAINDER') for i,identifier in enumerate(ids)]
    hashes={};values={}
    def add(name,value):
        path=V.OUTLINES/name;hashes[path]='a'*64;values[path]=value
    add('FINAL_PROTOCOL.json',dict(central_plan_sha256='c'*64,selection_profile=V.SELECTION_PROFILE,ordered_ids=ids,source_sha256={},input_sha256={},protected_user_sha256={},
        original_CAL_file_sha256={str(i):'d'*64 for i in range(53)},immutable_LCC_CAL_file_sha256={str(i):'e'*64 for i in range(53)}))
    add('OUTLINE_ROW_INDEX.json',{'entries':entries})
    add('OUTLINE_OBSERVATIONS.jsonl.gz',{})
    for part,count,selected,index in (('CAL',46,ids[:46],entries[:46]),('REST',192,ids[46:],entries[46:])):
        add(part+'_ROW_INDEX.json',{'entries':index})
        add(part+'_OUTLINE_OBSERVATIONS.jsonl.gz',{})
        add(part+'_OBSERVATION_SEAL.json',dict(rows=count,ordered_ids=selected,selection_profile=V.SELECTION_PROFILE,
            row_index_sha256='a'*64,observations_sha256='a'*64,union_exact_match_rows=count))
    add('OUTLINE_OBSERVATION_SEAL.json',dict(rows=238,ordered_ids=ids,partitions=V.PARTITIONS,
        selection_profile=V.SELECTION_PROFILE,protocol_sha256='a'*64,observations_sha256='a'*64,row_index_sha256='a'*64,
        canonical_validation=dict(rows=238,canonical_row_digests_verified=238,missing_duplicate_or_extra_rows=0),
        partition_seals={'CAL':'a'*64,'REST':'a'*64}))
    hashes[V.DECLARATION]='c'*64
    monkeypatch.setattr(V.C,'read',lambda path:deepcopy(values[Path(path)]))
    monkeypatch.setattr(V.C,'sha',lambda path:hashes[Path(path)])
    monkeypatch.setattr(V,'_verify',lambda *args:None)
    parent=dict(original_CAL_file_sha256=values[V.OUTLINES/'FINAL_PROTOCOL.json']['original_CAL_file_sha256'])
    return ids,entries,values,parent


def test_outline_CAL46_REST192_full238_closure(monkeypatch):
    ids,entries,values,parent=outline_metadata(monkeypatch)
    central=dict(ordered_ids=ids,source_sha256={},protected_user_sha256={})
    final,index=V._outline_metadata('a'*64,central,parent)
    assert index==entries and len(final['immutable_LCC_CAL_file_sha256'])==53
    values[V.OUTLINES/'REST_OBSERVATION_SEAL.json']['union_exact_match_rows']=191
    with pytest.raises(ValueError,match='partitions'):
        V._outline_metadata('a'*64,central,parent)


def test_outline_incomplete_or_changed_originalCAL_and_fullorder_reject(monkeypatch):
    ids,entries,values,parent=outline_metadata(monkeypatch)
    original=deepcopy(values)
    central=dict(ordered_ids=ids,source_sha256={},protected_user_sha256={})
    values[V.OUTLINES/'OUTLINE_OBSERVATION_SEAL.json']['rows']=237
    with pytest.raises(ValueError,match='all238'):
        V._outline_metadata('a'*64,central,parent)
    values.clear();values.update(deepcopy(original))
    values[V.OUTLINES/'FINAL_PROTOCOL.json']['original_CAL_file_sha256']['0']='f'*64
    with pytest.raises(ValueError,match='CAL53'):
        V._outline_metadata('a'*64,central,parent)
    values.clear();values.update(deepcopy(original))
    central['source_sha256']={'omitted.py':'f'*64}
    with pytest.raises(ValueError,match='inherit exact'):
        V._outline_metadata('a'*64,central,parent)


def test_central_full_inventory_and_every_current_source_are_required(monkeypatch):
    ids=[f'SYNTHETIC:{i}' for i in range(238)]
    value=dict(selection_profile=V.SELECTION_PROFILE,ordered_ids=ids,partitions=V.POPULATION_INVENTORY,
        arms=list(V.ARMS),source_sha256={'source.py':'b'*64},input_sha256={},
        protected_user_sha256={str(i):'c'*64 for i in range(6)})
    monkeypatch.setattr(V.C,'sha',lambda path:'a'*64)
    monkeypatch.setattr(V.C,'read',lambda path:deepcopy(value))
    monkeypatch.setattr(V,'_source_map',lambda:{'source.py':'b'*64})
    monkeypatch.setattr(V,'_verify',lambda *maps:None)
    monkeypatch.setattr(V,'_pins',lambda:None)
    assert V._central('a'*64)['partitions']==V.POPULATION_INVENTORY
    value['source_sha256']['source.py']='d'*64
    with pytest.raises(ValueError,match='every current'):
        V._central('a'*64)
    value['source_sha256']['source.py']='b'*64
    value['partitions']=V.PARTITIONS
    with pytest.raises(ValueError,match='Uniform'):
        V._central('a'*64)


def test_parent_closed_seals_and_direct_original_positive_E_bindings(monkeypatch):
    ids=[f'SYNTHETIC:{i}' for i in range(238)]
    records=[dict(id=identifier,partition='CAL' if i<46 else 'TEST' if i<95 else 'REMAINDER') for i,identifier in enumerate(ids)]
    protocol=dict(ordered_ids=ids,records=records,new_model_seal_sha256='a'*64)
    seal=dict(rows=238,ordered_ids=ids,arms=list(V.ARMS),arm_count=5,all_declared_arms_complete=True,
        before_fixed_reference_scoring=True,no_population_drop=True,
        **{key:'a'*64 for key in ('predictions_sha256','protocol_sha256','input_seal_sha256',
            'execution_protocol_sha256','row_index_sha256','separate_e_comparison_sha256')})
    central=dict(ordered_ids=ids,input_sha256={V._relative(V.PARENT/name):'a'*64 for name in
        ('PROTOCOL.json','INPUT_SEAL.json','INPUTS.jsonl.gz','E_COMPARISON.jsonl.gz','PREDICTION_SEAL.json')})
    monkeypatch.setattr(V.M,'_load_prepared',lambda folder:deepcopy(protocol))
    monkeypatch.setattr(V.C,'sha',lambda path:'a'*64)
    monkeypatch.setattr(V.C,'read',lambda path:deepcopy(seal))
    assert V._parent_metadata(central)==protocol
    del central['input_sha256'][V._relative(V.PARENT/'E_COMPARISON.jsonl.gz')]
    with pytest.raises(ValueError,match='positive/e'):
        V._parent_metadata(central)
    central['input_sha256'][V._relative(V.PARENT/'E_COMPARISON.jsonl.gz')]='a'*64
    seal['all_declared_arms_complete']=False
    with pytest.raises(ValueError,match='Completed original'):
        V._parent_metadata(central)


def toy_preparation(monkeypatch, tmp_path, *, broken=False):
    monkeypatch.setattr(V.C,'ROOT',tmp_path)
    parent=tmp_path/'parent';outlines=tmp_path/'outlines';planned=tmp_path/'plan'
    parent.mkdir();outlines.mkdir();planned.mkdir();(planned/'PREDECLARATION.json').write_text('{}')
    monkeypatch.setattr(V,'PARENT',parent);monkeypatch.setattr(V,'OUTLINES',outlines)
    value=invalid_packet();ids=[f'SYNTHETIC:{i}' for i in range(238)]
    packets=[];rows=[];original=[];entries=[]
    for i,identifier in enumerate(ids):
        p=deepcopy(value);part='CAL' if i<46 else 'TEST' if i<95 else 'REMAINDER'
        p['identifiers'].update(id=identifier,partition=part)
        p['clean_input']['identifiers'].update(id=identifier,partition=part)
        row=dict(id=identifier,partition=part,selection_profile=V.SELECTION_PROFILE,baseline_profile=V.LCC.BASELINE_PROFILE,
            positive_packet_sha256=V._digest(p),union_exact_match=True)
        packets.append(p);rows.append(row)
        original.append(dict(id=identifier,partition=part,positive_packet_sha256=V._digest(p)))
        entries.append(dict(id=identifier,partition=part,canonical_row_sha256=V._digest(row)))
    if broken:rows[0]['union_exact_match']=False
    V.C.save_rows(parent/'INPUTS.jsonl.gz',packets)
    V.C.save_rows(outlines/'OUTLINE_OBSERVATIONS.jsonl.gz',rows)
    (parent/'E_COMPARISON.jsonl.gz').write_bytes(b'opaque-toy-E-not-decoded-during-prepare')
    maps={key:{} for key in ('source_sha256','input_sha256','protected_user_sha256',
        'original_CAL_file_sha256','immutable_LCC_CAL_file_sha256')}
    peer=dict(path='peer.json',sha256='a'*64)
    declared=dict(**deepcopy(maps),central_plan_sha256='b'*64,outline_seal_sha256='c'*64,
        controller_peer_binding=peer,ordered_ids=ids,arms=list(V.ARMS))
    declared['input_sha256']={peer['path']:peer['sha256']}
    monkeypatch.setattr(V,'_load_plan',lambda folder:deepcopy(declared))
    monkeypatch.setattr(V,'_stage_inputs',lambda *args:dict(maps=deepcopy(maps),parent={'records':original},entries=entries,ordered_ids=ids))
    monkeypatch.setattr(V,'_verify',lambda *maps:None)
    monkeypatch.setattr(V.M,'solve_worker_packet',forbidden)
    return parent,planned,packets


def test_fake_preparation_streams238_and_copies_parent_E_bytes_without_decoding_or_PnP(monkeypatch,tmp_path):
    parent,planned,packets=toy_preparation(monkeypatch,tmp_path)
    output=tmp_path/'prepared'
    result=V.prepare(root_metadata_GO=True,plan_folder=planned,output=output)
    assert result['path']=='prepared/INPUT_SEAL.json'
    assert list(V.C.rows(output/'INPUTS.jsonl.gz'))==packets
    assert (output/'E_COMPARISON.jsonl.gz').read_bytes()==(parent/'E_COMPARISON.jsonl.gz').read_bytes()
    assert V.C.read(output/'INPUT_SEAL.json')['rows']==238
    assert V.C.read(output/'PROTOCOL.json')['e_comparison_parent_only'] is True


def test_fake_failed_preparation_is_preserved_and_cannot_silently_restart(monkeypatch,tmp_path):
    _,planned,_=toy_preparation(monkeypatch,tmp_path,broken=True)
    output=tmp_path/'failed'
    with pytest.raises(ValueError):
        V.prepare(root_metadata_GO=True,plan_folder=planned,output=output)
    failure=V.C.read(output/'PREPARATION_FAILURE_V1.json')
    assert failure['attempt_preserved'] and failure['prepared_rows']==0
    with pytest.raises(FileExistsError,match='Preserve'):
        V.prepare(root_metadata_GO=True,plan_folder=planned,output=output)


def test_actual_spawn_can_import_frozen_worker_and_retain_invalid_toy_row():
    value=invalid_packet()
    with ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn')) as pool:
        result=pool.submit(V.solve_worker_packet,value).result(timeout=30)
    assert result['row_retained'] and result['invalid_initial_coarse']
    assert len(result['arms'])==4 and all(not arm['pose_available'] for arm in result['arms'].values())


def test_exact_bounded_three_worker_helper_on_all238_toy_no_pose_packets():
    value=invalid_packet()
    ids=[f'SYNTHETIC-SPAWN:{i}' for i in range(238)]
    def rows():
        for index,identifier in enumerate(ids):
            row=deepcopy(value)
            row['identifiers']['id']=row['clean_input']['identifiers']['id']=identifier
            row['identifiers']['partition']=row['clean_input']['identifiers']['partition']='CAL' if index<46 else 'TEST' if index<95 else 'REMAINDER'
            yield row
    # Temporary toy-only files live under ROOT so the frozen repo-relative writer is exercised.
    with tempfile.TemporaryDirectory(prefix='.v33-toy-spawn-',dir=V.C.ROOT) as folder:
        output=Path(folder);(output/'prediction_spool').mkdir()
        entries=V._bounded_predict(rows(),ids,output,3)
        assert set(entries)==set(ids) and len(entries)==238
        progress=V.C.read(output/'LIVE_PROGRESS.json')
        assert progress['completed']==238 and progress['maximum_pending']==progress['workers']==3
        saved=list(V.C.rows(V.C.ROOT/entries[ids[0]]['path']))
        assert len(saved)==1 and saved[0]['arms'][V.ARMS[1]]['pose_available'] is False
