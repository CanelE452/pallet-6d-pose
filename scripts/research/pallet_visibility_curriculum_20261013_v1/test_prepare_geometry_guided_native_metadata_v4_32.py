"""Source/byte-link and positive extraction toy fixtures; actual32 IO=0."""
from copy import deepcopy
from pathlib import Path
import gc
import weakref

import numpy as np
import pytest

from . import prepare_geometry_guided_native_metadata_v4_32 as P
from .test_prepare_geometry_guided_native_evidence_v4_32 import fixture as factory_fixture, binding


class Poison:
    def __getitem__(self, key): raise AssertionError('Forbidden historical or unknown payload read')
    def __iter__(self): raise AssertionError('Forbidden payload iterated')
    def __array__(self, *args, **kwargs): raise AssertionError('Forbidden payload converted')
    def __deepcopy__(self, memo): raise AssertionError('Forbidden payload copied')
    def __eq__(self, other): raise AssertionError('Forbidden payload compared')


def rows(partial=True):
    _, inputs, metadata, _ = factory_fixture(partial=partial)
    family = deepcopy(inputs['family'])
    task = dict(id='SYNTHETIC:CAL', partition='CAL', session='SYNTHETIC',
        states=inputs['states'], initial_self_hidden=inputs['initial_self_hidden'], image_sha256='a'*64,
        spec=dict(K=inputs['K'], states=inputs['states'], visible=inputs['visible_measurements'],
            domains=inputs['domains'], image_hw=inputs['image_hw'], config=inputs['config']),
        boundary_merged=inputs['boundary_merged'], u_regions=inputs['u_regions'],
        families=dict(ORIGINAL_WD=family, SWAPPED_WD=Poison()),
        complete_BI_evidence=dict(initial_projection_xy=deepcopy(metadata['initial_geometry']['initial_projection_xy'])),
        baseline=Poison(), proposals=Poison(), initializations=Poison(), stage1_family_seed_bank=Poison(), extra_guides=Poison())
    task['families']['ORIGINAL_WD']['initializations'] = Poison()
    task['families']['ORIGINAL_WD']['coarse_pose']['old_solution'] = Poison()
    row = dict(id=task['id'], partition='CAL', session=task['session'], states=inputs['states'],
        visible_measurements=deepcopy(inputs['visible_measurements']), domains=deepcopy(inputs['domains']),
        u_regions=deepcopy(inputs['u_regions']), image_sha256='a'*64,
        initial_geometry=metadata['initial_geometry'], observation=metadata['observation'],
        selection_profile=P.G.SELECTION_PROFILE)
    return task, row


def closure():
    ids = [f'CAL:{i:02d}' for i in range(46)]
    hashes = {name: f'{i:064x}' for i, name in enumerate(P.CAL_FILES)}
    operator = str(Path(P.OP.__file__).resolve().relative_to(P.C.ROOT))
    protocol = dict(schema='V32_INITIAL_ONLY_CACHED_CAL46_THEN_REST192_PROTOCOL',
        ids=ids+[f'PRIMARY:{i:03d}' for i in range(192)], CAL_ids=ids, central_plan_sha256='a'*64,
        source_sha256={operator:'b'*64}, selection_profile=P.G.SELECTION_PROFILE,
        raw_observation_generation=P.G.RAW_OBSERVATION_GENERATION)
    index = [dict(id=ident, partition='CAL') for ident in ids]
    summary = dict(rows=46, records=[dict(id=ident) for ident in ids], selection_profile=P.G.SELECTION_PROFILE)
    seal = dict(rows=46, ordered_ids=ids, source_sha256=deepcopy(protocol['source_sha256']),
        selection_profile=P.G.SELECTION_PROFILE, raw_observation_generation=P.G.RAW_OBSERVATION_GENERATION,
        canonical_validation=dict(rows=46, canonical_row_digests_verified=46, missing_duplicate_or_extra_rows=0))
    fields = dict(protocol_sha256='PROTOCOL.json', preparation_seal_sha256='PREPARATION_SEAL.json',
        execution_protocol_sha256='CAL_EXECUTION_PROTOCOL.json', row_index_sha256='CAL_ROW_INDEX.json',
        observations_sha256='CAL_OBSERVATIONS.jsonl.gz', summary_sha256='CAL_SUMMARY.json')
    seal.update({key:hashes[name] for key,name in fields.items()})
    execution = dict(central_plan_sha256='a'*64, source_sha256='b'*64)
    return protocol, seal, index, summary, execution, hashes, 'a'*64, 'b'*64


def test_GO_and_explicit_central_before_all_IO(monkeypatch):
    def forbidden(*args): raise AssertionError('IO before GO/hash')
    monkeypatch.setattr(P.C, 'read', forbidden); monkeypatch.setattr(P.C, 'sha', forbidden)
    with pytest.raises(PermissionError): P.main()
    for value in (None, '', 'A'*64, 'a'*63):
        with pytest.raises(ValueError, match='Explicit exact central32'):
            P.main(root_metadata_GO=True, expected_central_plan_sha256=value)


def test_positive_fields_only_historical_banks_never_accessed_and_genuine_factory_connected(monkeypatch):
    task, row = rows()
    def forbidden(*args, **kwargs): raise AssertionError('No PnP/optimizer/old evaluator in CAL metadata')
    monkeypatch.setattr(P.CLEAN, 'prepare_clean_fresh_BI_task', forbidden)
    monkeypatch.setattr(P.M.FRESH, 'generate_fresh_observed_BI_pnp_proposals', forbidden)
    inputs, metadata = P.positive_inputs(task, row, initial_source_binding=binding())
    assert set(inputs['family']) == P.M.FAMILY_KEYS
    assert set(inputs['family']['coarse_pose']) == P.M.COARSE_KEYS
    assert not {'task','baseline','proposals','initializations','initial_solutions','pose_guides'} & set(inputs)
    prepared, audit = P.M.prepare_one(**inputs, **metadata)
    assert prepared['selection_profile'] == P.G.SELECTION_PROFILE
    assert set(prepared['profiles']) == set(P.G.SOURCE_POLICIES)
    assert audit['authenticated_groups'] == 1
    for profile in prepared['profiles'].values():
        for group in profile['groups']:
            component = group['components'][0]
            assert component['normal_variance_px2'] == component['normal_floor_px']**2
            assert component['angular_variance_rad2'] == component['angular_sine_floor']**2


def test_state_S_and_U_payloads_skipped_before_comparison_and_original_S_projection_unread():
    task, row = rows(partial=False)
    hidden = next(k for k,s in enumerate(task['states']) if s=='S')
    class Record(dict):
        def __getitem__(self, key):
            if key == 'native_corner': return super().__getitem__(key)
            raise AssertionError('S/U q/source/covariance queried')
        def __deepcopy__(self,memo): raise AssertionError('S/U q payload copied')
    for records in (task['spec']['visible'], row['visible_measurements']): records.append(Record(native_corner=hidden))
    task['complete_BI_evidence']['initial_projection_xy'][hidden] = Poison()
    row['initial_geometry']['initial_projection_xy'][hidden] = Poison()
    inputs, metadata = P.positive_inputs(task,row,initial_source_binding=binding())
    prepared,_ = P.M.prepare_one(**inputs,**metadata)
    assert prepared['initial_geometry']['initial_projection_xy'][hidden] is None
    # An unknown observed slot is not a model ray or a BI archive.
    task,row = rows(partial=False)
    unknown = next(k for k,s in enumerate(task['states']) if s=='B')
    task['states'][unknown] = 'U'; row['states'] = task['states']; task['spec']['states'] = task['states']
    task['spec']['visible'] = [m for m in task['spec']['visible'] if m['native_corner'] != unknown] + [Record(native_corner=unknown)]
    row['visible_measurements'] = [m for m in row['visible_measurements'] if m['native_corner'] != unknown] + [Record(native_corner=unknown)]
    inputs,_ = P.positive_inputs(task,row,initial_source_binding=binding())
    assert unknown not in [m['native_corner'] for m in inputs['visible_measurements']]


@pytest.mark.parametrize('change',['q','source','covariance','weight','image','U','domain','R','center','projection','profile'])
def test_exact_current_observation_known_physics_and_archive_join_failclosed(change):
    task,row=rows()
    if change=='q':row['visible_measurements'][0]['xy'][0]+=1.
    elif change=='source':row['visible_measurements'][0]['source']='changed'
    elif change=='covariance':row['visible_measurements'][0]['fit_covariance_px2'][0][0]+=1.
    elif change=='weight':row['visible_measurements'][0]['fit_reliability_weight']+=.1
    elif change=='image':row['image_sha256']='b'*64
    elif change=='U':row['u_regions']=[dict(native_corner=0)]
    elif change=='domain':row['domains']=[dict(native_corner=0)]
    elif change=='R':row['initial_geometry']['R_cf'][0][0]+=.01
    elif change=='center':row['initial_geometry']['centroid'][2]+=.01
    elif change=='projection':row['initial_geometry']['initial_projection_xy'][0][0]+=1.
    else:row['selection_profile']=P.G.PREVIOUS.SELECTION_PROFILE
    with pytest.raises(ValueError):P.positive_inputs(task,row,initial_source_binding=binding())


def test_primary_or_false_source_proof_gate_before_any_coordinate_access():
    with pytest.raises(ValueError,match='CAL-only'):
        P.positive_inputs(dict(partition='TEST'),dict(partition='TEST'),initial_source_binding=Poison())
    task,row=rows();task['spec']=Poison();proof=binding();proof['verified_by_caller']=False
    with pytest.raises(ValueError,match='closure'):P.positive_inputs(task,row,initial_source_binding=proof)


def test_full_CAL_closure_source_exec_inventory_and_all_link_hashes():
    args=closure();assert P.validate_CAL_closure(*args)==args[0]['CAL_ids']


@pytest.mark.parametrize('change',['source','execsource','preparation','execution','summary','index','raw','profile','duplicate','missing','nonCAL','canonical','central','schema','primarycount'])
def test_CAL_seal_inventory_or_immutable_link_failure(change):
    protocol,seal,index,summary,execution,hashes,expected,operator=closure()
    if change=='source':seal['source_sha256']={}
    elif change=='execsource':execution['source_sha256']='c'*64
    elif change=='preparation':seal['preparation_seal_sha256']='changed'
    elif change=='execution':seal['execution_protocol_sha256']='changed'
    elif change=='summary':summary['records'][0]['id']='changed'
    elif change=='index':seal['row_index_sha256']='changed'
    elif change=='raw':seal['raw_observation_generation']='renamed'
    elif change=='profile':protocol['selection_profile']='31'
    elif change=='duplicate':index[0]['id']=index[1]['id']
    elif change=='missing':index.pop()
    elif change=='nonCAL':index[0]['partition']='TEST'
    elif change=='canonical':seal['canonical_validation']['canonical_row_digests_verified']=45
    elif change=='central':execution['central_plan_sha256']='c'*64
    elif change=='schema':protocol['schema']='31'
    else:protocol['ids'].pop()
    with pytest.raises(ValueError):P.validate_CAL_closure(protocol,seal,index,summary,execution,hashes,expected,operator)


def test_initial_source_proofs_actual_hash_source_union_verified_before_attestation(monkeypatch):
    source=dict(verdicts=dict(original_coarse_q_S_and_originalguide='NO_DOWNSTREAM_E_POSE_COPY_FOUND_IN_TRACED_SOURCE',
        statistical_independence_claimed=False),clean_cached_measurement_candidate_design=dict(allow_guide_kinds=list(P.G.ASSOCIATION.ALLOWED_GUIDE_KINDS)),source_sha256={'one.py':'1'*64})
    raw=dict(verdict='NO_CROSS_QUERY_NMS_OR_PREGRAPH_CACHED_GUIDE_QUOTA_FOUND',actual_rows_or_images_or_arrays_read=0,source_sha256={'two.py':'2'*64})
    hashes={str(P.M.__file__):P.FACTORY_SHA256,str(P.SOURCE_CLOSURE):P.G.INITIAL_SOURCE_CLOSURE_SHA256,str(P.RAW_QUERY_AUDIT):P.RAW_QUERY_AUDIT_SHA256}
    verified=[]
    monkeypatch.setattr(P.C,'sha',lambda path:hashes[str(path)])
    monkeypatch.setattr(P.C,'read',lambda path:source if path==P.SOURCE_CLOSURE else raw)
    monkeypatch.setattr(P.PREFIX,'_verify',lambda mapping:verified.append(deepcopy(mapping)))
    proof,mapping=P._initial_source_closure()
    assert verified==[mapping] and mapping=={'one.py':'1'*64,'two.py':'2'*64}
    assert proof['verified_by_caller'] and proof['sha256']==P.G.INITIAL_SOURCE_CLOSURE_SHA256
    hashes[str(P.M.__file__)]='bad'
    monkeypatch.setattr(P.C,'read',lambda *args:pytest.fail('Changed factory before proof IO'))
    with pytest.raises(ValueError,match='Exact factory'):P._initial_source_closure()


def stream_fixture():
    ids=[f'CAL:{i}' for i in range(46)]
    tasks=[dict(id=ident,partition='CAL')for ident in ids]
    rows=[dict(id=ident,partition='CAL',selection_profile=P.G.SELECTION_PROFILE,
        observation=dict(selected_chain_refs=[],selected_physical_edges=0,joint_graph=dict(selected_line_names=[])),
        summary=dict(selected_physical_edges=0,zero_selected_edges=True))for ident in ids]
    entries=[dict(id=r['id'],canonical_row_sha256=P.OP.D.R._canonical_sha(r))for r in rows]
    return ids,tasks,rows,entries


def test_stream46_zero_rows_positive_factory_kwargs_only_and_no_PnP_calls(monkeypatch):
    ids,tasks,rows,entries=stream_fixture();records=[];errors=[]
    def extract(task,row,**kwargs):return dict(identifier=task['id']),dict(partition='CAL')
    def prepare(**kwargs):
        assert set(kwargs)=={'identifier','partition'}
        return dict(id=kwargs['identifier'],profiles={p:dict(groups=[],tangent_records=[])for p in P.G.SOURCE_POLICIES}),dict(id=kwargs['identifier'],profiles={})
    monkeypatch.setattr(P,'positive_inputs',extract);monkeypatch.setattr(P.M,'prepare_one',prepare)
    output=list(P.prepared_CAL_rows(iter(tasks),iter(rows),iter(entries),ids,records,errors,initial_source_binding=binding()))
    assert [r['id']for r in output]==ids and len(records)==46 and not errors
    assert all(set(r['profiles'])==set(P.G.SOURCE_POLICIES)for r in output)


def test_stream_releases_raw_row_before_yield_and_before_loading_next(monkeypatch):
    ids,tasks,rows,entries=stream_fixture();refs=[]
    class Row(dict):pass
    def raw():
        for row in rows:
            value=Row(row);refs.append(weakref.ref(value));yield value
            del value
    def extract(task,row,**kwargs):return dict(identifier=task['id']),dict(observation=row['observation'])
    monkeypatch.setattr(P,'positive_inputs',extract)
    monkeypatch.setattr(P.M,'prepare_one',lambda identifier,observation:(dict(id=identifier),dict(id=identifier)))
    prepared=P.prepared_CAL_rows(iter(tasks),raw(),entries,ids,[],[],initial_source_binding=binding())
    for i,result in enumerate(prepared):
        gc.collect()
        if i:assert refs[i-1]()is None
        assert result['id']==ids[i]


def test_stream_canonical_mismatch_or_counts_fail_before_positive_extraction(monkeypatch):
    monkeypatch.setattr(P,'positive_inputs',lambda *a,**kw:pytest.fail('Bad inventory reached extraction'))
    ids,tasks,rows,entries=stream_fixture();rows[0]['summary']['zero_selected_edges']=False
    with pytest.raises(ValueError,match='canonical'):
        list(P.prepared_CAL_rows(tasks,rows,entries,ids,[],[],initial_source_binding=binding()))
    ids,tasks,rows,entries=stream_fixture()
    with pytest.raises(ValueError,match='counts'):
        list(P.prepared_CAL_rows(tasks,[],entries,ids,[],[],initial_source_binding=binding()))


def test_failed_metadata_keeps_every_CAL_id_and_union_conflicts_fail(monkeypatch):
    ids,tasks,rows,entries=stream_fixture();errors=[]
    def fail(*args,**kwargs):raise ValueError('toy metadata failure')
    monkeypatch.setattr(P,'positive_inputs',fail)
    assert list(P.prepared_CAL_rows(tasks,rows,entries,ids,[],errors,initial_source_binding=binding()))==[]
    assert [e['id']for e in errors]==ids
    assert P._union({'a':'x'},{'a':'x','b':'y'})=={'a':'x','b':'y'}
    with pytest.raises(ValueError,match='conflict'):P._union({'a':'x'},{'a':'y'})
