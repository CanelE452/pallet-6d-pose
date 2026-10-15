"""Toy positive packets only; no actual first-CAL/model/source-plan execution."""
from copy import deepcopy
import inspect
import sys

import numpy as np
import pytest

from . import smoke_clean_fresh_BI_runtime_v4_32 as P
from .test_clean_fresh_BI_input_projection_v4_32 import project, attached
from .test_prepare_geometry_guided_native_metadata_v4_32 import rows, Poison
from .test_prepare_geometry_guided_native_evidence_v4_32 import binding


def packet():
    _,row,projected=project()
    return P.M.PROJECT.assemble_worker_input(projected,**attached(projected,row))


@pytest.fixture(scope='module')
def toy_result():
    value=packet();before=deepcopy(value)
    modules=(P.M,P.M.CLEAN,P.M.FACTORY,P.G)
    namespaces=[{k:id(v)for k,v in vars(module).items()}for module in modules]
    result=P.smoke_worker_packet(value)
    assert value==before
    assert namespaces==[{k:id(v)for k,v in vars(module).items()}for module in modules]
    return value,result


def test_root_GO_and_all_explicit_hashes_precede_IO(monkeypatch):
    def no(*args,**kwargs):raise AssertionError('IO before stage GO/explicit SHA')
    monkeypatch.setattr(P.C,'read',no);monkeypatch.setattr(P.C,'sha',no)
    with pytest.raises(PermissionError):P.declare()
    with pytest.raises(PermissionError):P.runtime()
    with pytest.raises(ValueError,match='central'):
        P.runtime(root_actual_CAL_smoke_GO=True)
    names=('central','controller','CAL_seal','CAL_owner','model_seal')
    for missing in names:
        args={'expected_'+name+('_plan_sha256' if name=='central' else '_sha256'):'a'*64 for name in names}
        args['expected_'+missing+('_plan_sha256' if missing=='central' else '_sha256')]=None
        with pytest.raises(ValueError,match='Explicit exact'):
            P.declare(root_metadata_GO=True,**args)


def test_declaration_binds_CAL53_and_first_id_without_raw_or_numeric_model_decode(monkeypatch,tmp_path):
    planpath=tmp_path/'PREDECLARATION.json';monkeypatch.setattr(P,'PLAN',planpath)
    expected=dict(central='a'*64,controller='b'*64,CALseal='c'*64,CALowner='d'*64,modelseal='e'*64)
    ids=[f'SYNTHETIC_CAL:{i}'for i in range(46)]
    maps=dict(source_sha256={},input_sha256={},protected_user_sha256={})
    immutable={f'SYNTHETIC/opaque_CAL_file_{i}':'f'*64 for i in range(53)}
    index=[dict(id=ident,canonical_row_sha256='f'*64)for ident in ids]
    monkeypatch.setattr(P.META,'_observation_closure',lambda value:(deepcopy(maps),deepcopy(maps),ids,index,immutable))
    monkeypatch.setattr(P.META,'_initial_source_closure',lambda:(binding(),{}))
    central_rel=str(P.M.CAL.CENTRAL_PLAN.relative_to(P.C.ROOT))
    modelseal=dict(observation_generation=P.G.OBSERVATION_FAMILY,noise_population=P.G.POPULATION,
        selection_profile=P.G.SELECTION_PROFILE,raw_observation_generation=P.G.RAW_OBSERVATION_GENERATION,
        source_sha256={},input_sha256={central_rel:expected['central']},assets={})
    readable={P.M.CAL.OUTPUT/'MODEL_SEAL.json':modelseal,
        P.M.TASKS/'INPUT_SEAL.json':dict(ordered_ids=ids,input_sha256='1'*64,protocol_sha256='2'*64),
        P.M.TASKS/'PROTOCOL.json':dict(ids=ids,protected_user_sha256={})}
    reads=[];writes={};verified=[]
    def read(path):
        path=P.Path(path);reads.append(path)
        if path not in readable:raise AssertionError('Raw row or numeric model decoded by declaration')
        return deepcopy(readable[path])
    hashes={P.Path(P.M.__file__):expected['controller'],P.Path(P.M.PROJECT.__file__):P.PROJECTOR_SHA256,
        P.OBS.OUTPUT/'CAL_OBSERVATION_SEAL.json':expected['CALseal'],
        P.OBS.OUTPUT/'CAL_OWNER_COMPLETION_RECEIPT_V32_V1.json':expected['CALowner'],
        P.M.CAL.OUTPUT/'MODEL_SEAL.json':expected['modelseal'],P.M.TASKS/'INPUTS.jsonl.gz':'1'*64,
        P.M.TASKS/'PROTOCOL.json':'2'*64,**{P.M.B/rel:value for rel,value in P.PROOF_PINS.items()}}
    monkeypatch.setattr(P.C,'read',read)
    monkeypatch.setattr(P.C,'sha',lambda path:hashes.get(P.Path(path),'f'*64))
    monkeypatch.setattr(P.C,'rows',lambda *a:pytest.fail('Declaration decoded raw row'))
    monkeypatch.setattr(P.C,'write',lambda path,value:writes.update({P.Path(path):deepcopy(value)}))
    monkeypatch.setattr(P.M,'_verify',lambda mapping:verified.append(deepcopy(mapping)))
    P.declare(root_metadata_GO=True,expected_central_plan_sha256=expected['central'],
        expected_controller_sha256=expected['controller'],expected_CAL_seal_sha256=expected['CALseal'],
        expected_CAL_owner_sha256=expected['CALowner'],expected_model_seal_sha256=expected['modelseal'])
    plan=writes[planpath]
    assert len(reads)==3 and plan['first_CAL_id']==ids[0]
    assert plan['original_CAL_file_sha256']==immutable and len(verified)==3
    assert plan['expected_sha256']==expected and plan['metadata_declaration_models_or_rows_decoded']==0
    assert plan['no_full_OBSERVATIONS238_needed']


def test_worker_numeric_interface_has_no_task_rawobs_e_or_model_input():
    assert set(inspect.signature(P.smoke_worker_packet).parameters)=={'packet'}
    value=packet()
    assert set(value)==P.M.PROJECT.WORKER_KEYS
    assert not {'baseline','proposals','initializations','observation','pose_guides'} & set(value['clean_input'])
    with pytest.raises(ValueError,match='seven-key'):
        P.smoke_worker_packet({**value,'baseline':Poison()})
    value['identifiers']['partition']='TEST';value['clean_input']=Poison()
    with pytest.raises(ValueError,match='CAL-only'):
        P.smoke_worker_packet(value)


def test_shared_controller_builder_called_once_not_duplicated(monkeypatch):
    original=P.M.build_clean_native_cores;seen=[]
    def shared(value):
        seen.append(id(value));return original(value)
    monkeypatch.setattr(P.M,'build_clean_native_cores',shared)
    value=packet();result=P.smoke_worker_packet(value)
    assert seen==[id(value)] and result['ready']


def test_fresh_seed_bank_all_rejected_and_accepted_evaluations_preserved(toy_result):
    value,result=toy_result
    assert result['ready'] and result['no_numeric_equality_to_e_filter']
    assert result['source_policy_count']==result['family_count']==2
    assert set(result['records'])==set(P.G.SOURCE_POLICIES)
    for policy,families in result['records'].items():
        assert set(families)==set(P.M.FAMILIES)
        for name,record in families.items():
            bank=record['fresh_seed_bank'];evaluations=record['all_fresh_seed_evaluations']
            assert [e['name']for e in evaluations]==[p['name']for p in bank['proposals']]
            assert len(evaluations)==len(bank['proposals'])>0
            assert all('accepted'in e['evaluation']and 'objective'in e['evaluation']for e in evaluations)
            assert all(e['origin']in(P.M.FACTORY.FRESH.COARSE_SOURCE,P.M.FACTORY.FRESH.FRESH_SOURCE)for e in evaluations)
            assert record['current_cost_plan']['source_policy']==policy
            assert record['current_cost_plan']['selection_profile']==P.G.SELECTION_PROFILE
            assert record['current_cost_plan_sha256']==P.G._digest(record['current_cost_plan'])
            assert record['family']['name']==name and record['pose_vector_scale']>0


def test_start_origins_binding_and_native_ledger_gradient_guard_snapshots(toy_result):
    value,result=toy_result
    for policy,families in result['records'].items():
        for name,record in families.items():
            start=record['start_packet'];bank=record['fresh_seed_bank']
            assert start['fresh_PnP_input_sha256']==bank['input_binding_sha256']
            assert len(start['start_packet_sha256'])==64 and start['native_ranked_start_budget']==6
            assert start['cached_or_baseline_pose_inputs']==0 and not start['legacy_reference_start_problem_used']
            assert len(record['samples'])==len(start['starts'])+1
            assert record['samples'][0]['name']=='INDEPENDENT_COARSE_STATIC'
            assert [s['name']for s in record['samples'][1:]]==[s['name']for s in start['starts']]
            for sample in record['samples']:
                assert np.isclose(sample['objective'],sample['evaluation']['objective'])
                assert np.isclose(sample['objective'],sample['native_cost_sum'])
                assert len(sample['objective_gradient'])==len(sample['pose_vector'])==6
                assert np.isfinite(sample['objective_gradient']).all()
                assert sample['native_t']==sample['t']
                for branch in sample['constraint_branches']:
                    assert np.asarray(branch['jacobian']).shape==(len(branch['slack']),6)
            assert record['native_factory_uses_full_original_BI_guard_archive']


def test_no_continuous_old_wrapper_or_optimizer_calls_and_profile_restored(toy_result):
    _,result=toy_result
    counts=result['observed_call_counts']
    assert counts['fresh_BI_task']==1 and counts['fresh_PnP_bank']==2 and counts['SQPNP_Generic']>0
    assert all(counts[k]==0 for k in('continuous_refiner','full_solver','scipy_optimizer',
                                    'legacy_evaluator_run','legacy_refiner_wrapper'))
    previous=sys.getprofile()
    assert previous is None


def test_profile_restored_even_if_same_worker_build_raises(monkeypatch):
    def fail(value):raise ValueError('toy failure')
    monkeypatch.setattr(P,'_worker',fail)
    def previous(frame,event,arg):pass
    initial=sys.getprofile()
    try:
        sys.setprofile(previous)
        with pytest.raises(ValueError,match='toy failure'):P.smoke_worker_packet({})
        assert sys.getprofile()is previous
    finally:sys.setprofile(initial)


def test_invalid_coarse_retained_without_guards_models_or_fresh_PnP(monkeypatch):
    task,row=rows();del task['families']['ORIGINAL_WD']['coarse_pose']['t']
    task['spec']['visible']=task['spec']['domains']=Poison();row['initial_geometry']=Poison()
    projected=P.M.PROJECT.project_clean_input(task,row,initial_source_binding=binding())
    value=P.M.PROJECT.assemble_worker_input(projected,typed_bundles=Poison(),regularizers=Poison(),
        model_availability=Poison(),tangent_parameters=Poison(),line_noise_provenance=Poison())
    monkeypatch.setattr(P.M,'build_native_cores',lambda *a:pytest.fail('invalid coarse read cores'))
    result=P.smoke_worker_packet(value)
    assert not result['ready']and result['row_retained']and result['records']=={}
    assert result['observed_call_counts']['fresh_PnP_bank']==0


def test_first_iterator_closes_without_decoding_second_row():
    events=[]
    def rows():
        try:
            yield {'id':'FIRST_CAL'}
            raise AssertionError('second row decoded')
        finally:events.append('closed')
    assert P._first(rows())=={'id':'FIRST_CAL'}and events==['closed']


def test_prepared_O_array_packet_hash_matches_saved_finite_JSON_without_mutation():
    value=dict(o_regions=[dict(center_xy=np.array([2.,3.]),basis=np.eye(2))])
    original=value['o_regions'][0]['center_xy']
    assert P._plain_digest(value)==P.G._digest(dict(o_regions=[dict(center_xy=[2.,3.],basis=[[1.,0.],[0.,1.]])]))
    assert value['o_regions'][0]['center_xy'] is original and np.array_equal(original,[2.,3.])


def test_map_conflicts_failclosed_and_no_parent_source_namespace_mutation(toy_result):
    assert P._map({'a':'1'},{'a':'1','b':'2'})=={'a':'1','b':'2'}
    with pytest.raises(ValueError,match='conflict'):P._map({'a':'1'},{'a':'2'})
    # The smoke uses imports/call profiling only; no private FunctionType
    # clone, setattr or monkeypatch of frozen numeric modules exists here.
    source=inspect.getsource(P)
    assert 'FunctionType('not in source and 'setattr('not in source
