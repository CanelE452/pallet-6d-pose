"""Positive worker/IO synthetic tests; only one small model-export schema check.

No actual task/observation/prediction/image/reference is decoded and no
operational stage or CAL fit is executed. Fresh PnP below uses toy geometry.
"""
from copy import deepcopy
from concurrent.futures import Future
import gzip
import inspect
import io
from pathlib import Path

import numpy as np
import pytest

from . import evaluate_clean_fresh_BI_poses_v4_32 as V
from .test_clean_fresh_BI_input_projection_v4_32 import project, attached
from .test_prepare_geometry_guided_native_metadata_v4_32 import Poison


def forbidden(*args, **kwargs):
    raise AssertionError('Forbidden old pose/IO/optimizer path reached')


def packet(*, unavailable=False):
    task,row,projected=project()
    args=attached(projected,row,unavailable=unavailable)
    args['regularizers']=dict(o_regions=[],o_parameters=dict(epsilon=.2,huber_scale=3.),
        original_S_prepared=None,outline_observation=dict(supported=False),outline_configuration={})
    value=V.PROJECT.assemble_worker_input(projected,**args)
    value['physics']['config'].update(max_joint_iterations=2,optimizer_ftol=1e-7)
    return task,row,projected,value


def test_root_GO_precedes_any_stage_IO(monkeypatch):
    monkeypatch.setattr(V.C,'read',forbidden);monkeypatch.setattr(V.C,'sha',forbidden)
    for function in (V.plan,V.prepare,V.predict):
        with pytest.raises(PermissionError):function()
    with pytest.raises(ValueError,match='before IO'):
        V.plan(root_metadata_GO=True)


def test_unknown_legacy_worker_keys_reject_before_fresh_PnP(monkeypatch):
    monkeypatch.setattr(V.CLEAN,'prepare_clean_fresh_BI_task',forbidden)
    for key in ('baseline','proposals','initializations','pose_guides','task','e'):
        with pytest.raises(ValueError,match='seven-key'):
            V.solve_worker_packet({key:Poison()})
    assert not {'task','baseline','proposals','pose_guides'} & set(inspect.signature(V.solve_worker_packet).parameters)


def test_positive_factory_receives_original_guard_archive_and_genuine_two_policy_two_WD(monkeypatch):
    _,_,_,value=packet()
    original=deepcopy(value)
    baseline=V.FACTORY.native_baseline;seen=[]
    def captured(**kwargs):
        seen.append(deepcopy(kwargs['visible_measurements']))
        return baseline(**kwargs)
    monkeypatch.setattr(V.FACTORY,'native_baseline',captured)
    clean,cores=V.build_clean_native_cores(value)
    assert clean['ready'] and set(cores)==set(V.G.SOURCE_POLICIES)
    assert len(seen)==4 and all(records==value['clean_input']['visible_measurements'] for records in seen)
    assert any('protection_px' in record for record in seen[0])
    assert all('protection_px' not in record for record in clean['spec']['visible'])
    for policy,families in cores.items():
        assert set(families)==set(V.FAMILIES)
        for core in families.values():
            assert isinstance(core,V.G.GeometryGuidedEdgeProblem)
            assert core.plan['source_policy']==policy
            assert core.plan['observation_family']==V.G.OBSERVATION_FAMILY
            assert not hasattr(core,'reference_start_problem')
    assert value==original


def test_invalid_coarse_no_pose_reads_no_other_payloads(monkeypatch):
    _,_,_,value=packet()
    value['clean_input']['initial_coarse_pose']={'source':V.FACTORY.FRESH.COARSE_SOURCE,
        'independent_of_cached_pose_bank':True}
    value['source_contract']['ready']=False
    value['physics']=value['typed_bundles']=value['regularizers']=Poison()
    monkeypatch.setattr(V,'build_native_cores',forbidden)
    result=V.solve_worker_packet(value)
    assert result['invalid_initial_coarse'] and result['row_retained']
    assert set(result['arms'])==set(V.SOLVER.ARMS)
    assert all(not entry['accepted'] and not entry['pose_available'] and entry['actual_pose'] is None
        for entry in result['arms'].values())


def test_unavailable_new_noise_has_typed_empty32_gamma0_and_original_guard_values():
    _,_,_,value=packet(unavailable=True)
    clean,cores=V.build_clean_native_cores(value)
    for policy,families in cores.items():
        for name,core in families.items():
            assert core.plan['new_owner_ids']==[] and core.plan['active_groups']==[]
            family=clean['families'][name];pose=family['coarse_pose']
            p=V.G.V._pose_vector(np.asarray(pose['R']),np.asarray(pose['t']),core.scale)
            actual=core.objective(p)
            expected=core.base.objective(p)
            assert actual[0]==expected[0] and np.array_equal(actual[1],expected[1])
            assert value['source_contract']['tangent_parameters'][policy] is None


def test_shared_worker_solver_path_and_positive_readout_without_old_wrapper(monkeypatch):
    _,_,_,value=packet()
    def solve(clean,cores,**kwargs):
        arms={}
        for policy,names in V.POLICY_ARMS.items():
            family=clean['families']['ORIGINAL_WD'];coarse=family['coarse_pose']
            for name in names:
                arms[name]=dict(accepted=False,pose_available=True,R=coarse['R'],t=coarse['t'],
                    WD_family='ORIGINAL_WD',source_policy=policy,
                    solution=dict(accepted=False),new_geometry_edge_cost_plans={})
        return dict(arms=arms,row_retained=True,baseline_inputs=0)
    monkeypatch.setattr(V.SOLVER,'solve_clean_fresh_BI_poses',solve)
    monkeypatch.setattr(V.SOLVER.SCALAR,'_run',forbidden)
    monkeypatch.setattr(V.SOLVER.SCALAR,'_REFINER',forbidden,raising=False)
    result=V.solve_worker_packet(value)
    assert result['parent_only_e_comparison_not_supplied_to_worker']
    for entry in result['arms'].values():
        readout=entry['readout']
        assert not readout['used_in_accepted_pose'] and not readout['OT_solved']
        for k,state in enumerate(value['clean_input']['states']):
            if state in ('U','O','T'):assert readout['reported_points'][k] is None
            elif state=='S':assert readout['point_sources'][k].startswith('MODEL_PNP_PROJECTION_ONLY')
        assert entry['actual_pose']['native_t']==entry['t']


def test_genuine_four_arm_worker_composition_uses_only_fresh_contenders(monkeypatch):
    _,_,_,value=packet()
    before=deepcopy(value)
    monkeypatch.setattr(V.SOLVER.SCALAR,'_run',forbidden)
    monkeypatch.setattr(V.SOLVER.SCALAR,'_REFINER',forbidden,raising=False)
    result=V.solve_worker_packet(value)
    assert set(result['arms'])==set(V.SOLVER.ARMS)
    assert result['row_retained'] and result['baseline_inputs']==0
    for entry in result['arms'].values():
        assert entry['pose_available'] and entry['actual_pose'] is not None
        assert entry['fallback_audit']['cached_pose_inputs']==0
        assert not entry['readout']['OT_unique_solution_claimed']
        assert not entry['candidate_selection']['e_numeric_pose_prior_or_fallback']
    assert value==before


def test_actual_pose_noncentered_geometry_and_registered_Q_are_explicit():
    xyz=np.asarray([[x,y,z] for x in (-1.,1.) for y in (-.2,.2) for z in (-.7,.7)])+[.4,-.2,.3]
    Q=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
    family=dict(name='SYNTHETIC',native_xyz=xyz.tolist(),physical_registration_Q=Q.tolist())
    pose=V._actual_pose(family,np.eye(3),[1.,2.,5.])
    assert pose['native_t']==[1.,2.,5.]
    assert np.allclose(pose['centroid'],[1.4,1.8,5.3])
    assert np.array_equal(pose['R_physical'],Q)


def test_parent_comparison_exact_and_center_convention_never_enters_worker():
    task,_,_,_=packet();task['baseline']={'actual_pose':{'centroid':[.1,.2,3.]},'accepted':False}
    before=deepcopy(task['baseline'])
    comparison=V._comparison_row(task)
    assert comparison['prediction']==before and comparison['prediction'] is not task['baseline']
    assert comparison['legacy_e_pose_semantics']['numeric_worker_exposure'] is False
    task['families']['ORIGINAL_WD']['native_xyz'][0][0]+=.1
    with pytest.raises(ValueError,match='centered current body'):V._comparison_row(task)


def test_actual_saved_small_model_export_is_four_files_plus_sixteen_nested_entries():
    # Actual model metadata only: never CAL errors, labels, task/OBS or RGB.
    folder=V.CAL.OUTPUT;seal=V.C.read(folder/'MODEL_SEAL.json')
    report=V.C.read(folder/'CAL_LOGO_REPORT.json')
    full={policy:dict(line=V.C.read(folder/policy/'MODEL.json'),
        tangent=V.C.read(folder/policy/'TANGENT_MODEL.json')) for policy in V.G.SOURCE_POLICIES}
    assert len(seal['assets'])==7
    assert V._validate_model_inventory(seal,full,report)==dict(full_model_files=4,nested_LOGO_model_entries=16,total_models=20)
    wrong=deepcopy(report);wrong[next(iter(wrong))]['LOGO_folds'].pop()
    with pytest.raises(ValueError,match='Four separately'):V._validate_model_inventory(seal,full,wrong)
    wrong=deepcopy(full);model=wrong[next(iter(wrong))]['line'];model['model_digest_sha256']='0'*64
    with pytest.raises(ValueError,match='model digest'):V._validate_model_inventory(seal,wrong,report)


def test_all238_positive_preparation_retains_zero_rows_and_parent_e_only(monkeypatch):
    ids=[f'SYNTHETIC:{i}' for i in range(238)];tasks=[];rows=[];entries=[]
    for i,identifier in enumerate(ids):
        partition='CAL' if i<46 else 'TEST' if i<95 else 'REMAINDER'
        row=dict(id=identifier,partition=partition,selection_profile=V.G.SELECTION_PROFILE,
            observation=dict(selected_chain_refs=[],selected_physical_edges=0,joint_graph=dict(selected_line_names=[])),
            summary=dict(selected_physical_edges=0,zero_selected_edges=True))
        rows.append(row);entries.append(dict(id=identifier,partition=partition,canonical_row_sha256=V.OBS.D.R._canonical_sha(row)))
        tasks.append(dict(id=identifier,partition=partition,families={'ORIGINAL_WD':{'native_xyz':np.zeros((8,3)).tolist()}},
            baseline=dict(accepted=i%3==0,actual_pose=None),proposals=Poison(),initializations=Poison()))
    def prepare(task,row,binding,models):
        return dict(identifiers=dict(id=task['id'],partition=task['partition']),source_contract=dict(ready=False))
    monkeypatch.setattr(V,'_prepare_packet',prepare)
    stream=io.StringIO();records=[]
    output=list(V._prepared_rows(tasks,rows,entries,ids,None,None,stream,records))
    comparison=[__import__('json').loads(line) for line in stream.getvalue().splitlines()]
    assert len(output)==len(records)==len(comparison)==238
    assert all('baseline' not in row and 'prediction' not in row for row in output)
    assert [row['prediction'] for row in comparison]==[task['baseline'] for task in tasks]
    assert all(not record['ready'] for record in records)


def test_worker_spool_then_parent_e_merge_is_exact_unicode_and_all_five_arms(tmp_path,monkeypatch):
    monkeypatch.setattr(V.C,'ROOT',tmp_path)
    (tmp_path/'prediction_spool').mkdir();(tmp_path/'frames').mkdir()
    identifiers=dict(id='SYNTHETIC:팔레트',partition='TEST',image_sha256='a'*64)
    value=dict(identifiers=identifiers,clean_input=dict(states=['U']*8,initial_self_hidden=[False]*8))
    fresh=dict(arms={name:dict(accepted=False,actual_pose=None) for name in V.SOLVER.ARMS},
        states=['U']*8,initial_self_hidden=[False]*8,unicode='木😀')
    monkeypatch.setattr(V,'solve_worker_packet',lambda packet:deepcopy(fresh))
    entry=V._worker(value,tmp_path)
    e=dict(id=identifiers['id'],partition='TEST',prediction=dict(accepted=True,actual_pose={'exact':[1.,-0.,3.]}),
        legacy_e_pose_semantics={'comparison_only':True})
    rows=list(V._merged_rows(tmp_path,[identifiers['id']],{identifiers['id']:entry},[e]))
    assert set(rows[0]['arms'])==set(V.ARMS)
    assert rows[0]['arms']['SAME_FROZEN_E']==e['prediction']
    raw=list(V.C.rows(V.C.ROOT/entry['path']))[0]
    assert 'SAME_FROZEN_E' not in raw['arms'] and raw['unicode']=='木😀'
    assert len(list((tmp_path/'frames').glob('*.json')))==1


def test_large_json_line_handles_unicode_and_nonfinite_without_micro_writes():
    value=dict(text='팔레트/木/😀',array=np.array([np.nan,np.inf,-np.inf,1.]))
    text=V._line(value)
    assert text.endswith('\n') and '팔레트' in text
    assert __import__('json').loads(text)['array']==[None,None,None,1.]


def test_bounded_pool_keeps_three_pending_and_all238_fallback_rows_without_worker_e(monkeypatch,tmp_path):
    ids=[f'SYNTHETIC:{i}' for i in range(238)];state=dict(active=0,maximum=0,shutdown=[]);writes=[]
    class Result(Future):
        def result(self,*args,**kwargs):
            state['active']-=1
            return super().result(*args,**kwargs)
    class Pool:
        _processes={}
        def __init__(self,**kwargs):assert kwargs['max_workers']==3
        def submit(self,function,packet,output):
            assert set(packet)=={'identifiers'}
            state['active']+=1;state['maximum']=max(state['maximum'],state['active'])
            result=Result();result.set_result(dict(id=packet['identifiers']['id'],accepted=False));return result
        def shutdown(self,**kwargs):state['shutdown'].append(kwargs)
    monkeypatch.setattr(V,'ProcessPoolExecutor',Pool)
    monkeypatch.setattr(V.C,'write',lambda path,value:writes.append(value))
    output=V._bounded_predict((dict(identifiers={'id':ident})for ident in ids),ids,tmp_path,3)
    assert set(output)==set(ids) and len(output)==238 and state['maximum']==3 and state['active']==0
    assert writes[-1]['completed']==238 and state['shutdown']==[dict(wait=True)]


def test_failed_future_reports_string_pending_IDs_and_preserves_attempt(monkeypatch,tmp_path):
    writes=[];closed=[]
    class Pool:
        _processes={}
        def __init__(self,**kwargs):pass
        def submit(self,function,packet,output):
            future=Future();future.set_exception(RuntimeError('toy worker failure'));return future
        def shutdown(self,**kwargs):closed.append(kwargs)
    monkeypatch.setattr(V,'ProcessPoolExecutor',Pool)
    monkeypatch.setattr(V.C,'write',lambda path,value:writes.append(value))
    ids=['a','b','c']
    with pytest.raises(RuntimeError,match='toy worker failure'):
        V._bounded_predict((dict(identifiers={'id':ident})for ident in ids),ids,tmp_path,3)
    assert writes[-1]['attempt_preserved']
    assert all(isinstance(value,str) for value in writes[-1]['pending_ids'])
    assert closed==[dict(wait=True,cancel_futures=True)]
