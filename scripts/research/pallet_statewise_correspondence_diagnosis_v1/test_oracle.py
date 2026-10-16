"""Synthetic JSON semantics only; no original reference/optimizer opened."""
from copy import deepcopy
import numpy as np
from . import oracle as O


def _pose(code):
    return dict(available=True, R_cf=np.eye(3).tolist(), R_physical=np.eye(3).tolist(),
                centroid=[code,0,3], physical_registration_Q=np.eye(3).tolist(), cf_extents=[2,1,3])


def _metric(pose):
    table={0:(20.,20.),1:(1.,20.),2:(20.,1.),3:(1.,1.),4:(20.,85.)}
    t,r=table[int(pose['centroid'][0])]
    return dict(pose_available=True,translation_cm=t,rotation_deg=r,success_le5=t<=5 and r<=5)


def _fixture(codes=()):
    family=dict(name='ORIGINAL_WD',native_xyz=[[-1,-.5,-1.5],[1,-.5,-1.5],[1,.5,-1.5],[-1,.5,-1.5],
        [-1,-.5,1.5],[1,-.5,1.5],[1,.5,1.5],[-1,.5,1.5]],cf_extents=[2,1,3],
        physical_registration_Q=np.eye(3).tolist(),feasible_samples=[],published_actual_pose=_pose(0))
    for code in codes:
        family['feasible_samples'].append(dict(pose={k:v for k,v in _pose(code).items() if k!='available'}))
    stage=dict(families=[family],selected_family='ORIGINAL_WD')
    original=dict(id='synthetic:0',stages=dict(G10=deepcopy(stage),G13=deepcopy(stage)),
                  original_e=dict(frozen_family='ORIGINAL_WD',actual_pose=_pose(0)))
    packet=dict(id='synthetic:0',partition='CAL',session='synthetic',
                solver_input=dict(boundary_observations=[dict(explicit_complete_BI_likelihood_revocation=True)]))
    replay=dict(id='synthetic:0',generated_candidates=[])
    return original,replay,packet


def test_different_candidates_minT_minR_never_form_joint_success():
    original,replay,packet=_fixture([1,2])
    out=O.analyze_frame(original,replay,packet,_metric)
    feasible=out['stages']['feasible']
    assert feasible['translation_min_cm']==1 and feasible['rotation_min_deg']==1
    assert feasible['joint_success_exists'] is False
    assert 'SELECTION_LOSS' not in out['failure_flags']
    assert 'CANDIDATE_RECORD_INCOMPLETE' in out['failure_flags']
    assert 'OBSERVATION_OR_PROPOSAL_LIMIT' not in out['failure_flags']


def test_one_actual_success_candidate_supports_existential_selection_loss():
    original,replay,packet=_fixture([3])
    out=O.analyze_frame(original,replay,packet,_metric)
    assert out['stages']['feasible']['joint_success_exists']
    assert 'SELECTION_LOSS' in out['failure_flags']
    assert out['earliest_observed_failure_stage']=='within_family_final_selection'


def test_refinement_loss_matches_same_start_family_stage_branch():
    original,replay,packet=_fixture()
    for code,name in ((3,'start_UNREFINED'),(0,'start')):
        replay['generated_candidates'].append(dict(R=np.eye(3).tolist(),t=[code,0,3],family='ORIGINAL_WD',
            stage='G13',kind='outline_consider',original_endpoint_record=dict(initialization=name,branch=0,joint_feasible=True)))
    out=O.analyze_frame(original,replay,packet,_metric)
    assert 'REFINEMENT_LOSS' in out['failure_flags']
    assert len(out['refinement_pairs'])==1
    assert len(out['support_unknown_candidate_ids'])==2
    assert not out['stages']['after_support']['joint_success_exists']


def test_fallback_that_is_not_E0_is_marked_not_local_without_GT():
    result=dict(actual_pose=_pose(1),accepted=False,fallback=True,generated_candidates=[])
    out=O.single_fallback_audit(result,_pose(0))
    assert out['strict_single_local_interpretation_blocked']
    assert out['fallback_matches_own_E0'] is False
    result['actual_pose']=_pose(0)
    out=O.single_fallback_audit(result,_pose(0))
    assert out['fallback_matches_own_E0'] is True
    assert not out['strict_single_local_interpretation_blocked']


def test_G10_success_is_never_mixed_into_G13_pruning_pool():
    original,replay,packet=_fixture()
    replay['generated_candidates']=[dict(R=np.eye(3).tolist(),t=[3,0,3],family='ORIGINAL_WD',
        stage='G10',kind='outline_consider',original_endpoint_record=dict(initialization='s',branch=0,joint_feasible=False))]
    out=O.analyze_frame(original,replay,packet,_metric)
    assert out['original_stage_oracles']['G10']['generated']['joint_success_exists']
    assert not out['stages']['generated']['joint_success_exists']
    assert 'PRUNING_LOSS' not in out['failure_flags']


def _rejected_good(replay):
    replay['generated_candidates']=[dict(R=np.eye(3).tolist(),t=[3,0,3],family='ORIGINAL_WD',
        stage='G13',kind='outline_consider',original_endpoint_record=dict(initialization='s',branch=0,joint_feasible=False))]


def test_one_rejected_good_is_not_pruning_loss_if_another_good_survives():
    original,replay,packet=_fixture([3]);_rejected_good(replay)
    out=O.analyze_frame(original,replay,packet,_metric)
    assert out['rejected_good_endpoint_count']==1
    assert 'PRUNING_LOSS' not in out['failure_flags']
    assert 'SELECTION_LOSS' in out['failure_flags']


def test_pruning_loss_requires_no_joint_good_remains_and_final_failure():
    original,replay,packet=_fixture();_rejected_good(replay)
    out=O.analyze_frame(original,replay,packet,_metric)
    assert 'PRUNING_LOSS' in out['failure_flags']
    original['original_e']['actual_pose']=_pose(3)
    out=O.analyze_frame(original,replay,packet,_metric)
    assert out['rejected_good_endpoint_count']==1
    assert 'PRUNING_LOSS' not in out['failure_flags']
    assert 'OBSERVED_REFINEMENT_OR_PRUNING_LOSS' not in out['failure_flags']


def test_aggregate_separates_candidate_damage_on_final_success_from_failed_frames():
    original,replay,packet=_fixture([3]);original['original_e']['actual_pose']=_pose(3)
    for code,name in ((3,'s_UNREFINED'),(0,'s')):
        replay['generated_candidates'].append(dict(R=np.eye(3).tolist(),t=[code,0,3],family='ORIGINAL_WD',
            stage='G13',kind='outline_consider',original_endpoint_record=dict(initialization=name,branch=0,joint_feasible=True)))
    base=O.analyze_frame(original,replay,packet,_metric)
    assert 'REFINEMENT_LOSS' in base['failure_flags']
    rows=[]
    for part,n in (('CAL',46),('TEST',49),('REMAINDER',143)):
        for i in range(n):
            row=deepcopy(base);row.update(partition=part,id=f'{part}:{i}');rows.append(row)
    out=O.aggregate(rows)['PRIMARY']
    assert out['final_success_n']==192 and out['final_failure_n']==0
    assert out['failure_flags']['REFINEMENT_LOSS']==192
    assert out['failure_flags_final_failure_only'].get('REFINEMENT_LOSS',0)==0


def test_large_rotation_alone_is_not_reference_or_registration_error():
    original,replay,packet=_fixture();original['original_e']['actual_pose']=_pose(4)
    out=O.analyze_frame(original,replay,packet,_metric)
    audit=out['reference_and_registration_audit']
    assert audit['large_physical_rotation_over45_diagnostic']
    assert audit['semantic_owner_status']=='SEMANTIC_OWNER_UNVERIFIED'
    assert audit['flag'] is None
    assert 'REFERENCE_OR_REGISTRATION_UNRESOLVED' not in out['failure_flags']


def _coverage_fixture(monkeypatch):
    orig,replay,packet=_fixture()
    solution=dict(accepted=True,R=np.eye(3).tolist(),t=[0,0,3],
                  feasible_pose_samples=[dict(R=np.eye(3).tolist(),t=[0,0,3])])
    monkeypatch.setattr(O,'original_stage_inputs',lambda p,n:([deepcopy(solution)],[]))
    for stage in ('G10','G13'):
        fam=orig['stages'][stage]['families'][0]
        fam.update(solver_schema='statewise_global_outline_solver_v4_10',original_search={},
                   original_outcomes=[dict(original_record=dict(objective=1.))])
        replay['generated_candidates'].append(dict(stage=stage,family='ORIGINAL_WD',kind='outline_consider'))
    return orig,replay,packet


def test_complete_e_consumer_does_not_require_rejected_parent_proposals(monkeypatch):
    orig,replay,packet=_coverage_fixture(monkeypatch)
    result=O.consumer_coverage(orig,replay,packet,parity_passed=True)
    assert result['original_e_consumer_generated_bank_complete']
    assert result['historical_WD_P1_full_generation_complete'] is False
    assert len(result['incoming_candidates'])==4
    assert sum(c['original_endpoint_record']['original_dedup_kept'] is False
               for c in result['incoming_candidates'])==2
    assert not O.consumer_coverage(orig,replay,packet,parity_passed=False)['original_e_consumer_generated_bank_complete']
    replay['generated_candidates'].pop()
    assert not O.consumer_coverage(orig,replay,packet,parity_passed=True)['original_e_consumer_generated_bank_complete']


def test_active_partial_initial_generator_gap_requires_exact_supplement(monkeypatch):
    orig,replay,packet=_coverage_fixture(monkeypatch)
    family=orig['stages']['G13']['families'][0]
    family.update(solver_schema='statewise_partial_BI_u_solver_v4_13',original_search=dict(initialization_count=1))
    packet['original_e_stage1_family_seed_bank']={'ORIGINAL_WD':{'family':{'coarse_pose':{'R':np.eye(3).tolist(),'t':[0,0,3]}}}}
    replay['generated_candidates'][-1]['kind']='partial_joint_fit'
    assert not O.consumer_coverage(orig,replay,packet,parity_passed=True)['original_e_consumer_generated_bank_complete']
    supplement=dict(id=packet['id'],families={'ORIGINAL_WD':dict(starts=[dict(R=np.eye(3).tolist(),t=[0,0,3],name='root')],
        initialization_count_parity=True,name_order_parity=True,search_metadata_parity=True)})
    assert O.consumer_coverage(orig,replay,packet,parity_passed=True,
        initial_start_supplement=supplement)['original_e_consumer_generated_bank_complete']


def test_proposal_limit_requires_confirmed_consumer_bank_and_no_joint_candidate(monkeypatch):
    orig,replay,packet=_fixture();packet['original_e_stage1_family_seed_bank']={}
    complete=dict(original_e_consumer_generated_bank_complete=True,
        original_e_consumer_numeric_endpoints_complete=True,incoming_candidates=[])
    monkeypatch.setattr(O,'consumer_coverage',lambda *args,**kwargs:complete)
    for code in (1,2):
        replay['generated_candidates'].append(dict(R=np.eye(3).tolist(),t=[code,0,3],family='ORIGINAL_WD',
            stage='G13',kind='outline_consider',original_endpoint_record=dict(initialization=str(code),branch=0,joint_feasible=True)))
    out=O.analyze_frame(orig,replay,packet,_metric)
    assert 'OBSERVATION_OR_PROPOSAL_LIMIT' in out['failure_flags']
    assert 'CANDIDATE_RECORD_INCOMPLETE' not in out['failure_flags']
    assert not out['entire_historical_generated_bank_complete']
    assert out['proposal_limit_is_fixed_original_consumer_scope_not_global_physical_impossibility']
