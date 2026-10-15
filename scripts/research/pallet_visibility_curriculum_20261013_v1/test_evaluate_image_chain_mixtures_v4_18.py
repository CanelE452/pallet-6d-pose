"""New actual-line worker wiring, without images/labels/model fits."""
from copy import deepcopy

from . import evaluate_image_chain_mixtures_v4_18 as NEW
from .test_evaluate_refined_native_edges_v4_17 import task_fixture
from .test_image_chain_mixture_problem_v4_18 import setup


def inputs(monkeypatch):
    task,_=task_fixture(True);_,_,observation,parameters=setup()
    parameters={k:v for k,v in parameters.items()if k not in('initial_self_hidden','image_binding')}
    monkeypatch.setattr(NEW.CAL,'mixture_noise_parameters',lambda model:deepcopy(parameters))
    task['new_chain_observation']=observation;task['outline']['configuration']={}
    return task


def test_uniform_worker_preserves_inputs_baseline_all_proposals_readout_and_parent_score_schema(monkeypatch):
    task=inputs(monkeypatch);before=deepcopy(task)
    row=NEW._run(task,{})
    assert task==before and row['arms']['SAME_FROZEN_E']==task['baseline']
    assert set(row['fixed_family'])==set(NEW.ARMS)
    assert set(row['tangent_control_audit'])==set(NEW.ARMS[1:])
    for name in NEW.ARMS[1:]:
        arm=row['arms'][name];assert arm['accepted']
        assert row['image_chain_mixture_audit'][name]['evaluated_candidates']==len(task['families'])*len(task['proposals'])
        assert not arm['fallback_audit']['old_best_pose_fallback']
        for k,state in enumerate(task['states']):
            if state=='U':assert arm['keypoint_readout']['reported_points'][k]is None
            elif state=='S':assert arm['keypoint_readout']['point_sources'][k]=='POSE_ONLY_MODEL_PROJECTION'
    assert row['arms'][NEW.ARMS[1]]['solution']['objective']==row['arms'][NEW.ARMS[2]]['solution']['objective']


def test_invalid_all_candidates_keep_frame_and_use_only_own_coarse_fallback(monkeypatch):
    task=inputs(monkeypatch)
    for proposal in task['proposals']:proposal['t']=[0.,0.,-4.]
    row=NEW._run(task,{})
    coarse=task['families']['ORIGINAL_WD']['coarse_pose']
    for name in NEW.ARMS[1:]:
        arm=row['arms'][name];assert not arm['accepted']
        assert arm['actual_pose']['R_cf']==coarse['R']and arm['actual_pose']['centroid']==coarse['t']
        assert arm['fallback_audit']['own_original_initial_coarse_only']
        assert arm['keypoint_readout']['OT_solved']is False


def test_coupled_control_adds_frozen_weight_only_after_current_feasibility_gate():
    candidates=[]
    for name,native,mask,accepted in(('native',1.,8.,True),('shape',2.,0.,True),('invalid',0.,0.,False)):
        candidates.append(dict(name=name,family='ORIGINAL_WD',accepted=accepted,
            native_costs=[dict(weighted_robust_cost=native)],global_mask_cost=mask))
    selected=NEW._coupled_selection(candidates,1.,dict(ambiguity={'global_uniqueness_proved':False}))
    assert selected['selected_name']=='shape'
    assert len(selected['candidates'])==2
    selected=NEW._coupled_selection(candidates,0.,dict(ambiguity={}))
    assert selected['selected_name']=='native'
