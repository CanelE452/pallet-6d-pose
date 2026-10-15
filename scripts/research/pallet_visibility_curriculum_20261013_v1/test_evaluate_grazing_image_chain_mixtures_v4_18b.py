"""Strict profile/generation join and unchanged private worker wiring."""
from copy import deepcopy

import pytest

from . import evaluate_grazing_image_chain_mixtures_v4_18b as F
from .test_evaluate_refined_native_edges_v4_17 import task_fixture
from .test_image_chain_mixture_problem_v4_18 import setup
from .test_calibrate_native_edge_mixture_v4_18 import component,row


def fixture():
    task,_=task_fixture(True);_,_,chain,_=setup();task['outline']['configuration']={}
    original=deepcopy(task);family=task['families']['ORIGINAL_WD']
    task['grazing_adaptation']=dict(original_states=task['states'],
        geometry_gate=dict(states=task['states'],new_U_native_ids=[]))
    digest=F.OLD.E._digest(task['grazing_adaptation'])
    observation={k:task[k]for k in ('id','partition','session','image','image_sha256')}
    observation.update(states=task['states'],visible_measurements=task['spec']['visible'],
        domains=task['spec']['domains'],u_regions=task['u_regions'],new_grazing_U_native_ids=[],
        observation_generation=F.OBS.OBSERVATION_GENERATION,sealed_grazing_adaptation_sha256=digest,
        chain_observation={**chain,'schema':'IMAGE_SUPPORTED_EDGE_CHAINS_V4_17C'},
        initial_geometry=dict(native_xyz=family['native_xyz'],physical_registration_Q=family['physical_registration_Q'],
            K=task['spec']['K'],image_hw=task['spec']['image_hw'],initial_self_hidden=task['initial_self_hidden']))
    producer={k:task[k]for k in ('id','partition','session','image','image_sha256')}
    producer.update(states=task['states'],observation_generation=F.OBS.OBSERVATION_GENERATION,
        sealed_grazing_adaptation_sha256=digest,sealed_grazing_filtered_boundary_sha256=F.OLD.E._digest(task['boundary_merged']))
    return task,observation,producer,original


def test_join_preserves_all_cached_physics_and_rejects_old_observer_generation():
    values=fixture();before=deepcopy(values);task=F._join_task(*values)
    assert values==before and task['proposals']==values[0]['proposals']
    bad=deepcopy(values);bad[1]['observation_generation']='OLD_UNGATED_GENERATION'
    with pytest.raises(ValueError,match='grazing-gated'):F._join_task(*bad)


def test_geometry_visible_noise_boundary_and_candidate_changes_fail_closed():
    for field in ('covariance','boundary','proposal','Q'):
        values=list(deepcopy(fixture()))
        if field=='covariance':values[1]['visible_measurements']=[]
        elif field=='boundary':values[2]['sealed_grazing_filtered_boundary_sha256']='0'*64
        elif field=='proposal':values[3]['proposals'][0]['t'][2]+=.1
        else:values[1]['initial_geometry']['physical_registration_Q']=[[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]
        with pytest.raises(ValueError):F._join_task(*values)


def test_worker_separate_CAL_model_baseline_exact_and_new_profile_metadata():
    task=F._join_task(*fixture());before=deepcopy(task)
    model=F.CAL.fit_native_edge_mixture_model([row('a','A',[component(0.,0.)])])
    result=F._run(task,model)
    assert task==before and result['arms']['SAME_FROZEN_E']==task['baseline']
    assert result['observation_generation']==F.OBS.OBSERVATION_GENERATION
    assert result['contract']['fresh_grazing_NMS_and_separate_CAL_model_used']
    assert set(result['fixed_family'])==set(F.ARMS)
    for arm in F.ARMS[1:]:
        assert not result['arms'][arm]['keypoint_readout']['OT_solved']
        assert result['image_chain_mixture_audit'][arm]['evaluated_candidates']==len(task['families'])*len(task['proposals'])
    old=F.CAL.OLD.fit_native_edge_mixture_model([row('a','A',[component(0.,0.)])])
    with pytest.raises(ValueError):F._run(task,old)


def test_worker_rejected_candidates_keep_own_original_coarse_fallback():
    task=F._join_task(*fixture());coarse=task['families']['ORIGINAL_WD']['coarse_pose']
    for proposal in task['proposals']:proposal['t']=[0.,0.,-4.]
    model=F.CAL.fit_native_edge_mixture_model([row('a','A',[component(0.,0.)])])
    result=F._run(task,model)
    for arm in F.ARMS[1:]:
        out=result['arms'][arm]
        assert not out['accepted']and out['actual_pose']['R_cf']==coarse['R']
        assert out['actual_pose']['centroid']==coarse['t']
        assert out['fallback_audit']['own_original_initial_coarse_only']
