"""Driver wiring only; no GT/RGB or optimizer invocation in these fixtures."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest

from . import evaluate_refined_native_edges_v4_17 as NEW
from . import statewise_two_wd_family_v4_9 as FAMILY
from .test_hierarchical_native_candidate_score_v4_17 import fixture


def task_fixture(typed=False):
    a,b,kw=fixture(typed);a[7]['enforce_visible_geometry']=False
    # The shared fixture changes cuboid dimensions after its original O
    # domain was made. Rebind this synthetic range to the new geometry.
    projection,_=NEW.BASE.NATIVE.V.project_native(a[0],np.asarray(a[6]['R']),np.asarray(a[6]['t']),a[1])
    for domain in a[4]:domain['center_xy']=projection[domain['native_corner']].tolist()
    families=FAMILY.make_wd_families(a[0],a[6]);family=families[0]
    pose=NEW.BASE._actual_pose(family,a[6]['R'],a[6]['t'],True,'SYNTHETIC_FROZEN_E')
    baseline=dict(accepted=True,actual_pose=pose,WD_family='ORIGINAL_WD',solution={})
    Q=np.asarray(family['physical_registration_Q']);Rp=np.asarray(a[6]['R'])@Q
    proposals=[dict(R_physical=Rp.tolist(),t=list(a[6]['t']),pose_digest='one',old_accepted=False,old_objective=-1e9),
        dict(R_physical=Rp.tolist(),t=(np.asarray(a[6]['t'])+[.01,0.,0.]).tolist(),pose_digest='two')]
    task=dict(id='synthetic:no_ID_route',partition='CAL',session='synthetic',image='does_not_exist.png',image_sha256='a'*64,
        states=a[2],initial_self_hidden=[s=='S'for s in a[2]],baseline=baseline,
        spec=dict(xyz=a[0],K=a[1],states=a[2],visible=a[3],domains=a[4],image_hw=a[5],coarse_pose=a[6],config=a[7]),
        families={f['name']:f for f in families},boundary_original=[b],boundary_merged=[b],u_regions=[],
        outline={'supported':False},proposals=proposals,proposal_inventory={'distinct_physical_poses':2})
    model=NEW.BASE.ANGLE.fit_native_edge_angle_model([dict(id='cal',partition='CAL',session='fit',compatible=True,
        edges=[dict(signed_angular_sine_error=.02,actual_inlier_span_px=24.)])])
    return NEW.C.finite(task),model


def no_optimizer(*args,**kwargs):
    return dict(solutions=[],candidates=[],search={'continuous_SE3':True,'new_P3P_or_global_starts':0},
        start_selection={'selected_start_names':[]})


@pytest.mark.parametrize('typed',[False,True])
def test_all_cached_contenders_survive_no_refined_endpoint_and_parent_score_schema(monkeypatch,typed):
    task,model=task_fixture(typed);before=deepcopy(task)
    monkeypatch.setattr(NEW.REFINE,'refine_native_edge_pose',no_optimizer)
    row=NEW._run(task,model)
    assert task==before and row['arms'][NEW.ARMS[0]]==task['baseline']
    assert set(row['fixed_family'])==set(NEW.ARMS)
    assert set(row['tangent_control_audit'])==set(NEW.ARMS[1:])
    for arm in NEW.ARMS[1:]:
        result=row['arms'][arm];audit=row['native_refinement_audit'][arm]
        assert result['accepted'] and result['keypoint_readout']['selected_pose_is_cached_contender']
        assert audit['evaluated_candidates']==len(task['families'])*len(task['proposals'])
        assert audit['selected_native_energy']<=audit['cached_minimum_native_energy']+1e-10
        assert result['keypoint_readout']['U_point_or_line_measurements']==0
        for k,s in enumerate(task['states']):
            if s=='U':assert result['keypoint_readout']['reported_points'][k] is None
            if s=='S':assert result['keypoint_readout']['point_sources'][k]=='POSE_ONLY_MODEL_PROJECTION'
        p=result['actual_pose'];assert np.allclose(np.asarray(p['R_cf'])@p['physical_registration_Q'],p['R_physical'])


def test_total_failure_uses_only_own_initial_coarse_and_never_old_pose(monkeypatch):
    task,model=task_fixture();monkeypatch.setattr(NEW.REFINE,'refine_native_edge_pose',no_optimizer)
    for p in task['proposals']:p['t']=[0.,0.,-4.]
    row=NEW._run(task,model);coarse=task['families']['ORIGINAL_WD']['coarse_pose']
    for arm in NEW.ARMS[1:]:
        result=row['arms'][arm];assert not result['accepted']
        assert result['actual_pose']['R_cf']==coarse['R'] and result['actual_pose']['centroid']==coarse['t']
        assert result['fallback_audit']['own_original_initial_coarse_only']
        assert not result['fallback_audit']['old_best_pose_fallback']
        assert result['keypoint_readout']['OT_solved'] is False


def test_mask_is_only_evaluated_for_machine_equal_native_minima(monkeypatch):
    states=['B','I','U','O','T','B','S','S'];candidates=[];details={};seen=[]
    for name,cost,depth in [('a',1.,4.),('b',1.,5.),('worse',3.,6.)]:
        candidates.append(dict(name=name,family='ORIGINAL_WD',accepted=True,all_joint_constraints_satisfied=True,
            native_costs=[dict(kind='BI_COMPLETE',native_corner=0,weighted_robust_cost=cost)],global_mask_cost=None))
        details[name]=dict(evaluation={'R':np.eye(3).tolist(),'t':[0.,0.,depth]},
            problem=SimpleNamespace(scale=1.,xyz=np.zeros((8,3)),K=np.eye(3)))
    def mask(p,*args):
        seen.append(float(p[-1]));return 10.-float(p[-1]),np.zeros(6),{'valid_projection':True}
    monkeypatch.setattr(NEW.BASE.OUTLINE,'outline_value_and_gradient',mask)
    selection,_=NEW._select_with_mask_ties(states,candidates,details,{'outline':{'supported':True,'configuration':{}}})
    assert seen==[4.,5.] and selection['selected_name']=='b'
    assert candidates[2]['global_mask_cost'] is None
