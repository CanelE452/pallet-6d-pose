"""Narrow outline integration tests; synthetic cuboid, no dataset inputs."""
from copy import deepcopy

import cv2
import numpy as np

from . import statewise_partial_BI_outline_solver_v4_13 as NEW
from . import statewise_partial_boundary_u_solver_v4_6 as PARENT
from .test_statewise_partial_BI_u_solver_v4_13 import typed_fixture
from .test_statewise_partial_boundary_u_solver_v4_6 import region
from . import joint_ot_pnp as V


def sample():
    a,b=typed_fixture();rr=[region(a)]
    old=PARENT.solve_boundary_u_region_joint_ot_pnp(*a[:7],config=a[7],u_regions=rr)
    q,_=V.project_native(a[0],np.asarray(a[6]['R']),np.asarray(a[6]['t']),a[1])
    mask=np.zeros(a[5],np.float32)
    hull=cv2.convexHull(q.astype(np.float32)).reshape(-1,2).astype(np.int32)
    cv2.fillConvexPoly(mask,hull,1.)
    outline=NEW.observe_global_outline(mask,a[5],image_binding={'image_sha256':'a'*64})
    return a,b,rr,old,outline


def test_no_typed_revocation_exact_parent_delegate(monkeypatch):
    a,b=typed_fixture();result={'schema':'untouched_parent','sentinel':[1,2,3]};seen=[]
    monkeypatch.setattr(NEW.OLD,'refine_global_outline_joint_pose',lambda *args,**kw:(seen.append(kw) or result))
    out=NEW.refine_global_outline_joint_pose(*a[:7],config=a[7],outline_observation={'supported':True},
        initial_solutions=[result],boundary_observations=[],u_regions=[])
    assert out is result and seen[0]['boundary_observations']==[]


def test_unavailable_outline_active_factors_fresh_data_only(monkeypatch):
    a,b=typed_fixture();seen=[];result={'accepted':False,'typed_data_only':True}
    def fresh(*args,**kw):seen.append((args,kw));return result
    monkeypatch.setattr(NEW.P,'solve_partial_BI_u_region_joint_ot_pnp',fresh)
    original={'accepted':True,'R':a[6]['R'],'t':a[6]['t'],'feasible_pose_samples':[]}
    out=NEW.refine_global_outline_joint_pose(*a[:7],config=a[7],outline_observation={'supported':False},
        initial_solutions=[original],boundary_observations=[b],u_regions=[],initializations=iter([]))
    assert out is result and seen[0][1]['config']['prior_weight']==0.
    assert seen[0][1]['boundary_observations']==[b] and seen[0][0][3]==a[3]
    assert seen[0][1]['initializations'][0]['name']=='SAVED_0_WINNER'


def test_active_typed_I_outline_readout_archives_and_all_joint_guards():
    a,b,rr,old,outline=sample();before=deepcopy(a[3]);bb=deepcopy(b)
    result=NEW.refine_global_outline_joint_pose(*a[:7],config=a[7],outline_observation=outline,
        initial_solutions=[old],boundary_observations=[b],u_regions=rr)
    assert result['accepted'] and result['U_constraints_satisfied'] and result['OT_constraints_satisfied']
    assert result['partial_BI_constraints_satisfied'] and result['states']==a[2]
    k=b['native_corner'];assert result['reported_native'][k] is None
    assert result['boundary_readout'][str(k)]['state']=='I'
    assert result['boundary_readout'][str(k)]['reported_full_2D_xy'] is None
    assert result['objective_components']['removed_complete_BI_2D_likelihood_ids']==[k]
    assert result['boundary_factor_audit']['explicit_complete_BI_revocation_ids']==[k]
    assert a[3]==before and b==bb
    assert all(x['feasible'] for x in result['joint_constraint_audit']+result['U_region_constraint_audit']+result['boundary_observation_audit'])
    assert np.allclose(result['R_physical'],np.asarray(result['R'])@a[6]['physical_registration_Q'])
