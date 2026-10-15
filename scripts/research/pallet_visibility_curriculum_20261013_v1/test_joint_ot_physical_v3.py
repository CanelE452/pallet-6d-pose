"""Physical state consistency and frozen observations without a seed clamp."""
from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from . import common as C
from .joint_ot_jacobian_v2 import constraint_values_and_jacobian
from .joint_ot_pnp import (_DEFAULTS, _constraint_values, _inputs, project_native,
                          solve_joint_ot_pnp)


FAST = dict(global_rotation_starts=0,max_p3p_triples=2,max_joint_iterations=150,analytic_jacobian=True)


def physical_three_BI_fixture():
    xyz = C.cuboid(1.,.4,1.2)
    K = np.array([[600.,25.,320.],[0.,620.,240.],[0.,0.,1.]])
    R = Rotation.from_rotvec([-.4611806149383468,.4522670776283566,-.7933895902003064]).as_matrix()
    t = np.array([.03840692188819134,-.037316089277514036,4.700156034328923])
    q,_ = project_native(xyz,R,t,K)
    ids = [0,3,7]
    states = ["B","U","U","B","S","U","U","B"]
    visible = [dict(native_corner=i,xy=q[i].tolist(),covariance_px2=np.eye(2).tolist(),protection_px=.05)
               for i in ids]
    return xyz,K,R,t,q,states,visible


def test_default_final_BI_geometry_rejects_positive_depth_selfhidden_P3P_branch():
    xyz,K,R,t,_,states,visible = physical_three_BI_fixture()
    legacy = solve_joint_ot_pnp(xyz,K,states,visible,[],[480,640],config={**FAST,
        "protect_seed_projection":True,"enforce_visible_geometry":False,"enforce_visible_canvas":False})
    corrected = solve_joint_ot_pnp(xyz,K,states,visible,[],[480,640],config=FAST)
    assert legacy["accepted"] and corrected["accepted"]
    legacy_B3 = next(a for a in legacy["visible_protection_audit"] if a["native_corner"]==3)
    assert legacy_B3["maximum_incident_outward_ray_cosine"] < -.25
    assert all(a["visible_geometry_pass"] and a["visible_canvas_pass"] for a in corrected["visible_protection_audit"])
    np.testing.assert_allclose(corrected["R"],R,atol=2e-5,rtol=0)
    np.testing.assert_allclose(corrected["t"],t,atol=2e-5,rtol=0)
    assert corrected["states"]==states
    assert corrected["final_model_visibility_audit"]["input_S_states_relabelled"] is False
    assert corrected["final_model_visibility_audit"]["S_image_queries"]==0
    assert corrected["provenance"]["seed_projection_protection_enabled"] is False


def test_joint_O_constraints_can_change_pose_beyond_old_seed_guard_with_frozen_BI():
    # Five in-frame, model-visible observations on a thin registered pallet.
    # The independent O domains represent a translated possible pose inside
    # the same declared BI covariance budgets. No O center is fit as a point.
    xyz=C.cuboid(1.3,.11,1.1)
    K=np.array([[600.,15.,320.],[0.,620.,240.],[0.,0.,1.]])
    R=Rotation.from_euler("xyz",[.25,-.3,.05]).as_matrix();t=np.array([0.,0.,3.2])
    q,_=project_native(xyz,R,t,K);target_t=t+[.06,0,0]
    possible,_=project_native(xyz,R,target_t,K)
    ids=[1,2,4,5,7];states=["O","B","B","O","B","B","S","B"]
    measurements=[dict(native_corner=i,xy=q[i].tolist(),covariance_px2=(np.eye(2)*25).tolist(),protection_px=2.)
                  for i in ids]
    domains=[dict(native_corner=i,state="O",shape="ellipse",center_xy=possible[i].tolist(),half_axes_px=[.2,.2])
             for i in [0,3]]
    before=deepcopy(measurements)
    initializations=[dict(R=R.tolist(),t=target_t.tolist())]
    protected=solve_joint_ot_pnp(xyz,K,states,measurements,domains,[480,640],initializations,
        {**FAST,"protect_seed_projection":True})
    free=solve_joint_ot_pnp(xyz,K,states,measurements,domains,[480,640],initializations,FAST)
    assert not protected["accepted"] and free["accepted"]
    assert measurements==before
    assert max(a["seed_projection_shift_px"] for a in free["visible_protection_audit"]) > 3.
    assert all(not a["seed_projection_guard_active"] for a in free["visible_protection_audit"])
    assert all(a["measurement_budget_pass"] and a["visible_geometry_pass"] and a["visible_canvas_pass"]
               for a in free["visible_protection_audit"])
    assert all(a["feasible"] for a in free["joint_constraint_audit"])
    for measurement in measurements:
        np.testing.assert_array_equal(free["reported_native"][measurement["native_corner"]],measurement["xy"])
    assert free["provenance"]["OT_used_as_precise_correspondences"] is False


def test_projected_direct_corner_must_remain_in_closed_canvas_despite_noise_budget():
    xyz=C.cuboid(1.,.4,.6);K=np.array([[100.,0.,29.],[0.,100.,50.],[0.,0.,1.]])
    R=np.eye(3);t=np.array([0.,0.,2.]);scale=np.linalg.norm(np.ptp(xyz,axis=0))
    q,_=project_native(xyz,R,t,K);assert q[0,0]<0
    observed=q[0].copy();observed[0]=.2
    states=["B"]+["U"]*7
    visible=[dict(native_corner=0,xy=observed.tolist(),covariance_px2=np.eye(2).tolist(),protection_px=2.)]
    cfg=dict(_DEFAULTS,analytic_jacobian=True)
    xyz,K,_,measurements,domains,hw=_inputs(xyz,K,states,visible,[],[100,100],cfg)
    p=np.r_[np.zeros(3),t/scale]
    old=dict(cfg,enforce_visible_canvas=False)
    old_values=_constraint_values(p,xyz,K,measurements,domains,(),q,hw,scale,old)
    new_values=_constraint_values(p,xyz,K,measurements,domains,(),q,hw,scale,cfg)
    assert np.min(old_values)>=0 and np.min(new_values)<0
    analytic,_=constraint_values_and_jacobian(p,xyz,K,measurements,domains,(),q,hw,scale,cfg)
    np.testing.assert_allclose(analytic,new_values,atol=1e-12,rtol=0)


def test_two_physical_BI_with_OT_explores_pose_family_without_fake_PnP(monkeypatch):
    import cv2
    xyz=C.cuboid(1.3,.11,1.1)
    K=np.array([[600.,15.,320.],[0.,620.,240.],[0.,0.,1.]])
    R=Rotation.from_euler("xyz",[.25,-.3,.05]).as_matrix();t=np.array([0.,0.,3.2])
    q,_=project_native(xyz,R,t,K)
    states=["O","B","B","O","U","U","S","U"]
    measurements=[dict(native_corner=i,xy=q[i].tolist(),covariance_px2=(np.eye(2)*4).tolist(),protection_px=2.)
                  for i in [1,2]]
    domains=[dict(native_corner=i,state="O",shape="ellipse",center_xy=q[i].tolist(),half_axes_px=[3.,3.])
             for i in [0,3]]
    before=deepcopy(measurements)
    def forbidden(*args,**kwargs):
        raise AssertionError("Two B/I points must not be passed to P3P or SQPNP")
    monkeypatch.setattr(cv2,"solveP3P",forbidden)
    monkeypatch.setattr(cv2,"solvePnPGeneric",forbidden)
    starts=[dict(R=R.tolist(),t=t.tolist()),
            dict(R=(Rotation.from_euler("y",.15,degrees=True).as_matrix()@R).tolist(),t=t.tolist())]
    result=solve_joint_ot_pnp(xyz,K,states,measurements,domains,[480,640],starts,
        {**FAST,"global_rotation_starts":24})
    no_OT=solve_joint_ot_pnp(xyz,K,["U" if s=="O" else s for s in states],measurements,[],[480,640],starts,FAST)
    assert not no_OT["accepted"] and no_OT["fallback_reason"]=="fewer_than_three_corrected_BI_for_pose_seed"
    assert result["accepted"] and result["status"]=="joint_feasible_pose_family"
    assert result["ambiguity"]["detected"] and result["ambiguity"]["pose_family_from_two_BI"]
    assert result["OT_constraints_satisfied"] and result["OT_feasible_hypothesis_available"]
    assert result["OT_unique_solution_claimed"] is False and result["OT_solved"] is False
    assert result["output_corner_sources"][0]=="AMBIGUOUS_JOINT_POSE_FAMILY_PROJECTION"
    assert len(result["pose_family_candidates"])>=2
    assert result["ambiguity"]["global_uniqueness_proved"] is False
    assert result["rank"]["visible"]["rank"] <= 4
    assert result["search"]["P3P_or_SQPNP_called"] is False
    assert result["search"]["initialization_count"] >= 24
    assert len(result["feasible_pose_samples"]) >= 2
    assert all(a["feasible"] for a in result["joint_constraint_audit"])
    assert measurements==before
    assert result["provenance"]["OT_used_as_precise_correspondences"] is False
    assert result["provenance"]["S_image_terms"]==0 and result["provenance"]["S_region_terms"]==0
    for measurement in measurements:
        np.testing.assert_array_equal(result["reported_native"][measurement["native_corner"]],measurement["xy"])


def test_misspelled_solver_guard_configuration_is_rejected():
    xyz,K,_,_,_,states,visible=physical_three_BI_fixture()
    with pytest.raises(ValueError,match="Unknown joint pose config fields: visible_geometry_tolerance_deg"):
        solve_joint_ot_pnp(xyz,K,states,visible,[],[480,640],config={"visible_geometry_tolerance_deg":2.})
