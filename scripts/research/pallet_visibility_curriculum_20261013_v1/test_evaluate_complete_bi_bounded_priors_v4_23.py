"""Integration of typed costs, visible readout, and fixed original starts."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest

from . import evaluate_complete_bi_bounded_priors_v4_23 as V
from .test_complete_bi_image_chain_evidence_v4_20d import fixture,auth


def test_new_complete_tangent_enters_authenticated_basis_once_without_full_q():
    base,obs,kw,(k,_,_),R,t=fixture(False)
    problem=V._CompleteProblem(base,auth(base,obs,kw),overlap_policy=V.COMP.PROTOTYPE_OVERLAP_POLICY)
    at=V.COMP.N.V._pose_vector(R,t,base.scale)+[.01,-.02,.005,.008,-.011,.025]
    value,_,parts=problem.objective(at);costs=V._bridge_costs(parts['native_costs'])
    row=dict(name='unit',family='ORIGINAL_WD',accepted=True,all_joint_constraints_satisfied=True,
        native_costs=costs,global_mask_cost=0.)
    authenticated=V.OLD.P.AUTH.select_hierarchical_native_candidates(base.states,[row])
    assert authenticated['selected_name']=='unit'
    assert authenticated['candidates'][0]['native_energy']==pytest.approx(value)
    assert sum(c['kind']=='ARCHIVED_TANGENT'and c['native_corner']==k for c in costs)==1
    assert not any(c['kind']=='BI_COMPLETE'and c['native_corner']==k for c in costs)
    tangent=next(c for c in costs if c['kind']=='ARCHIVED_TANGENT'and c['native_corner']==k)
    assert tangent['original_native_cost_kind']=='ARCHIVED_COMPLETE_BI_TANGENT'


def test_revoked_complete_readout_is_not_reported_as_precise_corrected_q():
    base,obs,kw,(k,_,_),R,t=fixture(False)
    problem=V._CompleteProblem(base,auth(base,obs,kw),overlap_policy=V.COMP.PROTOTYPE_OVERLAP_POLICY)
    task=dict(states=base.states,spec=dict(K=base.K.tolist(),visible=kw['visible_measurements']))
    selected=dict(family=dict(native_xyz=base.xyz.tolist()),accepted=True,
        complete_BI_replacement_audit=problem.plan)
    pose=dict(R_cf=R.tolist(),centroid=t.tolist())
    result=V._readout(task,[],selected,pose)
    assert result['reported_points'][k]is None and k in result['complete_BI_revoked_point_ids']
    assert 'UNCERTAIN_ARCHIVED_COMPLETE_TANGENT'in result['point_sources'][k]
    assert result['new_full_2D_corner_measurements']==0
    for j,state in enumerate(base.states):
        if state=='S':assert np.allclose(result['reported_points'][j],kw['initial_projection_xy'][j])


def test_changed_new_likelihood_does_not_rerank_original_native_starts():
    class Ranking:
        scale=1.
        def __init__(self,sign):self.sign=sign
        def evaluate(self,R,t):
            return dict(accepted=True,R=np.asarray(R).tolist(),t=np.asarray(t).tolist(),objective=self.sign*t[0])
    R=np.eye(3).tolist();proposals=[dict(name='A',R=R,t=[.1,0.,3.]),dict(name='B',R=R,t=[.2,0.,3.])]
    coarse=dict(R=R,t=[0.,0.,3.]);configuration=dict(maximum_native_ranked_starts=1)
    original=Ranking(1.);changed=Ranking(-1.)
    direct,_=V.REFINE.choose_native_starts(changed,proposals,coarse,configuration=configuration)
    fixed,_=V._choose_original_starts(SimpleNamespace(reference_start_problem=original),proposals,coarse,
        configuration=configuration)
    assert direct[0]['name']=='B'and fixed[0]['name']=='A'
    assert fixed[-1]['name']=='OWN_INITIAL_COARSE'


def test_actual_archive_join_binds_original_geometry_and_does_not_mutate_sources():
    base,obs,kw,_,R,t=fixture(False)
    identity=dict(id='unit:0',session='unit',partition='CAL',image='unused',image_sha256='a'*64)
    family=dict(native_xyz=base.xyz.tolist(),physical_registration_Q=base.Q.tolist(),
        coarse_pose=dict(R=R.tolist(),t=t.tolist()))
    task=dict(**identity,states=base.states,initial_self_hidden=kw['initial_self_hidden'],
        spec=dict(K=base.K.tolist(),image_hw=list(base.image_hw),visible=kw['visible_measurements'],domains=[]),
        families=dict(ORIGINAL_WD=family),u_regions=[],new_chain_observation=obs,
        O_localization_abstention=dict(records=[]))
    observation=dict(**identity,states=base.states,visible_measurements=kw['visible_measurements'],domains=[],
        u_regions=[],chain_observation=obs,junction_proposals=kw['junction_proposals'],observation_generation='unit',
        initial_geometry=dict(native_xyz=base.xyz.tolist(),physical_registration_Q=base.Q.tolist(),
            K=base.K.tolist(),image_hw=list(base.image_hw),initial_self_hidden=kw['initial_self_hidden'],
            initial_projection_xy=kw['initial_projection_xy']))
    before=V.E._digest(task);saved=V.E._digest(observation)
    joined=V._join(task,observation)
    assert joined['complete_BI_evidence']['initial_projection_xy']==kw['initial_projection_xy']
    assert V.E._digest(task)==before and V.E._digest(observation)==saved
    wrong=deepcopy(observation);wrong['initial_geometry']['initial_projection_xy'][0][0]+=1.
    with pytest.raises(ValueError,match='original-only model guide'):V._join(task,wrong)
