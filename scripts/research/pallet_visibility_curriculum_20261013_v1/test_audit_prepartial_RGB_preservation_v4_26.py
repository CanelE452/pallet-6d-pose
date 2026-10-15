"""Independent restored-cost/parser arithmetic tests; synthetic only."""
from copy import deepcopy

import numpy as np
import pytest

from . import audit_prepartial_RGB_preservation_v4_26 as A
from . import prepartial_RGB_point_preservation_v4_26 as H
from .test_prepartial_RGB_point_preservation_v4_26 import partial_fixture,prepare,problem
from .test_source_preserving_complete_bi_evidence_v4_24 import fixture as complete_fixture
from .test_coarse_pose_prior_v4_25 import ingredients


def independent(base,obs,kw):
    # Old auth with new family provenance; no new helper factory/problem used.
    bundle=A._SOURCE_AUTH(base,obs,**kw)
    plan,restored=A._independent_plan(base,kw['visible_measurements'],bundle)
    return A.IndependentRestoredNativeProblem(base,plan,restored)


@pytest.mark.parametrize('source',['B_TARGET_BACKGROUND_LINES',H.FALLBACK_SOURCE,'UNKNOWN_SOURCE'])
def test_independent_direct_native_matches_production_proxy_and_guards(source):
    base,obs,kw,_,R,t=partial_fixture(source)
    expected=problem(prepare(base,kw),obs,kw);actual=independent(base,obs,kw)
    at=H.N.V._pose_vector(R,t,base.scale)+[.01,-.02,.005,.008,-.011,.025]
    ev,eg,ep=expected.objective(at);av,ag,ap=actual.objective(at)
    assert av==ev and np.array_equal(ag,eg)
    assert A.A24._native_cost_buckets(ep['native_costs'])==A.A24._native_cost_buckets(ap['native_costs'])
    assert A.A24._plan_signature(actual.plan)==A.A24._plan_signature(expected.plan)
    v,j=actual.constraints(at,actual.sides[0]);vv,jj=base.constraints(at,base.sides[0])
    assert np.array_equal(v,vv) and np.array_equal(j,jj)
    assert actual.evaluate(R,t)['accepted']==base.evaluate(R,t)['accepted']


def test_shared_fallback_owner_stays_once_while_restored_owner_costs_removed():
    base,obs,kw,(k,mate,_),R,t=partial_fixture()
    other=deepcopy(obs['edge_records'][0]);other.update(native_corner=mate,native_neighbor=k)
    chain=other['chains'][0];chain.update(native_corner=mate,native_neighbor=k,native_edge=[mate,k],
        corrected_endpoint_normal_residual_px={str(mate):0.})
    if base.states[mate]=='I':
        chain['mask_context_kind']='TARGET_TARGET_INTERNAL';chain['mask_minus']=[.99]*len(chain['observed_points'])
    obs['edge_records'].append(other)
    next(r for r in kw['visible_measurements'] if r['native_corner']==mate)['source']=H.FALLBACK_SOURCE
    expected=problem(prepare(base,kw),obs,kw);actual=independent(base,obs,kw)
    at=H.N.V._pose_vector(R,t,base.scale)+[.01,-.02,.005,.008,-.011,.025]
    # TCAL is an algebraic variance replacement, not a new point or weight.
    for plan in (expected.plan,actual.plan):
        for r in plan['corner_plans']:
            if 'tangent_kind' not in r:continue
            axis=np.asarray(r['tangent_unit_xy'])@np.asarray(r['original_measurement_archive']['original_effective_fit_covariance_px2'])@np.asarray(r['tangent_unit_xy'])
            r['tangent_variance_px2']=1.+1.7*axis
    assert actual.objective(at)[0]==expected.objective(at)[0]
    assert np.array_equal(actual.objective(at)[1],expected.objective(at)[1])
    assert len(actual.plan['active_groups'])==1 and actual.restored=={k}
    assert actual.plan['active_groups'][0]['activation_owners']==[dict(kind='ROLE_CORRECT_SELF_COMPLETE_ANCHOR',native_corner=mate)]
    numeric=H.N.V._jacobian(lambda x:np.array([actual.objective(x)[0]]),at)[0]
    assert np.allclose(actual.objective(at)[1],numeric,atol=4e-5,rtol=4e-6)


def test_current_complete_fallback_revocation_and_TCAL_are_not_undone_by_restoration():
    base,obs,kw,(k,_,_),R,t=complete_fixture(fallback=True)
    kw['noise_provenance']['observation_family']=H.NOISE_OBSERVATION_FAMILY
    expected=problem(prepare(base,kw),obs,kw);actual=independent(base,obs,kw)
    for plan in (expected.plan,actual.plan):
        for r in plan['corner_plans']:
            if 'tangent_kind' in r:r['tangent_variance_px2']=1.+1.7*(r['tangent_variance_px2']/4.)
    at=H.N.V._pose_vector(R,t,base.scale)+[.01,-.02,.005,.008,-.011,.025]
    assert actual.objective(at)[0]==expected.objective(at)[0]
    costs=actual.objective(at)[2]['native_costs']
    assert not any(c['kind']=='COMPLETE_BI' and c['native_corner']==k for c in costs)
    assert sum(c['kind']=='ARCHIVED_COMPLETE_BI_TANGENT' and c['native_corner']==k for c in costs)==1


@pytest.mark.parametrize('mode',['FULL','TRANSLATION'])
def test_direct_old_P1_zero_placeholder_or_TP_weight_is_applied_exactly_once(mode):
    xyz,coarse,covariance,bias,_,_=ingredients(mode)
    task=dict(families={'ORIGINAL_WD':dict(native_xyz=xyz.tolist(),coarse_pose=coarse,
        physical_registration_Q=coarse['physical_registration_Q'])})
    model=dict(prior_covariance6=covariance.tolist(),prior_fit=dict(median_bias6=bias.tolist()))
    fields=A._original_prior_fields(task,model,'a'*64,mode)
    scale=np.linalg.norm(np.ptp(xyz,axis=0));p=np.r_[[.4,-.2,.15],np.array([.3,-.1,2.8])/scale]
    v,g,audit=A.direct_old_P1_prior(p,xyz,scale,fields,coarse['physical_registration_Q'])
    if mode=='FULL':
        assert fields['factor_weight']==0. and v==0. and np.array_equal(g,np.zeros(6))
        assert audit['FULL_schema_is_disabled_placeholder']
    else:
        old,oldg,_=A.PREVIOUS.direct_old_P1_prior(p,xyz,scale,fields,coarse['physical_registration_Q'])
        assert fields['factor_weight']==1. and v==old and np.array_equal(g,oldg)
    assert A._source_binding(fields)!=A._source_binding({**fields,'factor_weight':1.-fields['factor_weight']})


@pytest.mark.parametrize('accepted',[True,False])
def test_restored_readout_is_exact_archived_point_including_fallback_S_stays_pose_only(accepted):
    base,obs,kw,_,R,t=partial_fixture();p=problem(prepare(base,kw),obs,kw)
    family=dict(name='ORIGINAL_WD',native_xyz=base.xyz.tolist(),cf_extents=np.ptp(base.xyz,axis=0).tolist(),physical_registration_Q=base.Q.tolist())
    task=dict(states=list(base.states),initial_self_hidden=kw['initial_self_hidden'],boundary_merged=base.boundary,
        spec=dict(visible=kw['visible_measurements'],K=base.K.tolist()),families={'ORIGINAL_WD':family})
    selected=dict(accepted=accepted,family=family)
    pose=A.V.BASE._actual_pose(family,R,t,accepted,'synthetic')
    readout=A.V._readout(task,selected,pose,dict(runtime_complete_BI_replacement_plan=p.plan))
    result=dict(WD_family='ORIGINAL_WD',actual_pose=pose,keypoint_readout=readout,accepted=accepted)
    assert all(ok for ok,_ in A._model_readout_checks(task,result))
    restored=p.plan['restored_prepartial_complete_ids'][0]
    result['keypoint_readout']['reported_points'][restored]=None
    assert not all(ok for ok,_ in A._model_readout_checks(task,result))
