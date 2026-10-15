"""Synthetic objective/source/ownership/guard contracts; no real data or fits."""
from copy import copy,deepcopy

import numpy as np
import pytest

from . import prepartial_RGB_point_preservation_v4_26 as V
from .test_complete_bi_image_chain_evidence_v4_20 import fixture as base_fixture,Poison
from .test_complete_bi_image_chain_evidence_v4_20d import ray
from .test_source_preserving_complete_bi_evidence_v4_24 import fixture as complete_fixture
from .common import finite


def partial_fixture(source='B_ACTUAL_BOUNDARY_RECOVERY'):
    base,obs,kw,ids,R,t=base_fixture(partial=True)
    k,mate,_=ids
    next(r for r in kw['visible_measurements'] if r['native_corner']==k)['source']=source
    kw['noise_provenance']['observation_family']=V.NOISE_OBSERVATION_FAMILY
    obs['configuration']=deepcopy(V.OBS.DEFAULT_CONFIGURATION)
    obs['edge_records']=[ray(base,kw['initial_projection_xy'],k,mate)]
    kw['junction_proposals']=dict(schema='NATIVE_EDGE_TOPOLOGY_JUNCTION_PROPOSALS_V4_17B',
        configuration=deepcopy(V.OBS.DEFAULT_CONFIGURATION),image_binding=deepcopy(obs['image_binding']),
        mask_binding=dict(image_sha256=kw['image_binding']['sha256'],mask_pixel_sha256='e'*64),
        records=[dict(native_corner=i,state=s,candidates=[]) for i,s in enumerate(base.states)])
    return base,obs,kw,ids,R,t


def prepare(base,kw):
    return V.prepare_prepartial_native_problem(base,visible_measurements=kw['visible_measurements'],
        initial_self_hidden=kw['initial_self_hidden'])


def problem(proxy,obs,kw):
    bundle=V.authenticate_complete_bi_image_chain_groups(proxy,obs,**kw)
    return V.CompleteBIImageChainProblem(proxy,bundle,overlap_policy=V.PROTOTYPE_OVERLAP_POLICY)


@pytest.mark.parametrize('source',['B_ACTUAL_BOUNDARY_RECOVERY','B_TARGET_BACKGROUND_LINES',
    'I_INTERNAL_RGB_JUNCTION','UNKNOWN_SOURCE','visible_backbone_fallback'])
def test_nonfallback_partial_restored_as_exact_2D_and_owned_costs_removed(source):
    base,obs,kw,(k,_,_),R,t=partial_fixture(source)
    before=deepcopy((kw,base.complete,base.boundary,base.remaining_normals))
    proxy=prepare(base,kw);p=problem(proxy,obs,kw)
    assert p.plan['restored_prepartial_complete_ids']==[k]
    assert not p.groups and not p.plan['replaces_normal_factor_refs']
    assert not p.revoked and k in [m['native_corner'] for m in p.complete]
    at=V.N.V._pose_vector(R,t,base.scale)+[.01,-.02,.005,.008,-.011,.025]
    value,gradient,parts=p.objective(at)
    assert sum(c['kind']=='COMPLETE_BI' and c['native_corner']==k for c in parts['native_costs'])==1
    assert not any(c['native_corner']==k and c['kind'] in ('UNCERTAIN_ARCHIVED_TANGENT','UNREPLACED_PARTIAL_NORMAL',
        'ARCHIVED_COMPLETE_BI_TANGENT','PROJECTED_NATIVE_EDGE') for c in parts['native_costs'])
    expected=copy(base);expected.complete=list(base.measurements);expected.boundary=[];expected.remaining_normals=[]
    ev,eg,_=V.N.NativeEdgeProblem.objective(expected,at)
    assert value==ev and np.array_equal(gradient,eg)
    assert finite((kw,base.complete,base.boundary,base.remaining_normals))==finite(before)


def test_fallback_partial_keeps_old_normal_tangent_and_auth_owned_edge():
    base,obs,kw,(k,_,_),R,t=partial_fixture(V.FALLBACK_SOURCE)
    p=problem(prepare(base,kw),obs,kw)
    assert not p.plan['restored_prepartial_complete_ids'] and len(p.groups)==1
    at=V.N.V._pose_vector(R,t,base.scale)+[.01,-.02,.005,.008,-.011,.025]
    _,_,parts=p.objective(at)
    assert not any(c['kind']=='COMPLETE_BI' and c['native_corner']==k for c in parts['native_costs'])
    assert sum(c['kind']=='UNCERTAIN_ARCHIVED_TANGENT' and c['native_corner']==k for c in parts['native_costs'])==1
    assert p.plan['replaces_normal_factor_refs']==[dict(native_corner=k,line_index=0)]


def test_current_constructor_complete_revocations_are_used_not_a_stale_cost_view():
    base,obs,kw,(k,_,_),R,t=complete_fixture(fallback=True)
    kw['noise_provenance']['observation_family']=V.NOISE_OBSERVATION_FAMILY
    p=problem(prepare(base,kw),obs,kw)
    assert k in p.revoked and k not in [m['native_corner'] for m in p.complete]
    at=V.N.V._pose_vector(R,t,base.scale)+[.01,-.02,.005,.008,-.011,.025]
    _,_,parts=p.objective(at)
    assert not any(c['kind']=='COMPLETE_BI' and c['native_corner']==k for c in parts['native_costs'])
    assert sum(c['kind']=='ARCHIVED_COMPLETE_BI_TANGENT' and c['native_corner']==k for c in parts['native_costs'])==1
    numeric=V.N.V._jacobian(lambda x:np.array([p.objective(x)[0]]),at)[0]
    assert np.allclose(p.objective(at)[1],numeric,atol=4e-5,rtol=4e-6)


def test_restored_owner_removed_but_other_fallback_owner_keeps_shared_edge_once():
    base,obs,kw,(k,mate,_),R,t=partial_fixture()
    other=deepcopy(obs['edge_records'][0]);other.update(native_corner=mate,native_neighbor=k)
    chain=other['chains'][0]
    chain.update(native_corner=mate,native_neighbor=k,native_edge=[mate,k],
        corrected_endpoint_normal_residual_px={str(mate):0.})
    if base.states[mate]=='I':
        chain['mask_context_kind']='TARGET_TARGET_INTERNAL';chain['mask_minus']=[.99]*len(chain['observed_points'])
    obs['edge_records'].append(other)
    next(r for r in kw['visible_measurements'] if r['native_corner']==mate)['source']=V.FALLBACK_SOURCE
    p=problem(prepare(base,kw),obs,kw)
    assert len(p.groups)==1 and p.plan['replaces_normal_factor_refs']==[]
    assert p.groups[0]['activation_owners']==[dict(kind='ROLE_CORRECT_SELF_COMPLETE_ANCHOR',native_corner=mate)]
    assert p.groups[0]['guard_authorized_original_normal_factor_refs']==[dict(native_corner=k,line_index=0)]
    assert p.plan['prepartial_owner_removals']
    assert p.plan['ledger']['restored_RGB_q_and_other_owner_edge_may_remain_correlated']
    at=V.N.V._pose_vector(R,t,base.scale)+[.01,-.02,.005,.008,-.011,.025]
    _,_,parts=p.objective(at)
    assert sum(c['kind']=='PROJECTED_NATIVE_EDGE' for c in parts['native_costs'])==1
    assert sum(c['kind']=='COMPLETE_BI' and c['native_corner']==k for c in parts['native_costs'])==1
    assert not any(c['kind']=='UNCERTAIN_ARCHIVED_TANGENT' and c['native_corner']==k for c in parts['native_costs'])


def test_full_original_boundary_hardguard_values_jacobians_and_audits_stay_exact():
    base,obs,kw,(k,_,_),R,t=partial_fixture()
    p=problem(prepare(base,kw),obs,kw)
    at=V.N.V._pose_vector(R,t,base.scale)+[.01,-.02,.005,.008,-.011,.025]
    for sides in base.sides:
        v,j=base.constraints(at,sides);vv,jj=p.constraints(at,sides)
        assert np.array_equal(v,vv) and np.array_equal(j,jj)
    old,new=base.evaluate(R,t),p.evaluate(R,t)
    assert new['accepted']==old['accepted']
    for key in ('visible_protection_audit','joint_constraint_audit','U_region_constraint_audit','boundary_observation_audit'):
        assert new[key]==old[key]
    assert p.boundary==base.boundary and len(p.boundary)==1 and not p.base.cost_boundary
    assert new['complete_BI_likelihood_count']==len(base.complete)+1


def test_restored_RGB_objective_gradient_and_ledger_exact_sum():
    base,obs,kw,_,R,t=partial_fixture();p=problem(prepare(base,kw),obs,kw)
    at=V.N.V._pose_vector(R,t,base.scale)+[.01,-.02,.005,.008,-.011,.025]
    value,gradient,parts=p.objective(at)
    numeric=V.N.V._jacobian(lambda x:np.array([p.objective(x)[0]]),at)[0]
    assert np.allclose(gradient,numeric,atol=4e-5,rtol=4e-6)
    assert value==pytest.approx(sum(c['weighted_robust_cost'] for c in parts['native_costs']))
    assert parts['complete_BI_likelihood_count']==p.plan['ledger']['original_complete_q_factors_once']
    assert parts['replaced_old_normal_factor_count']==len(p.plan['replaces_normal_factor_refs'])


@pytest.mark.parametrize('change',['xy','covariance','reliability','source'])
def test_current_archive_cannot_change_after_proxy_preparation(change):
    base,obs,kw,(k,_,_),_,_=partial_fixture();proxy=prepare(base,kw)
    row=next(r for r in kw['visible_measurements'] if r['native_corner']==k)
    if change=='xy':row['xy'][0]+=.1
    elif change=='covariance':row['fit_covariance_px2'][0][0]+=.1
    elif change=='reliability':row['fit_reliability_weight']+=.01
    else:row['source']='VISIBLE_BACKBONE_FALLBACK'
    with pytest.raises(ValueError):V.authenticate_complete_bi_image_chain_groups(proxy,obs,**kw)


def test_initial_S_and_nonBI_gates_precede_payload_fields_and_double_policy_rejects():
    base,obs,kw,_,_,_=partial_fixture();hidden=next(i for i,s in enumerate(base.states) if s=='S')
    kw['visible_measurements'].append(dict(native_corner=hidden,xy=Poison(),source=Poison()))
    proxy=prepare(base,kw)
    assert hidden not in proxy.restored_prepartial_complete_ids
    with pytest.raises(ValueError):prepare(proxy,kw)
    base2,obs2,kw2,_,_,_=partial_fixture();outside=next(i for i,s in enumerate(base2.states) if s=='S')
    base2.states=list(base2.states);base2.states[outside]='U';kw2['initial_self_hidden'][outside]=False
    kw2['visible_measurements'].append(dict(native_corner=outside,xy=Poison(),source=Poison()))
    assert outside not in prepare(base2,kw2).restored_prepartial_complete_ids


def test_calibration_preparation_path_uses_new_family_and_forbids_runtime_until_fit():
    base,obs,kw,_,_,_=partial_fixture();proxy=prepare(base,kw)
    old=kw['noise_provenance']['observation_family'];kw['noise_provenance']['observation_family']=V.SOURCE.NOISE_OBSERVATION_FAMILY
    with pytest.raises(ValueError):V.authenticate_complete_bi_image_chain_groups(proxy,obs,**kw)
    kw['noise_provenance']['observation_family']=old;kw['calibration_preparation']=True
    bundle=V.authenticate_complete_bi_image_chain_groups(proxy,obs,**kw)
    plan=V.plan_complete_bi_replacement(proxy,bundle,overlap_policy=V.PROTOTYPE_OVERLAP_POLICY)
    assert plan['new_observation_population_requires_separate_CAL_model']
    with pytest.raises(ValueError,match='Fit/freeze'):V.CompleteBIImageChainProblem(proxy,bundle,overlap_policy=V.PROTOTYPE_OVERLAP_POLICY)
