"""Pure objective/ownership/current support guard contracts; no real data."""
from copy import deepcopy
import numpy as np
import pytest

from . import geometry_guided_edge_problem_v4_29 as G
from .test_complete_bi_image_chain_evidence_v4_20 import fixture as native_fixture
from .test_complete_bi_image_chain_evidence_v4_20d import ray
from .common import finite


def fixture(partial=False, two=False):
    native,_,kw,(k,a,b),R,t=native_fixture(partial=partial)
    base=G.BASELINE.prepare_prepartial_native_problem(native,visible_measurements=kw['visible_measurements'],
        initial_self_hidden=kw['initial_self_hidden'])
    groups=[]
    for i,mate in enumerate([a,b] if two else [a]):
        chain=ray(native,kw['initial_projection_xy'],k,mate)['chains'][0]
        n=np.asarray(chain['normal']);actual=np.asarray(chain['observed_points']);c=chain['offset']
        center=actual.mean(axis=0);anchor=center-n*(n@center+c);axis=np.array([-n[1],n[0]])
        groups.append(dict(native_corner=k,native_neighbor=mate,normal_unit_xy=n.tolist(),offset_px=c,
            support_anchor_xy=anchor.tolist(),normal_variance_px2=9.,angular_variance_rad2=.001,
            actual_support_tangent_interval_px=[float((actual@axis).min()),float((actual@axis).max())],
            actual_inlier_xy=actual.tolist(),component_id='line'+str(i),image_quality_energy=float(i),
            activation_owners=[dict(native_corner=u,owner_provenance='SELF_ANCHOR_ACTUAL_SUPPORT'if u==k else
                'NAMED_PHYSICAL_EDGE_MODEL_CONSTRAINT_ONLY',
                fixed_tangent_sign=1 if axis@(np.asarray(kw['initial_projection_xy'][mate if u==k else k])-
                    np.asarray(kw['initial_projection_xy'][u]))>=0 else -1,
                order_provenance='IMMUTABLE_BI_ENDPOINT_HEMISPHERE')for u in [k,mate]if native.states[u]in('B','I')]))
    pairs=[dict(native_corner=k,component_ids=['line0','line1'],passed=True)]if two else []
    bundle=dict(observation_family=G.OBSERVATION_FAMILY,population=G.POPULATION,groups=groups,ordered_corner_pairs=pairs)
    return base,bundle,(k,a,b),R,t


def test_no_new_edge_preserves_exact_existing_v26_cost_and_guard_state():
    base,bundle,_,R,t=fixture(partial=True);bundle['groups']=[]
    problem=G.GeometryGuidedEdgeProblem(base,bundle,source_policy='ALL_BI_REPLACE',tangent_gamma=.2)
    p=G.V._pose_vector(R,t,base.scale)+[.01,-.02,.005,.008,-.011,.025]
    a,ja,_=base.objective(p);b,jb,_=problem.objective(p)
    assert a==b and np.array_equal(ja,jb)
    assert np.array_equal(base.constraints(p,base.sides[0])[0],problem.constraints(p,problem.sides[0])[0])
    assert np.array_equal(base.constraints(p,base.sides[0])[1],problem.constraints(p,problem.sides[0])[1])
    assert problem.guard_boundary==base.boundary and base.cost_boundary


def test_one_shared_edge_removes_both_BI_normal_costs_and_keeps_actual_axis_tangents_once():
    base,bundle,(k,mate,_),R,t=fixture(partial=True);before=deepcopy((base.complete,base.boundary,base.remaining_normals))
    problem=G.GeometryGuidedEdgeProblem(base,bundle,source_policy='ALL_BI_REPLACE',tangent_gamma=.2)
    p=G.V._pose_vector(R,t,base.scale)+[.01,-.02,.005,.008,-.011,.025]
    value,gradient,parts=problem.objective(p);owners={k,mate}
    assert {t['native_corner']for t in problem.tangents}==owners
    assert not any(c['native_corner']in owners and c['kind']in('COMPLETE_BI','UNCERTAIN_ARCHIVED_TANGENT','UNREPLACED_PARTIAL_NORMAL')for c in parts['native_costs'])
    assert sum(c['kind']=='PROJECTED_NATIVE_EDGE'for c in parts['native_costs'])==1
    assert sum(c['kind']=='ARCHIVED_Q_SELECTED_MEASURED_LINE_TANGENT'for c in parts['native_costs'])==2
    normal=np.asarray(bundle['groups'][0]['normal_unit_xy'])
    for tangent in problem.tangents:
        assert abs(tangent['axis']@normal)<1e-12
        archive=G._archive(base,tangent['native_corner']);cov=np.asarray(archive['original_effective_fit_covariance_px2'])
        assert tangent['variance']==pytest.approx(1.+.2*float(tangent['axis']@cov@tangent['axis']))
    assert value==pytest.approx(sum(c['weighted_robust_cost']for c in parts['native_costs']))
    numerical=G.V._jacobian(lambda x:np.array([problem.objective(x)[0]]),p)[0]
    assert np.allclose(gradient,numerical,rtol=4e-6,atol=4e-5)
    assert finite((base.complete,base.boundary,base.remaining_normals))==finite(before)


def test_RGB_preservation_overrides_other_endpoint_model_ownership():
    base,bundle,(k,mate,_),R,t=fixture(partial=True)
    problem=G.GeometryGuidedEdgeProblem(base,bundle,source_policy='RGB_NONFALLBACK_PRESERVE',tangent_gamma=.2)
    assert problem.plan['new_owner_ids']==[k]
    assert len(problem.groups)==1 and problem.groups[0]['activation_owners']==[bundle['groups'][0]['activation_owners'][0]]
    assert mate in [m['native_corner']for m in problem.base.complete]
    assert [r['native_corner']for r in problem.tangents]==[k]


def test_two_finite_ordered_directions_remove_q_without_creating_KP():
    base,bundle,(k,_,_),R,t=fixture(two=True)
    problem=G.GeometryGuidedEdgeProblem(base,bundle,source_policy='ALL_BI_REPLACE',tangent_gamma=.2)
    plan=next(p for p in problem.plan['corner_plans']if p['native_corner']==k)
    assert plan['mode']=='TWO_DIRECTION_LINES_NO_Q_COST'
    assert k not in [r['native_corner']for r in problem.tangents]
    assert k not in [m['native_corner']for m in problem.base.complete]
    assert not plan['new_precise_KP'] and problem.plan['ledger']['new_precise_2D_KPs']==0
    assert problem.evaluate(R,t)['accepted']


def test_failed_ordered_pair_falls_back_to_single_measured_direction():
    base,bundle,(k,_,_),R,t=fixture(two=True);bundle['ordered_corner_pairs'][0]['passed']=False
    p=G.GeometryGuidedEdgeProblem(base,bundle,source_policy='ALL_BI_REPLACE',tangent_gamma=.2)
    plan=next(p for p in p.plan['corner_plans']if p['native_corner']==k)
    assert plan['mode']=='ONE_SELECTED_LINE_ARCHIVED_Q_TANGENT'
    assert len(plan['selected_component_ids'])==1


def test_support_guard_uses_full_actual_points_and_analytic_J_not_old_q_cap():
    base,bundle,_,R,t=fixture(partial=True)
    problem=G.GeometryGuidedEdgeProblem(base,bundle,source_policy='ALL_BI_REPLACE',tangent_gamma=.2)
    p=G.V._pose_vector(R,t,base.scale)+[.01,-.02,.005,.008,-.011,.025]
    values,jac=problem.constraints(p,problem.sides[0]);numeric=G.V._jacobian(lambda x:problem.constraints(x,problem.sides[0])[0],p)
    assert np.allclose(jac,numeric,atol=2e-4,rtol=2e-5)
    assert not problem.guard_boundary
    assert all(a['uses_original_actual_support_not_rounded_exclusion']for a in problem._actual_support_constraints(p)[2])
    assert len(values)>0


def test_S_gate_precedes_edge_payload_and_non_BI_owner_fields():
    base,bundle,_,_,_=fixture();hidden=next(k for k,s in enumerate(base.states)if s=='S')
    class Forbidden(dict):
        def __getitem__(self,key):
            if key in ('native_corner','native_neighbor'):return super().__getitem__(key)
            raise AssertionError('Forbidden S edge payload read')
        def __deepcopy__(self,memo):raise AssertionError('Forbidden S copy')
    bundle['groups']=[Forbidden(native_corner=hidden,native_neighbor=(hidden+1)%8)]
    with pytest.raises(ValueError,match='S physical'):
        G.plan_geometry_guided_replacement(base,bundle,source_policy='ALL_BI_REPLACE')


def test_unknown_policy_or_noise_family_and_duplicate_physical_edge_reject():
    base,bundle,_,_,_=fixture()
    with pytest.raises(ValueError,match='source policy'):
        G.plan_geometry_guided_replacement(base,bundle,source_policy='per_image_choice')
    bad=deepcopy(bundle);bad['observation_family']='old17c'
    with pytest.raises(ValueError,match='generation'):
        G.plan_geometry_guided_replacement(base,bad,source_policy='ALL_BI_REPLACE')
    bundle['groups'].append(deepcopy(bundle['groups'][0]))
    with pytest.raises(ValueError,match='One selected'):
        G.plan_geometry_guided_replacement(base,bundle,source_policy='ALL_BI_REPLACE')


def test_fixed_measured_order_cannot_be_rescued_by_flipping_model_endpoints(monkeypatch):
    base,bundle,(k,mate,_),R,t=fixture()
    problem=G.GeometryGuidedEdgeProblem(base,bundle,source_policy='ALL_BI_REPLACE',tangent_gamma=.2)
    p=G.V._pose_vector(R,t,base.scale)
    q,depth,jac,extra=G.J.projection_and_jacobian(p,base.xyz,base.K,base.scale)
    q=q.copy();q[[k,mate]]=q[[mate,k]]
    monkeypatch.setattr(G.J,'projection_and_jacobian',lambda *args:(q,depth,jac,extra))
    _,_,audit=problem._actual_support_constraints(p)
    assert any(d['projected_endpoint_order_slack_px']<0 for a in audit for d in a['directed_support'])
    assert not audit[0]['passed']


def test_explicit_model_only_ambiguous_order_is_finite_partial_not_pose_inferred():
    base,bundle,_,R,t=fixture()
    for owner in bundle['groups'][0]['activation_owners']:
        owner.update(fixed_tangent_sign=None,order_ambiguous=True)
        owner.pop('order_provenance',None)
    problem=G.GeometryGuidedEdgeProblem(base,bundle,source_policy='ALL_BI_REPLACE',tangent_gamma=.2)
    _,_,audit=problem._actual_support_constraints(G.V._pose_vector(R,t,base.scale))
    assert all(not d['hard_order_guard_applied'] and d['order_ambiguous'] for d in audit[0]['directed_support'])
