"""Pure driver wiring, frozen archives, and family-before-auth contracts."""
from copy import deepcopy
import numpy as np
import pytest

from . import evaluate_source_preserving_tangent_mask_v4_24 as NEW
from .test_complete_bi_image_chain_evidence_v4_20d import fixture,auth


@pytest.fixture(scope='module')
def tangent_model():
    def row(fid,session,error,axis):
        return dict(id=fid,session=session,partition='CAL',compatible=True,tangents=[dict(native_corner=0,state='B',
            source='VISIBLE_BACKBONE_FALLBACK',source_group='BACKBONE_FALLBACK_COMPLETE',
            signed_tangent_error_px=error,original_variance_axis_px2=axis,exact_saved_old_tangent_variance_px2=4.*axis)])
    return NEW.TCAL.fit_archived_complete_tangent_model([row('a','A',2.,9.),row('b','B',3.,16.)])


def all_problem(with_legacy=False):
    base,obs,kw,(k,a,b),R,t=fixture(False)
    if with_legacy:
        records=deepcopy(kw['visible_measurements'])
        j=next(i for i,s in enumerate(base.states)if s=='B'and i!=k)
        mate=next(m for m in NEW.V.COMP.BROAD.NEIGHBORS[j]if base.states[m]!='S')
        measurement=next(r for r in records if r['native_corner']==j);measurement['source']='VISIBLE_BACKBONE_FALLBACK'
        P=np.asarray(kw['initial_projection_xy']);tangent=P[mate]-P[j];tangent/=np.linalg.norm(tangent)
        normal=np.array([-tangent[1],tangent[0]]);along=float(tangent@P[j])
        boundary=[dict(native_corner=j,state='B',fit_reliability_weight=measurement['fit_reliability_weight'],
            replaces_raw_B_likelihood=True,image_binding=dict(sha256='a'*64,image_hw=[480,640]),
            association_provenance=dict(original_raw_xy=measurement['xy']),lines=[dict(normal_unit_xy=normal.tolist(),
            offset_px=-float(normal@P[j]),normal_variance_px2=25.,support_tangent_interval_px=[along-100.,along+100.],
            support_extrapolation_px=11.,provenance=dict(native_neighbor=mate))])]
        base=NEW.V.COMP.N.prepare_native_edge_problem(base.xyz,base.K,base.states,records,[],base.image_hw,
            dict(R=R.tolist(),t=t.tolist(),physical_registration_Q=np.eye(3).tolist()),
            config=dict(prior_weight=0.,analytic_jacobian=True),boundary_observations=boundary,u_regions=[],
            edge_bundle=dict(edges=[],replaces_normal_factor_refs=[]))
        kw['visible_measurements']=records
    bundle=auth(base,obs,kw)
    result=NEW._AllProblem(base,bundle,overlap_policy=NEW.V.COMP.PROTOTYPE_OVERLAP_POLICY)
    assert sum('tangent_kind'in r for r in result.plan['corner_plans'])==1
    return result,base,k,R,t


def test_CAL_tangent_uses_original_axis_not_old_four_axis_and_preserves_archives(tangent_model):
    p,base,k,_,_=all_problem(with_legacy=True)
    snapshot=NEW.E._digest(dict(measurements=base.measurements,boundary=base.boundary,
        archives=[r['original_measurement_archive']for r in p.plan['corner_plans']]))
    legacy_before=NEW.E._digest(NEW.V.COMP.N.T._tangent_records(base.boundary,4.))
    assert NEW.V.COMP.N.T._tangent_records(base.boundary,4.)
    record=next(r for r in p.plan['corner_plans']if r['native_corner']==k)
    old=record['tangent_variance_px2'];axis=np.asarray(record['tangent_unit_xy']);cov=np.asarray(record['original_measurement_archive']['original_effective_fit_covariance_px2'])
    a=float(axis@cov@axis);assert old==pytest.approx(4.*a)
    NEW._calibrate_plan_tangents(p,tangent_model)
    assert record['tangent_variance_px2']==pytest.approx(1.+tangent_model['gamma']*a)
    assert not np.isclose(record['tangent_variance_px2'],1.+tangent_model['gamma']*old)
    assert record['legacy_uncalibrated_variance_px2']==old and record['legacy_uncalibrated_variance_inflation']==4.
    assert 'tangent_variance_inflation'not in record
    assert NEW.E._digest(dict(measurements=base.measurements,boundary=base.boundary,
        archives=[r['original_measurement_archive']for r in p.plan['corner_plans']]))==snapshot
    assert NEW.E._digest(NEW.V.COMP.N.T._tangent_records(base.boundary,4.))==legacy_before
    assert p.plan['original_q_covariance_reliability_guide_unchanged']and p.plan['old_partial_T4_unchanged']
    with pytest.raises(ValueError,match='original archived-complete tangent'):NEW._calibrate_plan_tangents(p,tangent_model)


def test_AUTH_bridge_preserves_costs_and_each_archived_complete_tangent_once(tangent_model):
    p,base,k,R,t=all_problem(with_legacy=True);NEW._calibrate_plan_tangents(p,tangent_model)
    at=NEW.BASE.NATIVE.V._pose_vector(R,t,p.scale)+[.01,-.02,.005,.008,-.011,.025]
    value,_,audit=p.objective(at);costs=audit['native_costs'];bridged=NEW.BASE._bridge_costs(costs)
    complete=[x for x in costs if x['kind']=='ARCHIVED_COMPLETE_BI_TANGENT']
    after=[x for x in bridged if x.get('original_native_cost_kind')=='ARCHIVED_COMPLETE_BI_TANGENT']
    assert len(complete)==len(after)==1 and complete[0]['native_corner']==after[0]['native_corner']==k
    assert after[0]['kind']=='ARCHIVED_TANGENT'
    assert len(costs)==len(bridged)
    assert sum(x['weighted_robust_cost']for x in bridged)==pytest.approx(value)
    for before,after in zip(costs,bridged):assert before['weighted_robust_cost']==after['weighted_robust_cost']
    candidate=dict(name='one',family='ORIGINAL_WD',accepted=True,all_joint_constraints_satisfied=True,
        native_costs=bridged,global_mask_cost=0.)
    selected=NEW.P.AUTH.select_hierarchical_native_candidates(base.states,[candidate])
    assert selected['selected_name']=='one'
    assert selected['candidates'][0]['native_energy']==pytest.approx(value)


STATES=['B','I','U','O','T','B','S','S']


def candidate(name,family,costs=(1.,1.),mask=0.):
    return dict(name=name,family=family,accepted=True,all_joint_constraints_satisfied=True,
        native_costs=[dict(kind='BI_COMPLETE',native_corner=k,weighted_robust_cost=v)for k,v in zip((0,1),costs)],
        regularizer_costs=[dict(kind=kind,weighted_cost=0.)for kind in NEW.P.REGULARIZER_KINDS],
        global_mask_cost=mask,metrics=object(),actual_pose=object(),frame_id=object())


def test_locked_family_filters_all_raw_candidates_before_AUTH_even_if_globally_dominated():
    rows=[candidate('global_winner','ORIGINAL_WD',(1.,1.)),candidate('allowed_dominated','SWAPPED_WD',(2.,2.))]
    global_audit=NEW.P.AUTH.select_hierarchical_native_candidates(STATES,rows)
    assert global_audit['pareto_frontier_indices']==[0]
    locked=NEW._select_locked(STATES,rows,family='SWAPPED_WD',mask_weight=1.)
    assert locked['selected_name']=='allowed_dominated'and locked['selected_family']=='SWAPPED_WD'
    assert locked['eligible_family_candidate_count']==1
    assert locked['filter_applied_to_all_candidates_before_authentication']
    assert locked['global_native_Pareto_not_used_as_family_prefilter']and not locked['e_numeric_pose_used']
    class Poison:
        def __iter__(self):raise AssertionError('Other-family native fields authenticated before family gate')
    rows[0]['native_costs']=Poison()
    assert NEW._select_locked(STATES,rows,family='SWAPPED_WD',mask_weight=1.)['selected_name']=='allowed_dominated'
    rejected=dict(name='rejected',family='SWAPPED_WD',accepted=False,native_costs=Poison(),global_mask_cost=Poison())
    assert NEW._select_locked(STATES,rows+[rejected],family='SWAPPED_WD',mask_weight=1.)['selected_name']=='allowed_dominated'


def test_absent_locked_family_returns_no_selection_without_old_pose_fallback():
    selected=NEW._select_locked(STATES,[candidate('old','ORIGINAL_WD')],family='SWAPPED_WD',mask_weight=1.)
    assert selected['selected_name']is None and selected['selected_family']is None
    assert selected['candidates']==[]and selected['eligible_family_candidate_count']==0
    assert selected['old_best_pose_fallback']is False and selected['e_numeric_pose_used']is False
    assert not any(k in selected for k in ('actual_pose','R','t'))


@pytest.mark.parametrize('family',['ORIGINAL','SWAPPED',None,True])
def test_invalid_locked_family_fails_before_candidate_or_pose_access(family):
    class Poison:
        def __iter__(self):raise AssertionError('Candidates accessed before invalid family rejection')
    with pytest.raises(ValueError,match='Stored predicted e WD family'):
        NEW._select_locked(STATES,Poison(),family=family,mask_weight=1.)
