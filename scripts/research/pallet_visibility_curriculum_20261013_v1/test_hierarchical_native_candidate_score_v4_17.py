"""Pure metadata/selection contracts; no images, GT or pose fitting."""
from copy import deepcopy
import numpy as np
import pytest

from . import hierarchical_native_candidate_score_v4_17 as NEW
from .common import cuboid
from .test_statewise_partial_boundary_u_solver_v4_6 import arguments, observation
from .test_statewise_partial_BI_u_solver_v4_13 import typed_fixture


def fixture(typed=False):
    if typed:
        a, b = typed_fixture('I')
    else:
        a = arguments(); b = observation(a)
    a = list(a)
    a[0] = cuboid(.8, .44, .18)
    k = b['native_corner']
    mate = next(m for m in NEW.NEIGHBORS[k] if a[2][m] != 'S')
    n = np.asarray(b['lines'][0]['normal_unit_xy']); c = b['lines'][0]['offset_px']
    tangent = np.array([-n[1], n[0]])
    centre = np.asarray(a[3][0]['xy'])
    centre -= n * (n@centre+c)
    points = centre + np.linspace(-12, 12, 15)[:, None]*tangent
    interval = [float((points@tangent).min()), float((points@tangent).max())]
    summary = dict(native_neighbor=mate, actual_observed_inlier_tangent_interval_px=interval,
        inlier_count=15, inlier_fraction=1., inlier_span_fraction=1., inlier_rms_px=.2,
        rgb_supported_sample_count=15, legacy_mask_only_line_is_not_used_as_RGB_evidence=True,
        physical_native_edge=sorted((k, mate)))
    if typed:
        summary['line_record'] = dict(native_corner=k, native_neighbor=mate, endpoints=[k, mate],
            supported=True, measurement_source='OBSERVED_LOCAL_ANCHORED_FACE_RGB_WITH_MASK_CONTEXT',
            normal=n.tolist(), offset=c, observed_points=points.tolist(), inlier_mask=[True]*15,
            inlier_count=15, inlier_fraction=1., inlier_span_fraction=1., inlier_rms_px=.2,
            RGB_supported_sample_count=15, mask_context_kind='TARGET_TARGET_INTERNAL')
    b['lines'][0]['provenance'] = summary
    kw = dict(initial_self_hidden=[s=='S' for s in a[2]], image_binding=deepcopy(b['image_binding']),
        angular_variance_rad2=1e-9, angular_variance_provenance={'fit_partition':'CAL','model_sha256':'b'*64},
        config=a[7])
    return a, b, kw


def auth(a, b, kw):
    return NEW.authenticate_supported_native_edges(a[0], a[2], a[3], b, a[5], **kw)


@pytest.mark.parametrize('typed', [False, True])
def test_legacy_and_typed_authentication_support_not_precise_point_floor_and_immutability(typed):
    a, b, kw = fixture(typed); saved = deepcopy(b)
    r = auth(a, [b], kw); edge = r['edges'][0]
    assert b == saved and r['RGB_queries'] == r['S_coordinate_reads'] == r['U_coordinate_reads'] == 0
    assert edge['angular_variance_rad2'] == pytest.approx(1e-9 + 1./24**2)
    assert edge['provenance']['support_anchor_is_fitted_support_midpoint_not_actual_pixel_or_KP']
    assert edge['provenance']['actual_inlier_points_present'] == typed
    n = np.array(edge['normal_unit_xy']); p = np.array(edge['support_anchor_xy'])
    assert abs(n@p+edge['offset_px']) < 1e-10
    assert r['replaces_normal_factor_refs'] == [{'native_corner':b['native_corner'], 'line_index':0}]
    assert r['native_identity_or_physical_visibility_proved'] is False


def test_exact_typed_archive_and_actual_TLS_binding_reject_tampering():
    a, b, kw = fixture(True)
    altered = deepcopy(b); altered['association_provenance']['original_observed_xy'][0] += .1
    with pytest.raises(ValueError, match='archive'):
        auth(a, [altered], kw)
    altered = deepcopy(b); altered['lines'][0]['provenance']['line_record']['offset'] += .1
    with pytest.raises(ValueError, match='coefficients'):
        auth(a, [altered], kw)


def test_initial_S_and_nonBI_anchor_gates_before_poison_fields():
    for state in ('S', 'U', 'O', 'T'):
        a, b, kw = fixture(); k = b['native_corner']; a[2][k] = state
        a[3] = [{'native_corner':k, 'xy':object(), 'covariance_px2':object()}]
        bad = {'native_corner':k, 'lines':object(), 'association_provenance':object()}
        result = auth(a, [bad], kw)
        assert result['edges'] == [] and result['skipped']
    a, b, kw = fixture(); k=b['native_corner']; kw['initial_self_hidden'][k]=True
    assert auth(a, [{'native_corner':k,'lines':object()}], kw)['edges'] == []


def test_S_mate_gate_precedes_line_coefficient_and_support_reads():
    a, b, kw = fixture(); mate=b['lines'][0]['provenance']['native_neighbor']
    kw['initial_self_hidden'][mate]=True
    b['lines']=[{'provenance':{'native_neighbor':mate},'normal_unit_xy':object(),'offset_px':object()}]
    assert auth(a, [b], kw)['edges'] == []


def test_shared_edge_once_preserves_existing_inflation_and_other_normal_reference():
    a, b, kw = fixture(); k=b['native_corner']
    mate = next(m['native_corner'] for m in a[3] if m['native_corner'] in NEW.NEIGHBORS[k])
    a[2][mate] = 'B'
    next(m for m in a[3] if m['native_corner']==mate)['state'] = 'B'
    b['lines'][0]['provenance'].update(native_neighbor=mate, physical_native_edge=sorted((k,mate)))
    second=deepcopy(b); second['native_corner']=mate
    second['association_provenance']['original_raw_xy']=next(m for m in a[3] if m['native_corner']==mate)['xy']
    second['lines'][0]['provenance']['native_neighbor']=k
    second['lines'][0]['normal_variance_px2']=50.
    second['fit_reliability_weight']=.7
    next(m for m in a[3] if m['native_corner']==mate)['fit_reliability_weight']=.7
    result=auth(a,[b,second],kw); edge=result['edges'][0]
    assert len(result['edges'])==1 and len(edge['replaces_normal_factor_refs'])==2
    assert edge['normal_variance_px2']==50. and edge['fit_reliability_weight']==.7
    assert edge['provenance']['shared_RGB_independence_claimed'] is False


@pytest.mark.parametrize('change', ['mask_only','low_RGB','negative_angle','non_CAL','wrong_edge'])
def test_invalid_rgb_uncertainty_or_native_correspondence_rejected(change):
    a,b,kw=fixture(True)
    if change=='mask_only':b['lines'][0]['provenance']['line_record']['measurement_source']='MODEL_ONLY'
    elif change=='low_RGB':b['lines'][0]['provenance']['line_record']['RGB_supported_sample_count']=5
    elif change=='negative_angle':kw['angular_variance_rad2']=-1.
    elif change=='non_CAL':kw['angular_variance_provenance']['fit_partition']='TEST'
    else:b['lines'][0]['provenance']['native_neighbor']=next(i for i in range(8) if i!=b['native_corner'] and i not in NEW.NEIGHBORS[b['native_corner']])
    with pytest.raises(ValueError):auth(a,[b],kw)


def test_zero_CAL_placeholder_plus_pixel_floor_is_not_a_precise_angle():
    a,b,kw=fixture(True);kw['angular_variance_rad2']=0.
    b['lines'][0]['provenance']['line_record']['inlier_rms_px']=0.
    edge=auth(a,[b],kw)['edges'][0]
    assert edge['angular_variance_rad2']==pytest.approx(1./24**2)
    assert edge['provenance']['cached_search_saturation_independently_certified_absent'] is False


STATES=['B','I','U','O','T','B','S','S']


def candidate(name, family, costs, mask):
    return dict(name=name, family=family, accepted=True, all_joint_constraints_satisfied=True,
        native_costs=[dict(kind='BI_COMPLETE',native_corner=k,weighted_robust_cost=v) for k,v in zip((0,1),costs)],
        global_mask_cost=mask, metrics=object(), frame_id=object())


def test_better_global_mask_cannot_outvote_dominated_native_image_evidence():
    r=NEW.select_hierarchical_native_candidates(STATES,[candidate('native','ORIGINAL',[1.,1.],100.),candidate('mask','SWAPPED',[2.,2.],0.)])
    assert r['selected_name']=='native' and r['pareto_frontier_indices']==[0]
    assert r['mask_tie_break_indices']==[0] and r['pareto_domination'][1]['dominated_by']==[0]


def test_native_tradeoff_preserves_family_ambiguity_minimum_summed_energy():
    r=NEW.select_hierarchical_native_candidates(STATES,[candidate('sum3','ORIGINAL',[0.,3.],100.),candidate('sum4','SWAPPED',[2.,2.],0.)])
    assert r['selected_name']=='sum3' and r['ambiguity']['cross_family_native_frontier']
    assert r['representative_is_not_uniqueness_certificate']


def test_mask_only_numeric_native_energy_ties_and_no_evidence_ambiguity():
    r=NEW.select_hierarchical_native_candidates(STATES,[candidate('a','ORIGINAL',[1.,2.],10.),candidate('b','SWAPPED',[2.,1.],0.)])
    assert r['selected_name']=='b' and r['mask_tie_break_indices']==[0,1]
    rows=[candidate('a','ORIGINAL',[],10.),candidate('b','SWAPPED',[],0.)]
    r=NEW.select_hierarchical_native_candidates(STATES,rows)
    assert r['selected_name']=='b' and r['ambiguity']['native_evidence_absent'] and r['ambiguity']['cross_family_native_frontier']


def test_rejected_candidates_S_U_poison_fields_and_fixed_basis():
    invalid=dict(accepted=False, native_costs=object(), global_mask_cost=object())
    assert NEW.select_hierarchical_native_candidates(STATES,[invalid])['selected_index'] is None
    for k in (2,3,4,6,7):
        row=candidate('a','ORIGINAL',[],1.);row['native_costs']=[dict(native_corner=k,weighted_robust_cost=object())]
        with pytest.raises(ValueError,match='gate'):NEW.select_hierarchical_native_candidates(STATES,[row])
    a=candidate('a','ORIGINAL',[1.,1.],1.);b=candidate('b','SWAPPED',[1.],1.)
    with pytest.raises(ValueError,match='same complete native evidence basis'):
        NEW.select_hierarchical_native_candidates(STATES,[a,b])
