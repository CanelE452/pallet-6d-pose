"""Conservative actual-line replacement; no image or reference decoding."""
from copy import deepcopy

import numpy as np
import pytest

from . import image_chain_mixture_problem_v4_18 as NEW
from . import native_edge_problem_v4_17 as N
from .test_hierarchical_native_candidate_score_v4_17 import fixture


def setup():
    a,b,kw=fixture(True)
    base=N.prepare_native_edge_problem(*a[:7],config=a[7],boundary_observations=[b],
        u_regions=[],edge_bundle=dict(edges=[],replaces_normal_factor_refs=[]))
    old=b['lines'][0]['provenance']['line_record'];k=b['native_corner'];mate=old['native_neighbor']
    chain={**deepcopy(old),'native_edge':sorted((k,mate)), 'invented_bridge_points':0}
    xy=np.asarray(chain['observed_points'])[chain['inlier_mask']]
    chain['inlier_rms_px']=float(np.sqrt(np.mean((xy@np.asarray(chain['normal'])+chain['offset'])**2)))
    observation=dict(image_binding={**b['image_binding'],'image_pixel_sha256':'c'*64},edge_records=[dict(
        native_corner=k,native_neighbor=mate,supported=True,chains=[chain],
        measurement_source='ACTUAL_IMAGE_NMS_CHAIN_WITH_MASK_CONTEXT',response_is_not_window_clipped=True)])
    kw=dict(initial_self_hidden=kw['initial_self_hidden'],image_binding=b['image_binding'],
        normal_variance_component_px2=4.,angular_variance_component_sine2=.002,
        noise_provenance=dict(fit_partition='CAL',model_digest_sha256='d'*64,
            observation_family=NEW.NOISE_OBSERVATION_FAMILY))
    return a,base,observation,kw


def test_auth_preserves_all_parallel_alternatives_and_deduplicates_identical_actual_pixels():
    a,base,observation,kw=setup();saved=deepcopy(observation)
    other=deepcopy(observation['edge_records'][0]['chains'][0])
    n=np.asarray(other['normal']);other['observed_points']=(np.asarray(other['observed_points'])+3.*n).tolist()
    other['offset']-=3.;observation['edge_records'][0]['chains'].extend([other,deepcopy(other)])
    before=deepcopy(observation)
    result=NEW.authenticate_image_chain_groups(base,observation,**kw)
    assert len(result['groups'])==1 and len(result['groups'][0]['components'])==2
    assert result['new_full_BI_point_revocations']==result['new_full_2D_KPs']==0
    assert observation==before and saved!=observation
    for component in result['groups'][0]['components']:
        span=component['actual_inlier_span_px'];fn=component['normal_floor_px']
        assert component['normal_variance_px2']==pytest.approx(4.+fn**2)
        assert component['angular_variance_rad2']==pytest.approx(.002+2*fn**2/(span**2+2*fn**2))


def test_mixture_replaces_exact_old_normals_once_and_preserves_full_BI_T4_guards_gradient():
    a,base,observation,kw=setup()
    bundle=NEW.authenticate_image_chain_groups(base,observation,**kw)
    old_remaining=deepcopy(base.remaining_normals)
    problem=NEW.ImageChainMixtureProblem(base,bundle)
    assert len(base.remaining_normals)==len(old_remaining)==1 and not problem.base.remaining_normals
    assert problem.complete==base.complete
    at=N.V._pose_vector(np.asarray(a[6]['R']),np.asarray(a[6]['t']),base.scale)+np.array([.01,-.02,.005,.008,-.011,.025])
    value,gradient,parts=problem.objective(at)
    numeric=N.V._jacobian(lambda p:np.array([problem.objective(p)[0]]),at)[0]
    assert np.allclose(gradient,numeric,rtol=3e-6,atol=3e-5)
    assert value==pytest.approx(sum(r['weighted_robust_cost']for r in parts['native_costs']))
    _,_,old_parts=base.objective(at)
    old_t=[r for r in old_parts['native_costs']if r['kind']=='UNCERTAIN_ARCHIVED_TANGENT']
    new_t=[r for r in parts['native_costs']if r['kind']=='UNCERTAIN_ARCHIVED_TANGENT']
    assert old_t==new_t
    oldguard,oldjac=base.constraints(at,base.sides[0]);newguard,newjac=problem.constraints(at,base.sides[0])
    assert np.array_equal(oldguard,newguard)and np.array_equal(oldjac,newjac)
    checked=problem.evaluate(np.asarray(a[6]['R']),np.asarray(a[6]['t']))
    assert checked['objective']==pytest.approx(problem.objective(N.V._pose_vector(np.asarray(a[6]['R']),np.asarray(a[6]['t']),base.scale))[0])


def test_unsupported_or_unmatched_new_edges_leave_base_exactly_unchanged():
    a,base,observation,kw=setup();observation['edge_records'][0]['supported']=False
    bundle=NEW.authenticate_image_chain_groups(base,observation,**kw);assert not bundle['groups']
    problem=NEW.ImageChainMixtureProblem(base,bundle)
    at=N.V._pose_vector(np.asarray(a[6]['R']),np.asarray(a[6]['t']),base.scale)
    old=base.objective(at);new=problem.objective(at)
    assert old[0]==new[0]and np.array_equal(old[1],new[1])


def test_initial_S_gates_before_poison_points_and_wrong_archive_binding_rejected():
    a,base,observation,kw=setup()
    k=next(i for i,s in enumerate(base.states)if s=='S')
    observation['edge_records']=[dict(native_corner=k,native_neighbor=0,chains=object())]
    assert not NEW.authenticate_image_chain_groups(base,observation,**kw)['groups']
    bad=deepcopy(kw);bad['image_binding']['sha256']='z'*64
    with pytest.raises(ValueError,match='bind'):
        NEW.authenticate_image_chain_groups(base,observation,**bad)


def test_complete_BI_without_oldpartial_ref_is_not_silently_revoked():
    a,base,observation,kw=setup();base.boundary=[];base.remaining_normals=[]
    bundle=NEW.authenticate_image_chain_groups(base,observation,**kw)
    assert not bundle['groups']and not bundle['replaces_normal_factor_refs']


def test_weighted_TLS_line_need_not_pass_unweighted_centroid_and_fitted_coefficients_survive():
    a,base,observation,kw=setup();chain=observation['edge_records'][0]['chains'][0]
    chain['offset']+=.1
    xy=np.asarray(chain['observed_points'])[chain['inlier_mask']]
    chain['inlier_rms_px']=float(np.sqrt(np.mean((xy@np.asarray(chain['normal'])+chain['offset'])**2)))
    bundle=NEW.authenticate_image_chain_groups(base,observation,**kw)
    component=bundle['groups'][0]['components'][0]
    assert component['offset_px']==chain['offset']
    assert abs(np.asarray(component['normal_unit_xy'])@component['support_anchor_xy']+component['offset_px'])<1e-10
    assert np.allclose(component['actual_unweighted_inlier_centroid_xy'],xy.mean(axis=0))
