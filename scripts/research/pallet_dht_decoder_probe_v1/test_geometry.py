import inspect
import math

import numpy as np
import pytest

from . import geometry as G


def fixture():
    theta=np.arange(90)*math.pi/90
    rho=np.arange(-14,14.01,.5)
    valid=np.abs(rho)[None]<=(np.abs(np.cos(theta))+np.abs(np.sin(theta)))[:,None]*5+1e-6
    logits=np.full((12,90,len(rho)),-20.)
    # Analytic x=48 and y=48 input-pixel lines; baseline corner0=(55,55).
    for role in range(12):logits[role,0,np.argmin(abs(rho+2))]=20.
    logits[0]=-20.;logits[0,45,np.argmin(abs(rho+2))]=20.
    points=np.repeat(np.array([[55.,55.]]),9,axis=0)
    points[8]=[80.,80.]
    return dict(logits=logits,theta=theta,rho=rho,lattice_valid=valid,
        feature_shape_hw=(10,10),input_shape_hw=(160,160),
        raw_to_input_affine=np.array([[1.,0,0],[0,1.,0]]),
        points_xy=points,point_conf=np.ones(9)*.8)


def test_cube_incidence_is_three_per_corner_and_role7_is_4_7():
    assert G.EDGES[7]==(7,4)
    assert all(len(r)==3 for r in G.INCIDENT_ROLES)
    assert sorted(r for rs in G.INCIDENT_ROLES for r in rs)==sorted(list(range(12))*2)


def test_exact_two_line_intersection_and_antipodal_invariance():
    a=np.array([1.,0,-7]);b=np.array([0,1.,-11.])
    np.testing.assert_array_equal(G.intersect_lines(a,b,.1),[7.,11.])
    np.testing.assert_array_equal(G.intersect_lines(-a,b,.1),[7.,11.])


def test_parallel_and_nearparallel_rejected_before_division():
    assert G.intersect_lines([1,0,0],[1,0,-2],math.sin(math.radians(5))) is None
    assert G.intersect_lines([1,0,0],[math.cos(.01),math.sin(.01),-1],math.sin(math.radians(5))) is None


def test_canonical_antipodes_preserve_geometry():
    t,r=G.canonical_line(-.1,3.)
    assert t==pytest.approx(math.pi-.1) and r==-3.
    q=np.array([12.,4.])
    old=np.array([math.cos(-.1),math.sin(-.1)])@q-3
    new=np.array([math.cos(t),math.sin(t)])@q-r
    assert new==pytest.approx(-old)


def test_seam_nms_flips_rho_with_normal():
    theta=np.array([0.,math.radians(178)])
    rho=np.array([-2.,2.]);logits=np.array([[0.,10.],[9.,0.]])
    peaks=G.topk_lines(logits,theta,rho,np.ones((2,2),bool))
    assert peaks[0]['flat_index']==1
    assert 2 not in [p['flat_index'] for p in peaks]


def test_same_theta_different_far_rho_modes_survive():
    peaks=G.topk_lines([[4.,3.]], [0.], [-2.,2.],[[True,True]])
    assert len(peaks)==2


def test_nms_ties_are_deterministic_flat_index():
    peaks=G.topk_lines(np.zeros((2,2)),[0.,math.pi/2],[-2.,2.],np.ones((2,2),bool))
    assert [p['flat_index'] for p in peaks]==[0,1,2,3]


def test_affine_maps_line_exactly_including_rectangular_padding():
    affine=np.array([[.7,0,70.],[0,.7,81.]])
    h=G.feature_line_to_raw(0.,-2.,(30,40),(480,640),affine)
    raw_x=(16*(20-2)-70)/.7
    assert h@np.array([raw_x,99.,1.])==pytest.approx(0.,abs=1e-12)


def test_candidate_slot_contract_and_known_intersection():
    d=G.build_candidates(**fixture())
    assert d['candidates_xy'].shape==(8,49,2)
    assert d['candidate_features'].shape==(8,49,18)
    assert d['candidate_sources'][0,1].tolist()==[0,0,3,0]
    assert d['candidate_valid'][0,1]
    np.testing.assert_allclose(d['candidates_xy'][0,1],[48.,48.],atol=1e-12)
    np.testing.assert_array_equal(d['candidates_xy'][:,0],fixture()['points_xy'][:8])


def test_no_lines_falls_back_exactly_and_preserves_center():
    f=fixture();f['lattice_valid'][:]=False
    d=G.build_candidates(**f);s=G.fixed_select(d)
    assert d['candidate_valid'].sum()==8
    np.testing.assert_array_equal(s['points_xy'],f['points_xy'])
    assert (s['indices']==0).all()


def test_fixed_selector_uses_available_correct_intersection():
    f=fixture();d=G.build_candidates(**f);s=G.fixed_select(d)
    assert s['indices'][0]!=0
    np.testing.assert_allclose(s['points_xy'][0],[48.,48.],atol=1e-12)
    np.testing.assert_array_equal(s['points_xy'][8],f['points_xy'][8])


def test_original_raw_unit_change_keeps_input_geometry_and_selection():
    f=fixture();a=G.build_candidates(**f)
    f['points_xy']=f['points_xy']*3
    f['raw_to_input_affine'][:,:2]/=3
    b=G.build_candidates(**f)
    np.testing.assert_array_equal(a['candidate_valid'],b['candidate_valid'])
    np.testing.assert_allclose(b['candidates_xy'],a['candidates_xy']*3,atol=1e-12)
    np.testing.assert_allclose(b['candidate_features'],a['candidate_features'],atol=1e-12)


def test_every_valid_intersection_is_on_both_source_lines():
    d=G.build_candidates(**fixture())
    for corner,slot in np.argwhere(d['candidate_valid']):
        if slot==0:continue
        a,ka,b,kb=d['candidate_sources'][corner,slot]
        q=np.r_[d['candidates_xy'][corner,slot],1.]
        assert abs(d['line_peaks_h_raw'][a,ka]@q)<1e-10
        assert abs(d['line_peaks_h_raw'][b,kb]@q)<1e-10


def test_intersections_inside_network_footprint():
    d=G.build_candidates(**fixture())
    q=d['candidates_xy'][d['candidate_valid']]
    assert (q>=-1e-12).all() and (q<=160+1e-12).all()


@pytest.mark.parametrize('key',['logits','points_xy','raw_to_input_affine'])
def test_nonfinite_input_is_explicit_failure(key):
    f=fixture();f[key]=np.array(f[key],copy=True);f[key].flat[0]=np.nan
    with pytest.raises(ValueError):G.build_candidates(**f)


def test_no_gt_inputs_and_centroid_not_a_candidate():
    assert not any('gt' in p for p in inspect.signature(G.build_candidates).parameters)
    assert not any('gt' in p for p in inspect.signature(G.fixed_select).parameters)
    d=G.build_candidates(**fixture())
    assert d['candidates_xy'].shape[0]==8
    assert G.fixed_select(d)['uses_gt'] is False


def test_joint_mixture_feature_has_no_averaged_line_hallucination():
    # Equal x=-10,+10 modes: at x=0 neither mode fits, although their mean does.
    c=G._mixture_cost(np.array([[0.,0.],[-10.,0.]]),np.array([[1.,0.],[1.,0.]]),
                      np.array([-10.,10.]),np.array([.5,.5]),1.)
    assert c[0]>5 and c[1]<1
