"""Focused algebraic factors/parity fixtures; no image or dataset claims."""
from copy import deepcopy

import numpy as np
import pytest

from . import joint_ot_pnp as P
from . import statewise_uncertain_solver_v4_bias as BASE
from . import statewise_boundary_normal_solver_v4_4 as NEW
from .test_statewise_uncertain_solver_v4 import fixture


def arguments():
    a=fixture(4)
    # This synthetic camera actually hides vertex7. All observed B/I and the
    # extra U6 are physically nonself; S7 is never a supplied image factor.
    a[2][5]='U';a[2][7]='S'
    a[3][0]['fit_covariance_px2']=[[16.,1.],[1.,9.]]
    a[3][0]['fit_reliability_weight']=.8
    return a


def observation(a,k=6,normals=((0.,1.),),interval_halfspan=100.):
    q,_=P.project_native(a[0],np.asarray(a[6]['R']),np.asarray(a[6]['t']),a[1])
    lines=[]
    for normal in normals:
        n=np.asarray(normal);t=np.array([-n[1],n[0]]);s=float(t@q[k])
        lines.append(dict(normal_unit_xy=n.tolist(),offset_px=-float(n@q[k]),normal_variance_px2=25.,
            support_tangent_interval_px=[s-interval_halfspan,s+interval_halfspan],support_extrapolation_px=11.))
    return dict(native_corner=k,state=a[2][k],lines=lines,image_binding={'sha256':'a'*64,'image_hw':list(a[5])},
        association_provenance={'hypothesis_only':True})


def solve(a,records=None,weight=1.,**kwargs):
    return NEW.solve_boundary_normal_joint_ot_pnp(*a[:7],config=a[7],
        boundary_observations=records,boundary_factor_weight=weight,**kwargs)


def parameters(a):
    scale=float(np.linalg.norm(np.ptp(a[0],axis=0)))
    return scale,P._pose_vector(np.asarray(a[6]['R']),np.asarray(a[6]['t']),scale)


def test_empty_and_zero_factor_are_exact_parent_results_no_new_hard_rows():
    a=arguments();parent=BASE.solve_uncertain_joint_ot_pnp(*a[:7],config=a[7])
    assert solve(a)==parent
    assert solve(a,[])==parent
    assert solve(a,[observation(a)],weight=0.)==parent


@pytest.mark.parametrize('state',['S','O','T','I'])
def test_forbidden_states_rejected_before_line_fields_even_when_weight_zero(state):
    a=arguments();k=6;a[2][k]=state
    # Invalid line objects remain unread; the state gate has precedence.
    with pytest.raises(ValueError,match='only U/B'):
        solve(a,[{'native_corner':k,'lines':object()}],weight=0.)


def test_directional_information_parallel_and_nearly_parallel_have_weak_tangent():
    a=arguments();scale,p=parameters(a)
    records=NEW._boundary_inputs([observation(a,normals=((0.,1.),(0.,1.)))],a[2],a[5],a[0])
    r,j=NEW._normal_residual_and_jacobian(p,a[0],a[1],records,scale)
    shifted=p.copy();shifted[3]+=.04 # x-only translation changes no horizontal-line normal residual.
    rr,jj=NEW._normal_residual_and_jacobian(shifted,a[0],a[1],records,scale)
    assert np.array_equal(r,rr) and np.all(j[:,3]==0.)
    audit=NEW._boundary_audit(p,a[0],a[1],records,a[5],scale,BASE._configuration(a[7]),1.)[0]
    assert audit['normal_information_eigenvalues'][0]==0.
    assert audit['normal_information_eigenvalues'][1]>0
    angle=1e-4
    records=NEW._boundary_inputs([observation(a,normals=((0.,1.),(np.sin(angle),np.cos(angle))))],a[2],a[5],a[0])
    values=NEW._boundary_audit(p,a[0],a[1],records,a[5],scale,BASE._configuration(a[7]),1.)[0]['normal_information_eigenvalues']
    assert 0<values[0]/values[1]<1e-8


@pytest.mark.parametrize('normals',[((0.,1.),),((0.,1.),(.6,.8))])
def test_normal_residual_jacobian_and_radial_objective_gradient(normals):
    a=arguments();scale,p=parameters(a);p=p+np.array([.01,-.02,.005,.008,-.011,.025])
    records=NEW._boundary_inputs([observation(a,normals=normals)],a[2],a[5],a[0])
    residual,jacobian=NEW._normal_residual_and_jacobian(p,a[0],a[1],records,scale)
    numerical=P._jacobian(lambda x:NEW._normal_residual_and_jacobian(x,a[0],a[1],records,scale)[0],p)
    assert np.allclose(jacobian,numerical,atol=2e-6,rtol=2e-6)
    value,gradient=NEW._normal_objective_pair(p,a[0],a[1],records,scale,.7,3.)
    numeric=P._jacobian(lambda x:np.array([NEW._normal_objective_pair(x,a[0],a[1],records,scale,.7,3.)[0]]),p)[0]
    assert value>0 and np.allclose(gradient,numeric,atol=2e-5,rtol=2e-6)


def test_finite_support_canvas_geometry_jacobian_and_caller_profile_interval():
    a=arguments();scale,p=parameters(a);cfg=BASE._configuration(a[7])
    records=NEW._boundary_inputs([observation(a,normals=((0.,1.),(.6,.8)))],a[2],a[5],a[0])
    values,jac=NEW._boundary_constraint_pair(p,a[0],a[1],records,a[5],scale,cfg)
    assert len(values)==9 and min(values)>=0.
    numerical=P._jacobian(lambda x:NEW._boundary_constraint_pair(x,a[0],a[1],records,a[5],scale,cfg)[0],p)
    assert np.allclose(jac,numerical,atol=2e-5,rtol=2e-6)
    # A caller's profile interval is used verbatim, rather than an inferred raw x.
    records[0]['lines'][0]['support_tangent_interval_px']+=200.
    modified,_=NEW._boundary_constraint_pair(p,a[0],a[1],records,a[5],scale,cfg)
    assert modified[0]<0.


def test_active_joint_factor_preserves_old_BI_seed_Q_all_OT_and_frozen_inputs():
    a=arguments();extra=[observation(a,normals=((0.,1.),(.001,np.sqrt(1-.001**2))))]
    before=deepcopy(a);saved=deepcopy(extra)
    parent=BASE.solve_uncertain_joint_ot_pnp(*a[:7],config=a[7]);result=solve(a,extra)
    assert result['accepted'] and result['states']==a[2]
    assert result['BI_prior_seed']==parent['BI_prior_seed']
    assert result['visible_observations']==parent['visible_observations']
    assert result['reported_native'][6] is None # U remains partial; no new measured KP6.
    assert all(r['feasible'] for r in result['joint_constraint_audit'])
    assert all(r['feasible'] for r in result['boundary_observation_audit'])
    assert np.allclose(result['R_physical'],np.asarray(result['R'])@a[6]['physical_registration_Q'])
    assert result['boundary_factor_audit']['finite_support_changes_positive_weight_feasible_set']
    assert extra==saved
    assert all(np.array_equal(x,y) if isinstance(x,np.ndarray) else x==y for x,y in zip(a,before))
    assert result['provenance']['S_boundary_terms']==0


def test_boundary_B_adds_no_duplicate_original_canvas_geometry_rows():
    a=arguments();scale,p=parameters(a)
    record=NEW._boundary_inputs([observation(a,k=0)],a[2],a[5],a[0])
    values,_=NEW._boundary_constraint_pair(p,a[0],a[1],record,a[5],scale,BASE._configuration(a[7]))
    assert len(values)==2 # Only finite tangent interval; old B already guarded.


def test_invalid_variance_binding_interval_and_factor_weight_rejected():
    a=arguments();r=observation(a)
    for variance in [0.,-1.,np.nan]:
        bad=deepcopy(r);bad['lines'][0]['normal_variance_px2']=variance
        with pytest.raises(ValueError,match='strictly positive'):solve(a,[bad])
    bad=deepcopy(r);bad['image_binding']['image_hw']=[1,2]
    with pytest.raises(ValueError,match='image binding'):solve(a,[bad])
    bad=deepcopy(r);bad['lines'][0]['support_tangent_interval_px']=[1.,1.]
    with pytest.raises(ValueError,match='increasing tangent'):solve(a,[bad])
    with pytest.raises(ValueError,match='factor_weight'):solve(a,[r],weight=-1.)


def test_old_two_BI_no_OT_early_fallback_retained_with_partial_factor():
    a=arguments();a[2][2]='U';a[2][3]='U';a[3][:]=a[3][:2]
    a[2][a[4][0]['native_corner']]='U';a[4].clear()
    result=solve(a,[observation(a)])
    assert not result['accepted'] and result['fallback_reason']=='two_BI_requires_OT_ranges'
