"""Focused partial-normal/range contracts; synthetic algebra, no data fitting."""
from copy import deepcopy

import numpy as np
import pytest

from . import joint_ot_pnp as P
from . import statewise_uncertain_solver_v4_bias as BASE
from . import statewise_u_region_solver_v4_5 as U
from . import statewise_partial_boundary_u_solver_v4_6 as NEW
from . import boundary_partial_observations_v4_6 as OBS
from . import truncation_interval_v4_2 as T
from .visible_bi_correction_v2 import DEFAULT_CONFIG
from .test_statewise_uncertain_solver_v4 import fixture


def arguments():
    a=fixture(4)
    for m in a[3]:
        m.update(source='VISIBLE_BACKBONE_FALLBACK',fit_covariance_px2=[[16.,1.],[1.,9.]],fit_reliability_weight=.8)
    return a


def observation(a, normals=((0.,1.),)):
    k=a[3][0]['native_corner'];q,_=P.project_native(a[0],np.asarray(a[6]['R']),np.asarray(a[6]['t']),a[1])
    lines=[]
    for n in map(np.asarray,normals):
        tangent=np.array([-n[1],n[0]]);s=float(tangent@q[k])
        lines.append(dict(normal_unit_xy=n.tolist(),offset_px=-float(n@q[k]),normal_variance_px2=25.,
            support_tangent_interval_px=[s-100.,s+100.],support_extrapolation_px=11.))
    return dict(native_corner=k,state='B',lines=lines,fit_reliability_weight=.8,replaces_raw_B_likelihood=True,
        image_binding={'sha256':'a'*64,'image_hw':list(a[5])},association_provenance={'original_raw_xy':a[3][0]['xy']})


def region(a):
    k=next(k for k,s in enumerate(a[2]) if s=='U');q,_=P.project_native(a[0],np.asarray(a[6]['R']),np.asarray(a[6]['t']),a[1])
    return dict(native_corner=k,state='U',shape='ellipse',center_xy=q[k].tolist(),half_axes_px=[80.,90.],angle_deg=12.)


def solve(a,boundary=None,weight=1.,regions=None,**kw):
    return NEW.solve_boundary_u_region_joint_ot_pnp(*a[:7],config=a[7],boundary_observations=boundary,
        boundary_factor_weight=weight,u_regions=regions,**kw)


def parameters(a):
    scale=float(np.linalg.norm(np.ptp(a[0],axis=0)));p=P._pose_vector(np.asarray(a[6]['R']),np.asarray(a[6]['t']),scale)
    return scale,p


def producer_arguments():
    states=['B','B','U','I','O','T','B','S']
    geometry={'initial_roles':['B']*8,'initial_self_hidden':[False]*7+[True],'image_hw':[480,640]}
    points=[[54.+i*2.,100.] for i in range(15)]
    ray=dict(native_corner=0,native_neighbor=1,supported=True,
        measurement_source='OBSERVED_EXTERIOR_MASK_AND_RGB_TRANSITION',normal=[0.,1.],offset=-100.,
        observed_points=points,inlier_mask=[True]*15,inlier_rms_px=.2,angle_change_deg=0.,
        inlier_fraction=1.,inlier_span_fraction=1.,rgb_supported_sample_count=15,
        rgb_support_thresholds={k:DEFAULT_CONFIG[k] for k in ('minimum_gradient','minimum_gradient_alignment','minimum_color_transition')})
    legacy={**ray,'measurement_source':'OBSERVED_TARGET_BACKGROUND_TRANSITION'}
    correction=dict(records=[dict(native_corner=0,state='B',corrected=False,rays=[legacy]),
        dict(native_corner=7,state='S',rays=object())],boundary_recovery_records=[dict(native_corner=0,
        expanded_rays=[ray],observed_line_recentering=[{'remeasured_ray':deepcopy(ray)}])],
        configuration=deepcopy(DEFAULT_CONFIG),image_pixel_sha256='b'*64)
    fit=[dict(native_corner=0,state='B',xy=[50.,105.],source='VISIBLE_BACKBONE_FALLBACK',
        fit_covariance_px2=[[16.,0.],[0.,9.]],fit_reliability_weight=.7)]
    model=dict(schema=T.SCHEMA,fit_partition='CAL',radius_ratio=.5,
        recipe='ALL_VALID_INITIAL_NON_S_DIRECT_AND_EXTERNAL_UNTRIMMED_MAX');model['model_sha256']=T._model_digest(model)
    kw=dict(localization_model=model,nonself_scale_px=200.,image_binding={'sha256':'a'*64,'image_hw':[480,640]})
    return correction,geometry,states,fit,kw


def test_producer_single_rgb_line_kept_dedup_and_broad_corner_interval():
    correction,geom,states,fit,kw=producer_arguments();saved=deepcopy(fit)
    result=OBS.observe_single_supported_b_lines(correction,geom,states,fit,**kw)
    assert result['S_coordinate_reads']==result['RGB_queries']==result['new_line_fits']==0
    assert fit==saved and result['states_unchanged']
    record=result['boundary_observations'][0];line=record['lines'][0]
    assert len(result['boundary_observations'])==len(record['lines'])==1
    assert line['normal_variance_px2']==9.
    assert line['support_tangent_interval_px']==[-150.,50.] and line['support_extrapolation_px']==11.
    assert record['association_provenance']['normal_projected_display_xy']==[50.,100.]
    # A real corner can lie beyond observed 4..36px edge samples; locality
    # remains broad rather than treating an incident edge sample as the KP.
    assert -120.>=line['support_tangent_interval_px'][0]
    assert result['records'][7]['coordinate_read'] is False and result['records'][7]['ray_fields_read'] is False


def test_producer_mask_only_or_insufficient_rgb_never_promoted():
    c,g,s,f,kw=producer_arguments();c['boundary_recovery_records']=[]
    assert OBS.observe_single_supported_b_lines(c,g,s,f,**kw)['boundary_observations']==[]
    c,g,s,f,kw=producer_arguments();c['boundary_recovery_records'][0]['expanded_rays'][0]['rgb_supported_sample_count']=5
    c['boundary_recovery_records'][0]['observed_line_recentering']=[]
    assert OBS.observe_single_supported_b_lines(c,g,s,f,**kw)['boundary_observations']==[]
    c,g,s,f,kw=producer_arguments();c['records'][0]['corrected']=True
    assert OBS.observe_single_supported_b_lines(c,g,s,f,**kw)['boundary_observations']==[]


def test_empty_zero_exact_latest_U_parent_parity():
    a=arguments();rr=[region(a)];parent=U.solve_uncertain_u_region_joint_ot_pnp(*a[:7],config=a[7],u_regions=rr)
    assert solve(a,regions=rr)==parent
    assert solve(a,[],regions=rr)==parent
    assert solve(a,[observation(a)],weight=0.,regions=rr)==parent


@pytest.mark.parametrize('state',['S','U','O','T','I'])
def test_forbidden_state_gate_precedes_line_fields(state):
    a=arguments();k=a[3][0]['native_corner'];a[2][k]=state
    with pytest.raises(ValueError,match='only B'):
        solve(a,[{'native_corner':k,'lines':object()}],weight=0.)


def test_replacement_objective_no_raw_normal_or_tangent_double_count_and_FD():
    a=arguments();scale,p=parameters(a);cfg=BASE._configuration({**a[7],'prior_weight':0.})
    boundary=NEW._boundary_inputs([observation(a)],a[2],a[5],a[0]);k=boundary[0]['native_corner']
    measure=dict(native_corner=k,xy=np.asarray(a[3][0]['xy']),whitening=np.eye(2),fit_reliability_weight=.8)
    prior=dict(R=np.asarray(a[6]['R']),center=np.asarray(a[6]['t']),whitening=np.eye(6))
    at=p+np.array([.01,-.02,.005,.008,-.011,.025])
    value,grad,meta=NEW._joint_objective_pair(at,a[0],a[1],[measure],scale,prior,cfg,boundary,1.)
    numerical=P._jacobian(lambda pp:np.array([NEW._joint_objective_pair(pp,a[0],a[1],[measure],scale,prior,cfg,boundary,1.)[0]]),at)[0]
    assert np.allclose(grad,numerical,atol=2e-5,rtol=2e-6)
    altered={**measure,'xy':measure['xy']+1000.}
    vv,gg,_=NEW._joint_objective_pair(at,a[0],a[1],[altered],scale,prior,cfg,boundary,1.)
    assert vv==value and np.array_equal(gg,grad) and meta['removed_raw_B_2D_likelihood_ids']==[k]
    shifted=p.copy();shifted[3]+=.04
    first=NEW._joint_objective_pair(p,a[0],a[1],[measure],scale,prior,cfg,boundary,1.)
    second=NEW._joint_objective_pair(shifted,a[0],a[1],[measure],scale,prior,cfg,boundary,1.)
    assert first[0]==second[0] and first[1][3]==second[1][3]==0.


def test_joint_OT_U_and_partial_support_constraint_jacobian():
    a=arguments();scale,p=parameters(a);cfg=BASE._configuration(a[7])
    xyz,K,states,ms,ds,hw=P._inputs(*a[:6],BASE._helper_config(cfg))
    boundary=NEW._boundary_inputs([observation(a)],states,hw,xyz);regions=U._u_region_inputs([region(a)],states)
    values,jac=NEW._joint_constraint_pair(p,xyz,K,ms,ds,(None,),hw,scale,cfg,boundary,regions)
    numeric=P._jacobian(lambda pp:NEW._joint_constraint_pair(pp,xyz,K,ms,ds,(None,),hw,scale,cfg,boundary,regions)[0],p)
    assert np.allclose(jac,numeric,atol=2e-5,rtol=2e-6)
    oldvalues,_=BASE._constraints(p,xyz,K,ms,ds,(None,),hw,scale,cfg)
    assert np.array_equal(values[:len(oldvalues)],oldvalues)
    assert len(values)==len(oldvalues)+1+2 and values[-3]>=0
    regions[0]['center_xy']+=2000.
    failed,_=NEW._joint_constraint_pair(p,xyz,K,ms,ds,(None,),hw,scale,cfg,boundary,regions)
    assert failed[-3]<0 and np.array_equal(failed[:len(oldvalues)],oldvalues)


def test_active_solver_seed_Q_OT_U_preserved_partial_readout_and_generators():
    a=arguments();rr=[region(a)];line=[observation(a)];saved=deepcopy(a);savedline=deepcopy(line)
    parent=U.solve_uncertain_u_region_joint_ot_pnp(*a[:7],config=a[7],u_regions=rr)
    result=solve(a,line,regions=rr)
    assert result['accepted'] and result['states']==a[2]
    assert result['BI_prior_seed']==parent['BI_prior_seed']
    assert result['U_constraints_satisfied'] and result['partial_B_constraints_satisfied']
    assert all(x['feasible'] for x in result['joint_constraint_audit'])
    k=line[0]['native_corner'];assert result['reported_native'][k] is None
    assert result['boundary_readout'][str(k)]['reported_full_2D_xy'] is None
    assert result['objective_components']['raw_B_normal_cost_duplicated'] is False
    assert np.allclose(result['R_physical'],np.asarray(result['R'])@a[6]['physical_registration_Q'])
    assert a[3]==saved[3] and a[4]==saved[4] and line==savedline
    assert all(result['reported_native'][i] is None for i,s in enumerate(a[2]) if s=='U')
    ag=list(a);ag[3]=(x for x in a[3]);ag[4]=(x for x in a[4])
    generated=solve(ag,line,regions=rr,initializations=(x for x in []))
    assert generated==result
