"""Frozen-line pilot geometry. No GT, model forward, relabeling or PnP.

Discrete Top-K and intersection generation are intentionally outside the
learned selector graph. All output points retain camera_dynamic_0123_v4 IDs.
"""
from itertools import combinations
import math

import numpy as np

EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
INCIDENT_ROLES = tuple(tuple(r for r,e in enumerate(EDGES) if i in e) for i in range(8))
DEFAULT_CONFIG = dict(schema='decoder_probe_geometry_v1', top_k=4,
    nms_angle_degrees=4., nms_rho_feature_cells=1., minimum_pair_angle_degrees=5.,
    candidate_footprint='actual_network_input_closed_rectangle',
    line_sigma_input_diagonal_fraction=.01, line_sigma_min_input_px=1.,
    point_prior_weight=.25, point_prior_input_diagonal_fraction=.1,
    tie_policy='first_slot_original_point_then_fixed_pair_rank_order',
    role_assignment='fixed_camera_dynamic_0123_v4', centroid_policy='copy_original',
    score_policy='mean_three_role_single_point_mixture_cost_plus_weak_pseudohuber_anchor')
FEATURE_NAMES = ('is_baseline','delta_x_over_input_diagonal','delta_y_over_input_diagonal',
    'centered_x_over_input_diagonal','centered_y_over_input_diagonal','original_point_confidence',
    'mixture_cost_incident0','mixture_cost_incident1','mixture_cost_incident2',
    'normalized_entropy_incident0','normalized_entropy_incident1','normalized_entropy_incident2',
    'peak_log_probability_incident0','peak_log_probability_incident1','peak_log_probability_incident2',
    'source_pair_abs_sin_angle','source_a_log_probability','source_b_log_probability')


def canonical_line(theta, rho):
    """Normalize a normal angle to [0,pi), flipping rho for each antipode."""
    if not np.isfinite([theta,rho]).all():
        raise ValueError('Finite theta and rho required')
    turns = math.floor(theta/math.pi)
    return theta-turns*math.pi, rho*(-1. if turns % 2 else 1.)


def topk_lines(logits, theta, rho, valid, config=None):
    """Deterministic seam-safe NMS; no score/visibility threshold is fitted."""
    config=DEFAULT_CONFIG if config is None else config
    logits=np.asarray(logits,float);theta=np.asarray(theta,float);rho=np.asarray(rho,float)
    valid=np.asarray(valid,bool)
    if logits.shape!=(len(theta),len(rho)) or valid.shape!=logits.shape:
        raise ValueError('Malformed line lattice')
    if not np.isfinite(logits).all() or not np.isfinite(theta).all() or not np.isfinite(rho).all():
        raise ValueError('Nonfinite line evidence')
    tt,rr=np.meshgrid(theta,rho,indexing='ij')
    tt=tt.ravel();rr=rr.ravel(); indices=np.flatnonzero(valid.ravel())
    ordered=indices[np.lexsort((indices,-logits.ravel()[indices]))]
    result=[]
    angle=math.radians(config['nms_angle_degrees'])
    for index in ordered:
        t,r=canonical_line(float(tt[index]),float(rr[index]))
        duplicate=False
        for s in result:
            cosine=math.cos(t-s['theta'])
            distance=math.acos(min(1.,max(-1.,abs(cosine))))
            aligned_r=s['rho']*(1. if cosine>=0 else -1.)
            if distance<=angle+1e-8 and abs(r-aligned_r)<=config['nms_rho_feature_cells']+1e-8:
                duplicate=True;break
        if not duplicate:
            result.append(dict(theta=t,rho=r,flat_index=int(index),logit=float(logits.ravel()[index])))
            if len(result)==config['top_k']:break
    return result


def intersect_lines(h1, h2, minimum_abs_sin):
    """h=[unit nx,unit ny,c], line nx*x+ny*y+c=0; None if unstable."""
    h1=np.asarray(h1,float);h2=np.asarray(h2,float)
    if h1.shape!=(3,) or h2.shape!=(3,) or not np.isfinite([h1,h2]).all():
        raise ValueError('Finite line coefficients required')
    a=np.linalg.norm(h1[:2]);b=np.linalg.norm(h2[:2])
    if min(a,b)<=1e-12:return None
    h1=h1/a;h2=h2/b
    det=h1[0]*h2[1]-h1[1]*h2[0]
    if abs(det)<minimum_abs_sin:return None
    q=np.array([(h1[1]*h2[2]-h1[2]*h2[1])/det,
                (h1[2]*h2[0]-h1[0]*h2[2])/det])
    return q if np.isfinite(q).all() else None


def feature_line_to_raw(theta,rho,feature_shape_hw,input_shape_hw,raw_to_input_affine):
    h,w=map(int,feature_shape_hw);ih,iw=map(int,input_shape_hw)
    stride=iw/w
    if not math.isclose(ih/h,stride,abs_tol=1e-9,rel_tol=0):
        raise ValueError('Feature and input footprints have different aspect ratios')
    affine=np.asarray(raw_to_input_affine,float)
    if affine.shape!=(2,3) or not np.isfinite(affine).all():
        raise ValueError('Finite raw-to-input 2x3 affine required')
    n=np.array([math.cos(theta),math.sin(theta)])
    normal=affine[:,:2].T@n/stride
    c=n@affine[:,2]/stride-n@np.array([w/2,h/2])-rho
    norm=np.linalg.norm(normal)
    if norm<=1e-12:raise ValueError('Degenerate coordinate affine')
    return np.r_[normal,c]/norm


def _mixture_cost(q_feature,normals,offsets,weights,sigma):
    if not len(weights):return np.zeros(len(q_feature))
    residual=(q_feature[:,0,None]*normals[:,0]+q_feature[:,1,None]*normals[:,1]-offsets)/sigma
    costs=np.sqrt(1.+residual**2)-1.
    # Exact log-sum-exp, including large finite displacements.
    values=np.log(weights)[None]-costs
    maximum=values.max(-1)
    return -(maximum+np.log(np.exp(values-maximum[:,None]).sum(-1)))


def build_candidates(logits, theta, rho, lattice_valid, feature_shape_hw,
                     input_shape_hw, raw_to_input_affine, points_xy, point_conf,
                     config=None):
    """Return original-pixel [8,49,2] candidates and [8,49,18] features.

    Slot0 is each original semantic corner. Slots1..48 enumerate ascending
    incident-role pairs then ranks0..3 lexicographically. Invalid slots repeat
    the baseline coordinate but have a false mask. No per-corner GT mask is
    accepted. Inference uses all 12 predicted structural-role distributions.
    """
    config=dict(DEFAULT_CONFIG if config is None else config)
    if config['top_k']!=4:raise ValueError('Pilot contract fixes Top-K=4 and 49 slots')
    logits=np.asarray(logits,float);theta=np.asarray(theta,float);rho=np.asarray(rho,float)
    valid=np.asarray(lattice_valid,bool);points=np.asarray(points_xy,float);conf=np.asarray(point_conf,float)
    affine=np.asarray(raw_to_input_affine,float)
    if logits.shape!=(12,len(theta),len(rho)) or valid.shape!=logits.shape[1:]:
        raise ValueError('Expected12 role distributions on the same lattice')
    if points.shape!=(9,2) or conf.shape!=(9,) or not np.isfinite(points).all() or not np.isfinite(conf).all():
        raise ValueError('Finite nine original-pixel points/confidences required')
    if affine.shape!=(2,3) or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:,:2]))<1e-12:
        raise ValueError('Invertible finite raw-to-input affine required')
    if not np.isfinite(logits).all():raise ValueError('Nonfinite line logits')
    h,w=map(int,feature_shape_hw);ih,iw=map(int,input_shape_hw)
    if min(h,w,ih,iw)<=0 or not math.isclose(ih/h,iw/w,abs_tol=1e-9,rel_tol=0):
        raise ValueError('Invalid feature/input stride contract')
    stride=iw/w;diag=math.hypot(iw,ih)
    sigma=max(config['line_sigma_min_input_px'],config['line_sigma_input_diagonal_fraction']*diag)/stride
    cosine=np.cos(theta);sine=np.sin(theta)
    normals=np.broadcast_to(np.stack([cosine,sine],-1)[:,None],(*valid.shape,2))[valid]
    offsets=np.broadcast_to(rho[None],valid.shape)[valid]
    masses=[];entropies=[];peaklogs=[]
    line_peaks_h_raw=np.zeros((12,4,3));line_peaks_h_feature=np.zeros((12,4,3))
    line_peak_valid=np.zeros((12,4),bool);line_peak_theta_rho=np.zeros((12,4,2))
    line_peak_log_probability=np.zeros((12,4));line_peak_flat_index=np.full((12,4),-1,int)
    for role in range(12):
        logmass=-np.logaddexp(0.,-logits[role][valid])
        if len(logmass):
            # Avoid underflow even for extremely negative but finite logits.
            exp=np.exp(logmass-logmass.max());mass=exp/exp.sum()
            logprob=logmass-(logmass.max()+math.log(exp.sum()))
            entropy=float(-(mass*logprob).sum()/math.log(len(mass))) if len(mass)>1 else 0.
            pl=float(logprob.max())
        else:mass=np.empty(0);entropy=0.;pl=0.;logprob=np.empty(0)
        masses.append(mass);entropies.append(entropy);peaklogs.append(pl)
        dense_logprob=np.zeros(valid.shape);dense_logprob[valid]=logprob
        for rank,peak in enumerate(topk_lines(logits[role],theta,rho,valid,config)):
            t,r=peak['theta'],peak['rho']
            line_peaks_h_raw[role,rank]=feature_line_to_raw(t,r,(h,w),(ih,iw),affine)
            line_peaks_h_feature[role,rank]=[math.cos(t),math.sin(t),-r]
            line_peak_theta_rho[role,rank]=[t,r]
            line_peak_valid[role,rank]=True
            line_peak_flat_index[role,rank]=peak['flat_index']
            line_peak_log_probability[role,rank]=dense_logprob.ravel()[peak['flat_index']]
    candidates=np.repeat(points[:8,None],49,axis=1)
    mask=np.zeros((8,49),bool);mask[:,0]=True
    sources=np.full((8,49,4),-1,int)
    features=np.zeros((8,49,len(FEATURE_NAMES)))
    inverse=np.linalg.inv(affine[:,:2])
    minimum=math.sin(math.radians(config['minimum_pair_angle_degrees']))
    rejected_parallel=rejected_footprint=0
    for corner,roles in enumerate(INCIDENT_ROLES):
        slot=1
        for a,b in combinations(roles,2):
            for ka in range(4):
                for kb in range(4):
                    sources[corner,slot]=[a,ka,b,kb]
                    if line_peak_valid[a,ka] and line_peak_valid[b,kb]:
                        ha=line_peaks_h_feature[a,ka];hb=line_peaks_h_feature[b,kb]
                        q=intersect_lines(ha,hb,minimum)
                        if q is None:rejected_parallel+=1
                        else:
                            p=(q+np.array([w/2,h/2]))*stride
                            if (p>=0.).all() and (p<=np.array([iw,ih])).all():
                                candidates[corner,slot]=(p-affine[:,2])@inverse.T
                                mask[corner,slot]=True
                                features[corner,slot,15]=abs(ha[0]*hb[1]-ha[1]*hb[0])
                                features[corner,slot,16:18]=[line_peak_log_probability[a,ka],line_peak_log_probability[b,kb]]
                            else:rejected_footprint+=1
                    slot+=1
        qinput=candidates[corner]@affine[:,:2].T+affine[:,2]
        pinput=points[corner]@affine[:,:2].T+affine[:,2]
        features[corner,0,0]=1.
        features[corner,:,1:3]=(qinput-pinput)/diag
        features[corner,:,3:5]=(qinput-np.array([iw/2,ih/2]))/diag
        features[corner,:,5]=conf[corner]
        qfeature=qinput/stride-np.array([w/2,h/2])
        for j,role in enumerate(roles):
            features[corner,:,6+j]=_mixture_cost(qfeature,normals,offsets,masses[role],sigma)
            features[corner,:,9+j]=entropies[role]
            features[corner,:,12+j]=peaklogs[role]
    if not np.isfinite(features).all():raise ValueError('Nonfinite derived geometry features')
    return dict(schema='decoder_probe_candidates_v1',config=config,feature_names=list(FEATURE_NAMES),
        baseline_points_xy=points.copy(),point_confidence=conf.copy(),candidates_xy=candidates,
        candidate_valid=mask,candidate_features=features,candidate_sources=sources,
        incident_roles=np.asarray(INCIDENT_ROLES),edges=np.asarray(EDGES),
        line_peaks_h_raw=line_peaks_h_raw,line_peak_valid=line_peak_valid,
        line_peak_theta_rho=line_peak_theta_rho,line_peak_log_probability=line_peak_log_probability,
        line_peak_flat_index=line_peak_flat_index,input_shape_hw=np.array([ih,iw]),
        feature_shape_hw=np.array([h,w]),raw_to_input_affine=affine.copy(),
        diagnostics=dict(rejected_parallel=rejected_parallel,rejected_footprint=rejected_footprint,
                         generated_valid_intersections=int(mask[:,1:].sum()),physical_bins=int(valid.sum())))


def fixed_select(candidates, config=None):
    """Fixed GT-free scoring; baseline remains an ordinary selectable candidate."""
    config=candidates['config'] if config is None else config
    f=np.asarray(candidates['candidate_features']);mask=np.asarray(candidates['candidate_valid'])
    movement=np.linalg.norm(f[:,:,1:3],axis=-1)/config['point_prior_input_diagonal_fraction']
    scores=f[:,:,6:9].mean(-1)+config['point_prior_weight']*(np.sqrt(1+movement**2)-1)
    indices=np.argmin(np.where(mask,scores,np.inf),axis=-1)
    output=np.array(candidates['baseline_points_xy'],copy=True)
    output[:8]=np.asarray(candidates['candidates_xy'])[np.arange(8),indices]
    return dict(points_xy=output,indices=indices,scores=scores,
                original_point_selected=indices==0,uses_gt=False,centroid_preserved=True)
