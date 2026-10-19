"""Finite cold-start proxy pose refiners and explicit physical pose starts.

PL consumes original native point/normal/outline evidence.  CONTOUR uses the
visible structured mesh. IMAGE uses whole native grayscale render/observation
DIS correspondences and rendered depth, and is not a DeepIM reproduction.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import time

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .material_assets import project,unproject
from .material_support import boundary,compile_factors,generic_residuals


CONFIG=dict(max_starts=2,updates=4,max_internal_evaluations=10,min_correspondences=12,
    max_correspondences=2048,flow_grid=8,forward_backward_max_px=2.,flow_max_px=64.,
    target_mask_min=.25,max_step_degrees=30.,max_step_object_diagonal=.5,
    DIS_preset='FAST',PnP_refine_iterations=10)


def body_geometry(frame):
    initial=frame['initial_pose'];Q=np.asarray(initial['R_cf']).T@np.asarray(initial['R_physical'])
    body=np.asarray(frame['native_xyz'])@Q
    return body,Q,float(np.linalg.norm(np.ptp(body,axis=0)))


def pose_valid(frame,pose):
    if not pose or pose.get('available',True) is False:return False,'PRODUCER_UNAVAILABLE'
    try:R=np.asarray(pose['R_physical'],float);t=np.asarray(pose['centroid'],float)
    except (KeyError,TypeError,ValueError):return False,'NO_FINITE_POSE'
    if R.shape!=(3,3)or t.shape!=(3,)or not np.isfinite(R).all()or not np.isfinite(t).all():return False,'NO_FINITE_POSE'
    if not np.allclose(R.T@R,np.eye(3),atol=1e-6,rtol=0)or abs(np.linalg.det(R)-1)>1e-6:return False,'IMPROPER_ROTATION'
    body,_,_=body_geometry(frame)
    if (body@R.T+t)[:,2].min()<=1e-5:return False,'NONPOSITIVE_NATIVE_DEPTH'
    return True,'PROPER_POSITIVE_COMPLETE_ACTION'


def pose(frame,R,t):
    _,Q,_=body_geometry(frame)
    return dict(available=True,R_physical=np.asarray(R).tolist(),centroid=np.asarray(t).tolist(),
        R_cf=(np.asarray(R)@Q.T).tolist(),physical_registration_Q=Q.tolist(),
        cf_extents=list(frame['initial_pose']['cf_extents']))


def same_pose(a,b,scale):
    return bool(np.max(np.abs(np.asarray(a['R_physical'])-b['R_physical']))<=1e-7 and
                np.max(np.abs(np.asarray(a['centroid'])-b['centroid']))<=1e-7*scale)


def choose_starts(frame,d1_result,pf_candidates=()):
    """N0, E0, PF-anchor, YOLO-anchor; aliases are not extra physical starts."""
    _,_,scale=body_geometry(frame)
    pool=[('N0',frame['initial_pose'])]
    pool += [(c['candidate_id'],dict(available=True,R_physical=c['R_physical'],centroid=c['centroid']))
             for c in d1_result['candidates']if c.get('source')=='E0']
    for method in ('POSEFIX_SUBPIX_PF_ANCHOR','POSEFIX_SUBPIX_YOLO_ANCHOR'):
        pool += [(method,c['actual_pose'])for c in pf_candidates if c.get('method')==method]
    result=[];audit=[]
    for name,p in pool:
        valid,why=pose_valid(frame,p)
        if not valid:audit.append(dict(source=name,status=why));continue
        match=next((r for r in result if same_pose(r['actual_pose'],p,scale)),None)
        if match is not None:
            match['source_aliases'].append(name);audit.append(dict(source=name,status='SAME_PHYSICAL_POSE_ALIAS'));continue
        if len(result)<CONFIG['max_starts']:
            result.append(dict(start_id=f'START/{len(result)}/{name}',source_aliases=[name],actual_pose=deepcopy(p)))
            audit.append(dict(source=name,status='PHYSICALLY_DISTINCT_ADMITTED'))
        else:audit.append(dict(source=name,status='FIXED_TWO_START_BUDGET_REACHED'))
    return result,audit


def candidate(frame,p,method,start,update,status,**extra):
    p=deepcopy(p)
    if extra.get('producer_available') is False:p['available']=False
    return dict(id=frame['id'],method=method,
        candidate_id=f'{method}/{start["start_id"]}/update{update}',source=method,
        actual_pose=deepcopy(p),R_physical=p['R_physical'],centroid=p['centroid'],
        original_native_family='GENERIC_REGISTERED_PROXY_BODY',start_id=start['start_id'],
        initialization_source_aliases=start['source_aliases'],update=update,
        producer_status=status,GT_inputs=False,**extra)


def _step_valid(frame,old,new):
    valid,why=pose_valid(frame,new)
    if not valid:return False,why
    _,_,scale=body_geometry(frame)
    angle=Rotation.from_matrix(np.asarray(new['R_physical'])@np.asarray(old['R_physical']).T).magnitude()
    displacement=np.linalg.norm(np.asarray(new['centroid'])-old['centroid'])
    if angle>np.deg2rad(CONFIG['max_step_degrees']):return False,'STEP_ROTATION_GUARD'
    if displacement>CONFIG['max_step_object_diagonal']*scale:return False,'STEP_TRANSLATION_GUARD'
    return True,'UPDATED'


def optimize_factors(frame,start_pose,factors):
    if not factors:return deepcopy(start_pose),dict(status='NO_DIRECT_FACTORS_FALLBACK',objective_calls=0,nfev=0)
    K=np.asarray(frame['K']);_,_,scale=body_geometry(frame)
    c=compile_factors(factors);calls=0;jacobian_calls=0;initial_cost=None
    x=np.r_[cv2.Rodrigues(np.asarray(start_pose['R_physical']))[0].ravel(),np.asarray(start_pose['centroid'])/scale]
    def residual(value):
        nonlocal calls,initial_cost
        calls+=1;R=cv2.Rodrigues(value[:3])[0];t=value[3:]*scale
        result=generic_residuals(c,R,t,K)
        if initial_cost is None:initial_cost=float(result@result)
        return result
    def jacobian(value):
        nonlocal jacobian_calls
        jacobian_calls+=1
        R=cv2.Rodrigues(value[:3])[0];t=value[3:]*scale
        _,j=cv2.projectPoints(c['xyz'],value[:3],t,K,None)
        j=j[:,:6].reshape(-1,2,6);j[:,:,3:]*=scale
        return generic_residuals(c,R,t,K,projection_jacobian=j)[1]
    fit=least_squares(residual,x,jac=jacobian,max_nfev=CONFIG['max_internal_evaluations'],ftol=1e-8,xtol=1e-8,gtol=1e-8)
    p=pose(frame,cv2.Rodrigues(fit.x[:3])[0],fit.x[3:]*scale)
    valid,why=_step_valid(frame,start_pose,p)
    return (p if valid else deepcopy(start_pose)),dict(status=why,objective_calls=calls,jacobian_calls=jacobian_calls,nfev=int(fit.nfev),
        initial_cost=initial_cost,final_cost=float(2*fit.cost),least_squares_status=int(fit.status),
        proposed_pose=p,proposal_accepted=valid)


def old_point_line_factors(frame,d1_result):
    """Original typed observations converted to the same physical body basis."""
    spec=d1_result['packet']['solver_input'];_,Q,_=body_geometry(frame)
    # The original packet carries the ORIGINAL_WD native coordinates/Q.
    family=d1_result['families']['ORIGINAL_WD']
    xyz=np.asarray(family['native_xyz'])@np.asarray(family['physical_registration_Q'])
    partial={int(b['native_corner']):b for b in spec['boundary_observations']}
    factors=[]
    for m in spec['visible']:
        k=int(m['native_corner'])
        if k in partial:continue
        covariance=np.asarray(m.get('fit_covariance_px2',m['covariance_px2']),float)
        weight=float(m.get('fit_reliability_weight',1.))
        covariance=covariance/max(weight,1e-8)
        factors.append(dict(type='ORIGINAL_NATIVE_POINT',observation_dimension=2,object_xyz=xyz[k].tolist(),
            xy=m['xy'],effective_covariance_px2=covariance.tolist(),covariance_px2=covariance.tolist(),
            raw_image_support_xy=[m['xy']],support_id=f'ORIGINAL_NATIVE/{k}',
            physical_source_ids=m.get('physical_source_ids',[]),semantic_source_aliases=[f'ORIGINAL_NATIVE_POINT/{k}'],
            physical_owner_status='UNKNOWN_UNLESS_INDEPENDENTLY_CERTIFIED',
            source_original_image=frame['image_sha256']))
    for k,b in partial.items():
        for j,line in enumerate(b['lines']):
            record=line.get('provenance',{}).get('line_record',{})
            support=np.asarray(record.get('observed_points',[]),float).reshape(-1,2)
            inliers=np.asarray(record.get('inlier_mask',np.ones(len(support),bool)),bool)
            if len(inliers)!=len(support):raise ValueError('Original raw line inlier/support archive differs')
            support=support[inliers&np.isfinite(support).all(-1)]
            # Display mean describes observed support only. It never becomes a
            # 2D normal measurement or a fabricated native physical identity.
            display=support.mean(0).tolist()if len(support)else[None,None]
            weight=float(b.get('fit_reliability_weight',1.))
            variance=float(line['normal_variance_px2'])/max(weight,1e-8)
            pixel_sources=sorted({f'{frame["image_sha256"]}/RAW_GRADIENT_PIXEL/{int(round(x))}/{int(round(y))}'for x,y in support})
            factors.append(dict(type='ORIGINAL_TYPED_NATIVE_NORMAL',observation_dimension=1,
                object_xyz=xyz[k].tolist(),xy=display,display_coordinate_is_full_2D_measurement=False,
                raw_image_support_xy=support.tolist(),normal_unit_xy=line['normal_unit_xy'],
                offset_px=line['offset_px'],effective_variance_px2=variance,
                original_normal_variance_px2=float(line['normal_variance_px2']),fit_reliability_weight=weight,
                effective_variance_rule='original normal_variance_px2 / original fit_reliability_weight',
                support_id=f'ORIGINAL_NATIVE/{k}/line{j}',physical_source_ids=pixel_sources,
                semantic_source_aliases=[f'ORIGINAL_NATIVE_NORMAL/{k}/{j}',
                    f'ORIGINAL_RGB_LINE/{line.get("provenance",{}).get("actual_RGB_line_id","UNKNOWN")}'],
                physical_owner_status='RAW_IMAGE_SAMPLES_WITH_UNVERIFIED_PHYSICAL_LINE_OWNER'if len(support)else'UNKNOWN_NO_RAW_SUPPORT',
                source_original_image=frame['image_sha256']))
    return factors,xyz


def retain_nonoverlapping_original(original,generic):
    if not generic:return list(original)
    def support(r):
        raw=np.asarray(r.get('raw_image_support_xy',[]),float).reshape(-1,2)
        if not len(raw)and r['observation_dimension']==2:
            raw=np.asarray([r['xy']],float)
        return raw[np.isfinite(raw).all(-1)]
    raw_groups=[support(r)for r in generic]
    raw_groups=[p for p in raw_groups if len(p)]
    if not raw_groups:
        return [dict(r,correlation_with_generic='UNKNOWN_NO_GENERIC_RAW_SUPPORT')for r in original]
    def point_segment_distance(points,line):
        if len(line)==1:return float(np.min(np.linalg.norm(points-line[0],axis=1)))
        start=line[:-1];direction=np.diff(line,axis=0)
        delta=points[:,None,:]-start[None,:,:]
        fraction=np.einsum('psi,si->ps',delta,direction)/np.maximum(np.sum(direction**2,axis=1),1e-20)[None,:]
        nearest=start[None,:,:]+np.clip(fraction,0,1)[:,:,None]*direction[None,:,:]
        return float(np.min(np.linalg.norm(points[:,None,:]-nearest,axis=-1)))
    retained=[]
    for record in original:
        points=support(record)
        if not len(points):
            retained.append(dict(record,correlation_with_generic='UNKNOWN_NO_OLD_RAW_SUPPORT'));continue
        overlap=any(min(point_segment_distance(points,new),point_segment_distance(new,points))<=8. for new in raw_groups)
        if not overlap:
            retained.append(dict(record,correlation_with_generic='NO_8PX_RAW_SUPPORT_OR_SEGMENT_OVERLAP_FOUND'))
    return retained


def refine_material(frame,starts,packet,d1_result):
    original,_=old_point_line_factors(frame,d1_result)
    kept=retain_nonoverlapping_original(original,packet['factors'])
    factors=packet['factors']+kept
    method='S3_'+packet['arm'];candidates=[];iterations=[]
    for start in starts:
        current=deepcopy(start['actual_pose'])
        candidates.append(candidate(frame,current,method,start,0,'INITIALIZATION_ALIAS'))
        if not packet['factors']:
            iterations.append(dict(start_id=start['start_id'],status='NO_DIRECT_MATERIAL_SUPPORT_FALLBACK',updates=0));continue
        for update in range(1,CONFIG['updates']+1):
            tick=time.monotonic();new,audit=optimize_factors(frame,current,factors)
            audit.update(start_id=start['start_id'],update=update,wall_seconds=time.monotonic()-tick)
            iterations.append(audit)
            if audit.get('proposed_pose') is not None:
                candidates.append(candidate(frame,audit['proposed_pose'],method,start,update,
                    'VISITED_PROPOSAL_'+audit['status'],producer_available=bool(audit.get('proposal_accepted'))))
            current=new
            if not audit.get('proposal_accepted'):break
    return dict(candidates=candidates,iterations=iterations,material_factor_count=len(packet['factors']),
        preserved_nonoverlapping_original_factor_count=len(factors)-len(packet['factors']),
        preserved_original_unknown_correlation_count=sum(r.get('correlation_with_generic','').startswith('UNKNOWN')for r in kept),
        original_normal_source_policy='actual observed inliers only; unknown correlation retained explicitly; native IDs are semantic aliases',
        generic_factor_sha256=packet['factor_sha256'],new_feature_dimensions=0)


def _nearest_contour_factors(points,q,target_boundary,K,R,t,source,depth=None,max_points=2048):
    if not target_boundary.any()or not len(q):return []
    _,nearest=distance_transform_edt(~target_boundary,return_indices=True)
    yy,xx=np.rint(q).astype(int).T[::-1]
    h,w=target_boundary.shape
    inside=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
    points,q,xx,yy=points[inside],q[inside],xx[inside],yy[inside]
    if not len(points):return []
    match=np.c_[nearest[1,yy,xx],nearest[0,yy,xx]].astype(float)
    delta=q-match;distance=np.linalg.norm(delta,axis=1)
    # Nearest-boundary normal; exact matches use the target-mask derivative.
    normals=delta/np.maximum(distance[:,None],1e-8)
    good=distance<=CONFIG['flow_max_px']
    ids=np.where(good)[0]
    if len(ids)>max_points:ids=ids[np.linspace(0,len(ids)-1,max_points).astype(int)]
    factors=[]
    for i in ids:
        if distance[i]<.1:continue
        factors.append(dict(type=source,observation_dimension=1,object_xyz=points[i].tolist(),
            xy=match[i].tolist(),normal_unit_xy=normals[i].tolist(),offset_px=-float(normals[i]@match[i]),
            effective_variance_px2=4.,physical_source_ids=[f'{source}/{int(match[i,0])}/{int(match[i,1])}']))
    # Several rendered pixels can hit one observed contour pixel.  Retain one.
    unique={tuple(r['physical_source_ids']):r for r in factors}
    return list(unique.values())


def native_outline_factors(frame,body,current,full_mask):
    """Current cuboid convex-outline points, original automatic mask evidence."""
    K=np.asarray(frame['K']);R=np.asarray(current['R_physical']);t=np.asarray(current['centroid'])
    q,z=project(body,R,t,K)
    if np.any(z<=1e-5):return []
    hull=cv2.convexHull(q.astype(np.float32),returnPoints=False).ravel()
    points=[]
    for a,b in zip(hull,np.roll(hull,-1)):
        for f in np.linspace(0.,1.,9):points.append((1-f)*body[a]+f*body[b])
    points=np.asarray(points);pixels,_=project(points,R,t,K)
    return _nearest_contour_factors(points,pixels,boundary(full_mask),K,R,t,'ORIGINAL_NATIVE_CUBOID_OUTLINE')


def _image_update(frame,current,rendered_rgb,depth,render_mask,observed_gray,full_mask):
    K=np.asarray(frame['K']);R=np.asarray(current['R_physical']);t=np.asarray(current['centroid'])
    gray=cv2.cvtColor(rendered_rgb,cv2.COLOR_RGB2GRAY)
    composite=observed_gray.copy();composite[render_mask]=gray[render_mask]
    # New independent matcher per update; no temporal flow/state is supplied.
    forward=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST).calc(composite,observed_gray,None)
    backward=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST).calc(observed_gray,composite,None)
    h,w=depth.shape;yy,xx=np.mgrid[0:h:CONFIG['flow_grid'],0:w:CONFIG['flow_grid']]
    xy=np.c_[xx.ravel(),yy.ravel()];z=depth[xy[:,1],xy[:,0]]
    flow=forward[xy[:,1],xy[:,0]];observed=xy+flow
    valid=(z>0)&np.isfinite(flow).all(-1)&(np.linalg.norm(flow,axis=1)<=CONFIG['flow_max_px'])
    valid&=(observed[:,0]>=0)&(observed[:,0]<w-1)&(observed[:,1]>=0)&(observed[:,1]<h-1)
    reverse=cv2.remap(backward,observed[:,0].astype(np.float32).reshape(-1,1),observed[:,1].astype(np.float32).reshape(-1,1),cv2.INTER_LINEAR).reshape(-1,2)
    target=cv2.remap(np.asarray(full_mask,np.float32),observed[:,0].astype(np.float32).reshape(-1,1),observed[:,1].astype(np.float32).reshape(-1,1),cv2.INTER_LINEAR).ravel()
    fb=np.linalg.norm(flow+reverse,axis=1)
    valid&=(fb<=CONFIG['forward_backward_max_px'])&(target>=CONFIG['target_mask_min'])
    ids=np.where(valid)[0]
    if len(ids)>CONFIG['max_correspondences']:
        ids=ids[np.linspace(0,len(ids)-1,CONFIG['max_correspondences']).astype(int)]
    audit=dict(DIS_calls=2,grid_render_valid=int((z>0).sum()),accepted_correspondences=len(ids),
        observed_representation='WHOLE_NATIVE_GRAYSCALE',render_background='OBSERVATION_OUTSIDE_PROXY_RENDER_MASK',
        temporal_state_used=False,forward_backward_max_px=CONFIG['forward_backward_max_px'])
    if len(ids)<CONFIG['min_correspondences']:
        return deepcopy(current),dict(audit,status='INSUFFICIENT_CORRESPONDENCES_FALLBACK',PnP_calls=0)
    body=unproject(xy[ids],z[ids],R,t,K)
    object_points=np.ascontiguousarray(body,dtype=np.float64)
    image_points=np.ascontiguousarray(observed[ids],dtype=np.float64)
    rvec=cv2.Rodrigues(R)[0];tvec=t.reshape(3,1).copy()
    rvec,tvec=cv2.solvePnPRefineLM(object_points,image_points,K,None,rvec,tvec,
        criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,CONFIG['PnP_refine_iterations'],1e-6))
    proposed=pose(frame,cv2.Rodrigues(rvec)[0],tvec.ravel())
    valid,why=_step_valid(frame,current,proposed)
    audit.update(PnP_calls=1,proposed_pose=proposed,proposal_accepted=valid,status=why,
        matched_render_xy=xy[ids].tolist(),matched_observed_xy=observed[ids].tolist(),
        matched_object_xyz=body.tolist(),matched_render_depth_m=z[ids].tolist(),
        matched_forward_backward_px=fb[ids].tolist(),all_selected_correspondences_archived=True,
        median_forward_backward_px=float(np.median(fb[ids])),native_crop_transform=None,
        depth_recovered_object_support=True)
    return proposed if valid else deepcopy(current),audit


def refine_s4(frame,starts,d1_result,model,dimensions,renderer,image,full_mask,reuse_branches=None):
    K=np.asarray(frame['K']);gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    original,body=old_point_line_factors(frame,d1_result)
    target_boundary=boundary(full_mask)
    result={}
    for arm in ('PL','CONTOUR','IMAGE'):
        if reuse_branches and arm in reuse_branches:
            result[arm]=deepcopy(reuse_branches[arm]);continue
        candidates=[];iterations=[]
        for start in starts:
            current=deepcopy(start['actual_pose'])
            candidates.append(candidate(frame,current,'S4_'+arm,start,0,'INITIALIZATION_ALIAS'))
            for update in range(1,CONFIG['updates']+1):
                tick=time.monotonic();before_calls=renderer.calls
                if arm=='PL':
                    factors=original+native_outline_factors(frame,body,current,full_mask)
                    new,audit=optimize_factors(frame,current,factors)
                    audit['original_point_line_outline_factors']=len(factors)
                else:
                    R=np.asarray(current['R_physical']);t=np.asarray(current['centroid'])
                    rgb,depth,mask=renderer.render(model,dimensions,R,t,K,frame['raw_hw'])
                    render_binding=dict(rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                        depth_sha256=hashlib.sha256(depth.tobytes()).hexdigest(),
                        mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                        current_pose=deepcopy(current),K=frame['K'],raw_hw=frame['raw_hw'],
                        raster_arrays_retained=False,raster_reproducible_from_bound_inputs=True)
                    if arm=='IMAGE':
                        new,audit=_image_update(frame,current,rgb,depth,mask,gray,full_mask)
                    else:
                        contour=boundary(mask);yy,xx=np.where(contour)
                        if len(xx)>CONFIG['max_correspondences']:
                            ids=np.linspace(0,len(xx)-1,CONFIG['max_correspondences']).astype(int);yy,xx=yy[ids],xx[ids]
                        pixels=np.c_[xx,yy].astype(float)
                        points=unproject(pixels,depth[yy,xx],R,t,K)
                        factors=_nearest_contour_factors(points,pixels,target_boundary,K,R,t,'VISIBLE_PROXY_CONTOUR')
                        new,audit=optimize_factors(frame,current,factors)
                        audit['visible_proxy_contour_factors']=len(factors)
                        audit['selected_correspondences']=factors
                    audit['render_binding']=render_binding
                audit.update(start_id=start['start_id'],update=update,wall_seconds=time.monotonic()-tick,
                    render_calls=renderer.calls-before_calls)
                iterations.append(audit)
                if audit.get('proposed_pose') is not None:
                    candidates.append(candidate(frame,audit['proposed_pose'],'S4_'+arm,start,update,
                        'VISITED_PROPOSAL_'+audit['status'],producer_available=bool(audit.get('proposal_accepted'))))
                current=new
                if not audit.get('proposal_accepted'):break
        result[arm]=dict(candidates=candidates,iterations=iterations,
            exact_common_initializations=deepcopy(starts),cold_start_single_image=True,
            temporal_state_used=False,new_feature_dimensions=0)
    return result
