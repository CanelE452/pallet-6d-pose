"""Evidence-driven generic mesh support, independent of eight cuboid corners.

COMMON and BI see identical geometry, visibility and three image cues.  They
intervene on cue adoption/recovery order.  Dimensionality follows supported
directions in both arms; conditional variance is shared (no invented role fit).
All ownership and visibility here are automatic proxy hypotheses.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import hashlib
import json

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from .material_assets import project


CONFIG = dict(max_edges=512, crease_degrees=35, samples=9, search_radius=8,
    minimum_support=3, tangent_max_degrees=35, line_RMS_max_px=3.,
    junction_angle_min_degrees=30., junction_distance_max_px=8.,
    conditional_variance_floor_px2=4., visibility_absolute_m=.005,
    visibility_depth_fraction=.005, canny_low=50, canny_high=150,
    role_boundary_distance_px=2., mask_threshold=.5, none_prior=.01)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def boundary(mask):
    binary = np.asarray(mask) >= CONFIG['mask_threshold']
    return binary & ~cv2.erode(binary.astype(np.uint8), np.ones((3,3),np.uint8)).astype(bool)


def cue_images(image, full_mask, tight_mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb = cv2.Canny(gray, CONFIG['canny_low'], CONFIG['canny_high']) > 0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    result = {'RGB_FULL': (rgb, gx, gy)}
    for key, mask in [('MASK_TIGHT', tight_mask), ('MASK_FULL', full_mask)]:
        mask = np.asarray(mask, np.float32)
        result[key] = (boundary(mask), cv2.Sobel(mask,cv2.CV_32F,1,0,ksize=3),
                       cv2.Sobel(mask,cv2.CV_32F,0,1,ksize=3))
    return result


def visible_samples(xyz, R, t, K, depth):
    q, z = project(xyz, R, t, K)
    h, w = depth.shape
    rounded = np.rint(q).astype(int)
    okay = np.isfinite(q).all(-1) & (z > 1e-5)
    matched = np.zeros(len(q), bool)
    tolerance = np.maximum(CONFIG['visibility_absolute_m'], CONFIG['visibility_depth_fraction']*z)
    for dx, dy in ((x,y) for y in (-1,0,1) for x in (-1,0,1)):
        xx, yy = rounded[:,0]+dx, rounded[:,1]+dy
        inside = okay & (xx>=0)&(xx<w)&(yy>=0)&(yy<h)
        ids = np.where(inside)[0]
        values = depth[yy[ids],xx[ids]]
        matched[ids] |= (values>0)&(np.abs(values-z[ids]) <= tolerance[ids])
    return q, z, matched


def _line_candidate(q, tangent, visible, cue, full_mask, image_sha):
    edge, gx, gy = cue
    normal = np.array([-tangent[1],tangent[0]])
    offsets = np.arange(-CONFIG['search_radius'],CONFIG['search_radius']+1)
    # Nearest distance first, then the negative offset, independent of accuracy.
    offsets = sorted(offsets, key=lambda x:(abs(x),x))
    hits, sample_ids, shifts = [], [], []
    h,w = edge.shape
    for j in np.where(visible)[0]:
        for shift in offsets:
            xy = q[j] + shift*normal
            x,y = np.rint(xy).astype(int)
            if x<0 or x>=w or y<0 or y>=h or not edge[y,x] or full_mask[y,x] < .25:
                continue
            grad = np.array([gx[y,x],gy[y,x]],float)
            norm = np.linalg.norm(grad)
            if norm <= 1e-8 or abs(grad@normal)/norm < np.cos(np.deg2rad(CONFIG['tangent_max_degrees'])):
                continue
            hits.append([float(x),float(y)]);sample_ids.append(int(j));shifts.append(int(shift));break
    if len(hits) < CONFIG['minimum_support']:
        return None, 'INSUFFICIENT_DIRECT_IMAGE_SUPPORT'
    hits=np.asarray(hits);center=hits.mean(0)
    _,_,vh=np.linalg.svd(hits-center,full_matrices=False)
    direction=vh[0]
    if abs(direction@tangent) < np.cos(np.deg2rad(CONFIG['tangent_max_degrees'])):
        return None, 'IMAGE_TANGENT_INCONSISTENT'
    n=np.array([-direction[1],direction[0]])
    if n@normal<0:n=-n
    offset=-float(n@center)
    rms=float(np.sqrt(np.mean((hits@n+offset)**2)))
    if rms>CONFIG['line_RMS_max_px']:
        return None,'NONLINEAR_OR_INCONSISTENT_IMAGE_SUPPORT'
    variance=CONFIG['conditional_variance_floor_px2']+rms*rms
    tangential=hits@np.array([n[1],-n[0]])
    # Same image line/interval is a single physical evidence source, whatever
    # cue or proxy support proposed it.  Cue names are deliberately omitted.
    sign=1 if n[np.argmax(np.abs(n))]>=0 else -1
    payload=dict(image_sha256=image_sha,normal=np.round(sign*n,3).tolist(),
        offset=round(sign*offset,1),interval=np.round([tangential.min(),tangential.max()],1).tolist())
    source='RAW_IMAGE_LINE/'+_digest(payload)
    pixel_sources=sorted({f'{image_sha}/RAW_GRADIENT_PIXEL/{int(x)}/{int(y)}'for x,y in hits})
    return dict(normal_unit_xy=n.tolist(),offset_px=offset,xy=center.tolist(),
        normal_variance_px2=variance,conditional_variance_px2=variance,
        effective_variance_px2=variance,normal_RMS_px=rms,
        raw_image_support_xy=hits.tolist(),visible_sample_ids=sample_ids,
        normal_shifts_px=shifts,physical_source_ids=pixel_sources,raw_line_summary_id=source,
        observed_tangent_interval_px=[float(tangential.min()),float(tangential.max())],
        normalized_cue_cost=float((np.mean(np.asarray(shifts,float)**2)+rms*rms)/variance)), 'SUPPORTED'


def extract_supports(frame, model, dimensions, R, t, K, rendered_depth, full_mask, tight_mask, image):
    """One shared archive then two actual policy packets; no role dimension rule."""
    cues=cue_images(image,full_mask,tight_mask)
    render_boundary=boundary(rendered_depth>0)
    boundary_distance=distance_transform_edt(~render_boundary)
    edges=model.body_edges(dimensions)
    vertices=model.body_vertices(dimensions)
    records=[]
    for index,(ends,vertex_ids) in enumerate(zip(edges,model.edge_indices)):
        fractions=np.linspace(0.,1.,CONFIG['samples'])
        points=ends[0][None,:]*(1-fractions[:,None])+ends[1][None,:]*fractions[:,None]
        q,z,visible=visible_samples(points,R,t,K,rendered_depth)
        delta=q[-1]-q[0];length=float(np.linalg.norm(delta))
        record=dict(support_id=f'{model.key}/GEOMETRIC_EDGE/{int(vertex_ids[0])}-{int(vertex_ids[1])}',
            edge_index=index,vertex_ids=vertex_ids.tolist(),object_segment_xyz=ends.tolist(),
            predicted_projection_xy=q.tolist(),predicted_visibility=visible.tolist(),
            visibility_source='AUTOMATIC_CURRENT_N0_PROXY_ZBUFFER',
            owner_evidence_status='AUTOMATIC_PROXY_HYPOTHESIS_NOT_PHYSICAL_GROUND_TRUTH',
            image_sha256=frame['image_sha256'],cue_candidates={},cue_rejections={})
        if visible.sum()<CONFIG['minimum_support'] or length<2.:
            record.update(role='UNKNOWN',support_status='NO_VISIBLE_SEGMENT');records.append(record);continue
        tangent=delta/length
        inside=np.rint(q[visible]).astype(int)
        inside[:,0]=np.clip(inside[:,0],0,rendered_depth.shape[1]-1)
        inside[:,1]=np.clip(inside[:,1],0,rendered_depth.shape[0]-1)
        record['role']='B' if np.mean(boundary_distance[inside[:,1],inside[:,0]]<=CONFIG['role_boundary_distance_px'])>=.5 else 'I'
        record['predicted_visible_fraction_intervals']=[[float(fractions[j]),float(fractions[j])]for j in np.where(visible)[0]]
        for cue_name,cue in cues.items():
            candidate,why=_line_candidate(q,tangent,visible,cue,full_mask,frame['image_sha256'])
            if candidate is not None:
                record['cue_candidates'][cue_name]=dict(candidate,cue=cue_name)
            else:record['cue_rejections'][cue_name]=why
        record['support_status']='IMAGE_SUPPORTED' if record['cue_candidates']else'NO_DIRECT_IMAGE_SUPPORT'
        records.append(record)
    packets={arm:_policy(records,vertices,R,t,K,arm)for arm in ('COMMON','BI')}
    return dict(shared_mesh_support_archive=records,policy_packets=packets,
        geometry_and_visibility_archive_sha256=_digest(records),
        physical_owner_oracle='NOT_APPLICABLE_WITH_REASON_NO_INDEPENDENT_PHYSICAL_LABELS',
        structural_review_source='AUTOMATIC_RENDER_AND_IMAGE_ALGORITHM; NO_HUMAN_CONFIRMATION',
        interventions=dict(cue_adoption='COMMON minimum normalized score; BI role cue priority',
            recovery_order='COMMON joint fullRGB/tight/fullmask ranking; BI B tight/fullmask thenRGB; I RGB then tight/fullmask',
            dimension='identical two/one/zero independent-direction rule, actual evidence may differ',
            covariance='shared4+RMS² for both; role-specific floor not fitted; no bias correction'))


def _policy(records, vertices, R, t, K, arm):
    selected=[];none=[]
    priority={'RGB_FULL':0,'MASK_TIGHT':1,'MASK_FULL':2}
    for record in records:
        candidates=record['cue_candidates']
        if not candidates:
            none.append(dict(support_id=record['support_id'],reason=record['support_status']));continue
        if arm=='COMMON':
            cue=min(candidates,key=lambda x:(candidates[x]['normalized_cue_cost'],priority[x]))
        else:
            order=('MASK_TIGHT','MASK_FULL','RGB_FULL')if record['role']=='B'else('RGB_FULL','MASK_TIGHT','MASK_FULL')
            cue=next(x for x in order if x in candidates)
        factor=dict(candidates[cue],support_id=record['support_id'],role=record['role'],
            observation_dimension=1,type='GENERIC_OBJECT_LINE_NORMAL',
            object_xyz=np.mean(record['object_segment_xyz'],axis=0).tolist(),
            object_segment_xyz=record['object_segment_xyz'],vertex_ids=record['vertex_ids'],
            visible_object_segment_intervals=record.get('predicted_visible_fraction_intervals',[]),
            visibility_source=record['visibility_source'],owner_evidence_status=record['owner_evidence_status'],
            source_original_image=record['image_sha256'],uncertainty_calibration='CONDITIONAL_UNCALIBRATED_PROXY_SUPPORT')
        selected.append(factor)
    incident=defaultdict(list)
    for i,r in enumerate(selected):
        for vertex in r['vertex_ids']:incident[int(vertex)].append(i)
    consumed=set();used_sources=set();junctions=[]
    for vertex,indices in sorted(incident.items()):
        q,z=project(vertices[[vertex]],R,t,K)
        if z[0]<=1e-5:continue
        options=[]
        for a,b in combinations(indices,2):
            if a in consumed or b in consumed:continue
            first,second=selected[a],selected[b]
            if set(first['physical_source_ids']) & set(second['physical_source_ids']):continue
            n=np.asarray([first['normal_unit_xy'],second['normal_unit_xy']])
            if abs(np.linalg.det(n))<np.sin(np.deg2rad(CONFIG['junction_angle_min_degrees'])):continue
            xy=np.linalg.solve(n,-np.asarray([first['offset_px'],second['offset_px']]))
            gap=float(np.linalg.norm(xy-q[0]))
            if gap<=CONFIG['junction_distance_max_px']:
                options.append((gap,a,b,xy,n))
        if not options:continue
        gap,a,b,xy,n=min(options,key=lambda x:(x[0],x[1],x[2]))
        first,second=selected[a],selected[b]
        sources=first['physical_source_ids']+second['physical_source_ids']
        if used_sources.intersection(sources):continue
        inverse=np.linalg.inv(n)
        covariance=inverse@np.diag([first['normal_variance_px2'],second['normal_variance_px2']])@inverse.T
        junctions.append(dict(support_id=f'GENERIC_MESH_VERTEX/{vertex}',type='GENERIC_OBJECT_JUNCTION',
            observation_dimension=2,object_xyz=vertices[vertex].tolist(),xy=xy.tolist(),
            covariance_px2=covariance.tolist(),conditional_covariance_px2=covariance.tolist(),
            effective_covariance_px2=covariance.tolist(),physical_source_ids=sources,
            supporting_mesh_edges=[first['support_id'],second['support_id']],
            role='B'if first['role']=='B'or second['role']=='B'else'I',
            source_original_image=first['source_original_image'],
            raw_image_support_xy=first.get('raw_image_support_xy',[])+second.get('raw_image_support_xy',[]),
            owner_evidence_status='AUTOMATIC_PROXY_JUNCTION_NOT_PHYSICAL_GROUND_TRUTH',
            visibility_source='SHARED_N0_PROXY_ZBUFFER',observed_intersection_distance_from_initial_px=gap,
            uncertainty_calibration='CONDITIONAL_UNCALIBRATED_PROXY_SUPPORT',
            underlying_line_factors_replaced=True))
        consumed.update((a,b));used_sources.update(sources)
    factors=list(junctions);duplicates=[]
    for i,r in enumerate(selected):
        if i in consumed:continue
        if used_sources.intersection(r['physical_source_ids']):
            duplicates.append(dict(support_id=r['support_id'],reason='RAW_SOURCE_ALREADY_CONSUMED'));continue
        factors.append(r);used_sources.update(r['physical_source_ids'])
    ids=[source for r in factors for source in r['physical_source_ids']]
    assert len(ids)==len(set(ids))
    return dict(arm=arm,factors=factors,no_direct_support=none,source_duplicates_removed=duplicates,
        evidence_dimensions=dict(Counter(str(r['observation_dimension'])for r in factors)),
        role_counts=dict(Counter(r['role']for r in factors)),raw_source_ids=sorted(used_sources),
        role_does_not_force_observation_dimension=True,
        factor_sha256=_digest(factors),calibration_parameter_count=0,
        conditional_variance_rule='4+measured line RMS squared; junction covariance propagated from two actual image normals',
        physical_owner_oracle='NOT_APPLICABLE_WITH_REASON_NO_INDEPENDENT_LABELS')


def compile_factors(factors):
    return dict(xyz=np.asarray([r['object_xyz']for r in factors],float).reshape(-1,3),
        dimensions=np.asarray([r['observation_dimension']for r in factors],int),
        xy=np.asarray([r['xy']for r in factors],float).reshape(-1,2),
        inverse_cholesky=np.asarray([np.linalg.inv(np.linalg.cholesky(r['effective_covariance_px2']))
            if r['observation_dimension']==2 else np.eye(2)for r in factors]).reshape(-1,2,2),
        variance=np.asarray([r.get('effective_variance_px2',1.)for r in factors]),
        determinant=np.asarray([np.linalg.det(r['effective_covariance_px2'])if r['observation_dimension']==2
            else r['effective_variance_px2']for r in factors]),
        normal=np.asarray([r.get('normal_unit_xy',[0.,0.])for r in factors]).reshape(-1,2),
        offset=np.asarray([r.get('offset_px',0.)for r in factors]))


def generic_residuals(factors, R, t, K, none_prior=CONFIG['none_prior'], *, projection_jacobian=None):
    """Normalized Gaussian/uniform-none mixture on general object 3D supports."""
    c=factors if isinstance(factors,dict)else compile_factors(factors)
    if not len(c['xyz']):
        return (np.empty(0),np.empty((0,6)))if projection_jacobian is not None else np.empty(0)
    q,z=project(c['xyz'],R,t,K);dims=c['dimensions'];point=dims==2
    residual=np.einsum('nij,nj->ni',c['inverse_cholesky'],q-c['xy'])
    residual[~point,0]=(np.einsum('ni,ni->n',c['normal'][~point],q[~point])+c['offset'][~point])/np.sqrt(c['variance'][~point])
    residual[~point,1]=0.
    norm2=np.einsum('ni,ni->n',residual,residual)
    log_density=-.5*dims*np.log(2*np.pi)-.5*np.log(np.maximum(c['determinant'],1e-20))
    log_none=np.log(none_prior)-dims*np.log(2*CONFIG['search_radius'])
    log_signal=np.log1p(-none_prior)+log_density
    energy=np.maximum(0.,-2*(np.logaddexp(log_signal-.5*norm2,log_none)-np.logaddexp(log_signal,log_none)))
    factor=np.sqrt(energy/np.maximum(norm2,1e-30))
    posterior=np.exp(log_signal-.5*norm2-np.logaddexp(log_signal-.5*norm2,log_none))
    small=norm2<1e-10
    factor[small]=np.sqrt(np.exp(log_signal[small]-np.logaddexp(log_signal[small],log_none[small])))
    jac=None
    if projection_jacobian is not None:
        j=np.asarray(projection_jacobian)
        jac=np.einsum('nij,njk->nik',c['inverse_cholesky'],j)
        jac[~point,0]=np.einsum('ni,nij->nj',c['normal'][~point],j[~point])/np.sqrt(c['variance'][~point])[:,None]
        jac[~point,1]=0.
        coefficient=np.zeros(len(q))
        coefficient[~small]=posterior[~small]/np.maximum(energy[~small],1e-30)-1/norm2[~small]
        derivative=np.eye(2)[None,:,:]+coefficient[:,None,None]*residual[:,:,None]*residual[:,None,:]
        jac=factor[:,None,None]*np.einsum('nij,njk->nik',derivative,jac)
    residual*=factor[:,None]
    residual[(z<=1e-5)|~np.isfinite(q).all(-1)]=1e3
    keep=np.c_[np.ones(len(q),bool),point]
    if jac is not None:
        jac[(z<=1e-5)|~np.isfinite(q).all(-1)]=0.
        return residual[keep],jac[keep]
    return residual[keep]
