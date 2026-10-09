"""Fixed image-only wide refinement and shared cuboid-boundary refinement.

Inputs contain only native grayscale, predicted points and inherited support.
An edge is estimated once from many normal-search observations and reused by
both incident corners. The method does not establish the semantic identity of
a strong, coherent image edge; that limitation is evaluated rather than hidden.
"""
from __future__ import annotations

from collections import Counter
import copy
import math

import cv2
import numpy as np

EDGES=((0,1),(1,2),(2,3),(3,0),
       (4,5),(5,6),(6,7),(7,4),
       (0,4),(1,5),(2,6),(3,7))
PAIRS=((0,3,3),(1,2,1),(4,7,7),(5,6,5))
METHODS=('WIDE_SUBPIX','BOUNDARY')
CONFIG={
    'WIDE_SUBPIX':dict(winSize=[25,25],zeroZone=[-1,-1],
        criteria=dict(type=int(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT),maxCount=40,epsilon=.001),
        minimum_image_extent=55,arithmetic='float32 OpenCV seeds/output, returned array float64'),
    'BOUNDARY':dict(edges=[list(e) for e in EDGES],normal_search_radius_px=32,
        normal_grid_step_px=1,along_interval=[.1,.9],sample_spacing_px=6,
        minimum_samples=9,maximum_samples=49,minimum_edge_length_px=8,
        gradient='unblurred uint8 -> float32 Sobel ksize3 scale1/8; bilinear remap at normal search locations',
        minimum_normal_gradient=6.,maximum_gradient_normal_angle_deg=30.,
        polarity='fit positive and negative normal gradients separately; deterministic best supported coherent line',
        point_selection='one strongest signed aligned gradient per along-edge sample and polarity; quadratic peak interpolation',
        robust_fit='weighted orthogonal TLS with Huber IRLS; start from predicted normal and median offset',
        huber_delta_px=1.5,irls_iterations=8,minimum_inlier_count=6,
        minimum_inlier_fraction=.60,minimum_inlier_along_span_fraction=.60,
        maximum_inlier_rms_px=1.5,maximum_line_angle_change_deg=15.,
        minimum_intersection_sin_angle=.20,maximum_native_corner_move_px=math.sqrt(2)*32,
        paired_corner_edges=[list(p) for p in PAIRS],
        corner_rule='both endpoints must share their predicted cuboid height-edge line; each endpoint intersects that same line with its best supported nonparallel other incident line; pair applied atomically or both retain seeds',
        pair_fallback='either endpoint unsupported/outside/missing, shared line unavailable, or either intersection unsupported -> both original seeds retained',
        height_edge_semantics='cuboid indexing only; no image-space vertical or 90 degree constraint',
        no_image_right_angle_constraint=True,
        semantic_limit='coherent unrelated edges can be selected; response, polarity and conditioning do not prove pallet boundary identity'),
    'preservation':dict(center=8,missing='original slots retained',unsupported='retained',
        initial_outside_image='retained',nonfinite='retained',new_points_filled=False),
    'external_cap':'not applied here; caller separately records native and original Base q0 anchored CAP1',
    'inference_inputs':['gray','initial_points','prediction_support','method'],
    'GT_inputs':False,'visibility_inputs':False,'pose_inputs':False,'new_weight_training':0,
    'opencv_version':cv2.__version__,
}


def method_configuration():
    return copy.deepcopy(CONFIG)


def _inputs(gray,initial_points,prediction_support):
    gray=np.asarray(gray)
    if gray.ndim!=2 or gray.dtype!=np.uint8 or not gray.size:
        raise ValueError('Expected nonempty raw-resolution grayscale uint8 image')
    points=np.array(initial_points,np.float64,copy=True);support=np.array(prediction_support,bool,copy=True)
    if points.shape!=(9,2) or support.shape!=(9,):
        raise ValueError('Expected initial_points[9,2] and prediction_support[9]')
    usable=support & np.isfinite(points).all(-1) & ~(points==-1).all(-1)
    h,w=gray.shape;inside=usable & (points[:,0]>=0) & (points[:,0]<w) & (points[:,1]>=0) & (points[:,1]<h)
    return gray,points,support,usable,inside


def _initial_reason(k,points,support,inside):
    if not support[k]:return 'unsupported_prediction'
    if not np.isfinite(points[k]).all():return 'nonfinite_initial'
    if (points[k]==-1).all():return 'missing_initial_sentinel'
    if not inside[k]:return 'outside_initial'
    return None


def _wide(gray,points,support,usable,inside):
    output=points.copy();rows=[];calls=0;h,w=gray.shape
    for k in range(8):
        reason=_initial_reason(k,points,support,inside)
        if reason is None and min(h,w)<CONFIG['WIDE_SUBPIX']['minimum_image_extent']:reason='image_too_small'
        row=dict(corner=k,attempted=False,status=reason,exception=None)
        if reason is None:
            calls+=1;row['attempted']=True
            try:
                value=cv2.cornerSubPix(gray,points[k].astype(np.float32).reshape(1,1,2).copy(),
                    (25,25),(-1,-1),(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.001))
                if value is None or np.asarray(value).size!=2:raise ValueError('cornerSubPix returned no coordinate')
                chosen=np.asarray(value,np.float64).reshape(2)
                if not np.isfinite(chosen).all():row['status']='nonfinite_refined'
                elif not (0<=chosen[0]<w and 0<=chosen[1]<h):row['status']='outside_refined'
                else:output[k]=chosen;row['status']='refined'
            except Exception as error:
                row['status']='function_error';row['exception']=dict(type=type(error).__name__,message=str(error))
        row['fallback_reason']=None if row['status']=='refined' else row['status'];rows.append(row)
    return output,dict(corner_records=rows,algorithm_corner_calls=calls,shared_edge_fits=0)


def _remap(values,positions):
    return cv2.remap(values,positions[...,0].astype(np.float32),positions[...,1].astype(np.float32),
                     cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)


def _robust_line(observations,strength,sample_t,seed_a,seed_n):
    cfg=CONFIG['BOUNDARY'];n=seed_n.copy()
    b=-float(np.median(observations@n));base=np.clip(strength/strength.max(),.1,1.)
    for _ in range(cfg['irls_iterations']):
        residual=observations@n+b
        weights=base*np.minimum(1.,cfg['huber_delta_px']/np.maximum(np.abs(residual),1e-12))
        center=np.average(observations,axis=0,weights=weights);delta=observations-center
        covariance=(delta*weights[:,None]).T@delta/weights.sum()
        _,vectors=np.linalg.eigh(covariance);candidate=vectors[:,0]
        if candidate@seed_n<0:candidate=-candidate
        angle=float(np.degrees(np.arccos(np.clip(candidate@seed_n,-1,1))))
        if angle>cfg['maximum_line_angle_change_deg']:
            return None,dict(status='line_direction_inconsistent',angle_change_deg=angle)
        n=candidate;b=-float(n@center)
    residual=np.abs(observations@n+b);inliers=residual<=cfg['huber_delta_px']
    count=int(inliers.sum());fraction=float(inliers.mean())
    span=float(sample_t[inliers].max()-sample_t[inliers].min()) if count else 0.
    # Span refers to the sampled .1-.9 edge interval, not the full cuboid edge.
    span_fraction=span/(cfg['along_interval'][1]-cfg['along_interval'][0])
    rms=float(np.sqrt(np.mean(residual[inliers]**2))) if count else None
    diag=dict(status='supported',inlier_count=count,inlier_fraction=fraction,
        inlier_along_span_fraction=span_fraction,inlier_rms_px=rms,normal=n,offset=b,
        observed_points=observations,normal_gradient_strength=strength,sample_t=sample_t,inliers=inliers,
        angle_change_deg=angle,median_normal_shift_px=float(np.median((observations-seed_a)@seed_n)))
    if count<cfg['minimum_inlier_count']:diag['status']='insufficient_inlier_count'
    elif fraction<cfg['minimum_inlier_fraction']:diag['status']='inconsistent_observations'
    elif span_fraction<cfg['minimum_inlier_along_span_fraction']:diag['status']='insufficient_along_span'
    elif rms>cfg['maximum_inlier_rms_px']:diag['status']='large_line_residual'
    if diag['status']!='supported':return None,diag
    # Independent along-edge support is the primary score. Strength only breaks
    # otherwise similar coherent fits; it is not a semantic confidence score.
    score=float(count*span_fraction/(1.+rms))
    return dict(normal=n,offset=b,score=score,inliers=count,polarity=None),diag


def _edge(gray,gx,gy,points,inside,index,endpoints):
    cfg=CONFIG['BOUNDARY'];a,b=endpoints;row=dict(edge=index,endpoints=[a,b],supported=False)
    if not (inside[a] and inside[b]):
        row['status']='unsupported_or_outside_endpoint';return None,row
    seed_a,seed_b=points[a],points[b];delta=seed_b-seed_a;length=float(np.linalg.norm(delta))
    if length<cfg['minimum_edge_length_px']:
        row['status']='short_predicted_edge';return None,row
    tangent=delta/length;normal=np.array([-tangent[1],tangent[0]])
    samples=int(np.clip(math.ceil(length/cfg['sample_spacing_px'])+1,cfg['minimum_samples'],cfg['maximum_samples']))
    t=np.linspace(*cfg['along_interval'],samples);centers=seed_a+t[:,None]*delta
    offsets=np.arange(-32.,33.);positions=centers[:,None,:]+offsets[None,:,None]*normal
    h,w=gray.shape;valid=(positions[...,0]>=1)&(positions[...,0]<w-1)&(positions[...,1]>=1)&(positions[...,1]<h-1)
    xx=_remap(gx,positions);yy=_remap(gy,positions);response=xx*normal[0]+yy*normal[1]
    magnitude=np.hypot(xx,yy)
    aligned=np.abs(response)>=math.cos(math.radians(cfg['maximum_gradient_normal_angle_deg']))*magnitude
    fits=[];polarity_rows=[]
    for polarity in (1,-1):
        strength=np.where(valid&aligned,polarity*response,-np.inf)
        chosen=np.argmax(strength,axis=1);values=strength[np.arange(samples),chosen]
        available=np.isfinite(values)&(values>=cfg['minimum_normal_gradient'])
        count=int(available.sum())
        pr=dict(polarity=polarity,available_samples=count,total_samples=samples,
            coverage_fraction=count/samples,status='insufficient_gradient_support')
        if count>=cfg['minimum_inlier_count'] and count/samples>=cfg['minimum_inlier_fraction']:
            locations=offsets[chosen].copy()
            # Subpixel peak interpolation along the same normal profile.
            for j in np.flatnonzero(available):
                k=chosen[j]
                if 0<k<len(offsets)-1 and np.isfinite(strength[j,k-1:k+2]).all():
                    left,peak,right=[float(v) for v in strength[j,k-1:k+2]]
                    curvature=left-2*peak+right
                    if curvature < -1e-12:locations[j]+=float(np.clip(.5*(left-right)/curvature,-.5,.5))
            observations=centers[available]+locations[available,None]*normal
            line,fit_diag=_robust_line(observations,values[available].astype(np.float64),t[available],seed_a,normal)
            pr.update(fit_diag)
            # Coverage/inlier fractions must refer to ALL along-edge samples,
            # including samples where image/gradient support is unavailable.
            if line is not None:
                full_fraction=line['inliers']/samples;pr['full_sample_inlier_fraction']=full_fraction
                if full_fraction<cfg['minimum_inlier_fraction']:
                    line=None;pr['status']='insufficient_full_sample_inlier_fraction'
            if line is not None:line['polarity']=polarity;fits.append((line,pr))
        polarity_rows.append(pr)
    row.update(seed_endpoints=[seed_a,seed_b],seed_normal=normal,samples=samples,length_px=length,polarity_fits=polarity_rows)
    if not fits:row['status']='no_supported_coherent_line';return None,row
    line,chosen=max(fits,key=lambda pair:(pair[0]['score'],float(np.median(pair[1]['normal_gradient_strength'])),-abs(pair[1]['median_normal_shift_px'])))
    row.update(status='supported',supported=True,normal=line['normal'],offset=line['offset'],score=line['score'],
        selected_polarity=line['polarity'],selected_fit=chosen)
    return line,row


def _boundary(gray,points,support,usable,inside):
    image=gray.astype(np.float32)
    gx=cv2.Sobel(image,cv2.CV_32F,1,0,ksize=3,scale=1/8,borderType=cv2.BORDER_REFLECT_101)
    gy=cv2.Sobel(image,cv2.CV_32F,0,1,ksize=3,scale=1/8,borderType=cv2.BORDER_REFLECT_101)
    lines=[];edge_rows=[]
    for index,edge in enumerate(EDGES):
        line,row=_edge(gray,gx,gy,points,inside,index,edge);lines.append(line);edge_rows.append(row)
    result=points.copy();corners=[];pair_rows=[];h,w=gray.shape
    for k in range(8):
        reason=_initial_reason(k,points,support,inside)
        corners.append(dict(corner=k,attempted=False,status=reason,selected_edges=None))
    for first,second,shared in PAIRS:
        pair=dict(corners=[first,second],shared_edge=shared,accepted=False,
                  status=None,endpoint_proposals={},shared_line_native_residual_px=None)
        if not (inside[first] and inside[second]):
            pair['status']='paired_endpoint_unavailable'
        elif lines[shared] is None:
            pair['status']='shared_height_line_unsupported'
        else:
            common=lines[shared];chosen={}
            for k in (first,second):
                incident=[i for i,e in enumerate(EDGES) if k in e and i!=shared and lines[i] is not None]
                corners[k]['supported_other_incident_edges']=incident;proposals=[]
                for other in incident:
                    line=lines[other];matrix=np.stack([common['normal'],line['normal']]);sine=abs(float(np.linalg.det(matrix)))
                    if sine<CONFIG['BOUNDARY']['minimum_intersection_sin_angle']:continue
                    value=np.linalg.solve(matrix,-np.array([common['offset'],line['offset']]))
                    if not np.isfinite(value).all() or not (0<=value[0]<w and 0<=value[1]<h):continue
                    movement=float(np.linalg.norm(value-points[k]))
                    if movement>CONFIG['BOUNDARY']['maximum_native_corner_move_px']:continue
                    score=min(common['score'],line['score'])*sine
                    proposals.append((score,value,other,sine))
                if proposals:
                    chosen[k]=max(proposals,key=lambda x:(x[0],-np.linalg.norm(x[1]-points[k])))
                    pair['endpoint_proposals'][str(k)]=dict(other_edge=chosen[k][2],intersection=chosen[k][1],sin_angle=chosen[k][3])
            if len(chosen)==2:
                for k in (first,second):
                    _,value,other,sine=chosen[k];result[k]=value
                    corners[k].update(status='refined',attempted=True,selected_edges=[shared,other],
                        shared_paired_edge=shared,paired_corner=second if k==first else first,intersection_sin_angle=sine)
                residual=result[[first,second]]@common['normal']+common['offset']
                assert np.max(np.abs(residual))<1e-8
                pair.update(accepted=True,status='refined_pair',shared_line_native_residual_px=residual)
            else:
                pair['status']='paired_two_endpoint_intersections_unavailable'
        if not pair['accepted']:
            for k in (first,second):
                if corners[k]['status'] is None:corners[k]['status']=pair['status']
                assert np.array_equal(result[k],points[k],equal_nan=True)
        pair_rows.append(pair)
    for row in corners:row['fallback_reason']=None if row['status']=='refined' else row['status']
    usage=Counter(i for r in corners if r['selected_edges'] is not None for i in r['selected_edges'])
    return result,dict(corner_records=corners,edge_records=edge_rows,paired_records=pair_rows,
        accepted_pairs=sum(p['accepted'] for p in pair_rows),atomic_paired_endpoints=True,algorithm_corner_calls=0,
        shared_edge_searches=len(EDGES),shared_edge_fits=sum(l is not None for l in lines),
        line_usage_counts=[usage[i] for i in range(len(EDGES))],
        shared_edges_used_by_both_endpoints=[i for i in range(len(EDGES)) if usage[i]==2],
        sobel_calls=2,edges_estimated_once=True,independent_corner_candidate_ranking=False)


def correct(gray,initial_points,prediction_support,method):
    if method not in METHODS:raise ValueError('Method must be WIDE_SUBPIX or BOUNDARY')
    gray,initial,support,usable,inside=_inputs(gray,initial_points,prediction_support)
    result,diag=(_wide if method=='WIDE_SUBPIX' else _boundary)(gray,initial,support,usable,inside)
    assert result.shape==(9,2) and result.dtype==np.float64
    assert np.array_equal(result[8],initial[8],equal_nan=True)
    assert np.array_equal(result[~inside],initial[~inside],equal_nan=True)
    assert np.isfinite(result[inside]).all()
    for row in diag['corner_records']:
        k=row['corner'];row['changed']=not np.array_equal(result[k],initial[k],equal_nan=True)
        row['movement_px']=float(np.linalg.norm(result[k]-initial[k])) if usable[k] else None
    diag.update(method=method,input_shape_hw=list(gray.shape),
        status_counts=dict(Counter(r['status'] for r in diag['corner_records'])),
        fallback_counts=dict(Counter(r['fallback_reason'] for r in diag['corner_records'] if r['fallback_reason'] is not None)),
        changed_corners=sum(r['changed'] for r in diag['corner_records']),
        center_preserved=True,missing_prediction_slots_preserved=True,unsupported_and_outside_preserved=True,
        GT_inputs=False,visibility_inputs=False,pose_inputs=False,new_weight_training=0,external_cap_applied=False)
    return result,diag
