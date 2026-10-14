"""Frozen prompt construction and mask evidence; no label or pose fitting input.

The instance mask is class agnostic. Neither its quality score nor foreground
support establishes ownership of a particular physical 3D corner.
"""
from __future__ import annotations

from copy import deepcopy
import cv2
import numpy as np

FACES=((0,1,5,4),(3,2,6,7),(0,3,7,4),(1,2,6,5),(0,1,2,3),(4,5,6,7))


def build_prompts(image_hw, points, self_hidden, raw_appearance, xyz, R, t, raw_mask):
    """Two fixed policies: q8 box, or box plus RGB-confirmed interior anchors."""
    h,w=map(int,image_hw);q=np.asarray(points,float)[:8];H=np.asarray(self_hidden,bool)
    valid=np.isfinite(q).all(1)&~(q==-1).all(1)
    inside=valid&(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)
    if not valid.any():
        return []
    lo=np.maximum(q[valid].min(0),[0.,0.]);hi=np.minimum(q[valid].max(0),[w-1.,h-1.])
    if np.any(hi-lo<2):
        return []
    box=np.r_[lo,hi];r=max(1,int(raw_appearance['diagnostics']['patch_radius_px']))
    probability=np.asarray(raw_appearance['visibility_score'],float)
    candidates=[];X=np.asarray(xyz,float);R=np.asarray(R,float);t=np.asarray(t,float)
    mask=np.asarray(raw_mask)>0
    for face_index,face in enumerate(FACES):
        ids=np.asarray(face,int)
        if not (inside[ids]&~H[ids]).all():
            continue
        center3=X[ids].mean(0);normal=np.zeros(3)
        axis=int(np.argmin(np.ptp(X[ids],axis=0)));normal[axis]=np.sign(center3[axis])
        camera_center=R@center3+t
        front=float((-camera_center)@(R@normal)/max(np.linalg.norm(camera_center),1e-12))
        if front<=np.sin(np.deg2rad(2)):
            continue
        center2=q[ids].mean(0)
        for k in ids:
            if probability[k]<.85:
                continue
            anchor=.7*q[k]+.3*center2
            x,y=np.rint(anchor).astype(int);rr=2
            if x-rr<0 or y-rr<0 or x+rr>=w or y+rr>=h:
                continue
            support=float(mask[y-rr:y+rr+1,x-rr:x+rr+1].mean())
            if support<.8:
                continue
            candidates.append(dict(point=anchor.tolist(),source_corner=int(k),face=face_index,
                                   initial_face_front_cosine=front,RGB_probability=float(probability[k]),
                                   RGB_material_5x5_fraction=support))
    candidates.sort(key=lambda a:(-a['RGB_probability'],-a['RGB_material_5x5_fraction'],a['source_corner'],a['face']))
    anchors=[]
    for candidate in candidates:
        if all(np.linalg.norm(np.asarray(candidate['point'])-a['point'])>=2*r for a in anchors):
            anchors.append(candidate)
        if len(anchors)==3:
            break
    mid=(lo+hi)/2;gap=2*r
    negatives=[]
    for point in ([lo[0]-gap,mid[1]],[hi[0]+gap,mid[1]],[mid[0],lo[1]-gap],[mid[0],hi[1]+gap]):
        if 0<=point[0]<w and 0<=point[1]<h:
            negatives.append(list(map(float,point)))
    points_with_anchors=[a['point'] for a in anchors]+negatives
    common=dict(box=box.tolist(),coordinate_source='initial independent q8; never geometric reprojections')
    return [dict(policy='BOX_ONLY',points=[],labels=[],anchors=[],negatives=[],**common),
            dict(policy='BOX_RGB_ANCHORS',points=points_with_anchors,
                 labels=[1]*len(anchors)+[0]*len(negatives),anchors=anchors,negatives=negatives,**common)]


def choose_mask(masks, quality):
    """Highest uncalibrated predicted mask quality; exact ties keep first."""
    masks=np.asarray(masks,bool);quality=np.asarray(quality,float)
    if masks.ndim!=3 or quality.shape!=(len(masks),) or not np.isfinite(quality).all():
        raise ValueError('Expected finite mask-quality scores and [M,H,W] masks')
    index=int(np.argmax(quality))
    return masks[index].copy(),index


def fixed_support_appearance(mask_appearance, raw_appearance):
    """Raw-trained RGB body confidence plus fixed mask occupancy >= .5.

    No raw RGB head is run on changed SAM binary feature meanings. Continuous
    evidence is min(raw body score, mask inward occupancy), not a probability.
    """
    result=deepcopy(mask_appearance)
    rows=result['diagnostics']['corners'];body=np.asarray(raw_appearance['visibility_score'],float)
    for k,state in enumerate(result['states']):
        if rows[k].get('reason') in ('missing_or_nonfinite_prediction','predicted_coordinate_outside_frame',
                                     'supplied_initial_pose_self_occlusion'):
            continue
        support=float(np.clip(rows[k].get('inward_material_fraction') or 0.,0.,1.))
        score=min(float(body[k]),support)
        visible=body[k]>=.35 and support>=.5
        result['visibility_score'][k]=score
        result['states'][k]=('VISIBLE_BOUNDARY' if result['boundary_score'][k]>=.35 else 'VISIBLE_INTERIOR') if visible else 'UNKNOWN'
        if visible:
            result['local_accuracy_score'][k]=max(result['local_accuracy_score'][k],.12)
        rows[k]['foundation_inward_occupancy']=support
        rows[k]['frozen_raw_RGB_body_score']=float(body[k])
        rows[k]['reason']='frozen_foundation_mask_and_raw_RGB_body_support'
    result['diagnostics']['foundation_support']=dict(mask_support_threshold=.5,raw_body_threshold=.35,
        continuous_score='min(raw_RGB_body_score,foundation_mask_inward_occupancy)',
        score_is_calibrated_probability=False,ownership_proven=False,GT_inputs=False)
    return result
