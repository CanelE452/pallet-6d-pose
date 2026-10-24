"""Source-only exact/radius2 target support rules; no RGB, model or pose code."""
import numpy as np

V,E,O=0,1,2
CLASS_NAMES=['VISIBLE','EXTERNAL_SUPPORT','OUTSIDE']
JOINT_STATES=[(V,V),(E,V),(E,E),(O,V),(O,E),(O,O)]
JOINT_NAMES=['VISIBLE_EXACT_AND_NEAR','HIDDEN_EXACT_VISIBLE_NEAR','HIDDEN_EXACT_AND_NEAR','OUTSIDE_EXACT_VISIBLE_NEAR','OUTSIDE_EXACT_HIDDEN_SUPPORT_NEAR','OUTSIDE_EXACT_AND_NEAR']


def decode_mask(values,source):
    values=np.asarray(values)
    if values.ndim!=2:raise ValueError('Single channel native-resolution mask required')
    if source.upper()in{'SYNTH','SYNTHETIC'}:return values>127
    if source.upper()in{'YCB','YCBV','YCB-V'}:
        if not set(np.unique(values))<={0,255}:raise ValueError('Unexpected BOP binary mask encoding')
        return values>0
    raise ValueError('Unregistered source mask encoding')


def validate_masks(full,visible):
    f,v=np.asarray(full,bool),np.asarray(visible,bool)
    if f.ndim!=2 or f.shape!=v.shape:raise ValueError('Aligned2D masks required')
    if (v&~f).any():raise ValueError('Visible is not a subset of target full silhouette')
    return f,v


def query_pixel(query,hw):
    q=np.asarray(query,float);h,w=hw
    if q.shape!=(2,)or not np.isfinite(q).all():return None
    p=np.floor(q+.5).astype(int)
    return tuple(p)if 0<=q[0]<w and 0<=q[1]<h and 0<=p[0]<w and 0<=p[1]<h else None


def patch_slices(px,py,radius,hw):
    if radius not in{0,2}:raise ValueError('Only registered exact0/radius2 support targets permitted')
    h,w=hw
    return(slice(max(0,py-radius),min(h,py+radius+1)),slice(max(0,px-radius),min(w,px+radius+1)))


def state_from_bits(full,visible):
    if visible and not full:raise ValueError('Invalid full/visible bits')
    return V if visible else E if full else O


def query_labels(full,visible,query):
    f,v=validate_masks(full,visible);pixel=query_pixel(query,f.shape)
    if pixel is None:return dict(valid=False,classes=[None,None],joint=None,semantic_self=None)
    px,py=pixel;classes=[];counts=[]
    for radius in[0,2]:
        sl=patch_slices(px,py,radius,f.shape);fp,vp=f[sl],v[sl]
        classes.append(state_from_bits(bool(fp.any()),bool(vp.any())))
        counts.append(dict(radius=radius,window_pixels=int(fp.size),full_pixels=int(fp.sum()),visible_pixels=int(vp.sum()),hidden_support_pixels=int((fp&~vp).sum()),mixed_visible_and_hidden=bool(vp.any()and(fp&~vp).any())))
    pair=tuple(classes)
    if pair not in JOINT_STATES:raise AssertionError('Impossible exact/near ground-truth pair')
    return dict(valid=True,pixel=[int(px),int(py)],classes=classes,joint=JOINT_STATES.index(pair),support_counts=counts,semantic_self=None)


def joint_codes(exact,near,valid=None):
    exact,near=np.broadcast_arrays(np.asarray(exact),np.asarray(near))
    valid=np.ones(exact.shape,bool)if valid is None else np.broadcast_to(np.asarray(valid,bool),exact.shape)
    out=np.full(exact.shape,-1,np.int64)
    for j,pair in enumerate(JOINT_STATES):out[valid&(exact==pair[0])&(near==pair[1])]=j
    if (valid&(out<0)).any():raise ValueError('Invalid or impossible exact/near label pair')
    return out


def joint_marginals(probabilities):
    p=np.asarray(probabilities,float)
    if p.shape[-1]!=6 or not np.isfinite(p).all()or(p<0).any()or not np.allclose(p.sum(-1),1,atol=1e-6):raise ValueError('Six-state softmax probabilities required')
    exact=np.stack([p[...,0],p[...,1]+p[...,2],p[...,3]+p[...,4]+p[...,5]],-1)
    near=np.stack([p[...,0]+p[...,1]+p[...,3],p[...,2]+p[...,4],p[...,5]],-1)
    return np.stack([exact,near],-2)


def external_evidence(full,visible,query,radius=0,other_visible=None,measured_depth=None,target_render_depth=None,depth_tolerance=None):
    """Evidence attributes only: never alters a support-state class.

    Measured depth alone, object-center z, and per-instance valid-pixel totals
    cannot confirm pixel occlusion. Optional scene and target-render depth must
    be aligned native arrays in the same units, with a prespecified tolerance.
    """
    f,v=validate_masks(full,visible);pixel=query_pixel(query,f.shape)
    if pixel is None:return dict(valid=False,evidence='INVALID_QUERY',label_changed=False)
    px,py=pixel;sl=patch_slices(px,py,radius,f.shape);fp,vp=f[sl],v[sl]
    state=state_from_bits(bool(fp.any()),bool(vp.any()));support=fp&~vp
    y0,y1=max(0,py-radius-1),min(f.shape[0],py+radius+2);x0,x1=max(0,px-radius-1),min(f.shape[1],px+radius+2)
    f_context,v_context=f[y0:y1,x0:x1],v[y0:y1,x0:x1]
    boundary=bool((f_context.any()and not f_context.all())or(v_context.any()and not v_context.all()))
    clipped=bool(px-radius<0 or py-radius<0 or px+radius>=f.shape[1]or py+radius>=f.shape[0])
    other_count=0;depth_valid=0;depth_front=0;depth_conflict=0
    if other_visible is not None:
        other=np.asarray(other_visible,bool)
        if other.shape!=f.shape:raise ValueError('Other-instance visible mask must share canvas')
        other_count=int((support&other[sl]).sum())
    if measured_depth is not None and target_render_depth is not None:
        if depth_tolerance is None or depth_tolerance<=0:raise ValueError('Explicit source-locked depth tolerance required')
        d,t=np.asarray(measured_depth,float),np.asarray(target_render_depth,float)
        if d.shape!=f.shape or t.shape!=f.shape:raise ValueError('Depths must share mask canvas and units')
        dv,tv=d[sl],t[sl];valid=support&np.isfinite(dv)&np.isfinite(tv)&(dv>0)&(tv>0)
        depth_valid=int(valid.sum());depth_front=int((valid&(dv<tv-depth_tolerance)).sum());depth_conflict=int((valid&(dv>tv+depth_tolerance)).sum())
    tier='NOT_EXTERNAL_SUPPORT'
    if state==E:
        tier='EXTERNAL_CONFLICTING_DEPTH'if depth_conflict else'EXTERNAL_DEPTH_FRONT_EVIDENCE'if depth_front else'EXTERNAL_OTHER_VISIBLE_EVIDENCE'if other_count else'EXTERNAL_AMBIGUOUS_RENDER_BOUNDARY'if boundary or clipped else'EXTERNAL_RENDER_ONLY_INTERIOR_UNVERIFIED'
    return dict(valid=True,radius=radius,source_state=state,source_state_name=CLASS_NAMES[state],evidence=tier,hidden_support_pixels=int(support.sum()),other_visible_overlap_pixels=other_count,paired_scene_target_depth_valid_pixels=depth_valid,scene_depth_in_front_pixels=depth_front,scene_depth_behind_conflict_pixels=depth_conflict,render_boundary_uncertain=boundary,window_clipped_by_image=clipped,depth_has_target_reference=target_render_depth is not None,label_changed=False,semantic_self=None)


def hierarchical_states(probabilities):
    """Registered .5 logical-support diagnostic; valid paired outputs.

    Visible takes precedence; remaining target-full support is external-support.
    Nested sums preserve the exact/near threshold ordering numerically as well
    as mathematically. This can call E with pE below .5; user-facing mechanism
    decisions instead use the separate confidence-qualified joint MAP decoder.
    """
    p=np.asarray(probabilities,float);joint_marginals(p)
    visible0=p[...,0]
    full0=(p[...,0]+p[...,1])+p[...,2]
    visible2=(p[...,0]+p[...,1])+p[...,3]
    full2=(full0+p[...,3])+p[...,4]
    exact=np.where(visible0>=.5,V,np.where(full0>=.5,E,O))
    near=np.where(visible2>=.5,V,np.where(full2>=.5,E,O))
    paired=np.stack([exact,near],-1)
    # Any failure is a contract error, never a post-hoc relabeling rule.
    joint_codes(exact,near)
    return paired


def confident_joint_map(probabilities):
    """Registered user-facing jointMAP only ifmaxjoint>=.5, otherwise U(-1).

    U applies to the whole pair, even if one scale's marginal is confident.
    This is a fixed confidence heuristic, not calibrated pallet reliability.
    """
    p=np.asarray(probabilities,float);marginal=joint_marginals(p)
    best=p.argmax(-1);maximum=p.max(-1);accepted=maximum>=.5
    decision=np.where(accepted,best,-1)
    pair=np.asarray(JOINT_STATES,np.int64)[best]
    pair=np.where(accepted[...,None],pair,-1)
    for scale in[0,1]:
        chosen=np.take_along_axis(marginal[...,scale,:],np.asarray(JOINT_STATES,np.int64)[best][...,scale,None],axis=-1)[...,0]
        if np.any(accepted&(chosen<.5-1e-12)):raise AssertionError('Accepted joint does not support chosen scale class')
    return dict(joint=decision,classes=pair,accepted=accepted,max_joint_probability=maximum)
