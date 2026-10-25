"""V235 pure in-memory same-q opaque donor pair; no filesystem/model access.

Caller admits only original FIT-D parents and an approved different-object FIT
YCB donor. This function does not infer D, identity, native or split from pixels.
"""
import hashlib
import numpy as np

MAX_SIDES=(32,40,48,56)
CANDIDATES_PER_ROLE=256


def _resize(rgb,mask,height,width):
    """Half-pixel bilinear RGB with border clamp + round-half-up; mask nearest."""
    old_h,old_w=mask.shape
    sx=(np.arange(width,dtype=np.float64)+.5)*old_w/width-.5
    sy=(np.arange(height,dtype=np.float64)+.5)*old_h/height-.5
    ix=np.floor(sx).astype(np.int64);iy=np.floor(sy).astype(np.int64)
    wx=(sx-ix)[None,:,None];wy=(sy-iy)[:,None,None]
    x0=np.clip(ix,0,old_w-1);x1=np.clip(ix+1,0,old_w-1)
    y0=np.clip(iy,0,old_h-1);y1=np.clip(iy+1,0,old_h-1)
    nw=rgb[y0[:,None],x0[None,:]].astype(np.float64)
    ne=rgb[y0[:,None],x1[None,:]].astype(np.float64)
    sw=rgb[y1[:,None],x0[None,:]].astype(np.float64)
    se=rgb[y1[:,None],x1[None,:]].astype(np.float64)
    value=((1-wy)*((1-wx)*nw+wx*ne)+wy*((1-wx)*sw+wx*se))
    result=np.clip(np.floor(value+.5),0,255).astype(np.uint8)
    nx=np.minimum(((np.arange(width,dtype=np.float64)+.5)*old_w/width).astype(np.int64),old_w-1)
    ny=np.minimum(((np.arange(height,dtype=np.float64)+.5)*old_h/height).astype(np.int64),old_h-1)
    return result,mask[ny[:,None],nx[None,:]].copy()


def _plan(row_id,bbox_height,bbox_width):
    digest=hashlib.sha256(('V235:'+row_id).encode('utf-8')).digest()
    seed=int.from_bytes(digest,'big') ^ 235
    rng=np.random.Generator(np.random.PCG64(seed))
    side=MAX_SIDES[int(rng.integers(0,len(MAX_SIDES)))]
    maximum=max(bbox_height,bbox_width)
    height=max(1,int(np.floor(bbox_height*side/maximum+.5)))
    width=max(1,int(np.floor(bbox_width*side/maximum+.5)))
    positions={}
    # Pregenerate BOTH complete pools before any query/reference acceptance.
    # Sampling is with replacement: all x draws, then y draws, COVER then SHAM.
    for role in ('COVER','SHAM'):
        x=rng.integers(0,97-width,size=CANDIDATES_PER_ROLE,dtype=np.int64)
        y=rng.integers(0,97-height,size=CANDIDATES_PER_ROLE,dtype=np.int64)
        positions[role]=np.stack((x,y),axis=1)
    return seed,side,height,width,positions


def _reference_taps(query,L,reference):
    nearest=np.rint(reference).astype(np.int64)  # nearest integer, ties-to-even
    samples=np.array([(nearest[0]+dx,nearest[1]+dy) for dy in (-1,0,1) for dx in (-1,0,1)],np.int64)
    mapped=(samples.astype(np.float64)-query[None,:])/L*96.+47.5
    corner=np.floor(mapped).astype(np.int64)
    offsets=np.array([[0,0],[1,0],[0,1],[1,1]],np.int64) # NW,NE,SW,SE
    taps=corner[:,None,:]+offsets[None,:,:]
    return nearest,samples,mapped,taps


def _alpha(mask,x,y,valid):
    h,w=mask.shape
    result=np.zeros((96,96),bool)
    result[y:y+h,x:x+w]=mask
    return result & valid  # original invalid pixels are never overwritten


def _paste(base,donor,alpha,x,y):
    h,w=donor.shape[:2]
    result=base.copy()
    local=result[:,y:y+h,x:x+w]
    take=alpha[y:y+h,x:x+w]
    local[:,take]=donor.transpose(2,0,1)[:,take]
    return result


def compose_counterfactual_pair(baseCHWuint8,validbool96,donorHWCuint8,donorMaskbool,qF64,L,refF64,row_id):
    """Return one matched-size COVER/SHAM pair or explicit masked failure.

    Native/condition/role sidecars stay caller-owned and unchanged. Reference is
    teacher-only acceptance data, never used in RNG/placement or source recrop.
    Technical malformed input raises ValueError; valid construction failure is
    returned with arrays/targets null and masked=True. No fallback semantic E.
    """
    base=np.asarray(baseCHWuint8);valid=np.asarray(validbool96)
    donor=np.asarray(donorHWCuint8);mask=np.asarray(donorMaskbool)
    query=np.asarray(qF64);reference=np.asarray(refF64);scale=np.asarray(L)
    if base.dtype!=np.uint8 or base.shape!=(3,96,96) or valid.dtype!=bool or valid.shape!=(96,96):
        raise ValueError('Exact base uint8[3,96,96] and valid bool[96,96] required')
    if donor.dtype!=np.uint8 or donor.ndim!=3 or donor.shape[2]!=3 or min(donor.shape[:2])<1 or mask.dtype!=bool or mask.shape!=donor.shape[:2]:
        raise ValueError('Matching donor uint8[H,W,3] and bool[H,W] required')
    if query.dtype!=np.float64 or reference.dtype!=np.float64 or query.shape!=(2,) or reference.shape!=(2,) or not np.isfinite(query).all() or not np.isfinite(reference).all():
        raise ValueError('Finite original q/reference float64[2] required')
    if scale.shape!=() or scale.dtype!=np.float64 or not np.isfinite(scale) or scale<=0 or not isinstance(row_id,str) or not row_id:
        raise ValueError('Positive F64 scalar L and original row_id required')
    result={'success':False,'status':'CONSTRUCTION_FAILED','masked':True,'failure_reason':None,
            'COVER':None,'SHAM':None,'alpha_COVER':None,'alpha_SHAM':None,
            'proposed_states':None,'original_query_xy':query.copy(),'original_L':np.float64(scale),
            'teacher_reference_xy':reference.copy(),'row_id':row_id,
            'source_generated_not_natural':True,'native_gold_claim':False,
            'q_L_crop_and_validity_unchanged':True}
    def failure(reason):
        result['failure_reason']=reason
        return result
    ys,xs=np.nonzero(mask)
    if not len(xs):return failure('EMPTY_DONOR_FOREGROUND')
    x0,x1=int(xs.min()),int(xs.max())+1;y0,y1=int(ys.min()),int(ys.max())+1
    foreground_rgb=donor[y0:y1,x0:x1];foreground_mask=mask[y0:y1,x0:x1]
    seed,side,height,width,positions=_plan(row_id,y1-y0,x1-x0)
    resized_rgb,resized_mask=_resize(foreground_rgb,foreground_mask,height,width)
    result['donor_plan']={'foreground_bbox_xyxy_exclusive':[x0,y0,x1,y1],
        'seed_integer':seed,'PCG64_seed_formula':"int.from_bytes(SHA256(('V235:'+row_id).UTF8),'big') XOR235",
        'chosen_max_side':side,'resized_hw':[height,width],
        'resized_foreground_pixels':int(resized_mask.sum()),'candidate_budget_per_role':256,
        'candidate_positions_xy':positions,
        'candidate_positions_sha256':{role:hashlib.sha256(p.astype('<i8').tobytes()).hexdigest() for role,p in positions.items()}}
    if not resized_mask.any():return failure('RESIZED_DONOR_FOREGROUND_EMPTY')
    nearest,samples,mapped,taps=_reference_taps(query,np.float64(scale),reference)
    inbounds=bool(((taps>=0)&(taps<96)).all())
    checks={'reference_nearest_xy':nearest,'reference_original_3x3_xy':samples,'mapped_RAW96_xy':mapped,
            'required_taps_xy':taps,'all_required_taps_in96':inbounds,
            'all_required_taps_original_valid':False,'acceptance_uses_reference_only':True,
            'zero_changed_RGB_is_not_an_extra_acceptance_gate':True}
    result['teacherchecks']=checks
    if not inbounds:return failure('REFERENCE_3X3_FOUR_TAP_SUPPORT_OUTSIDE_RAW96')
    tx=taps[:,:,0].reshape(-1);ty=taps[:,:,1].reshape(-1)
    checks['all_required_taps_original_valid']=bool(valid[ty,tx].all())
    checks['required_unique_tap_pixels']=len(set(zip(tx.tolist(),ty.tolist())))
    if not checks['all_required_taps_original_valid']:return failure('REFERENCE_REQUIRED_TAP_ORIGINAL_INVALID')
    selected={}
    for role in ('COVER','SHAM'):
        for index,(x,y) in enumerate(positions[role]):
            alpha=_alpha(resized_mask,int(x),int(y),valid)
            support=alpha[ty,tx]
            accepted=bool(support.all()) if role=='COVER' else bool((~support).all() and alpha.sum()>=64)
            if accepted:
                selected[role]=(int(x),int(y),index,alpha)
                break
        if role not in selected:
            checks['selected_roles']=list(selected)
            return failure(role+'_CANDIDATE_BUDGET_EXHAUSTED')
    rolechecks={}
    for role,(x,y,index,alpha) in selected.items():
        image=_paste(base,resized_rgb,alpha,x,y)
        changed=np.any(image!=base,axis=0)
        assert not np.any(changed & ~alpha) and not np.any(alpha & ~valid)
        result[role]=image;result['alpha_'+role]=alpha
        rolechecks[role]={'position_xy':[x,y],'candidate_index':index,
            'alpha_visible_pixels':int(alpha.sum()),'changed_RGB_pixels':int(changed.sum()),
            'exact_base_RGB_same':bool(np.array_equal(image,base)),
            'all_changes_only_original_valid':bool(not np.any(changed & ~valid)),
            'unchanged_outside_alpha':bool(np.array_equal(image[:,~alpha],base[:,~alpha])),
            'all_required_taps_covered':bool(alpha[ty,tx].all()),
            'all_required_taps_uncovered':bool((~alpha[ty,tx]).all())}
    result.update(success=True,status='CONSTRUCTED',masked=False,proposed_states={'COVER':'E','SHAM':'D'},
                  resized_donor_RGB=resized_rgb,resized_donor_mask=resized_mask)
    checks.update(roles=rolechecks,same_donor_RGB_and_resize=True,
                  COVER_all_required_taps_opaque=rolechecks['COVER']['all_required_taps_covered'],
                  SHAM_all_required_taps_uncovered=rolechecks['SHAM']['all_required_taps_uncovered'],
                  SHAM_visible_paste_at_least64=rolechecks['SHAM']['alpha_visible_pixels']>=64)
    return result
