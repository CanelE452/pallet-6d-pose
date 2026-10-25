"""SOURCE native layer再構成/teacher proofのみ。ファイル/producer/model importなし。"""
import hashlib
import numpy as np

def need(ok,message):
    if not ok:raise ValueError(message)

def sha(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

def lift_scene(base,carrier_CHW,alpha96,q,L):
    """独立 nearest-cell raster。native p -> (p-q)*96/L+47.5 -> floor(u+.5)."""
    need(base.dtype==np.uint8 and base.ndim==3 and base.shape[2]==3,'base HWC U8')
    need(carrier_CHW.dtype==np.uint8 and carrier_CHW.shape==(3,96,96)
         and alpha96.dtype==bool and alpha96.shape==(96,96),'stored carrier/alpha')
    q=np.asarray(q,np.float64);L=np.float64(L)
    need(q.shape==(2,) and np.isfinite(q).all() and np.isfinite(L) and L>0,'immutable q/L')
    height,width=base.shape[:2]
    address_x=(np.arange(width,dtype=np.float64)-q[0])*96./L+47.5
    address_y=(np.arange(height,dtype=np.float64)-q[1])*96./L+47.5
    good_x=(address_x>=-.5)&(address_x<95.5)
    good_y=(address_y>=-.5)&(address_y<95.5)
    cell_x=np.clip(np.floor(address_x+.5).astype(np.int64),0,95)
    cell_y=np.clip(np.floor(address_y+.5).astype(np.int64),0,95)
    native_alpha=good_y[:,None]&good_x[None,:]&alpha96[cell_y[:,None],cell_x[None,:]]
    native_colour=carrier_CHW.transpose(1,2,0)[cell_y[:,None],cell_x[None,:]]
    scene=base.copy();scene[native_alpha]=native_colour[native_alpha]
    need(np.array_equal(scene[~native_alpha],base[~native_alpha]),'opaque outside-alpha preservation')
    return scene,native_alpha

def footprints(reference,hw):
    if reference is None:return None
    r=np.asarray(reference,np.float64)
    if r.shape!=(2,) or not np.isfinite(r).all():return None
    nearest=np.floor(r+.5).astype(np.int64)
    sets={'continuous_reference':[tuple(r)],'nearest_half_up_3x3':
        [(float(nearest[0]+dx),float(nearest[1]+dy)) for dy in (-1,0,1) for dx in (-1,0,1)]}
    result={}
    for name,locations in sets.items():
        pixels=set()
        for x,y in locations:
            left,top=int(np.floor(x)),int(np.floor(y));fx,fy=x-left,y-top
            for dx,dy,weight in ((0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),
                                 (0,1,(1-fx)*fy),(1,1,fx*fy)):
                if weight>0:pixels.add((left+dx,top+dy))
        points=sorted(pixels)
        inside=all(0<=x<hw[1] and 0<=y<hw[0] for x,y in points)
        result[name]=dict(pixels=points,supported=inside)
    return result

def teacher_proof(base,cover,sham,cover_alpha,sham_alpha,reference):
    fp=footprints(reference,base.shape[:2])
    if fp is None:return dict(PASS=False,reason='UNKNOWN_REFERENCE',checks={})
    checks={};passed=True
    for name,f in fp.items():
        supported=f['supported'];c=same=clear=None
        if supported:
            x,y=np.asarray(f['pixels'],np.int64).T
            c=bool(cover_alpha[y,x].all());clear=bool((~sham_alpha[y,x]).all())
            same=bool(np.array_equal(base[y,x],sham[y,x]))
        checks[name]=dict(supported=supported,positive_native_pixels=len(f['pixels']),
            COVER_all_opaque=c,SHAM_all_clear=clear,SHAM_reference_RGB_unchanged=same)
        passed=passed and bool(supported and c and clear and same)
    return dict(PASS=passed,checks=checks,teacher_reference_only=True,
        q_center_visibility_inferred=False,whole_intended_part_gold=False)

def sample_all(scene,group,crop90,context107):
    """Pinned純粋V90/V107 sampling。availabilityでq8/cropを消さない。"""
    q=np.asarray(group['q8'],np.float64);L=np.asarray(group['L8'],np.float64)
    avail=np.asarray(group['original_available8'],bool)
    need(q.shape==(8,2) and L.shape==(8,) and avail.shape==(8,),'originalq8/L8')
    nodes=np.zeros((8,3,96,96),np.uint8);valid=np.zeros((8,96,96),bool)
    for slot in range(8):
        if np.isfinite(q[slot]).all() and np.isfinite(L[slot]) and L[slot]>0:
            rgb,v=crop90(scene,q[slot],L[slot],96);nodes[slot]=rgb;valid[slot]=v[0]
    box=context107.query_extent(q)
    need((box is None and group['box_xyxy'] is None) or
         (box is not None and np.array_equal(np.asarray(box,np.float64),np.asarray(group['box_xyxy'],np.float64))),
         'finite/non-sentinel originalq8 box; no availability/GT override')
    if box is None:
        context=np.zeros((3,224,224),np.float32);context_valid=np.zeros((224,224),bool)
    else:
        value,context_valid=context107.sample_target_context(scene,box)
        context=value.transpose(2,0,1).copy()
    return dict(nodes=nodes,node_valid=valid,context=context,context_valid=context_valid,
        node_support=avail&valid[:,47:49,47:49].all((1,2))&valid.any((1,2)),
        node_fraction=valid.mean((1,2)).astype(np.float32),context_fraction=np.float32(context_valid.mean()))

def changed_counts(base,scene,before,after,active_slot):
    return dict(native=int(np.any(base!=scene,axis=2).sum()),
        active_RAW96=int(np.any(before['nodes'][active_slot]!=after['nodes'][active_slot],axis=0).sum()),
        context_F32_samples=int(np.any(before['context']!=after['context'],axis=0).sum()))
