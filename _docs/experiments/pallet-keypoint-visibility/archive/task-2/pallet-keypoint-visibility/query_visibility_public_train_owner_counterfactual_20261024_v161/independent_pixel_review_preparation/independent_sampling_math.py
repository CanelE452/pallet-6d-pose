"""Independent NumPy sampling/footprint checks; invented-array preparation only.
No source body reads at import or direct execution. No model dependencies.
"""
import numpy as np

def bilinear_grid(rgb,xs,ys,output_uint8):
    """Explicit four-neighbor interpolation in native pixel-center coordinates."""
    h,w=rgb.shape[:2];x,y=np.meshgrid(np.asarray(xs,np.float64),np.asarray(ys,np.float64))
    ix=np.floor(x).astype(np.int64);iy=np.floor(y).astype(np.int64);dx=x-ix;dy=y-iy
    result=np.zeros((*x.shape,3),np.float64);all_positive_valid=np.ones(x.shape,bool)
    for ox,oy,weight in ((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)):
        xx=ix+ox;yy=iy+oy;valid=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
        result+=rgb[np.clip(yy,0,h-1),np.clip(xx,0,w-1)].astype(np.float64)*(weight*valid)[...,None]
        all_positive_valid &= valid | (weight==0)
    if output_uint8:return np.floor(np.clip(result,0,255)+.5).astype(np.uint8),all_positive_valid
    return result.astype(np.float32),all_positive_valid

def crop96(rgb,q,L):
    u=(np.arange(96,dtype=np.float64)+.5)/96-.5
    xs=q[0]+u*L;ys=q[1]+u*L;out,_=bilinear_grid(rgb,xs,ys,True)
    valid=(xs[None,:]>=-.5)&(xs[None,:]<rgb.shape[1]-.5)&(ys[:,None]>=-.5)&(ys[:,None]<rgb.shape[0]-.5)
    out[~valid]=0
    return out,valid

def context224(rgb,box):
    lo=np.asarray(box,np.float64)
    if lo.shape!=(4,) or not np.isfinite(lo).all() or lo[2]<=lo[0] or lo[3]<=lo[1]:
        return np.zeros((224,224,3),np.float32),np.zeros((224,224),bool)
    xs=lo[0]+(np.arange(224)+.5)*(lo[2]-lo[0])/224
    ys=lo[1]+(np.arange(224)+.5)*(lo[3]-lo[1])/224
    return bilinear_grid(rgb,xs,ys,False)

def center4_footprints(q,L,hw):
    """All strict positive-weight native neighbors; zero-weight cells excluded."""
    h,w=hw;records=[];union=set()
    for row in (47,48):
        for col in (47,48):
            x=float(q[0]+((col+.5)/96-.5)*L);y=float(q[1]+((row+.5)/96-.5)*L)
            ix=int(np.floor(x));iy=int(np.floor(y));dx=x-ix;dy=y-iy;neighbors=[]
            for ox,oy,weight in ((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)):
                if weight>0:
                    xx,yy=ix+ox,iy+oy;inside=0<=xx<w and 0<=yy<h
                    neighbors.append({'x':xx,'y':yy,'weight':float(weight),'inside':inside});union.add((yy,xx))
            records.append({'row96':row,'col96':col,'native_xy':[x,y],'neighbors':neighbors})
    return records,sorted(union)

def center4_from_footprints(rgb,records):
    result=[]
    for r in records:
        value=np.zeros(3,np.float64)
        for n in r['neighbors']:
            if n['inside']:value+=rgb[n['y'],n['x']].astype(np.float64)*n['weight']
        result.append(np.floor(np.clip(value,0,255)+.5).astype(np.uint8))
    return np.asarray(result).reshape(2,2,3)

def generated_checks():
    rgb=np.arange(13*17*3,dtype=np.uint16).reshape(13,17,3).astype(np.uint8)
    q=[8.,6.];L=96.;records,cells=center4_footprints(q,L,rgb.shape[:2]);a,v=crop96(rgb,q,L)
    assert np.array_equal(center4_from_footprints(rgb,records),a[47:49,47:49])
    assert len(cells)==9 and all(n['inside'] for r in records for n in r['neighbors'])
    # Integer raw samples have one contributor, not four zero-weight neighbors.
    records,cells=center4_footprints([8.,6.],0.,rgb.shape[:2]);assert all(len(r['neighbors'])==1 for r in records) and len(cells)==1
    records,cells=center4_footprints([-.25,2.],2.,rgb.shape[:2]);assert any(not n['inside'] for r in records for n in r['neighbors'])
    assert not context224(rgb,[0,0,0,1])[1].any()
    return {'PASS':True,'invented_arrays_only':True,'literal_center4_matches_independent_raw_footprint_sum':True,'positive_weight_only_and_integer_zero_weight_cases':True,'out_of_bounds_positive_neighbors_explicit':True,'no_source_pixels_or_NPZ':True}
