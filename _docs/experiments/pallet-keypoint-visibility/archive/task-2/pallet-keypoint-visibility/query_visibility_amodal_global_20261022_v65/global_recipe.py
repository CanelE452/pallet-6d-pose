"""Input-only target-global grid recipe; no reference masks or poses."""
import numpy as np

SIZE=64
EXPANSION=1.25

def grid(box):
    box=np.asarray(box,np.float64)
    assert box.shape==(4,) and np.isfinite(box).all()
    step=EXPANSION*(box[2:]-box[:2])/SIZE
    assert (step>0).all()
    center=(box[:2]+box[2:])/2
    a=np.arange(SIZE,dtype=np.float64)+.5-SIZE/2
    xx,yy=np.meshgrid(a,a)
    xy=center+np.stack((xx,yy),-1)*step
    pixel=np.floor(xy+.5).astype(np.int64)
    return xy,pixel,center,step

def native_valid(xy,pixel,hw):
    h,w=hw
    return ((xy[...,0]>=0)&(xy[...,0]<w)&(xy[...,1]>=0)&(xy[...,1]<h)&
            (pixel[...,0]>=0)&(pixel[...,0]<w)&(pixel[...,1]>=0)&(pixel[...,1]<h))

def local_mapping(query,side,box,hw):
    a=(np.arange(9)-4)*side/9
    xx,yy=np.meshgrid(a,a)
    xy=np.asarray(query,np.float64)[:,None,None,:]+np.stack((xx,yy),-1)[None]
    pixel=np.floor(xy+.5).astype(np.int64)
    _,_,center,step=grid(box)
    cropxy=(pixel-center)/step+SIZE/2-.5
    cropidx=np.floor(cropxy+.5).astype(np.int64)
    valid=native_valid(xy,pixel,hw)&(cropidx>=0).all(-1)&(cropidx<SIZE).all(-1)
    index64=np.where(valid,cropidx[...,1]*SIZE+cropidx[...,0],0)
    index8=np.where(valid,(cropidx[...,1]//8)*8+cropidx[...,0]//8,0)
    return index64.astype(np.int64),index8.astype(np.int64),valid
