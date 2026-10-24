"""Pure geometric source-mask coverage, no file/model access."""
import hashlib
import numpy as np

VALID_NAMES=['RAW_CONTINUOUS','NN_BOUNDS','SAMPLER_CONTINUOUS','NATIVE_CONTINUOUS_AND_NN','CACHED_CANVAS','CACHED_AND_NN','NATIVE_CACHED_INTERSECTION']
CLASS_NAMES=['M4_VISIBLE','M0_AND_NOT_M4_COVERED_SUPPORT','NOT_M0_OUTSIDE_SUPPORT']
SUBSET_NAMES=['NATIVE_CONTINUOUS_AND_NN','CACHED_AND_NN','NATIVE_CACHED_INTERSECTION']

def coordinates(q,L,n):
    q=np.asarray(q,np.float64)
    u=(np.arange(n,dtype=np.float64)+.5)/n-.5
    xx,yy=np.meshgrid(u*L,u*L)
    return q[:,None,None,:]+np.stack([xx,yy],-1)[None]

def cached_formula(grid,hw,n,q):
    h,w=hw;good=np.isfinite(q).all(1)&~(q==-1).all(1)
    lo=0 if n==9 else-.5;hi_x=w if n==9 else w-.5;hi_y=h if n==9 else h-.5
    return(grid[...,0]>=lo)&(grid[...,0]<hi_x)&(grid[...,1]>=lo)&(grid[...,1]<hi_y)&good[:,None,None]

def fingerprint(n,valid,m0,m4):
    bits=np.stack([valid,m0&valid,m4&valid]).ravel(order='C')
    return hashlib.sha256(f'V79_MAP_V1|grid{n}'.encode()+np.packbits(bits,bitorder='little').tobytes()).hexdigest()

def count_grid(m0,m4,q,L,n,cached):
    h,w=m0.shape;grid=coordinates(q,L,n);finite=np.isfinite(grid).all(-1)
    good=np.isfinite(q).all(1)&~(q==-1).all(1)
    safe=np.where(np.isfinite(grid),grid,-1e9);pixel=np.floor(safe+.5).astype(np.int64)
    rawcont=finite&(grid[...,0]>=0)&(grid[...,0]<w)&(grid[...,1]>=0)&(grid[...,1]<h)
    nn=finite&(pixel[...,0]>=0)&(pixel[...,0]<w)&(pixel[...,1]>=0)&(pixel[...,1]<h)
    samplercont=rawcont&good[:,None,None];native=samplercont&nn
    assert cached.shape==(8,n,n)
    known=cached&nn;inter=native&cached
    v=np.zeros((8,n,n),bool);f=v.copy();v[nn]=m4[pixel[...,1][nn],pixel[...,0][nn]];f[nn]=m0[pixel[...,1][nn],pixel[...,0][nn]]
    assert not(v&~f).any()
    classes=np.stack([v,f&~v,~f],1)
    valsets=[rawcont,nn,samplercont,native,cached,known,inter]
    subsets=[native,known,inter]
    counts=np.stack([(classes&s[:,None]).sum((2,3))for s in subsets],1).astype(np.int32)
    valid_counts=np.stack([s.sum((1,2))for s in valsets],1).astype(np.int32)
    assert np.array_equal(counts.sum(2),valid_counts[:,[3,5,6]])
    unique=np.stack([[len(np.unique(pixel[k,...,1][s[k]]*w+pixel[k,...,0][s[k]]))for k in range(8)]for s in subsets],1).astype(np.int32)
    a=np.asarray([fingerprint(n,native[k],f[k],v[k])for k in range(8)],dtype='S64')
    b=np.asarray([fingerprint(n,inter[k],f[k],v[k])for k in range(8)],dtype='S64')
    # Informative tie counts; literal producer geometry remains the audit primary.
    distance=np.abs(safe+.5-np.rint(safe+.5));near=(distance.min(-1)<=1e-6)&nn
    if n==9:
        uu=(np.arange(n,dtype=np.float64)-4)*L/9
        ax,ay=np.meshgrid(uu,uu);algebra=q[:,None,None,:]+np.stack([ax,ay],-1)[None]
    else:
        algebra=grid # Exact same literal already used by crop; no alternate phase.
    ap=np.floor(np.where(np.isfinite(algebra),algebra,-1e9)+.5).astype(np.int64)
    tie_changes=((ap!=pixel).any(-1)&nn).sum((1,2)).astype(np.int32)
    return dict(class_counts=counts,valid_counts=valid_counts,unique_native_pixels=unique,
      gt_map_native_sha256=a,gt_map_intersection_sha256=b,near_NN_half_tie_counts=near.sum((1,2)).astype(np.int32),literal_vs_algebra_NN_changes=tie_changes),grid,pixel,native

def roi_float_phase(grid,gain,offset,hw,native_valid,m0,m4):
    # Exact norm float32 sent to grid_sample; inverse interpreted in float64.
    # A numerical grid-anchor diagnostic, not a physical receptive-field claim.
    network=(grid+100)*gain+offset;index=network/8.;norm=2*(index+.5)/80-1
    normalized32=norm.astype(np.float32)
    recovered_index=(normalized32.astype(np.float64)+1)*80/2-.5
    recovered_native=(recovered_index*8-offset)/gain-100
    h,w=hw;pix=np.floor(grid+.5).astype(np.int64);p2=np.floor(recovered_native+.5).astype(np.int64)
    inside=(p2[...,0]>=0)&(p2[...,0]<w)&(p2[...,1]>=0)&(p2[...,1]<h)&native_valid
    diff=((pix!=p2).any(-1)&native_valid)
    a=np.zeros(native_valid.shape,np.uint8);b=a.copy()
    a[native_valid]=(m0[pix[...,1][native_valid],pix[...,0][native_valid]].astype(np.uint8)+m4[pix[...,1][native_valid],pix[...,0][native_valid]].astype(np.uint8))
    b[inside]=(m0[p2[...,1][inside],p2[...,0][inside]].astype(np.uint8)+m4[p2[...,1][inside],p2[...,0][inside]].astype(np.uint8))
    return dict(ROI_float32_norm_max_native_abs_error=np.max(abs(recovered_native-grid),axis=(1,2,3)),ROI_float32_norm_NN_pixel_changes=diff.sum((1,2)).astype(np.int32),ROI_float32_norm_mask_class_changes=((a!=b)&inside).sum((1,2)).astype(np.int32),ROI_float32_norm_recovered_outside=(native_valid&~inside).sum((1,2)).astype(np.int32)),norm
