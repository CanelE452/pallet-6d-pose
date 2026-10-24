"""Vectorized native exact/radius2 partial-joint labels, no images/models/poses.

Consumes logically closed vp/vn/fp/fn facts from either registered contract.
Never converts UNKNOWN to a hard negative. Does not edit/import frozen helpers.
"""
import cv2
import numpy as np

PAIRS=((0,0),(1,0),(1,1),(2,0),(2,1),(2,2))


def _counts(a):
    # Integer counts0..25 are represented exactly in float32. Constant-zero
    # padding is separately accounted for by the clipped native pixel count.
    return cv2.boxFilter(a.astype(np.float32),-1,(5,5),normalize=False,
                         borderType=cv2.BORDER_CONSTANT).astype(np.int16)


def _options(vp,vn,fp,fn,pixels):
    uv=pixels-vp-vn
    if (uv<0).any():raise AssertionError('Contradictory visibility count')
    return np.stack([(vp>0)|(uv>0),(vp==0)&(fn<pixels),(vp==0)&(fp==0)])


def _state(options):
    known=options.sum(axis=0)==1
    return np.where(known,options.argmax(axis=0),-1).astype(np.int8)


def partial_truth_grid(cert):
    arrays=[np.asarray(cert[k])for k in ('visible_positive','visible_negative','full_positive','full_negative')]
    vp,vn,fp,fn=arrays
    if vp.ndim!=2 or min(vp.shape)<1 or any(a.shape!=vp.shape or a.dtype!=np.bool_ for a in arrays):
        raise ValueError('Aligned nonempty native Boolean2D evidence required')
    if (vp&vn).any()or(fp&fn).any()or(vp&~fp).any()or(fn&~vn).any():
        raise ValueError('Logically closed noncontradictory support facts required')
    counts=[_counts(a)for a in arrays];pixels=_counts(np.ones_like(vp))
    center=[a.astype(np.int16)for a in arrays]
    exact_options=_options(*center,np.ones_like(pixels))
    radius_options=_options(*counts,pixels)
    other_options=_options(*[a-b for a,b in zip(counts,center)],pixels-1)
    mask=np.zeros(vp.shape,np.uint8)
    for a in range(3):
        for b in range(3):
            near=min(a,b) # V0 priority, then E1, then O2.
            code=PAIRS.index((a,near))
            mask|=((exact_options[a]&other_options[b]).astype(np.uint8)<<code)
    if (mask==0).any():raise AssertionError('Empty compatible physical completion set')
    lookup=np.full(64,-1,np.int8)
    for code in range(6):lookup[1<<code]=code
    joint=lookup[mask];exact=_state(exact_options);near=_state(radius_options)
    known=joint>=0
    if not np.array_equal(known,(exact>=0)&(near>=0)):
        raise AssertionError('Unique joint/marginal certainty mismatch')
    return dict(exact_state=exact,radius2_state=near,joint_state=joint,joint_known=known,
                joint_compatible_mask=mask,exact_possible_mask=(exact_options*np.array([1,2,4])[:,None,None]).sum(0).astype(np.uint8),
                radius2_possible_mask=(radius_options*np.array([1,2,4])[:,None,None]).sum(0).astype(np.uint8),
                radius2_window_pixels=pixels.astype(np.uint8),radius2_clipped_by_frame=pixels!=25,
                radius2_strict_E_core=(pixels==25)&(_counts(fp&vn)==25),
                radius2_strict_O_core=(pixels==25)&(counts[3]==25))
