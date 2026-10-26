"""Input-only per-frame crop preparation. Caller supplies pinned V90 crop/model.

No GT pose/reference/visibility/direct-cue argument is accepted by this API.
Root constants: originalL*1; initial projected bbox maxspan*.25, simple fallback.
"""
import numpy as np
import torch

SIMPLE_FACTOR=1.0
OBJECT_FRACTION=.25


def prepare_frame(spatial, v90_crop, rgb, q0, original_L, original_available,
                  initial_T, K, registered_dimensions, initial_solver_success):
    rgb=np.asarray(rgb);q=np.asarray(q0,np.float64);L=np.asarray(original_L,np.float64)
    available=np.asarray(original_available)
    n=len(q)
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:
        raise ValueError('Explicit original RGB uint8 HWC only')
    if q.shape!=(n,2) or L.shape!=(n,) or available.shape!=(n,) or available.dtype!=bool:
        raise ValueError('Unchanged q0/L/original_available per native row')
    if not np.isfinite(q[available]).all() or not np.isfinite(L[available]).all() or not (L[available]>0).all():
        raise ValueError('Available original input must be finite and positive')
    simple=L*SIMPLE_FACTOR
    simple_ok=np.isfinite(simple)&(simple>0)&np.isfinite(q).all(1)
    geo=np.full(n,np.nan);geo_ok=np.zeros(n,bool)
    if initial_solver_success:
        T=torch.as_tensor(np.asarray(initial_T,np.float64)[None])
        intr=torch.as_tensor(np.asarray(K,np.float64)[None])
        dims=torch.as_tensor(np.asarray(registered_dimensions,np.float64)[None])
        size,ok=spatial.geometric_crop_side(T,intr,dims,OBJECT_FRACTION)
        geo[:]=float(size[0]);geo_ok[:]=bool(ok[0])
    sides=np.stack([simple,np.where(geo_ok,geo,simple)])
    geometric_fallback=(~geo_ok)&simple_ok
    crops=np.zeros((2,n,3,96,96),np.uint8);valid=np.zeros((2,n,1,96,96),bool)
    for variant in range(2):
        for j in range(n):
            if available[j] and simple_ok[j]:
                crops[variant,j],valid[variant,j]=v90_crop(rgb,q[j],float(sides[variant,j]),96)
    # Source-invalid rows remain zero; no GT-based support restoration.
    support=valid[:,:,0].reshape(2,n,16,6,16,6).any(axis=(3,5))
    any_valid=valid.reshape(2,n,-1).any(2)&available[None]
    central4=valid[:,:,0,47:49,47:49].all(axis=(2,3))&available[None]
    return {'RGB96':crops,'valid96':valid,'token_support':support,
            'crop_side':sides,'original_q0':q.copy(),'original_L':L.copy(),
            'original_available':available.copy(),'model_input_supported':any_valid,
            'LOC_input_available':central4&any_valid,'central4_valid':central4,
            'initial_solver_success':bool(initial_solver_success),
            'geometry_fallback_to_simple':geometric_fallback,
            'simple_factor':SIMPLE_FACTOR,'object_fraction':OBJECT_FRACTION}


def encode_crops(train_api, model, preprocess_views, crop_packet, batch_size=32):
    """Two variants, only input-supported rows. No GT fields, no CLS pooling."""
    device=next(model.parameters()).device
    count=crop_packet['RGB96'].shape[1]
    features=np.zeros((2,count,256,384),np.float16)
    for variant in range(2):
        indices=np.flatnonzero(crop_packet['model_input_supported'][variant])
        for start in range(0,len(indices),batch_size):
            batch=indices[start:start+batch_size]
            rgb=torch.from_numpy(crop_packet['RGB96'][variant,batch])
            valid=torch.from_numpy(crop_packet['valid96'][variant,batch])
            views,_=preprocess_views(rgb,valid)
            features[variant,batch]=train_api.extract_spatial(model,views[:len(batch)].to(device)).cpu().numpy()
    return features
