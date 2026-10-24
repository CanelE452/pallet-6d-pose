"""Independent NumPy math for frozen V78 validity pools and FP16 storage.

No model/torch/files read on import. These checks certify sampling/storage,
not semantic ownership or D/E/S correctness.
"""
import numpy as np

# Fixed before actual features are read; engineering numeric bound only.
VALID_FRACTION_ABS_BOUND=2**-11+1e-6
POOL_REFERENCE_ABS_FACTOR=2e-5

def linear_weights(native_size,first=0,last=224):
 """Mean bilinear224 align_cornersFalse output over [first,last)."""
 assert native_size in (96,128)and 0<=first<last<=224
 u=(np.arange(first,last,dtype=np.float64)+.5)*(native_size/224.)-.5
 left=np.floor(u).astype(np.int64);alpha=u-left
 w=np.zeros(native_size,np.float64)
 np.add.at(w,np.clip(left,0,native_size-1),(1-alpha)/(last-first))
 np.add.at(w,np.clip(left+1,0,native_size-1),alpha/(last-first))
 assert abs(w.sum()-1)<1e-14 and (w>=0).all()
 return w

def patch_weights(mask):
 m=np.asarray(mask);assert m.shape[-2]==m.shape[-1]and m.shape[-1]in(96,128)
 assert np.isin(m,[0,1]).all()
 size=m.shape[-1];W=np.stack([linear_weights(size,k*14,(k+1)*14)for k in range(16)])
 return np.einsum('ai,...ij,bj->...ab',W,m,W,optimize=True)

def expected_valid_fractions(local_valid,target_valid):
 local=np.asarray(local_valid);target=np.asarray(target_valid)
 assert local.ndim==5 and local.shape[1:]==(8,1,96,96)and target.shape==(len(local),1,128,128)
 c=linear_weights(96,98,126);g=linear_weights(96);t=linear_weights(128)
 cm=np.einsum('i,nkij,j->nk',c,local[:,:,0],c,optimize=True)
 gm=np.einsum('i,nkij,j->nk',g,local[:,:,0],g,optimize=True)
 tm=np.einsum('i,nij,j->n',t,target[:,0],t,optimize=True)
 return np.stack([cm,gm,np.broadcast_to(tm[:,None],cm.shape)],axis=-1)

def pooled_reference(tokens,mask):
 tok=np.asarray(tokens);m=np.asarray(mask)
 assert tok.shape[-2:]==(256,384)
 assert m.shape[:-2]==tok.shape[:-2]
 weights=patch_weights(m);grid=tok.reshape(*tok.shape[:-2],16,16,384).astype(np.float64)
 def pool(g,w):
  mass=w.sum(axis=(-2,-1));num=(g*w[...,None]).sum(axis=(-3,-2));v=num/np.maximum(mass[...,None],1e-12)
  return np.where(mass[...,None]>0,v,0.)
 center=pool(grid[...,7:9,7:9,:],weights[...,7:9,7:9]);global_=pool(grid,weights)
 fractions=np.stack([weights[...,7:9,7:9].mean(axis=(-2,-1)),weights.mean(axis=(-2,-1))],axis=-1)
 return center,global_,fractions

def assert_fp16_storage(fp32,stored):
 x=np.asarray(fp32);h=np.asarray(stored)
 assert x.dtype==np.float32 and h.dtype==np.float16 and x.shape==h.shape
 assert np.isfinite(x).all()and np.isfinite(h).all()
 assert np.array_equal(x.astype(np.float16),h),'Stored FP16 must be the exact cast of bound FP32 pooled vectors'
 error=abs(x.astype(np.float64)-h.astype(np.float64));bound=abs(x.astype(np.float64))*2**-11+2**-25
 assert (error<=bound).all()
 return dict(elements=x.size,max_abs=float(error.max(initial=0)),max_relative_normal=float(np.max(np.divide(error,abs(x),where=abs(x)>=2**-14,out=np.zeros_like(error)),initial=0)),exact_numpy_half_cast=True,rounding_error_bound='abs(FP32)*2^-11 +2^-25')

def explicit_mask_resize(mask):
 m=np.asarray(mask,np.float64);size=m.shape[-1];v=(np.arange(224)+.5)*size/224-.5;l=np.floor(v).astype(int);f=v-l;l0=np.clip(l,0,size-1);l1=np.clip(l+1,0,size-1)
 a=m[...,l0,:]*(1-f)[:,None]+m[...,l1,:]*f[:,None]
 return a[...,l0]*(1-f)+a[...,l1]*f

def mock():
 rng=np.random.default_rng(20261023)
 for size in [96,128]:
  for m in [np.zeros((size,size),bool),np.ones((size,size),bool),rng.integers(0,2,(size,size),dtype=np.uint8)]:
   w=patch_weights(m);full=explicit_mask_resize(m);direct=full.reshape(16,14,16,14).mean(axis=(1,3));assert np.allclose(w,direct,rtol=0,atol=2e-15)
 tokens=np.zeros((3,256,384),np.float32);tokens[:,:,0]=(np.arange(256,dtype=np.float32)**2)[None,:];tokens[:,:,1]=rng.normal(size=(3,256));mask=np.ones((3,96,96),bool);mask[1]=0;mask[2,:48]=0
 c,g,f=pooled_reference(tokens,mask);assert not c[1].any()and not g[1].any()and not f[1].any()
 grid=tokens[0].reshape(16,16,384);assert np.allclose(c[0],grid[7:9,7:9].mean(axis=(0,1)))and np.allclose(g[0],grid.mean(axis=(0,1)))
 examples=np.r_[np.linspace(-12,12,10000),[0,1e-8,-1e-8,2**-14,2**-24,65504,-65504]].astype(np.float32)
 quant=assert_fp16_storage(examples,examples.astype(np.float16))
 local=np.ones((2,8,1,96,96),bool);target=np.ones((2,1,128,128),bool);local[1]=0;target[1]=0;vf=expected_valid_fractions(local,target)
 assert np.allclose(vf[0],1,rtol=0,atol=1e-14)and not vf[1].any()
 return dict(PASS=True,bilinear_mean_patch_validity_reference=True,zero_mass_pool_allzero=True,center2x2_and_global_pools=True,fractional_validity_no_threshold=True,fp16_quantization=quant,valid_fraction_absolute_engineering_bound=VALID_FRACTION_ABS_BOUND,pool_FP64_reference_engineering_bound_factor=POOL_REFERENCE_ABS_FACTOR,source_RGB_GT_features_weights_current_forward_fit_PnP=0)
