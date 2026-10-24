"""Pure generated tensors only; no source RGB, model or weights."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from pooling import image_input,token_weights,pool_local,pool_target

HERE=Path(__file__).resolve().parent
torch.set_num_threads(1)
g=torch.Generator().manual_seed(20261023)
rgb=torch.zeros(4,3,96,96,dtype=torch.uint8);rgb[1]=255;rgb[2,:,24:72,24:72]=127
v=torch.ones(4,1,96,96,dtype=torch.bool);v[0]=False;v[2,:,:,48:]=False;v[3,:,:1,:1]=False
x=image_input(rgb);w=token_weights(v)
assert x.shape==(4,3,224,224)and x.dtype==torch.float32
assert w.shape==(4,16,16)and w.min()>=0 and w.max()<=1 and w[0].sum()==0
assert torch.allclose(w[2].mean(),torch.tensor(.5),atol=1e-7)
assert torch.allclose(x[1,:,17,13],(torch.ones(3)-torch.tensor([.485,.456,.406]))/torch.tensor([.229,.224,.225]))
t=torch.randn(4,256,384,generator=g);a,f=pool_local(t,w);b,fb=pool_target(t,w)
assert a.shape==(4,2,384)and b.shape==(4,384)and f.shape==(4,2)
assert not a[0].any()and not b[0].any()and not f[0].any()and fb[0]==0
assert torch.equal(a[:,1],b)
maxdiff=0.
for i in range(4):
 for j,(sl,sc)in enumerate([(slice(7,9),slice(7,9)),(slice(None),slice(None))]):
  tt=t[i].reshape(16,16,384).numpy().astype(np.float64)[sl,sc];ww=w[i].numpy().astype(np.float64)[sl,sc]
  ref=(tt*ww[...,None]).sum((0,1))/max(ww.sum(),1e-12)
  maxdiff=max(maxdiff,float(np.max(np.abs(a[i,j].numpy()-ref))))
assert maxdiff<2e-5
ramp=torch.arange(256,dtype=torch.float32)[None,:,None].expand(1,256,384)
rr,_=pool_local(ramp,torch.ones(1,16,16));assert torch.equal(rr,torch.full((1,2,384),127.5))
p=HERE/'GENERATED_POOLING_TESTS.json'
assert not p.exists()
p.write_text(json.dumps(dict(PASS=True,tests=['RGB normalization/bilinear224','fractional validity/avg_pool14','zero mass exact zero','center 7:9 phase','full token spatial ramp','independent float64 pool'],max_pool_abs_diff=maxdiff,atol_declared=2e-5,RGB_GT_model_weights_current_forward_fit_reads=0,pooling_sha256=hashlib.sha256((HERE/'pooling.py').read_bytes()).hexdigest()),indent=2)+'\n')
print(p)
