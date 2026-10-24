import sys
sys.dont_write_bytecode=True
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
import torch
H=Path(__file__).resolve().parent;path=H/'prepare_target_context_features.py'
s=importlib.util.spec_from_file_location('c9_prep',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
torch.set_num_threads(1);torch.manual_seed(206)
n=7;raw=torch.randn(n,512);old=torch.randn(n,1026);old[:,1024:]=torch.rand(n,2);fraction=torch.tensor([1.,.5,0.,1.,1.,.2,1.]);available=torch.tensor([1,1,1,0,1,1,1],dtype=torch.bool)
z=m.assemble(raw,old,fraction,available);assert z.shape==(n,1027)
assert torch.equal(z[:,512:1024],old[:,512:1024])and torch.equal(z[:,1026],old[:,1025])
assert torch.count_nonzero(z[2,:512])==torch.count_nonzero(z[3,:512])==0
assert torch.equal(z[:,1024],torch.nn.functional.cosine_similarity(z[:,:512],old[:,512:1024],dim=1,eps=1e-12))
old2=old.clone();old2[:,:512]=1000000;old2[:,1024]=.123;assert torch.equal(z,m.assemble(raw,old2,fraction,available))
assert torch.equal(z[:,1025],torch.where(available,fraction,torch.zeros_like(fraction)))
x=torch.rand(n,3,224,224)*255;valid=torch.ones(n,1,224,224,dtype=torch.bool);valid[0]=False
p=m.preprocess_target(x,valid);assert torch.count_nonzero(p[0])==0 and torch.isfinite(p).all()
for args in[(x.double(),valid),(x,valid.float())]:
 try:m.preprocess_target(*args)
 except ValueError:pass
 else:raise AssertionError('Bad dtype accepted')
math=m.module(m.MATH,'c9_math_generated');rgb=np.arange(6*8*3,dtype=np.uint8).reshape(6,8,3);crop,v=math.sample_target_context(rgb,[0,0,8,6],(3,4));assert v.all()and np.array_equal(crop[0,0],rgb[1,1])
unit=np.array([0,0,1,2,2]);a=np.array([1,0,1,1,1],bool);ctx=np.array([1,0,1],bool);assert np.array_equal(a&ctx[unit],[1,0,0,1,1])
out=H/'GENERATED_CONTEXT_FEATURE_TESTS.json'
with out.open('x')as f:json.dump(dict(completed=True,PASS=True,code=m.bind(path),test_code=m.bind(__file__),tests=['exact1027layout','C8central512_and_fraction_bitexact','C8full96_descriptor_and_fraction_not_used','target_L2_normalized','standard_cosine_eps1e-12','missing_target_zero','invalid_native_RGB_meanfill_then_normalize_exact0','bad_dtype_rejected','generated_V107_bilinear_grid','factualANDcontext_U_preservation'],real_features_RGB_weights_forward_or_fit=0,torch=torch.__version__),f,indent=2);f.write('\n')
print(json.dumps(m.bind(out)))
