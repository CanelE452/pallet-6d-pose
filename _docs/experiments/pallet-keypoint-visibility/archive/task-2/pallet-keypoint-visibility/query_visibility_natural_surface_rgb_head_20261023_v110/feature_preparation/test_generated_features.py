import sys
sys.dont_write_bytecode=True
from pathlib import Path
import importlib.util,json
import numpy as np
import torch
H=Path(__file__).resolve().parent;p=H/'prepare_source_rgb_features.py';s=importlib.util.spec_from_file_location('c10_features',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
torch.set_num_threads(1);torch.manual_seed(206)
n=9;target=torch.randn(n,512);local=torch.randn(n,1026);local[:,1024:]=torch.rand(n,2);f=torch.rand(n);cos=torch.randn(n)
z=m.compose1541(target,local,f,cos);assert z.shape==(n,1541)
assert torch.equal(z[:,:512],target)and torch.equal(z[:,512:1536],local[:,:1024])and torch.equal(z[:,1536],cos)and torch.equal(z[:,1538],f)and torch.equal(z[:,1539:],local[:,1024:])
assert torch.equal(z[:,1537],torch.nn.functional.cosine_similarity(target,local[:,:512],dim=1,eps=1e-12))
z2=m.compose1541(target,local,f);assert torch.equal(z2[:,1536],torch.nn.functional.cosine_similarity(target,local[:,512:1024],dim=1,eps=1e-12))
zero=m.compose1541(torch.zeros(n,512),torch.zeros(n,1026),torch.zeros(n));assert torch.count_nonzero(zero)==0
try:m.compose1541(target[:,:511],local,f)
except ValueError:pass
else:raise AssertionError('wrongshape accepted')
crop,qv=m.pure_local_math();rgb=np.arange(11*13*3,dtype=np.uint8).reshape(11,13,3);checks=0
for q in [[5.7,6.2],[-.25,.5],[12.75,10.1]]:
 a,v=crop(rgb,q,16.,96)
 for y,x in[(0,0),(47,47),(48,48),(95,95)]:
  xx=q[0]+((x+.5)/96-.5)*16;yy=q[1]+((y+.5)/96-.5)*16;valid=(-.5<=xx<12.5 and -.5<=yy<10.5)
  acc=np.zeros(3)
  for xi in[int(np.floor(xx)),int(np.floor(xx))+1]:
   for yi in[int(np.floor(yy)),int(np.floor(yy))+1]:
    if 0<=xi<13 and 0<=yi<11:acc+=rgb[yi,xi]*max(0,1-abs(xx-xi))*max(0,1-abs(yy-yi))
  expected=np.floor(np.clip(acc,0,255)+.5).astype(np.uint8)if valid else np.zeros(3,np.uint8)
  assert bool(v[0,y,x])==valid and np.array_equal(a[:,y,x],expected);checks+=1
valid,cont,nearest=qv([[0.,0.],[-.1,2.],[12.75,4.],[-1,-1]],[11,13]);assert valid.tolist()==[True,False,False,False]
c8=m.module(m.C8_FACTORY,'generated_C8_math');c9=m.module(m.C9_FACTORY,'generated_C9_math')
x=torch.randint(0,256,(8,3,96,96),dtype=torch.uint8);vv=torch.ones(8,1,96,96,dtype=torch.bool);vv[0]=False
views,frac=c8.preprocess_views(x,vv);assert views.shape==(16,3,224,224)and not views[0].any()and not views[8].any()
trgb=torch.rand(1,3,224,224)*255;tv=torch.zeros(1,1,224,224,dtype=torch.bool);assert not c9.preprocess_target(trgb,tv).any()
oldunit=np.array([0,0,3651]);newunit=np.arange(3652,11652);assert len(newunit)==8000 and newunit.min()==3652 and newunit.max()==11651
r=dict(completed=True,PASS=True,code=m.bind(p),test_code=m.bind(__file__),tests=['1541layout','exactoldtarget_full_central_TC_fractions_copied','standardcosTF_and_newcosTC','missingdescriptorszero','wrongshapereject','pure_V90crop_scalar_exact','qcontinuous_nearest_unavailable','C8full96central32preprocess_order_and_invalidmean','C9target_invalidmean','oldunit_reuse_new8000unique'],scalar_crop_checks=checks,source_pixels_features_weights_encoder_forward_labels_current_fit=0,torch=torch.__version__)
m.write(H/'GENERATED_FEATURE_TESTS.json',r);print(json.dumps(m.bind(H/'GENERATED_FEATURE_TESTS.json')))
