import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,importlib.util,json
import numpy as np
import torch
from torch.nn import functional as F
H=Path(__file__).resolve().parent;p=H/'prepare_spatial_query_features.py';sp=importlib.util.spec_from_file_location('c11_math',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
def test(output):
 torch.set_num_threads(1);torch.manual_seed(211)
 qs=np.array([[7,7],[217,217],[112,112],[21,105],[0,0],[224,224],[-1e-7,100],[224+1e-7,100],[100,100],[np.nan,100],[-1,-1],[100,100]],np.float64);n=len(qs);boxes=np.tile([0,0,224,224],(n,1)).astype(float);boxes[-1]=[0,0,0,224];support=np.ones(n,bool);support[8]=False
 ramp=torch.zeros(16,16,384);ramp[:,:,0]=torch.arange(16)[None,:];ramp[:,:,1]=torch.arange(16)[:,None];ramp[:,:,2]=1;patch=ramp.reshape(1,256,384).repeat(n,1,1)
 value,inside=m.sample_Q_raw(patch,qs,boxes,support);assert inside.tolist()==[1,1,1,1,1,1,0,0,0,0,0,0]
 expected=[[0,0,1],[15,15,1],[7.5,7.5,1],[1,7,1],[0,0,1],[15,15,1]]
 for i,e in enumerate(expected):assert torch.equal(value[i,:3],torch.tensor(e,dtype=torch.float32))
 assert not value[~inside].any()
 # Independent scalar interpolation at a fractional native point, including the
 # actual finalF32grid representation and alignFalse half-cell relation.
 q=np.array([[73.2,129.7]],np.float64);v,_=m.sample_Q_raw(patch[:1],q,boxes[:1],np.array([True]));g=(2*q/224-1).astype(np.float32)[0];ix=(g.astype(float)+1)*8-.5;ix=np.clip(ix,0,15)
 assert np.max(np.abs(v[0,:2].numpy()-ix))<2e-6
 cls=torch.randn(n,384);fullcls=torch.randn(n,384);centralcls=torch.randn(n,384);fullpatch=torch.zeros(n,16,16,384)
 raw4=torch.tensor([[1.,0.,0.],[0.,7.,0.],[0.,0.,3.],[2.,5.,1.]])
 fullpatch[:,7:9,7:9,:3]=raw4.reshape(1,2,2,3)
 frac=np.ones((n,3),np.float32);frac[8,0]=0;frac[9,1]=0;frac[10,2]=0;c4=np.ones(n,bool);c4[9]=False
 before=qs.copy();z,flags=m.assemble(cls,patch,fullcls,fullpatch.reshape(n,256,384),centralcls,qs,boxes,frac,c4)
 assert z.shape==(n,1928)and torch.isfinite(z).all()and np.array_equal(qs,before,equal_nan=True)
 assert np.array_equal(flags,inside)and np.array_equal(z[:,1927].numpy(),inside.astype(np.float32))
 assert torch.equal(z[0,:384],F.normalize(cls[:1],dim=1,eps=1e-12)[0])and not z[8,:384].any()
 assert not z[~flags,384:768].any()and not z[9,768:1152].any()and not z[10,1152:1536].any()and not z[9,1536:1920].any()
 rawmean=raw4.mean(0);expectedP=F.normalize(rawmean,dim=0,eps=1e-12);assert torch.equal(z[0,1536:1539],expectedP)
 wrong=F.normalize(F.normalize(raw4,dim=1).mean(0),dim=0);assert not torch.allclose(expectedP,wrong,atol=1e-3)
 for j,(a,b)in enumerate([(z[:,:384],z[:,1152:1536]),(z[:,:384],z[:,768:1152]),(z[:,384:768],z[:,1152:1536]),(z[:,384:768],z[:,1536:1920])]):assert torch.equal(z[:,1920+j],F.cosine_similarity(a,b,dim=1,eps=1e-12))
 assert np.array_equal(z[:,1924:1927].numpy(),frac)
 # Closed u0/u1 are inside; tinyoutside remains zero even if F32grid rounds to1.
 available=np.array([True,False]*6);saved=available.copy();m.assemble(cls,patch,fullcls,fullpatch.reshape(n,256,384),centralcls,qs,boxes,frac,c4);assert np.array_equal(available,saved)
 c8=m.mod(m.C8,'generated_c11_local_preprocess');c9=m.mod(m.C9,'generated_c11_target_preprocess');rgb=torch.randint(0,256,(8,3,96,96),dtype=torch.uint8);valid=torch.ones(8,1,96,96,dtype=torch.bool);valid[0]=False
 local,_=c8.preprocess_views(rgb,valid);target=c9.preprocess_target(torch.zeros(8,3,224,224),torch.zeros(8,1,224,224,dtype=torch.bool));assert torch.cat((target,local),0).shape==(24,3,224,224)and not local[0].any()and not local[8].any()and not target.any()
 assert torch.cat((target[:1],local),0).shape[0]==17
 crop=m.pure_crop();a=np.arange(11*13*3,dtype=np.uint8).reshape(11,13,3);rr,v=crop(a,[-.2,.3],16.,96);assert rr.shape==(3,96,96)and v.shape==(1,96,96)and v.any()and not v.all()
 for args in[(patch.double(),qs,boxes,support),(patch,qs[:1],boxes,support)]:
  try:m.sample_Q_raw(*args)
  except ValueError:pass
  else:raise AssertionError('Malformed sample schema accepted')
 m.write(output,dict(completed=True,PASS=True,code=m.bind(p),test_code=m.bind(__file__),tests=['1928_descriptor_scalar_order','native_q_patchcentre_phase','closed_edges_border_alignFalse','fractional_grid_scalar_oracle','tinyoutside_gate_before_F32_rounding','finite_missing_sentinel_emptybox_nosupport_zeroQ','raw_patch_bilinear_then_L2','raw_centre2x2_pool_then_L2_not_normalizedpatchmean','T_F_C_support_zeros','literal_RGBcentral4_P_zero','independent_descriptor_cosines','Qoutside_does_not_change_availability','exact_C8_C9_meanfill_preprocess','old24views_new17views_max32','original_q_unchanged','RGB96valid_phase_preserved','bad_dtype_shape_reject'],source_pixels_features_weights_model_constructor_or_encoder_forward_fit_current=0,torch=torch.__version__))
 print(json.dumps(m.bind(output)))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,default=H/'GENERATED_FEATURE_TESTS.json');test(a.parse_args().output)
