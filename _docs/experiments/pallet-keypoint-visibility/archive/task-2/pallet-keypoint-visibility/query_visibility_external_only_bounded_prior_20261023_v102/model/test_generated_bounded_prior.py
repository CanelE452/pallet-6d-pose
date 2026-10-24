"""Generated C1-format checkpoint and tensors only; never actual source assets."""
from pathlib import Path
import argparse,inspect,json,tempfile
import numpy as np
import torch
import bounded_prior_residual as m
H=Path(__file__).resolve().parent

def bind(p):return dict(path=str(Path(p).resolve()),sha256=m.file_sha256(p))
def inputs(n):
 p=torch.randn(n,65,9,9);p[:,64]=1
 return p,torch.randint(0,256,(n,3,96,96),dtype=torch.uint8),torch.ones(n,1,96,96,dtype=torch.bool)
def zero_or_none(x):return x is None or torch.count_nonzero(x)==0

def deterministic_addition(device):
 tiny=torch.nextafter(torch.tensor(0.,dtype=torch.float32),torch.tensor(1.,dtype=torch.float32)).item()
 raw=torch.tensor([-10.,-2.,-tiny,-0.,0.,tiny,2.,10.],device=device)
 answer={}
 for arm in m.ARMS:
  anchor=m.prior_anchor(raw,arm)
  assert torch.equal(anchor,raw if arm==m.ARMS[0]else raw.clamp(-2,2))
  assert torch.equal(torch.signbit(anchor),torch.signbit(raw))
  initial=anchor+torch.zeros_like(anchor)
  assert torch.equal(initial>=0,raw>=0),'Initial sign lost for finite subnormal/signedzero'
  grads=[]
  for repeat in range(2):
   delta=torch.linspace(-.4,.4,8,device=device).requires_grad_();combined=anchor+delta
   loss=torch.nn.functional.binary_cross_entropy_with_logits(combined,torch.tensor([0.,1.,0.,1.,0.,1.,0.,1.],device=device));loss.backward();grads.append(delta.grad.detach().clone())
  assert torch.equal(grads[0],grads[1])and torch.isfinite(grads[0]).all()
  answer[arm]=dict(sign_parity=True,clamp_exact=True,signedzero_anchor_signbit=True,deterministic_backward_bitexact=True)
 return answer

def tests():
 assert torch.__version__=='2.1.1+cu118';torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.manual_seed(194);base=m._BASE.QueryCenterScalar()
 with torch.no_grad():base.readout[-1].weight.normal_(0,.4);base.readout[-1].bias.fill_(4.25)
 original=m.state_sha256(base.state_dict());models=[]
 with tempfile.TemporaryDirectory(prefix='v102_invented_C1_')as tmp:
  path=Path(tmp)/'INVENTED_C1_FORMAT.pt';torch.save(dict(state_dict=base.state_dict(),update=100,seed=194),path)
  for arm in m.ARMS:
   torch.manual_seed(205);model,receipt=m.load_frozen_c1(arm,path,m.file_sha256(path));models.append(model)
   assert receipt['loaded_base_state_sha256']==original and receipt['C5_trained_branch_loaded']is False
  try:m.load_frozen_c1(m.ARMS[0],path,'0'*64)
  except RuntimeError:pass
  else:raise AssertionError('Wrong checkpoint SHA accepted')
  try:m.load_frozen_c1('UNDECLARED_ARM',path,m.file_sha256(path))
  except ValueError:pass
  else:raise AssertionError('Undeclared arm accepted')
 c6,c7=models
 assert all(torch.equal(c6.state_dict()[k],c7.state_dict()[k])for k in c6.state_dict())
 for model in models:
  assert sum(p.numel()for p in model.parameters())==29794
  assert sum(p.numel()for p in model.parameters()if p.requires_grad)==3221
  assert all(not p.requires_grad for p in model.base.parameters())and all(p.requires_grad for p in model.residual_encoder.parameters())
  assert tuple(inspect.signature(model.forward).parameters)==('p3_roi','local_rgb96','local_valid96')
  assert all(torch.equal(a,b)for a,b in zip(base.local_encoder.state_dict().values(),model.residual_encoder.state_dict().values()))
 x=inputs(8);base.eval()
 with torch.no_grad():reference=base(*x)
 out=[model(*x)for model in models]
 assert(reference['query_logit'].abs()>2).any()
 for model,o in zip(models,out):
  assert torch.equal(o['reference_base_logit'],reference['query_logit'])and torch.equal(o['query_available'],reference['query_available'])
  assert not o['reference_base_logit'].requires_grad and not o['base_logit'].requires_grad
  assert torch.count_nonzero(o['delta_logit'])==0 and o['delta_logit'].requires_grad
  assert torch.equal(o['query_logit']>=0,reference['query_logit']>=0)
  model.train();assert all(not module.training for module in model.base.modules())
  model.eval();q=model(*x);assert all(torch.equal(o[k],q[k])for k in q)
 assert torch.equal(out[0]['query_logit'],reference['query_logit'])
 assert torch.equal(out[1]['query_logit'],reference['query_logit'].clamp(-2,2))
 # Phase/locality law is inherited from pinned C5 and must be nonvacuous.
 yy,xx=torch.meshgrid(torch.arange(4),torch.arange(4),indexing='ij');ramp=(xx+10*yy).float()[None,None].expand(2,16,-1,-1).clone()
 assert torch.equal(m.sample_query_feature(ramp),torch.full((2,16),15.5/8*11))
 for model in models:
  with torch.no_grad():model.residual_readout[-1].weight.normal_(0,.2);model.residual_readout[-1].bias.fill_(.1)
  p,r,v=inputs(4);r2=255-r;r2[:,:,32:64,32:64]=r[:,:,32:64,32:64];v2=v.clone();v2[:,:,:16,:16]=False
  assert model(p,r,v)['delta_logit'].abs().sum()>0
  assert torch.equal(model(p,r,v)['delta_logit'],model(p,r2,v2)['delta_logit'])
  v[:,:,36:42,36:42]=False;r3=r.clone();r3[:,:,36:42,36:42]=255-r3[:,:,36:42,36:42]
  assert torch.equal(model(p,r,v)['delta_logit'],model(p,r3,v)['delta_logit'])
  p[0,64]=0;v[0]=False;v[1,0,47,47]=False;p[2,64,4,4]=0;o=model(p,r,v)
  assert o['query_available'].tolist()==[False,False,False,True]
  for key in('query_logit','base_logit','reference_base_logit','delta_logit'):assert torch.equal(o[key][:3],torch.zeros(3))
  assert m.decode_binary(o)[:3].tolist()==[-1,-1,-1]
  model.assert_base_unchanged()
 rejected=0
 for key in('reference_xy','GT_region','native_id','source_id','variant','arm','label','pose'):
  try:c6(*x,**{key:torch.zeros(8,2)})
  except TypeError:rejected+=1
  else:raise AssertionError('Forbidden tensor/metadata accepted')
 try:c6({'p3_roi':x[0]},x[1],x[2])
 except TypeError:rejected+=1
 else:raise AssertionError('Dictionary accepted')
 try:c6(x[0].reshape(1,8,65,9,9),x[1],x[2])
 except ValueError:rejected+=1
 else:raise AssertionError('Implicit native slot accepted')
 # Unknown and factual U losszero independent of clamp.
 z=torch.tensor([-.3,.7,1.,-2.],requires_grad=True)
 l=m.known_binary_loss(dict(query_logit=z,query_available=torch.tensor([True,True,False,True])),torch.tensor([0.,1.,1.,float('nan')]),torch.tensor([True,True,True,False]))
 l['loss'].backward();assert torch.equal(z.grad[2:],torch.zeros(2))
 device_proofs={'CPU':deterministic_addition('cpu')}
 assert torch.cuda.is_available(),'Root requires exact learned CUDA generated proof'
 device_proofs['CUDA']=deterministic_addition('cuda:0')
 # Full generated model backward twice: no source tensors or actual weights.
 backward={}
 for model in models:
  model=model.cuda().train();xx=tuple(t.cuda()for t in inputs(4));grad=[]
  for repeat in range(2):
   model.zero_grad(set_to_none=True);o=model(*xx);loss=m.known_binary_loss(o,torch.tensor([0.,1.,0.,1.],device='cuda'),torch.ones(4,dtype=torch.bool,device='cuda'))['loss'];loss.backward()
   assert all(p.grad is None for p in model.base.parameters())
   grad.append({k:v.grad.detach().clone()for k,v in model.named_parameters()if v.grad is not None})
  assert all(torch.equal(grad[0][k],grad[1][k])for k in grad[0])
  assert sum(float(v.abs().sum())for k,v in grad[0].items()if k.startswith('residual_encoder'))>0
  model.assert_base_unchanged();backward[model.arm]=dict(full_generated_CUDA_backward_exact=True,base_gradients_absent=True,copied_encoder_gradient_nonzero=True)
 return dict(completed=True,PASS=True,model_code=bind(H/'bounded_prior_residual.py'),test_code=bind(__file__),inherited_C5_code=dict(path=str(m.C5_CODE),sha256=m.C5_CODE_SHA256),inherited_C1_code=dict(path=str(m.BASE_CODE),sha256=m.BASE_CODE_SHA256),arms=list(m.ARMS),parameters=29794,trainable_parameters=3221,base_parameters=26573,same_seed205_initial_fullstate_exact=True,invented_nonzero_C1_raw_and_clipped_initial_parity=True,base_frozen_eval_hash_unchanged=True,fresh_branch_copied_C1_not_C5_weights=True,phase=dict(feature4_index=1.9375,align_corners=True,grid=7/24),residual_nonzero_locality_and_valid_mask_invariance=True,factual_U_losszero_and_availability_exact=True,forbidden_shape_GT_metadata_rejections=rejected,deterministic_clamp_addition=device_proofs,full_generated_backward=backward,actual_source_RGB_features_checkpoint_weights_forward_fit_private_current_PnP=0,generated_optimizer_updates=0,runtime=dict(torch=torch.__version__,device=torch.cuda.get_device_name(0),CPUthreads1=True,strict_deterministic=True,TF32=False))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();r=tests()
 with a.output.open('x')as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(r))
