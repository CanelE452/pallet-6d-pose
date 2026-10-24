"""Generated model/loss tests only; no source caches/GT/checkpoints/current."""
from pathlib import Path
import hashlib,importlib.util,json
import numpy as np
import torch
from torch.nn import functional as F
from dense_support_head import DenseSupportHead
from dense_auxiliary_loss import dense_support_losses,combined_support_loss

HERE=Path(__file__).resolve().parent;P=HERE.parents[1]
oldpath=P/'query_visibility_reference_local_support_20261023_v76/mixed_fit/query_center_support_head.py'
s=importlib.util.spec_from_file_location('generated_v80_old_primary_only',oldpath);old=importlib.util.module_from_spec(s);s.loader.exec_module(old)
torch.set_num_threads(1);torch.manual_seed(20261023);baseline=old.NativeContextSupportHead();oldrng=torch.get_rng_state().clone()
torch.manual_seed(20261023);new=DenseSupportHead();assert torch.equal(oldrng,torch.get_rng_state())
assert sum(p.numel()for p in baseline.parameters())==41237
assert sum(p.numel()for p in new.parameters())==41287
for k,v in baseline.state_dict().items():assert torch.equal(v,new.state_dict()[k]),k
assert not new.aux_decoder.weight.any()and not new.aux_decoder.bias.any()
g=torch.Generator().manual_seed(127)
Fcount=1;n=Fcount*8
p3=torch.randn(Fcount,8,65,9,9,generator=g);p3[:,:,64]=(torch.rand(Fcount,8,9,9,generator=g)>.1).float()
rgb=torch.randint(0,256,(Fcount,8,3,96,96),dtype=torch.uint8,generator=g)
lv=torch.rand(Fcount,8,1,96,96,generator=g)>.1
trgb=torch.randint(0,256,(Fcount,3,128,128),dtype=torch.uint8,generator=g);tv=torch.ones(Fcount,1,128,128,dtype=torch.bool)
ctx=torch.randn(Fcount,8,42,generator=g)
whole=(p3,rgb,lv,trgb,tv,ctx)
flat=(p3.flatten(0,1),rgb.flatten(0,1),lv.flatten(0,1),trgb.repeat_interleave(8,0),tv.repeat_interleave(8,0),ctx.flatten(0,1))
# Use a nonzero output and all nonzero inputs for meaningful primary parity.
with torch.no_grad():
 baseline.classifier[2].weight.copy_(torch.linspace(-.2,.2,32).reshape(1,32));baseline.classifier[2].bias.fill_(.173)
 st=new.state_dict();st.update(baseline.state_dict());new.load_state_dict(st)
wo=new(*whole);fo=new(*flat);bo=baseline(*flat)
assert torch.equal(wo['support_logit'].flatten(),fo['support_logit'])and torch.equal(fo['support_logit'],bo['support_logit'])
assert torch.equal(wo['aux_mask_logits'].flatten(0,1),fo['aux_mask_logits'])and not fo['aux_mask_logits'].any()
assert fo['support_logit'].abs().sum()>0 and fo['support_logit'].std()>0
valid=torch.rand(n,9,9,generator=g)>.15;m0=torch.rand(n,9,9,generator=g)>.4;m4=m0&(torch.rand(n,9,9,generator=g)>.5)
targets=(torch.arange(n)%2).float()
base_loss=F.binary_cross_entropy_with_logits(bo['support_logit'],targets);base_loss.backward()
loss0=combined_support_loss(fo['support_logit'],targets,fo['aux_mask_logits'],valid,m0,m4,0.);loss0['loss'].backward()
assert torch.equal(base_loss,loss0['loss'])
maxdiff=0.
for name,p in baseline.named_parameters():
 q=dict(new.named_parameters())[name];assert p.grad is not None and q.grad is not None
 diff=float((p.grad-q.grad).abs().max());maxdiff=max(maxdiff,diff);assert diff==0,name
assert new.aux_decoder.weight.grad is not None and not new.aux_decoder.weight.grad.any()
# Dense gradient reaches its decoder initially; zero decoder gives zero shared aux gradient.
new.zero_grad(set_to_none=True);o=new(*flat);la=dense_support_losses(o['aux_mask_logits'],valid,m0,m4);la['auxiliary_mean'].backward()
initial_decoder_grad=float(new.aux_decoder.weight.grad.norm());assert initial_decoder_grad>0
initial_shared_auxgrad=max(float(p.grad.abs().max())for name,p in new.named_parameters()if name.startswith('fusion.')and p.grad is not None);assert initial_shared_auxgrad==0
with torch.no_grad():new.aux_decoder.weight.uniform_(-.1,.1);new.aux_decoder.bias.fill_(.07)
new.zero_grad(set_to_none=True);o=new(*flat);ll=dense_support_losses(o['aux_mask_logits'],valid,m0,m4);ll['auxiliary_mean'].backward()
later_shared_grad=max(float(p.grad.abs().max())for name,p in new.named_parameters()if name.startswith('fusion.')and p.grad is not None);assert later_shared_grad>0
# Independent float64 class-loop loss with empty, O-only, V-only, mixed cells.
z=torch.linspace(-3.,2.,4*2*9*9,dtype=torch.float64).reshape(4,2,9,9).requires_grad_();v=torch.zeros(4,9,9,dtype=torch.bool);v[1:]=True
a=torch.zeros_like(v);a[2]=True;a[3,:4]=True;t=a.clone();t[3,2:4]=False
got=dense_support_losses(z,v,a,t);refs=[[],[]]
for i in range(4):
 for head in(0,1):
  region=v[i].numpy()if head==0 else(v[i]&a[i]).numpy();truth=a[i].numpy()if head==0 else t[i].numpy();parts=[]
  for cls in(0,1):
   select=region&(truth==cls)
   if select.any():
    zz=z.detach()[i,head].numpy()[select];parts.append(float(np.mean(np.logaddexp(0,zz)-cls*zz)))
  refs[head].append(float(np.mean(parts))if parts else 0.)
assert np.allclose(got['M0_per_query'].detach().numpy(),refs[0],atol=1e-14,rtol=0)
assert np.allclose(got['VgivenM0_per_query'].detach().numpy(),refs[1],atol=1e-14,rtol=0)
got['auxiliary_mean'].backward();assert not z.grad[0].any()and not z.grad[1,1].any()
empty=torch.randn(3,2,9,9,generator=g,requires_grad=True);zero=torch.zeros(3,9,9,dtype=torch.bool)
ze=dense_support_losses(empty,zero,zero,zero);ze['auxiliary_mean'].backward();assert ze['auxiliary_mean']==0 and not empty.grad.any()
# Invalid GT inconsistencies are ignored; valid inconsistencies fail closed.
dense_support_losses(empty.detach(),zero,zero,~zero)
try:dense_support_losses(empty.detach(),~zero,zero,~zero);raise AssertionError('M4subset guard missing')
except ValueError:pass
for field in('GT','reference_xy','perm','matched','source_id','expected_support_M0'):
 try:new(*flat,**{field:torch.zeros(n)});raise AssertionError('GTforward guard missing')
 except TypeError:pass
try:new(*flat[:-1],{'context42':ctx});raise AssertionError('Metadata dict forward accepted')
except TypeError:pass
out=HERE/'GENERATED_DENSE_SUPPORT_TESTS.json';assert not out.exists()
out.write_text(json.dumps(dict(PASS=True,parameters=41287,new_auxiliary_parameters=50,primary_initial_state_and_RNG_exact=True,nonzero_primary_forward_exact=True,whole8_flatten_forward_exact=True,lambda0_primary_loss_exact=True,lambda0_shared_gradient_max_abs_diff=maxdiff,zero_aux_first_update_shared_gradient=initial_shared_auxgrad,zero_aux_decoder_gradient_norm=initial_decoder_grad,nonzero_aux_shared_gradient_max=later_shared_grad,independent_float64_present_class_loss=True,empty_invalid_graph_zero_gradient=True,conditional_outside_M0_gradient_zero=True,M4subset_validity_guard=True,GT_metadata_forward_blocked=True,old_model_code_sha256=hashlib.sha256(oldpath.read_bytes()).hexdigest(),model_sha256=hashlib.sha256((HERE/'dense_support_head.py').read_bytes()).hexdigest(),loss_sha256=hashlib.sha256((HERE/'dense_auxiliary_loss.py').read_bytes()).hexdigest(),generated_test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_GT_cache_RGB_checkpoint_current_reads=0,actual_source_forward_training_PnP=0),indent=2)+'\n')
print(out)
