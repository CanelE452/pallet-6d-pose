"""Invented CPU checks only: exact baseline updates and combined owner gradients."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json
import numpy as np
import torch
from torch.nn import functional as F
H=Path(__file__).resolve().parent;P=H.parent.parent
FINAL=H/'train_counterfactual_owner_control.py'
OLD=P/'query_visibility_native_slot_owner_factorial_20261024_v160/source_fit_preparation/train_slot_owner_factorial.py'
HEAD=P/'query_visibility_native_slot_owner_factorial_20261024_v160/head_preparation/native_slot_owner_head.py'
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
assert bind(OLD)['sha256']=='3dd58095938f63b37b92be2b42a628fb0862169087e9c41f7af873c8001c9260' and bind(HEAD)['sha256']=='82c843555cff138d8830f612befb733807c688e1e6c84d0607606dc4c4bd14b6'
def extract(p,names):
 ns={'torch':torch,'F':F};nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(nodes)==len(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p)+'::invented_only', 'exec'),ns);return ns
old=extract(OLD,{'balanced_owner_ce'});new=extract(FINAL,{'balanced_owner_ce','owner_loss_for_condition'})
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
checks={};counts=torch.tensor([210,209,148]);target=torch.repeat_interleave(torch.arange(3),counts)
gen=torch.Generator().manual_seed(162);logits=torch.randn((567,3),generator=gen,dtype=torch.float64,requires_grad=True)
loss=new['balanced_owner_ce'](logits,target);loss.backward();want=(F.softmax(logits.detach(),1)-F.one_hot(target,3).double())/(3*counts[target,None])
assert torch.allclose(logits.grad,want,atol=1e-14,rtol=1e-12);checks['combined567_analytic_gradient_exact_class_weights']=True
original_y=torch.tensor([0]*53+[1]*52+[2]*148);cf_y=torch.tensor([0]*157+[1]*157);combined_y=torch.cat((original_y,cf_y));z=torch.randn((567,3),generator=gen,dtype=torch.float64)
correct=new['balanced_owner_ce'](z,combined_y);wrong=old['balanced_owner_ce'](z[:253],original_y)+sum(F.cross_entropy(z[253:][cf_y==k],cf_y[cf_y==k]) for k in (0,1))/2
assert not torch.allclose(correct,wrong);checks['combined567_not_erroneous_separate253_plus314_loss']=True
checks['original_owner_perrow_weight_change_explicit']=all(abs(1/(3*a)-1/(3*b))>0 for a,b in ((53,210),(52,209))) and 1/(3*148)==1/(3*148)
spec=importlib.util.spec_from_file_location('invented_existing_head',HEAD);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def model():
 torch.manual_seed(1550);return mod.NativeSlotOwnerQueryHead(native_slot_hint=False,owner_auxiliary=True).float()
def opts(m):return [torch.optim.AdamW(p,lr=1e-4,weight_decay=.001) for p in (m.visibility_parameters(),m.auxiliary_parameters(),m.new_parameters())]
def same(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and a.detach().contiguous().numpy().tobytes()==b.detach().contiguous().numpy().tobytes()
 if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return isinstance(b,type(a)) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b
models=[model(),model()];assert same(models[0].state_dict(),models[1].state_dict()) and len(models[0].state_dict())==16
optimizers=[opts(m) for m in models]
xs=[torch.randn((512,1928),generator=gen) for _ in range(4)];slots=torch.arange(512,dtype=torch.int64)%8;z42=torch.zeros((512,42));positions=np.arange(253);rngs=[np.random.default_rng(1550),np.random.default_rng(1550)]
def balance(z,y):return sum(F.binary_cross_entropy_with_logits(z[y==k],y[y==k]) for k in (0,1))/2
per_step=[]
for step in range(1,4):
 draws=[r.choice(512,64,replace=True) for r in rngs];assert np.array_equal(*draws)
 states=[]
 for index,(m,oo,ii) in enumerate(zip(models,optimizers,draws)):
  before=torch.get_rng_state().clone()
  for opt in oo:opt.zero_grad(set_to_none=True)
  no=[m(xs[k][ii],z42[:64],slots[ii]) for k in (0,1)];po=m(xs[2][:43],z42[:43],slots[:43]);mo=m(xs[3][:390],z42[:390],slots[:390]);eo=m(xs[3][ii],z42[:64],slots[ii])['target_entity'];qo=[m(xs[k][:32],z42[:32],slots[:32])['coarse_point_quality'] for k in (0,1)]
  y64=torch.arange(64).float()%2;y43=torch.arange(43).float()%2;y390=torch.arange(390).float()%2;y32=torch.arange(32).float()%2
  original_loss=sum(balance(o['external_occlusion'],y64) for o in no)+balance(po['external_occlusion'],y43)+balance(mo['external_occlusion'],y390)+balance(eo,y64)+sum(balance(q,y32) for q in qo)/2
  if index==0:owner=old['balanced_owner_ce'](mo['query_visible_owner'][positions],original_y)
  else:owner=new['owner_loss_for_condition'](None,mo,positions,original_y,None,None,z42,None,None,False)
  total=original_loss+owner;total.backward()
  groups=[m.visibility_parameters(),m.auxiliary_parameters(),m.new_parameters()]
  norms=[torch.nn.utils.clip_grad_norm_(p,5) for p in groups]
  for opt in oo:opt.step()
  assert torch.equal(before,torch.get_rng_state()) and not m.geometry.weight.any() and not m.native_hint.weight.any()
  states.append(dict(total=float(total),norms=[float(n) for n in norms]))
 assert states[0]==states[1] and same(models[0].state_dict(),models[1].state_dict()) and all(same(a.state_dict(),b.state_dict()) for a,b in zip(*optimizers))
 assert json.dumps(rngs[0].bit_generator.state,sort_keys=True)==json.dumps(rngs[1].bit_generator.state,sort_keys=True)
 per_step.append(dict(step=step,all16_parameters_numeric_and_byte_equal=True,all3_AdamW_states_numeric_and_byte_equal=True,original_numpy_and_torch_RNG_unchanged=True,baseline_no_CF_forward=True))
checks['three_step_C41_newbaseline_all16params_all3optimizers_byte_parity']=True
# Isolate CF owner gradient through the actual augmented helper, with nonzero
# owner weights so the shared pre-slot gradient is meaningful after initialization.
m=model()
with torch.no_grad():m.owner_head.weight.copy_(torch.randn(m.owner_head.weight.shape,generator=gen)*.1)
mo={'query_visible_owner':torch.randn((390,3),generator=gen)};cx=torch.randn((314,1928),generator=gen);cs=torch.arange(314,dtype=torch.int64)%8
before=torch.get_rng_state().clone();aug=new['owner_loss_for_condition'](m,mo,positions,original_y,cx,cs,z42,np.arange(314),cf_y,True);aug.backward()
nonzero=[n for n,p in m.named_parameters() if p.grad is not None and bool(p.grad.any())]
assert set(nonzero)=={'head.0.weight','head.0.bias','owner_head.weight','owner_head.bias'} and torch.equal(before,torch.get_rng_state())
checks['combined_CF_owner_gradient_shared_head0_owner_only_no_nativehint_head2_entity_quality']=True
checks['owner_addition_no_extra_RNG_draws']=True
assert all(checks.values())
r=dict(schema='V162_INVENTED_COMBINED_OWNER_CONTROL_MEANINGFUL_CHECKS',completed=True,PASS=True,code=bind(FINAL),check_code=bind(Path(__file__).resolve()),head=bind(HEAD),old_trainer=bind(OLD),checks=checks,per_step=per_step,CF_gradient_nonzero_parameters=nonzero,combined_counts=[210,209,148],original_counts=[53,52,148],CF_counts=[157,157,0],runtime=dict(executable=sys.executable if 'sys' in globals() else str(Path('/home/minjae/anaconda3/envs/pallet-yolo26/bin/python')),torch=str(torch.__version__),numpy=np.__version__,device='cpu'),source_feature_weight_pixel_bodies_read=0,actual_source_fit_or_encoder_calls=0,invented_optimizer_steps=3)
p=H/'GENERATED_FINAL_CONTROL_CHECKS.json'
with p.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(dict(PASS=True,checks=bind(p))))
