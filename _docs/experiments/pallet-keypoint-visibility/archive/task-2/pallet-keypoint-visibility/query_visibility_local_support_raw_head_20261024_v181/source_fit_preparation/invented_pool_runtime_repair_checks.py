"""Explicitly authorized invented CPU/CUDA checks for the pool-only repair.

No source/cache/checkpoint file is opened. The prior head is source code only.
"""
from pathlib import Path
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import importlib.util,sys,hashlib,json
import numpy as np
import torch

H=Path(__file__).resolve().parent;V=H.parent
OLD=V/'provenance/BEFORE_DETERMINISTIC_POOL_RUNTIME_REPAIR/head_preparation/local_support_head.py'
NEW=V/'head_preparation/local_support_head.py'
OUT=H/'GENERATED_POOL_RUNTIME_REPAIR_CHECKS.json'

def module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def state_sha(m):
 h=hashlib.sha256()
 for k,a in sorted(m.state_dict().items()):h.update(k.encode());h.update(a.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def obj_sha(x):
 h=hashlib.sha256()
 def visit(v):
  if torch.is_tensor(v):a=v.detach().cpu().contiguous().numpy();h.update(str((a.dtype,a.shape)).encode());h.update(a.tobytes())
  elif isinstance(v,dict):
   for k in sorted(v,key=str):h.update(str(k).encode());visit(v[k])
  elif isinstance(v,(list,tuple)):
   for a in v:visit(a)
  else:h.update(repr(v).encode())
 visit(x);return h.hexdigest()
def b(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def run():
 assert Path(sys.executable).resolve()==Path('/home/minjae/anaconda3/envs/pallet-yolo26/bin/python').resolve()
 assert torch.__version__=='2.1.1+cu118' and np.__version__=='1.26.4' and torch.cuda.is_available()
 assert not OUT.exists()
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True,warn_only=False)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 old,new=module(OLD,'v181_old_pool_source'),module(NEW,'v181_new_pool_source')
 checks=[];numeric={}
 def passed(name,ok):
  assert bool(ok),name
  assert torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
  checks.append(dict(name=name,PASS=True,strict_deterministic_still_on=True))
 torch.manual_seed(1810);a=old.Raw96SupportHead();ra=torch.get_rng_state().clone()
 torch.manual_seed(1810);c=new.Raw96SupportHead();rc=torch.get_rng_state().clone()
 passed('Same115157 parameter tensors/statekeys/initialization/RNG, pool has no state',sum(p.numel() for p in a.parameters())==sum(p.numel() for p in c.parameters())==115157 and list(a.state_dict())==list(c.state_dict()) and state_sha(a)==state_sha(c) and torch.equal(ra,rc))
 # Direct fixed partition: no overlap, each output averages the same16 pixels.
 z=(torch.arange(2*64*12*12,dtype=torch.float32).reshape(2,64,12,12).remainder(113)-56)/37
 oldcpu=torch.nn.AdaptiveAvgPool2d((3,3));newcpu=torch.nn.AvgPool2d(4,4)
 oc=oldcpu(z);nc=newcpu(z);og=oldcpu(z.cuda());ng=newcpu(z.cuda())
 numeric['pool_oldCPU_newCPU_maxdiff']=float((oc-nc).abs().max());numeric['pool_oldGPU_newGPU_maxdiff']=float((og-ng).abs().max())
 numeric['pool_oldCPU_newGPU_maxdiff']=float((oc-ng.cpu()).abs().max())
 passed('CPU Adaptive12to3 and fixed4stride4 forward bytes equal',torch.equal(oc,nc))
 passed('OldGPUforward and newGPUfixed forward bytes equal',torch.equal(og,ng))
 passed('Exact fixed pool CPU and CUDA forward bytes equal',torch.equal(oc,ng.cpu()))
 weights=(torch.arange(2*64*3*3,dtype=torch.float32).reshape(2,64,3,3).remainder(13)-6)/11
 z1=z.clone().requires_grad_();z2=z.clone().requires_grad_();z3=z.cuda().requires_grad_()
 (oldcpu(z1)*weights).sum().backward();(newcpu(z2)*weights).sum().backward();(newcpu(z3)*weights.cuda()).sum().backward()
 passed('Strict newCUDAbackward equals oldCPUAdaptive and newCPU input gradients bytes',torch.equal(z1.grad,z2.grad) and torch.equal(z1.grad,z3.grad.cpu()))
 x=(torch.arange(2*8*96*96,dtype=torch.float32).reshape(2,8,96,96).remainder(97)-48)/53
 ao=a(x);co=c(x)
 passed('Whole-head oldCPU/newCPU outputs byteequal on fixed96 inputs',all(torch.equal(ao[k],co[k]) for k in ao))
 loss=lambda o:torch.nn.functional.binary_cross_entropy_with_logits(o['support_logits'],torch.tensor([[1.,1.],[0.,1.]],device=o['support_logits'].device))+torch.nn.functional.cross_entropy(o['owner_logits'],torch.tensor([0,2],device=o['owner_logits'].device))
 loss(ao).backward();loss(co).backward()
 passed('Whole-head oldCPU/newCPU all parameter gradients byteequal',all(torch.equal(pa.grad,pb.grad) for pa,pb in zip(a.parameters(),c.parameters())))
 ag=old.Raw96SupportHead().cuda();cg=new.Raw96SupportHead().cuda();ag.load_state_dict(a.state_dict());cg.load_state_dict(c.state_dict())
 agout=ag(x.cuda());cgout=cg(x.cuda())
 passed('Whole-head oldGPU/newGPU forward outputs byteequal',all(torch.equal(agout[k],cgout[k]) for k in agout))
 loss(cgout).backward()
 passed('Whole-head strict newCUDAbackward succeeds with finite all parameter gradients',all(p.grad is not None and torch.isfinite(p.grad).all() for p in cg.parameters()))
 # Both repaired arms, absent SAM: actual FP32 CUDA update/clip/AdamW sequence.
 torch.manual_seed(1810);ma=new.Raw96SupportHead().cuda();torch.manual_seed(1810);mb=new.Raw96SupportHead().cuda()
 opts=[torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4) for m in [ma,mb]]
 rngs=[np.random.default_rng(1810),np.random.default_rng(1810)];records=[]
 for step in range(3):
  draws=[r.choice(2,2,replace=True) for r in rngs];assert np.array_equal(*draws)
  torch_state=torch.get_rng_state().clone();cuda_state=torch.cuda.get_rng_state().clone()
  for m,op,draw in zip([ma,mb],opts,draws):
   op.zero_grad(set_to_none=True);ll=loss(m(x[draw].cuda()));ll.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);op.step()
  equal=state_sha(ma)==state_sha(mb) and obj_sha(opts[0].state_dict())==obj_sha(opts[1].state_dict()) and torch.equal(torch_state,torch.get_rng_state()) and torch.equal(cuda_state,torch.cuda.get_rng_state())
  assert equal;records.append(dict(step=step+1,all_parameter_optimizer_CPU_CUDA_RNG_draw_bytes_equal=True))
 passed('Three inventedCUDA AdamW updates share init/draw/loss/clip/optimizer/RNG bytes',len(records)==3)
 report=dict(completed=True,PASS=True,code=b(Path(__file__)),old_head_source=b(OLD),new_head_source=b(NEW),runtime=dict(executable=sys.executable,torch=torch.__version__,numpy=np.__version__,device=torch.cuda.get_device_name(0),FP32=True,TF32=False,strict_deterministic=True,warn_only=False),checks=checks,numeric_differences=numeric,three_steps=records,actual_source_cache_label_weight_or_fit_reads=0,actual_source_fit_updates=0,pool_only_change_no_parameters_or_statekeys_or_input_loss_draw_seed_changes=True)
 with OUT.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(completed=True,PASS=True,checks=len(checks),receipt=b(OUT))),flush=True)

if __name__=='__main__':run()
