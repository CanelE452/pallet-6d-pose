"""Generated tensors/labels only. No actual pack or ledger body reads."""
import importlib.util
import json
import math
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v94_generated_trainer',HERE/'train_source_once.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)

def main():
 torch.set_num_threads(1);torch.manual_seed(194)
 labels=np.array([0,1,1,-1],np.int8);known=labels>=0;splits=np.array(['VAL_HALL']*4)
 z=np.array([-1.,1.,0.,123.],np.float32);available=np.array([True,True,False,True])
 value,means,n=t.macro_BCE(z,labels,known,splits)
 d=math.log1p(math.exp(-1));e=(d+math.log(2))/2
 assert abs(value-(d+e)/2)<1e-14 and n==[1,2]
 m=t.cm_metrics(z,available,labels,known,np.ones(4,bool));assert m['matrix']==[[1,0,0],[0,1,1]] and m['E_recall']==.5 and m['known_U_rate']==1/3
 checks,passed=t.gates(dict(D_recall=.85,E_recall=.75,known_U_rate=.05));assert passed
 assert not t.gates(dict(D_recall=.849999,E_recall=.75,known_U_rate=.05))[1]
 hist=[(0,2.),(100,1.),(200,1.),(300,1.2)];best=float('inf');selected=None
 for u,v in hist:
  if v<best:best=v;selected=u
 assert selected==100
 lab=np.array([0,0,1,1,0,1,-1]);kn=lab>=0;sp=np.array(['TRAIN_LAB']*4+['VAL_HALL']*3)
 pools=t.make_pools(lab,kn,sp);g=torch.Generator().manual_seed(194);g2=torch.Generator().manual_seed(194)
 for _ in range(2000):
  ix=t.draw(pools,g);assert np.array_equal(ix,t.draw(pools,g2));assert np.all(sp[ix]=='TRAIN_LAB') and np.sum(lab[ix]==0)==np.sum(lab[ix]==1)==32
 try:t.require_entry(dict(approved_actual_source_fit=False))
 except AssertionError:pass
 else:raise AssertionError('Unapproved actual reads must fail')
 op=t.operator();model=op.QueryCenterScalar();assert sum(x.numel() for x in model.parameters())==26573
 p=torch.randn(4,65,9,9);p[:,64]=1;rgb=torch.randint(0,256,(4,3,96,96),dtype=torch.uint8);valid=torch.ones(4,1,96,96,dtype=torch.bool);valid[2]=False
 target=torch.tensor([0,1,1,-1]);known_t=target>=0
 out=model(p,rgb,valid);decoded=op.decode_binary(out);assert decoded.tolist()==[1,1,-1,1]
 lo=op.known_binary_loss(out,target,known_t);assert int(lo['known_count'])==3 and int(lo['available_known_count'])==2 and int(lo['known_unavailable_count'])==1
 target2=target.clone();target2[3]=12345;assert torch.equal(lo['loss'],op.known_binary_loss(out,target2,known_t)['loss'])
 # Nonzero final layer ensures upstream encoders and attention are exercised.
 with torch.no_grad():model.readout[-1].weight.fill_(.031);model.readout[-1].bias.fill_(.07)
 model.zero_grad();nonzero=model(p,rgb,valid);op.known_binary_loss(nonzero,target,known_t)['loss'].backward()
 assert torch.isfinite(nonzero['query_logit']).all() and any(x.grad is not None and x.grad.abs().sum()>0 for x in model.p3_encoder.parameters())
 try:model(p,rgb,valid,labels=target)
 except TypeError:pass
 else:raise AssertionError('GT keyword rejected')
 cuda_result='NOT_RUN'
 if torch.cuda.is_available():
  torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  cm=op.QueryCenterScalar().cuda()
  with torch.no_grad():cm.readout[-1].weight.fill_(.031)
  co=cm(p.cuda(),rgb.cuda(),valid.cuda());cl=op.known_binary_loss(co,target.cuda(),known_t.cuda())['loss'];cl.backward();torch.cuda.synchronize()
  cuda_result='GENERATED_NONZERO_FORWARD_BACKWARD_DETERMINISTIC_PASS'
 t.write(HERE/'GENERATED_SOURCE_TRAINER_TESTS.json',dict(PASS=True,torch=torch.__version__,runtime_executable=__import__('sys').executable,source_pack_or_actual_GT_body_reads=0,actual_source_fit_updates=0,generated_sampler_draws=2000,allknown_selector_includes_U_safezero=True,rawsign_and_2x3_U_denoms=True,earliest_tie=True,model_param_count=26573,nonzero_upstream_backward=True,GT_forward_rejected=True,CUDA_generated_preflight=cuda_result,trainer=t.bind(HERE/'train_source_once.py'),operator=t.bind(t.MODEL)))
 print(json.dumps(t.read(HERE/'GENERATED_SOURCE_TRAINER_TESTS.json')))

if __name__=='__main__':main()
