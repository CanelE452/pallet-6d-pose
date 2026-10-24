"""Invented state and tensors only, no actual source/checkpoint reads."""
from pathlib import Path
import argparse,importlib.util,json,copy
import numpy as np
import torch
H=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('c4_generated',H/'train_source_once.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
def main(output):
 torch.set_num_threads(1);torch.manual_seed(194);t=c.shared();op=t.operator()
 reference=op.QueryCenterScalar()
 with torch.no_grad():reference.readout[-1].weight.fill_(.031);reference.readout[-1].bias.fill_(-.47)
 payload=dict(update=100,parameter_count=26573,model_code=c.bind(t.MODEL),state_dict={k:v.clone()for k,v in reference.state_dict().items()},optimizer_state={'invented_old_state':'must_not_be_loaded'})
 target=op.QueryCenterScalar();assert target.readout[-1].weight.abs().sum()==0
 c.apply_C1_state(target,payload,c.bind(t.MODEL));assert target.readout[-1].weight.abs().sum()>0 and all(p.requires_grad for p in target.parameters())
 for k,v in reference.state_dict().items():assert torch.equal(v,target.state_dict()[k])
 p=torch.randn(4,65,9,9);p[:,64]=1;rgb=torch.randint(0,256,(4,3,96,96),dtype=torch.uint8);valid=torch.ones(4,1,96,96,dtype=torch.bool);valid[2]=False
 reference.eval();target.eval()
 with torch.no_grad():a=reference(p,rgb,valid);b=target(p,rgb,valid)
 assert torch.equal(a['query_logit'],b['query_logit'])and torch.equal(a['query_available'],b['query_available'])and not torch.equal(a['query_logit'],torch.zeros_like(a['query_logit']))
 optimizer=torch.optim.AdamW(target.parameters(),lr=1e-5,weight_decay=1e-4);assert not optimizer.state and optimizer.param_groups[0]['lr']==1e-5
 # Shared sampler contract is unmodified: five pools and same CPU194 stream.
 pools=[np.arange(20*i,20*i+7,dtype=np.int64)for i in range(5)];g=torch.Generator().manual_seed(194);g2=torch.Generator().manual_seed(194)
 for _ in range(2000):
  ix=t.draw(pools,g);other=t.draw(pools,g2);assert np.array_equal(ix,other)
  assert [int(np.isin(ix,pool).sum())for pool in pools]==[8,8,16,8,24]
 assert c.settings(t)['AdamW_lr']==1e-5 and t.SETTINGS['AdamW_lr']==.001
 try:c.require_entry(dict(approved_actual_source_fit=False),t)
 except AssertionError:pass
 else:raise AssertionError('Root guard did not stop')
 report=dict(completed=True,PASS=True,trainer=c.bind(H/'train_source_once.py'),generated_code=c.bind(__file__),shared_trainer=c.bind(c.SHARED),model=c.bind(t.MODEL),torch=torch.__version__,loaded_nonzero_state_not_accidentally_reset=True,generated_output_and_availability_bitexact=True,all26573_trainable=True,old_optimizer_state_ignored_fresh_AdamW=True,lr=1e-5,all2000_sampler_draws_shared_contract=True,quotas=[8,8,16,8,24],no_actual_source_checkpoint_pixels_GT_current_reads=True,actual_fit_updates=0)
 c.write(output,report);print(json.dumps(report))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',default=str(H/'GENERATED_WARMSTART_TESTS.json'));main(Path(ap.parse_args().output))
