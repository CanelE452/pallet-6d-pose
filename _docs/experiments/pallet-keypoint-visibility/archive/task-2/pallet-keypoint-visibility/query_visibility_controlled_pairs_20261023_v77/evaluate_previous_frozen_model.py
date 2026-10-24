"""Source-only diagnostic V76 baseline on identical original/controlled inputs."""
from pathlib import Path
import hashlib,importlib.util,json,time
import numpy as np
import torch
R=Path(__file__).resolve().parent;V=R.parent/'query_visibility_reference_local_support_20261023_v76';OUT=R/'frozen_V76_source_baseline'
def b(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for q in iter(lambda:f.read(1<<20),b''):h.update(q)
 return dict(path=str(p),sha256=h.hexdigest())
def load(p):return json.loads(p.read_text())
def check(x):assert b(Path(x['path']))==x,x['path']
def write(p,x):assert not p.exists();p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 gate=load(R/'FROZEN_V76_BASELINE_ENTRY.json')
 for x in gate['bindings'].values():check(x)
 assert gate['bindings']['code']==b(Path(__file__)) and not OUT.exists();OUT.mkdir()
 spec=importlib.util.spec_from_file_location('v77_locked_source_helpers',R/'train_paired_once.py');t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t);p=t.decisions()
 torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 base,gpu,labels,meta=t.load_source(p,'cuda:0');M=t.imp(p['bindings']['unchanged_model_code'],'same_model_baseline');model=M.NativeContextSupportHead().cuda();model.load_state_dict(torch.load(gate['bindings']['checkpoint']['path'],map_location='cuda',weights_only=True));model.eval().requires_grad_(False);before={k:v.detach().cpu().clone()for k,v in model.state_dict().items()};start=time.monotonic()
 original=base.predict(model,gpu,np.arange(41624),labels);after=base.predict(model,gpu,np.arange(41624,meta['N']),labels)
 with np.load(gate['bindings']['old_original_scores']['path'],allow_pickle=False)as z:
  old=z['support_logit'].reshape(-1);assert np.array_equal(original['logits'],old),'Identicaloriginalsourcebaseline shouldreproduce exactly'
 pred={k:np.concatenate((original[k],after[k]))for k in original};assert np.array_equal(pred['input_available'],meta['input_available'])
 t.save_scores(OUT/'SOURCE_SCORES.npz',pred,meta);write(OUT/'SOURCE_RESULTS.json',dict(completed=True,model='V76_FIXED_SOURCE_SELECTED200',checkpoint=gate['bindings']['checkpoint'],input_entry=b(R/'COMPLETED_SOURCE_INPUT_BINDINGS.json'),original41624_logits_bitexact=True,results=t.report(base,pred,meta),model_forward_source_only=True,fit_updates=0,current_PnP=0))
 assert all(torch.equal(before[k],v.detach().cpu())for k,v in model.state_dict().items());check(gate['bindings']['checkpoint'])
 write(OUT/'BASELINE_SEAL.json',dict(completed=True,PASS=True,entry=b(R/'FROZEN_V76_BASELINE_ENTRY.json'),scores=b(OUT/'SOURCE_SCORES.npz'),results=b(OUT/'SOURCE_RESULTS.json'),original_logit_differences=0,weights_unchanged=True,fit_updates=0,current_reads=0,PnP=0,seconds=time.monotonic()-start));print(json.dumps(b(OUT/'BASELINE_SEAL.json')))
if __name__=='__main__':main()
