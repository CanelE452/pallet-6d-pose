"""V94 C1 one source-only fit. Import has no source data/weight reads."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import importlib.util
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
MODEL=ROOT/'operator_preparation/query_center_scalar.py'
PROTOCOL=HERE/'FIT_PROTOCOL_BEFORE_SOURCE_ARRAYS.json'
RUN=ROOT/'runs/C1_NATURAL_KNOWNSET_SOURCE_V1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):
 p=Path(p);return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);assert not p.exists(),str(p);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def verify(b):assert sha(b['path'])==b['sha256'],b['path']
def operator():
 spec=importlib.util.spec_from_file_location('v94_query_scalar_frozen',MODEL);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def require_entry(e):
 assert e.get('approved_actual_source_fit') is True,'Root source execution entry required before actual data reads'
 assert e.get('source_only') is True and e.get('current_allowed') is False and e.get('PnP_allowed') is False
 assert e.get('allow_restart_after_optimizer_update') is False
 for k in ['trainer','protocol','model_code','candidate_definition','label_admission','input_pack_seal','independent_input_audit','GT_ledger','generated_tests']:
  verify(e[k])
 assert e['trainer']['sha256']==sha(__file__) and e['protocol']['sha256']==sha(PROTOCOL)
 assert e['model_code']['sha256']==sha(MODEL)
 assert Path(e['run_directory']).resolve()==RUN.resolve() and not RUN.exists()

def make_pools(labels,known,splits):
 train=(splits=='TRAIN_LAB') & known
 pools=[np.flatnonzero(train & (labels==k)) for k in [0,1]]
 assert all(len(p)>0 for p in pools)
 return pools
def draw(pools,g):
 parts=[torch.as_tensor(p,dtype=torch.long)[torch.randint(len(p),(32,),generator=g)] for p in pools]
 z=torch.cat(parts);return z[torch.randperm(64,generator=g)].numpy()
def macro_BCE(logits,labels,known,splits):
 # Literal selector includes unavailable-known safezero logits as fixed BCE.
 losses=np.logaddexp(0.,logits.astype(np.float64))-labels*logits.astype(np.float64)
 means=[];counts=[]
 for k in [0,1]:
  m=known & (splits=='VAL_HALL') & (labels==k);assert m.any()
  means.append(float(losses[m].mean()));counts.append(int(m.sum()))
 return float(sum(means)/2),means,counts
def cm_metrics(logits,available,labels,known,mask):
 pred=np.where(available,(logits>=0).astype(np.int8),-1)
 cm=np.zeros((2,3),np.int64)
 for k in [0,1]:
  for j,v in enumerate([0,1,-1]):cm[k,j]=np.sum(mask & known & (labels==k) & (pred==v))
 n=int(cm.sum());d=int(cm[0].sum());e=int(cm[1].sum())
 return dict(rows=['D_CONDITIONAL','E_CONDITIONAL'],columns=['D','E','U'],matrix=cm.tolist(),known_N=n,D_recall=float(cm[0,0]/d) if d else None,E_recall=float(cm[1,1]/e) if e else None,known_U_rate=float(cm[:,2].sum()/n) if n else None,known_U_count=int(cm[:,2].sum()),all_rows_N=int(mask.sum()),unknown_GT_N=int(np.sum(mask & ~known)),unknown_GT_pred_distribution={str(v):int(np.sum(mask & ~known & (pred==v))) for v in [0,1,-1]})
def gates(metrics):
 checks={'VAL_D_recall':dict(value=metrics['D_recall'],minimum=.85),'VAL_E_recall':dict(value=metrics['E_recall'],minimum=.75),'VAL_known_U_rate':dict(value=metrics['known_U_rate'],maximum=.05)}
 for d in checks.values():d['PASS']=d['value'] is not None and (d['value']>=d['minimum'] if 'minimum'in d else d['value']<=d['maximum'])
 return checks,all(x['PASS'] for x in checks.values())

def load_actual(e):
 # Reached only after approved entry verifies every metadata binding above.
 seal=read(e['input_pack_seal']['path']);assert seal['completed'] and seal['PASS']
 mb=seal['manifest'];verify(mb);manifest=read(mb['path']);assert manifest['completed'] and manifest['frame_count']==100 and manifest['query_count']==800
 ledger=[json.loads(x) for x in Path(e['GT_ledger']['path']).read_text().splitlines() if x.strip()];assert len(ledger)==800
 chunks=[];position=0
 for c in manifest['chunks']:
  assert c['start']==position and c['end']-c['start']==c['count'];verify(c['file'])
  a=np.load(c['file']['path'],allow_pickle=False);chunks.append({k:np.ascontiguousarray(a[k]) for k in a.files});position=c['end']
 assert position==800
 keys=['p3_roi','rgb96','crop_valid96','query_id','frame_index','frame_input_id','native_slot','split','image_sha256','query_xy_raw','L_raw','query_present']
 data={k:np.concatenate([x[k] for x in chunks]) for k in keys}
 assert data['p3_roi'].shape==(800,65,9,9) and data['p3_roi'].dtype==np.float16 and np.isfinite(data['p3_roi']).all()
 assert data['rgb96'].shape==(800,3,96,96) and data['rgb96'].dtype==np.uint8
 assert data['crop_valid96'].shape==(800,1,96,96) and np.isin(data['crop_valid96'],[0,1]).all()
 for i,r in enumerate(ledger):
  for k in ['query_id','frame_index','frame_input_id','native_slot','split','image_sha256']:assert data[k][i]==r[k]
  if r['original_q'] is None:assert not data['query_present'][i] and np.isnan(data['query_xy_raw'][i]).all() and np.isnan(data['L_raw'][i])
  else:assert data['query_present'][i] and np.array_equal(data['query_xy_raw'][i],r['original_q']) and data['L_raw'][i]==r['original_L']
  assert r['native_expected_part_state']=='UNKNOWN' and r['human_gold'] is False
 labels=np.array([r['scalar_target_D0_E1_Uminus1'] for r in ledger],np.int8)
 known=np.array([r['candidate_known'] for r in ledger],bool);splits=data['split']
 assert np.array_equal(known,np.isin(labels,[0,1]))
 for s,n in [('TRAIN_LAB',(56,9)),('VAL_HALL',(81,11))]:assert tuple(int(np.sum(known & (labels==k) & (splits==s))) for k in [0,1])==n
 return data,labels,known,splits,manifest

def forward(model,data,indices,device):
 ix=np.asarray(indices)
 return model(torch.as_tensor(np.ascontiguousarray(data['p3_roi'][ix]),device=device),torch.as_tensor(np.ascontiguousarray(data['rgb96'][ix]),device=device),torch.as_tensor(np.ascontiguousarray(data['crop_valid96'][ix]),device=device))
@torch.no_grad()
def predict(model,data,device):
 model.eval();zs=[];avs=[]
 for start in range(0,800,128):
  out=forward(model,data,np.arange(start,min(start+128,800)),device)
  zs.append(out['query_logit'].float().cpu().numpy());avs.append(out['query_available'].cpu().numpy())
 return np.concatenate(zs),np.concatenate(avs)

def checkpoint(model,opt,update,e):
 return dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},optimizer_state=opt.state_dict(),update=update,seed=194,model_code=e['model_code'],parameter_count=26573,protocol=e['protocol'],source_input_bindings=e)

def run(entry):
 e=read(entry);require_entry(e)
 assert torch.__version__=='2.1.1+cu118' and torch.cuda.is_available()
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 random.seed(194);np.random.seed(194);torch.manual_seed(194);torch.cuda.manual_seed_all(194)
 device='cuda:0';op=operator();data,labels,known,splits,manifest=load_actual(e)
 model=op.QueryCenterScalar().to(device);assert sum(p.numel() for p in model.parameters())==26573
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
 pools=make_pools(labels,known,splits);g=torch.Generator(device='cpu').manual_seed(194)
 RUN.mkdir(parents=True);write(RUN/'RUN_STARTED.json',dict(entry=bind(entry),trainer=bind(__file__),runtime=dict(torch=torch.__version__,executable=os.sys.executable,device=device,threads=torch.get_num_threads()),initial_updates=0,restart_forbidden_after_any_update=True))
 torch.save(checkpoint(model,opt,0,e),RUN/'INITIAL_CHECKPOINT.pt')
 history=[];best=float('inf');best_update=None;start=time.monotonic()
 def evaluate(update):
  nonlocal best,best_update
  z,a=predict(model,data,device);criterion,means,counts=macro_BCE(z,labels,known,splits)
  val=cm_metrics(z,a,labels,known,splits=='VAL_HALL')
  record=dict(update=update,criterion_macro_BCE_all_VAL_candidateknown=criterion,VAL_D_BCE_all_known=means[0],VAL_E_BCE_all_known=means[1],VAL_D_N=counts[0],VAL_E_N=counts[1],VAL_known_unavailable=val['known_U_count'],VAL_metrics=val)
  history.append(record)
  with (RUN/'HISTORY.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
  if criterion<best:
   best=criterion;best_update=update;torch.save(checkpoint(model,opt,update,e),RUN/'BEST_QUERY_CENTER_SCALAR.pt')
 evaluate(0)
 for update in range(1,2001):
  ix=draw(pools,g);model.train();opt.zero_grad(set_to_none=True)
  output=forward(model,data,ix,device)
  target=torch.as_tensor(labels[ix],device=device);kn=torch.as_tensor(known[ix],device=device)
  lo=op.known_binary_loss(output,target,kn);assert torch.isfinite(lo['loss'])
  lo['loss'].backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(grad)
  opt.step()
  record=dict(update=update,query_indices=ix.tolist(),batch_D=int(np.sum(labels[ix]==0)),batch_E=int(np.sum(labels[ix]==1)),candidate_known_count=int(lo['known_count']),loss_available_known_count=int(lo['available_known_count']),known_unavailable_count=int(lo['known_unavailable_count']),loss=float(lo['loss'].detach()),gradient_norm_before_clip=float(grad))
  assert record['batch_D']==record['batch_E']==32 and np.all(splits[ix]=='TRAIN_LAB')
  with (RUN/'TRAIN_UPDATES.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
  if update%100==0:evaluate(update)
 torch.save(checkpoint(model,opt,2000,e),RUN/'FINAL_CHECKPOINT.pt')
 selected=torch.load(RUN/'BEST_QUERY_CENTER_SCALAR.pt',map_location='cpu');model.load_state_dict(selected['state_dict']);model.eval()
 z,a=predict(model,data,device);criterion,means,counts=macro_BCE(z,labels,known,splits)
 assert criterion==best and selected['update']==best_update
 np.savez_compressed(RUN/'SELECTED_ALL800_SOURCE_SCORES.npz',query_logit=z.astype(np.float32),query_available=a,prediction=np.where(a,(z>=0).astype(np.int8),-1),candidate_known=known,scalar_target_D0_E1_Uminus1=labels,query_id=data['query_id'],frame_index=data['frame_index'],native_slot=data['native_slot'],split=splits,original_q_raw=data['query_xy_raw'],original_L_raw=data['L_raw'])
 metrics={s:cm_metrics(z,a,labels,known,splits==s) for s in ['TRAIN_LAB','VAL_HALL']};checks,passed=gates(metrics['VAL_HALL'])
 write(RUN/'SOURCE_GATE_RESULTS.json',dict(completed=True,PASS=passed,all_gates_pass=passed,gates=checks,threshold='rawlogit0; unavailableU',current_inference_authorized=False))
 write(RUN/'SOURCE_RESULTS.json',dict(completed=True,source_only=True,selected_update=best_update,selector_macro_BCE_all_VAL_known=criterion,VAL_D_BCE=means[0],VAL_E_BCE=means[1],metrics=metrics,all800_preserved=True,unknown_GT_is_not_negative=True,human_native_gold=False,controlled_pairs_included=False))
 write(RUN/'MODEL_FREEZE.json',dict(completed=True,source_only=True,model_code=e['model_code'],checkpoint=bind(RUN/'BEST_QUERY_CENTER_SCALAR.pt'),initial_checkpoint=bind(RUN/'INITIAL_CHECKPOINT.pt'),final_checkpoint=bind(RUN/'FINAL_CHECKPOINT.pt'),parameter_count=26573,selected_update=best_update,updates=2000,selector='earliest strict minimum allknown VAL class macroBCE including unavailable-safezero',threshold_rawlogit=0,protocol=e['protocol'],entry=bind(entry),source_input_bindings=e,source_gates=bind(RUN/'SOURCE_GATE_RESULTS.json'),source_results=bind(RUN/'SOURCE_RESULTS.json'),native_expected_part_state='UNKNOWN',human_gold=False))
 write(RUN/'FIT_COMPLETE.json',dict(completed=True,PASS_execution=True,optimizer_updates=2000,selected_update=best_update,seconds=time.monotonic()-start,MODEL_FREEZE=bind(RUN/'MODEL_FREEZE.json'),source_results=bind(RUN/'SOURCE_RESULTS.json'),source_gates=bind(RUN/'SOURCE_GATE_RESULTS.json'),scores=bind(RUN/'SELECTED_ALL800_SOURCE_SCORES.npz'),history=bind(RUN/'HISTORY.jsonl'),train_updates=bind(RUN/'TRAIN_UPDATES.jsonl'),new_current_PnP=0))
 print(json.dumps(read(RUN/'FIT_COMPLETE.json')))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--entry',type=Path,required=True);args=p.parse_args();run(args.entry)
