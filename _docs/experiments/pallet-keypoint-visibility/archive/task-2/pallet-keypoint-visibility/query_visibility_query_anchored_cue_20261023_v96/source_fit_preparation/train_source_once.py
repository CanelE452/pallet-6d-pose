"""C3 source-only: unchanged scalar, query-anchored data, sparse input units.
Import is inert: no source arrays, labels, checkpoint or current data are read.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import argparse,hashlib,importlib.util,json,random,time,traceback
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
DESIGN=ROOT/'C3_DESIGN_BEFORE_SOURCE_FIT.json'
DESIGN_SHA='ac1cc040f3bb1a8caaa742e8c12015785d27f6484ef2845b54e1168c52a0c215'
MODEL=ROOT.parent/'query_visibility_matched_query_scalar_20261023_v94/operator_preparation/query_center_scalar.py'
MODEL_SHA='d62e1610ca45f24e3cbf90e66dff3a7d954f87713afea4f5ae7bdc404411b0cb'
PROTOCOL=HERE/'FIT_PROTOCOL_BEFORE_SOURCE_ARRAYS.json';RUN=ROOT/'runs/C3_QUERY_ANCHORED_SOURCE_V1'
VARIANTS=('BEFORE','OPAQUE_COVER','CUE_PRESERVING_SHAM');FIELDS=('p3_roi','rgb96','crop_valid96')
COHORTS=('LOCO_BEFORE_D','PALLOX_BEFORE_D','SHAM_D','BEFORE_E','COVER_E')
QUOTAS=(8,8,16,8,24);NBASE=960;K=16;STRIDE=33;N=31680
SETTINGS=dict(seed=194,updates=2000,batch=64,cohorts=list(COHORTS),quotas=list(QUOTAS),AdamW_lr=.001,AdamW_weight_decay=.0001,clip_norm=5.,validation_every=100,validation_update0_eligible=True,selector='earliest strict minimum equal mean BCE of five VAL cohorts',threshold_raw_logit=0.,online_augmentation=False,unavailable_VAL_policy='safe zero logit BCE ln2 included; prediction U',unique_input_prediction_batch=128)
BINDINGS=('trainer','protocol','model_code','candidate_definition','input_pack_manifest','input_pack_seal','base_input_lineage','GT_ledger','generated_tests','independent_preparation_review','independent_input_audit')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def verify(b):assert sha(b['path'])==b['sha256'],b['path']
def same_binding(a,b):return Path(a['path']).resolve()==Path(b['path']).resolve()and a['sha256']==b['sha256']
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x)for x in Path(p).read_text().splitlines()if x.strip()]
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def operator():
 assert sha(MODEL)==MODEL_SHA
 s=importlib.util.spec_from_file_location('c3_unchanged_scalar',MODEL);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def canonical_split(source,raw):
 mapping={('LOCO','TRAIN'):'TRAIN',('LOCO','VALIDATION'):'VALIDATION',('PALLOX','TRAIN_LAB'):'TRAIN',('PALLOX','VAL_HALL'):'VALIDATION'}
 assert (source,raw)in mapping,'Unfrozen source/raw split pair'
 return mapping[source,raw]

def require_entry(e):
 assert e.get('approved_actual_source_fit')is True,'Root source-fit entry required before source reads'
 assert e.get('source_only')is True and e.get('current_allowed')is False and e.get('PnP_allowed')is False
 assert e.get('allow_restart_after_optimizer_update')is False
 for k in BINDINGS:verify(e[k])
 assert same_binding(e['trainer'],bind(__file__))and same_binding(e['protocol'],bind(PROTOCOL))
 assert same_binding(e['model_code'],bind(MODEL))and same_binding(e['candidate_definition'],bind(DESIGN))
 assert e['model_code']['sha256']==MODEL_SHA and e['candidate_definition']['sha256']==DESIGN_SHA
 p=read(PROTOCOL);assert p['settings']==SETTINGS and p['source_only']is True and p['current_allowed']is False
 for k,b in p['bindings'].items():assert k in e and same_binding(e[k],b),k
 for k in ('generated_tests','independent_preparation_review','independent_input_audit'):assert read(e[k]['path'])['PASS']is True,k
 assert same_binding(read(e['generated_tests']['path'])['trainer'],e['trainer'])
 assert same_binding(read(e['independent_preparation_review']['path'])['trainer'],e['trainer'])
 assert Path(e['run_directory']).resolve()==RUN.resolve()and not RUN.exists()
 return p

def join_metadata(lineage,ledger):
 assert len(lineage)==len(ledger)==N
 y=[];ready=np.zeros((NBASE,K),bool)
 for i,(r,g)in enumerate(zip(lineage,ledger)):
  j=i%STRIDE;rank=None if j==0 else(j-1)//2;variant='BEFORE'if j==0 else VARIANTS[1]if j%2 else VARIANTS[2]
  assert r['occurrence_index']==g['occurrence_index']==i
  assert r['slot_index']==g['slot_index']==g['base_row_index']==i//STRIDE
  assert r['donor_rank']==g['donor_rank']==rank and r['variant']==g['variant']==variant
  assert r['occurrence_id']==g['occurrence_id'];canonical_split(r['source'],r['split'])
  assert g['native_expected_part_state']=='UNKNOWN'and g['human_gold']is False
  label=g['candidate_aux_hidden_label_GT_only'];assert label in(-1,0,1)and g['known_candidate_GT_only']==(label>=0)
  y.append(label)
  if j>0:
   if j%2:ready[i//STRIDE,rank]=bool(g['pair_ready'])
   else:assert ready[i//STRIDE,rank]==bool(g['pair_ready'])
 y=np.asarray(y,np.int8)
 for i in range(NBASE):
  start=i*STRIDE;rr=lineage[start:start+STRIDE];gg=ledger[start:start+STRIDE];original=gg[0]['query_anchored_before_label']
  assert original in(-1,0,1)and y[start]==original
  assert len({r['source']for r in rr})==len({r['split']for r in rr})==1
  assert all(r['query_xy']==rr[0]['query_xy']and r['L']==rr[0]['L']for r in rr)
  assert all(g['query_anchored_before_label']==original and g['original_C1_V95_label']==gg[0]['original_C1_V95_label']for g in gg)
  for k in range(K):
   labels=y[start+1+2*k:start+3+2*k].tolist();assert labels==([1,0]if ready[i,k]else[-1,-1])
   if ready[i,k]:assert original==0
  assert gg[0]['pair_ready']is False,'BEFORE is not a rank; pair status lives on rank after rows'
 assert [int(np.sum(y[::STRIDE]==k))for k in(0,1,-1)]==[61,4,895]
 source=np.array([r['source']for r in lineage]);raw=np.array([r['split']for r in lineage]);split=np.array([canonical_split(r['source'],r['split'])for r in lineage]);variant=np.array([r['variant']for r in lineage])
 meta=dict(labels=y,known=y>=0,source=source,original_split=raw,split=split,variant=variant,pair_ready=ready)
 meta['counts']={s:dict(rows=int(np.sum(split==s)),before_D=int(np.sum((split==s)&(variant=='BEFORE')&(y==0))),before_E=int(np.sum((split==s)&(variant=='BEFORE')&(y==1))),before_U=int(np.sum((split==s)&(variant=='BEFORE')&(y<0))),accepted_pairs=int(ready[split[::STRIDE]==s].sum()),known_D=int(np.sum((split==s)&(y==0))),known_E=int(np.sum((split==s)&(y==1))),unknown=int(np.sum((split==s)&(y<0))))for s in('TRAIN','VALIDATION')}
 return meta

def cohort_masks(meta,split):
 m=meta['split']==split;y=meta['labels'];v=meta['variant'];s=meta['source']
 return dict(LOCO_BEFORE_D=m&(s=='LOCO')&(v=='BEFORE')&(y==0),PALLOX_BEFORE_D=m&(s=='PALLOX')&(v=='BEFORE')&(y==0),SHAM_D=m&(v==VARIANTS[2])&(y==0),BEFORE_E=m&(v=='BEFORE')&(y==1),COVER_E=m&(v==VARIANTS[1])&(y==1))
def feasibility(meta):
 return {s:{name:int(mask.sum())for name,mask in cohort_masks(meta,s).items()}for s in('TRAIN','VALIDATION')}
def make_pools(meta):
 allcounts=feasibility(meta);assert all(v>0 for c in allcounts.values()for v in c.values()),'Empty required TRAIN/VAL cohort; no fit'
 return [np.flatnonzero(x)for x in cohort_masks(meta,'TRAIN').values()]
def draw(pools,g):
 a=torch.cat([torch.as_tensor(p)[torch.randint(len(p),(q,),generator=g)]for p,q in zip(pools,QUOTAS)])
 return a[torch.randperm(64,generator=g)].numpy()
def selector(z,available,meta):
 zz=np.where(available,z,0.).astype(np.float64);loss=np.logaddexp(0.,zz)-meta['labels']*zz;strata={}
 for name,mask in cohort_masks(meta,'VALIDATION').items():
  assert mask.any(),'Empty prospective VAL cohort '+name
  strata[name]=dict(count=int(mask.sum()),mean_BCE=float(loss[mask].mean()),unavailable=int((mask&~available).sum()))
 return float(np.mean([x['mean_BCE']for x in strata.values()])),strata

def load_metadata(e,p):
 seal=read(e['input_pack_seal']['path']);assert seal['completed']and seal['PASS']and same_binding(seal['manifest'],e['input_pack_manifest'])
 verify(seal['QA']);assert read(seal['QA']['path'])['PASS']is True
 manifest=read(e['input_pack_manifest']['path']);assert manifest['completed']and manifest['base_rows']==960 and manifest['occurrences']==N
 verify(manifest['entry']);assert same_binding(read(manifest['entry']['path'])['gt_only_occurrences'],e['GT_ledger'])
 assert same_binding(manifest['base_input_lineage'],e['base_input_lineage'])
 verify(manifest['input_lineage']);lineage=rows(manifest['input_lineage']['path']);ledger=rows(e['GT_ledger']['path']);meta=join_metadata(lineage,ledger)
 assert meta['counts']==p['expected_counts'],'Frozen counts differ, before any optimizer update'
 return manifest,lineage,meta

def reference_key(r,base_binding):
 ref=r['input_ref']
 if ref['kind']=='BASE_ROW':
  assert not r['full_rgb_changed']and ref['slot_index']==r['slot_index']and same_binding(ref['base_lineage'],base_binding)
  return ('BASE_ROW',ref['slot_index'])
 assert ref['kind']=='CHANGED_CHUNK'and r['full_rgb_changed']is True
 return ('CHANGED_CHUNK',str(Path(ref['file']['path']).resolve()),ref['file']['sha256'],ref['index'])

def materialize_unique_inputs(base,lineage,base_binding):
 """All occurrences retained; only repeated sparse references share tensors."""
 occurrence_unit=np.array([r['input_unit_index']for r in lineage],np.int64);assert len(occurrence_unit)==len(lineage)and np.all(occurrence_unit>=0)
 keys={};unit_rows={};ref_to_unit={}
 for r,u in zip(lineage,occurrence_unit):
  key=reference_key(r,base_binding)
  if u in keys:assert keys[u]==key and r['tensor_value_sha256']==unit_rows[u]['tensor_value_sha256']
  else:
   assert key not in ref_to_unit,'A repeated sparse reference must use the same input unit'
   keys[u]=key;unit_rows[u]=r;ref_to_unit[key]=u
 assert set(keys)==set(range(len(keys)))
 arrays={k:[]for k in FIELDS};cache={}
 def get(file,key,index):
  path=file['path']
  if path not in cache:verify(file);cache[path]=dict(sha256=file['sha256'],arrays={})
  assert cache[path]['sha256']==file['sha256']
  if key not in cache[path]['arrays']:
   with np.load(path,allow_pickle=False)as f:cache[path]['arrays'][key]=np.array(f[key],copy=True)
  return np.array(cache[path]['arrays'][key][index],copy=True)
 for u in range(len(keys)):
  r=unit_rows[u];ref=r['input_ref']
  refs=base[r['slot_index']]['before_input_refs']if ref['kind']=='BASE_ROW'else{k:dict(file=ref['file'],key=k,index=ref['index'])for k in FIELDS}
  x={k:get(refs[k]['file'],refs[k]['key'],refs[k]['index'])for k in FIELDS}
  assert set(r['tensor_value_sha256'])==set(FIELDS)
  for k in FIELDS:
   assert hashlib.sha256(np.ascontiguousarray(x[k]).tobytes()).hexdigest()==r['tensor_value_sha256'][k]
   arrays[k].append(x[k])
 return {k:np.ascontiguousarray(np.stack(v))for k,v in arrays.items()},occurrence_unit

def load_inputs(e,lineage):
 base=rows(e['base_input_lineage']['path']);assert len(base)==960
 data,unit=materialize_unique_inputs(base,lineage,e['base_input_lineage']);n=len(data['p3_roi'])
 assert data['p3_roi'].shape==(n,65,9,9)and data['p3_roi'].dtype==np.float16 and np.isfinite(data['p3_roi']).all()
 assert data['rgb96'].shape==(n,3,96,96)and data['rgb96'].dtype==np.uint8
 assert data['crop_valid96'].shape==(n,1,96,96)and np.isin(data['crop_valid96'],[0,1]).all()and np.isin(data['p3_roi'][:,64],[0,1]).all()
 avail=(data['p3_roi'][:,64,4,4]>0)&data['crop_valid96'][:,0,47:49,47:49].all((1,2))
 assert np.array_equal(avail[unit],[r['input_facts']['operator_query_available']for r in lineage])
 return data,unit,avail[unit]

def decode(z,a):return np.where(a,(z>=0).astype(np.int8),-1)
def cm_metrics(z,a,m,mask):
 p=decode(z,a);y=m['labels'];known=m['known'];cm=np.array([[np.sum(mask&known&(y==k)&(p==j))for j in(0,1,-1)]for k in(0,1)],np.int64)
 ratio=lambda x,n:float(x/n)if n else None
 return dict(matrix=cm.tolist(),rows=['conditional_AUX_D','conditional_AUX_hidden_E'],columns=['D','E','U'],D_recall=ratio(cm[0,0],cm[0].sum()),E_recall=ratio(cm[1,1],cm[1].sum()),D_precision=ratio(cm[0,0],cm[:,0].sum()),E_precision=ratio(cm[1,1],cm[:,1].sum()),known_U_rate=ratio(cm[:,2].sum(),cm.sum()),all_N=int(mask.sum()),known_N=int(cm.sum()),unknown_GT_N=int(np.sum(mask&~known)),unknown_GT_predictions={str(k):int(np.sum(mask&~known&(p==k)))for k in(0,1,-1)})
def pair_metrics(pred,meta,split,source=None):
 pp=pred.reshape(NBASE,STRIDE);base=(meta['split'][::STRIDE]==split)
 if source is not None:base&=meta['source'][::STRIDE]==source
 mask=meta['pair_ready']&base[:,None];good=(pp[:,0,None]==0)&(pp[:,1::2]==1)&(pp[:,2::2]==0)
 n=int(mask.sum());byrank={str(k):dict(N=int(mask[:,k].sum()),all3_correct=int((good[:,k]&mask[:,k]).sum()))for k in range(K)}
 return dict(accepted_pairs=n,all3_correct=int((mask&good).sum()),all3_correct_rate=float((mask&good).sum()/n)if n else None,rank_counts=byrank,controlled_tested=n>0)
def report(z,a,meta):
 pred=decode(z,a);result={}
 for s in('TRAIN','VALIDATION'):
  m=meta['split']==s;result[s]=dict(all_known=cm_metrics(z,a,meta,m),by_variant={v:cm_metrics(z,a,meta,m&(meta['variant']==v))for v in VARIANTS},five_cohorts={name:cm_metrics(z,a,meta,mask)for name,mask in cohort_masks(meta,s).items()},paired=pair_metrics(pred,meta,s),paired_by_source={source:pair_metrics(pred,meta,s,source)for source in('LOCO','PALLOX')})
 v=result['VALIDATION'];c=v['five_cohorts']
 checks={name:dict(value=c[cohort][metric],minimum=limit)for name,cohort,metric,limit in [('LOCO_before_D_recall','LOCO_BEFORE_D','D_recall',.85),('PALLOX_before_D_recall','PALLOX_BEFORE_D','D_recall',.85),('natural_before_E_recall','BEFORE_E','E_recall',.75),('sham_D_recall','SHAM_D','D_recall',.85),('cover_E_recall','COVER_E','E_recall',.75)]}
 checks['accepted_VAL_pair_all3_correct_rate']=dict(value=v['paired']['all3_correct_rate'],minimum=.75)
 checks['known_VAL_U_rate']=dict(value=v['all_known']['known_U_rate'],maximum=.05)
 for c in checks.values():c['PASS']=c['value']is not None and(c['value']>=c['minimum']if'minimum'in c else c['value']<=c['maximum'])
 return result,checks,all(x['PASS']for x in checks.values())

def forward(model,data,unit_indices,device):return model(*(torch.as_tensor(np.ascontiguousarray(data[k][unit_indices]),device=device)for k in FIELDS))
@torch.no_grad()
def predict(model,data,occurrence_unit,device):
 model.eval();zs=[];avs=[]
 for start in range(0,len(data['p3_roi']),128):
  out=forward(model,data,np.arange(start,min(start+128,len(data['p3_roi']))),device);zs.append(out['query_logit'].float().cpu().numpy());avs.append(out['query_available'].cpu().numpy())
 return np.concatenate(zs)[occurrence_unit],np.concatenate(avs)[occurrence_unit]
def checkpoint(model,opt,update,e):return dict(state_dict={k:v.detach().cpu().clone()for k,v in model.state_dict().items()},optimizer_state=opt.state_dict(),update=update,seed=194,parameter_count=26573,model_code=e['model_code'],protocol=e['protocol'],source_input_bindings=e)

def run(entry):
 e=read(entry);p=require_entry(e);assert torch.__version__=='2.1.1+cu118'and torch.cuda.is_available()
 RUN.mkdir(parents=True);write(RUN/'RUN_STARTED.json',dict(entry=bind(entry),trainer=bind(__file__),optimizer_updates=0,restart_forbidden_after_any_update=True));updates=0
 try:
  manifest,lineage,meta=load_metadata(e,p);cohort_counts=feasibility(meta)
  if any(x==0 for counts in cohort_counts.values()for x in counts.values()):
   write(RUN/'NO_FIT_FEASIBILITY_ONLY.json',dict(completed=True,fit_executed=False,reason='Required TRAIN or VAL cohort empty',cohort_counts=cohort_counts,counts=meta['counts'],current_allowed=False));return
  pools=make_pools(meta);data,unit,factual=load_inputs(e,lineage)
  write(RUN/'INPUT_PREFLIGHT.json',dict(completed=True,PASS=True,occurrences=N,unique_tensor_units=len(data['p3_roi']),occurrence_to_unit_sha256=hashlib.sha256(unit.tobytes()).hexdigest(),known_unavailable_by_split={s:int(np.sum(meta['known']&(meta['split']==s)&~factual))for s in('TRAIN','VALIDATION')},cohort_counts=cohort_counts,unique_unit_predict_batch=128,rows_not_deduplicated_for_loss_or_metrics=True))
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  random.seed(194);np.random.seed(194);torch.manual_seed(194);torch.cuda.manual_seed_all(194)
  op=operator();device='cuda:0';model=op.QueryCenterScalar().to(device);assert sum(x.numel()for x in model.parameters())==26573
  opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);g=torch.Generator(device='cpu').manual_seed(194)
  torch.save(checkpoint(model,opt,0,e),RUN/'INITIAL_CHECKPOINT.pt');best=float('inf');best_update=None;best_state=None;started=time.monotonic()
  def evaluate(update):
   nonlocal best,best_update,best_state
   z,a=predict(model,data,unit,device);assert np.array_equal(a,factual);value,cohorts=selector(z,a,meta)
   with(RUN/'HISTORY.jsonl').open('a')as f:f.write(json.dumps(dict(update=update,criterion_five_VAL_cohorts_BCE=value,VAL_cohorts=cohorts),allow_nan=False)+'\n')
   if value<best:
    best=value;best_update=update;best_state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()};torch.save(checkpoint(model,opt,update,e),RUN/'BEST_QUERY_CENTER_SCALAR.pt')
  evaluate(0)
  for update in range(1,2001):
   ix=draw(pools,g);model.train();opt.zero_grad(set_to_none=True);out=forward(model,data,unit[ix],device)
   loss=op.known_binary_loss(out,torch.as_tensor(meta['labels'][ix],device=device),torch.as_tensor(meta['known'][ix],device=device));assert torch.isfinite(loss['loss'])
   loss['loss'].backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(grad);opt.step();updates=update
   quota={name:int(mask[ix].sum())for name,mask in cohort_masks(meta,'TRAIN').items()};assert list(quota.values())==list(QUOTAS)and meta['known'][ix].all()
   with(RUN/'TRAIN_UPDATES.jsonl').open('a')as f:f.write(json.dumps(dict(update=update,occurrence_indices=ix.tolist(),input_unit_indices=unit[ix].tolist(),cohort_quotas=quota,loss=float(loss['loss'].detach()),available_known_count=int(loss['available_known_count']),known_unavailable_count=int(loss['known_unavailable_count']),gradient_norm_before_clip=float(grad)),allow_nan=False)+'\n')
   if update%100==0:evaluate(update)
  torch.save(checkpoint(model,opt,2000,e),RUN/'FINAL_CHECKPOINT.pt');model.load_state_dict(best_state);z,a=predict(model,data,unit,device);assert np.array_equal(a,factual)
  value,cohorts=selector(z,a,meta);assert value==best;results,checks,passed=report(z,a,meta);label=meta['labels'];nll=(np.logaddexp(np.float32(0),z)-np.where(label>=0,label,0)*z).astype(np.float32)
  np.savez_compressed(RUN/'SELECTED_ALL31680_SOURCE_SCORES.npz',query_logit=z.astype(np.float32),query_available=a,prediction=decode(z,a),NLL_FP32=nll,candidate_known=meta['known'],scalar_target_D0_E1_Uminus1=label,source=meta['source'],split=meta['split'],original_split=meta['original_split'],variant=meta['variant'],occurrence_index=np.arange(N),occurrence_id=np.array([r['occurrence_id']for r in lineage]),base_row_index=np.arange(N)//STRIDE,donor_rank=np.array([-1 if r['donor_rank']is None else r['donor_rank']for r in lineage],np.int8),input_unit_index=unit,input_id=np.array([r['input_id']for r in lineage]),native_slot_logical_only=np.array([r['native_slot_logical_only']for r in lineage]),image_sha256=np.array([r['image_sha256']for r in lineage]),original_image_sha256=np.array([r['original_image_sha256']for r in lineage]),query_xy_raw=np.array([r['query_xy']if r['query_xy']is not None else[np.nan,np.nan]for r in lineage],np.float64),L_raw=np.array([r['L']if r['L']is not None else np.nan for r in lineage],np.float64),accepted_pair=meta['pair_ready'])
  write(RUN/'SOURCE_GATE_RESULTS.json',dict(completed=True,PASS=passed,all_gates_pass=passed,gates=checks,current_inference_authorized=False))
  write(RUN/'SOURCE_RESULTS.json',dict(completed=True,source_only=True,selected_update=best_update,selector_five_VAL_cohorts_BCE=value,VAL_cohorts=cohorts,counts=meta['counts'],metrics=results,all31680_preserved=True,input_units=len(data['p3_roi']),native_expected_part_state='UNKNOWN',human_gold=False,NLL_unknown_diagnostic_only=True))
  write(RUN/'MODEL_FREEZE.json',dict(completed=True,source_only=True,model_code=e['model_code'],checkpoint=bind(RUN/'BEST_QUERY_CENTER_SCALAR.pt'),initial_checkpoint=bind(RUN/'INITIAL_CHECKPOINT.pt'),final_checkpoint=bind(RUN/'FINAL_CHECKPOINT.pt'),parameter_count=26573,selected_update=best_update,updates=2000,selector=SETTINGS['selector'],threshold_rawlogit=0,protocol=e['protocol'],entry=bind(entry),source_input_bindings=e,current_inference_authorized=False))
  write(RUN/'FIT_COMPLETE.json',dict(completed=True,PASS_execution=True,optimizer_updates=2000,selected_update=best_update,seconds=time.monotonic()-started,MODEL_FREEZE=bind(RUN/'MODEL_FREEZE.json'),source_results=bind(RUN/'SOURCE_RESULTS.json'),source_gates=bind(RUN/'SOURCE_GATE_RESULTS.json'),scores=bind(RUN/'SELECTED_ALL31680_SOURCE_SCORES.npz'),history=bind(RUN/'HISTORY.jsonl'),train_updates=bind(RUN/'TRAIN_UPDATES.jsonl'),new_current_PnP=0))
  print(json.dumps(read(RUN/'FIT_COMPLETE.json')),flush=True)
 except BaseException:
  write(RUN/'FAILED_EXECUTION.json',dict(completed=False,optimizer_updates_completed=updates,exception=traceback.format_exc(),restart_automatically=False));raise
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--entry',type=Path,required=True);run(parser.parse_args().entry)
