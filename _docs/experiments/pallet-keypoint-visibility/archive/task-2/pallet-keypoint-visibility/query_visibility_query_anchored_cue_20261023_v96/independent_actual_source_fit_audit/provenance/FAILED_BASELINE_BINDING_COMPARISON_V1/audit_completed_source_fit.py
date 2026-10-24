"""Independent saved-only C3/C1 audit. No trainer/model/optimizer import/forward.
Torch is used only for the CPU seeded randint/randperm sampler replay.
"""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,math
import numpy as np

H=Path(__file__).resolve().parent;V=H.parent
TRAINER=V/'source_fit_preparation/train_source_once.py';TRAINER_SHA='330f817e45fe8cbb3cda2c48bea5f2502b2f9053ce8e2921892fef44920ade03'
BASECODE=V/'frozen_C1_baseline_preparation/run_frozen_C1_baseline.py';BASECODE_SHA='77dcf08df245038bb6c4f3bc92932bb948846e2e19e1565de128eaf3fd194504'
RUN=V/'runs/C3_QUERY_ANCHORED_SOURCE_V1';BASE=V/'source_baselines/C1_FROZEN_ON_V96'
N=31680;B=960;STRIDE=33;K=16
NAMES=['LOCO_BEFORE_D','PALLOX_BEFORE_D','SHAM_D','BEFORE_E','COVER_E'];QUOTAS=[8,8,16,8,24]
VARIANTS=['BEFORE','OPAQUE_COVER','CUE_PRESERVING_SHAM']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s)for s in Path(p).read_text().splitlines()if s]
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def equal(actual,expected,path=''):
 if isinstance(expected,dict):
  assert set(actual)==set(expected),(path,'keys',set(actual)^set(expected))
  for k,v in expected.items():equal(actual[k],v,path+'/'+str(k))
 elif isinstance(expected,list):
  assert len(actual)==len(expected),(path,'length')
  for i,v in enumerate(expected):equal(actual[i],v,path+'/'+str(i))
 elif isinstance(expected,float):
  assert math.isfinite(actual)and math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12),(path,actual,expected)
 else:assert actual==expected,(path,actual,expected)
def canonical(source,raw):
 return {('LOCO','TRAIN'):'TRAIN',('LOCO','VALIDATION'):'VALIDATION',('PALLOX','TRAIN_LAB'):'TRAIN',('PALLOX','VAL_HALL'):'VALIDATION'}[(source,raw)]
def metadata(lineage,gt):
 assert len(lineage)==len(gt)==N
 y=np.full(N,-1,np.int8);ready=np.zeros((B,K),bool)
 for oi,(r,g)in enumerate(zip(lineage,gt)):
  i,j=divmod(oi,STRIDE);k=None if j==0 else(j-1)//2;variant=VARIANTS[0]if j==0 else VARIANTS[1+(j-1)%2]
  assert r['occurrence_index']==g['occurrence_index']==oi and r['slot_index']==g['slot_index']==g['base_row_index']==i
  assert r['donor_rank']==g['donor_rank']==k and r['variant']==g['variant']==variant and r['occurrence_id']==g['occurrence_id']
  assert g['native_expected_part_state']=='UNKNOWN'and not g['human_gold']
  y[oi]=g['candidate_aux_hidden_label_GT_only'];assert int(y[oi])in[-1,0,1]and g['known_candidate_GT_only']==(y[oi]>=0)
  assert g['query_anchored_before_label']==gt[33*i]['query_anchored_before_label']and g['original_C1_V95_label']==gt[33*i]['original_C1_V95_label']
  assert r['query_xy']==lineage[33*i]['query_xy']and r['L']==lineage[33*i]['L']and r['split']==lineage[33*i]['split']
  if j==0:assert not g['pair_ready']and y[oi]==g['query_anchored_before_label']
  elif j%2:ready[i,k]=g['pair_ready']
  else:assert ready[i,k]==g['pair_ready']
 for i in range(B):
  for k in range(K):assert y[33*i+1+2*k:33*i+3+2*k].tolist()==([1,0]if ready[i,k]else[-1,-1])
 assert Counter(y[::33].tolist())=={0:61,1:4,-1:895}
 return dict(y=y,known=y>=0,source=np.array([r['source']for r in lineage]),raw=np.array([r['split']for r in lineage]),split=np.array([canonical(r['source'],r['split'])for r in lineage]),variant=np.array([r['variant']for r in lineage]),ready=ready)
def cohorts(m,split):
 a=m['split']==split;y=m['y'];v=m['variant'];s=m['source']
 return {'LOCO_BEFORE_D':a&(s=='LOCO')&(v=='BEFORE')&(y==0),'PALLOX_BEFORE_D':a&(s=='PALLOX')&(v=='BEFORE')&(y==0),'SHAM_D':a&(v==VARIANTS[2])&(y==0),'BEFORE_E':a&(v=='BEFORE')&(y==1),'COVER_E':a&(v==VARIANTS[1])&(y==1)}
def counts(m):
 out={}
 for s in ['TRAIN','VALIDATION']:
  a=m['split']==s;y=m['y'];v=m['variant'];known=m['known']
  out[s]=dict(rows=int(a.sum()),before_D=int((a&(v=='BEFORE')&(y==0)).sum()),before_E=int((a&(v=='BEFORE')&(y==1)).sum()),before_U=int((a&(v=='BEFORE')&(y<0)).sum()),accepted_pairs=int(m['ready'][m['split'][::33]==s].sum()),known_D=int((a&(y==0)).sum()),known_E=int((a&(y==1)).sum()),unknown=int((a&~known).sum()))
 return out
def predictions(z,available):
 result=np.full(len(z),-1,np.int8);result[available&(z<0)]=0;result[available&(z>=0)]=1;return result
def cm(p,m,mask):
 matrix=np.zeros((2,3),np.int64)
 for yy in [0,1]:
  for col,pred in enumerate([0,1,-1]):matrix[yy,col]=int((mask&(m['y']==yy)&(p==pred)).sum())
 div=lambda a,b:float(a/b)if b else None
 return dict(matrix=matrix.tolist(),rows=['conditional_AUX_D','conditional_AUX_hidden_E'],columns=['D','E','U'],D_recall=div(matrix[0,0],matrix[0].sum()),E_recall=div(matrix[1,1],matrix[1].sum()),D_precision=div(matrix[0,0],matrix[:,0].sum()),E_precision=div(matrix[1,1],matrix[:,1].sum()),known_U_rate=div(matrix[:,2].sum(),matrix.sum()),all_N=int(mask.sum()),known_N=int(matrix.sum()),unknown_GT_N=int((mask&~m['known']).sum()),unknown_GT_predictions={str(k):int((mask&~m['known']&(p==k)).sum())for k in [0,1,-1]})
def pairs(p,m,split,source=None):
 pp=p.reshape(B,33);base=m['split'][::33]==split
 if source is not None:base&=m['source'][::33]==source
 mask=m['ready']&base[:,None];good=(pp[:,0,None]==0)&(pp[:,1::2]==1)&(pp[:,2::2]==0);n=int(mask.sum());correct=int((mask&good).sum())
 return dict(accepted_pairs=n,all3_correct=correct,all3_correct_rate=correct/n if n else None,rank_counts={str(k):dict(N=int(mask[:,k].sum()),all3_correct=int((mask[:,k]&good[:,k]).sum()))for k in range(16)},controlled_tested=n>0)
def all_metrics(z,a,m):
 p=predictions(z,a);result={}
 for split in ['TRAIN','VALIDATION']:
  mask=m['split']==split
  result[split]=dict(all_known=cm(p,m,mask),by_variant={v:cm(p,m,mask&(m['variant']==v))for v in VARIANTS},five_cohorts={name:cm(p,m,sub)for name,sub in cohorts(m,split).items()},paired=pairs(p,m,split),paired_by_source={s:pairs(p,m,split,s)for s in ['LOCO','PALLOX']})
 c=result['VALIDATION']['five_cohorts'];g={}
 for name,cohort,metric,minimum in [('LOCO_before_D_recall','LOCO_BEFORE_D','D_recall',.85),('PALLOX_before_D_recall','PALLOX_BEFORE_D','D_recall',.85),('natural_before_E_recall','BEFORE_E','E_recall',.75),('sham_D_recall','SHAM_D','D_recall',.85),('cover_E_recall','COVER_E','E_recall',.75)]:g[name]=dict(value=c[cohort][metric],minimum=minimum)
 g['accepted_VAL_pair_all3_correct_rate']=dict(value=result['VALIDATION']['paired']['all3_correct_rate'],minimum=.75)
 g['known_VAL_U_rate']=dict(value=result['VALIDATION']['all_known']['known_U_rate'],maximum=.05)
 for r in g.values():r['PASS']=r['value']is not None and (r['value']>=r['minimum']if'minimum'in r else r['value']<=r['maximum'])
 return result,g,all(r['PASS']for r in g.values())
def criterion(z,a,m):
 zz=np.where(a,z,0.).astype(np.float64);loss=np.logaddexp(0.,zz)-m['y']*zz
 per={name:dict(count=int(mask.sum()),mean_BCE=float(loss[mask].mean()),unavailable=int((mask&~a).sum()))for name,mask in cohorts(m,'VALIDATION').items()}
 assert all(r['count']>0 for r in per.values())
 return float(np.mean([r['mean_BCE']for r in per.values()])),per
def audit_scores(path,lineage,m):
 with np.load(path,allow_pickle=False)as f:z={k:np.array(f[k],copy=True)for k in f.files}
 zz=z['query_logit'];a=z['query_available'];assert zz.shape==(N,)and zz.dtype==np.float32 and np.isfinite(zz).all()and a.shape==(N,)and a.dtype==bool
 assert np.array_equal(a,[r['input_facts']['operator_query_available']for r in lineage])and not zz[~a].any()
 assert np.array_equal(z['prediction'],predictions(zz,a))and np.array_equal(z['scalar_target_D0_E1_Uminus1'],m['y'])and np.array_equal(z['candidate_known'],m['known'])
 mapping={'source':m['source'],'split':m['split'],'original_split':m['raw'],'variant':m['variant'],'occurrence_index':np.arange(N),'base_row_index':np.arange(N)//33,'donor_rank':np.array([-1 if r['donor_rank']is None else r['donor_rank']for r in lineage],np.int8),'input_unit_index':np.array([r['input_unit_index']for r in lineage]),'occurrence_id':np.array([r['occurrence_id']for r in lineage]),'input_id':np.array([r['input_id']for r in lineage]),'native_slot_logical_only':np.array([r['native_slot_logical_only']for r in lineage]),'image_sha256':np.array([r['image_sha256']for r in lineage]),'original_image_sha256':np.array([r['original_image_sha256']for r in lineage]),'query_xy_raw':np.array([r['query_xy']if r['query_xy']is not None else[np.nan,np.nan]for r in lineage],np.float64),'L_raw':np.array([r['L']if r['L']is not None else np.nan for r in lineage],np.float64),'accepted_pair':m['ready']}
 for key,expected in mapping.items():assert np.array_equal(z[key],expected,equal_nan=True)if z[key].dtype.kind in'f'else np.array_equal(z[key],expected),key
 expected_nll=(np.logaddexp(np.float32(0),zz)-np.where(m['y']>=0,m['y'],0)*zz).astype(np.float32)
 assert np.array_equal(z['NLL_FP32'],expected_nll),'NLL_FP32 exact matching-runtime recipe'
 metrics,gates,passed=all_metrics(zz,a,m);value,per=criterion(zz,a,m)
 return dict(metrics=metrics,gates=gates,all_gates_pass=passed,criterion=value,VAL_cohorts=per,known_NLL_sum_FP64=float((np.logaddexp(0.,zz.astype(float))-m['y']*zz)[m['known']].sum()),unknown_NLL_excluded=int((~m['known']).sum()),unavailable=int((~a).sum()),units=int(len(set(z['input_unit_index'].tolist())))),zz,a
def audit_sampler(log,m,lineage):
 import torch
 assert torch.__version__=='2.1.1+cu118','Matching CPU RNG runtime required'
 saved=rows(log);assert len(saved)==2000
 pools=[np.flatnonzero(mask)for mask in cohorts(m,'TRAIN').values()];assert all(len(p)>0 for p in pools)
 rng=torch.Generator(device='cpu').manual_seed(194);digest=hashlib.sha256()
 for u,r in enumerate(saved,1):
  pieces=[]
  for pool,q in zip(pools,QUOTAS):pieces.append(torch.tensor(pool)[torch.randint(len(pool),(q,),generator=rng)])
  indices=torch.cat(pieces);ix=indices[torch.randperm(64,generator=rng)].numpy()
  assert r['update']==u and r['occurrence_indices']==ix.tolist()and r['input_unit_indices']==[lineage[i]['input_unit_index']for i in ix]
  expected_quotas={name:int(mask[ix].sum())for name,mask in cohorts(m,'TRAIN').items()};assert list(expected_quotas.values())==QUOTAS and r['cohort_quotas']==expected_quotas
  assert (m['split'][ix]=='TRAIN').all()and m['known'][ix].all()
  av=np.array([lineage[i]['input_facts']['operator_query_available']for i in ix],bool)
  assert r['available_known_count']==int(av.sum())and r['known_unavailable_count']==int((~av).sum())
  assert math.isfinite(r['loss'])and r['loss']>=0 and math.isfinite(r['gradient_norm_before_clip'])
  digest.update(np.asarray(ix,np.int64).tobytes())
 return dict(updates=2000,all128000_occurrence_draws_exact=True,CPU_rng_seed=194,quotas=QUOTAS,draw_trace_sha256=digest.hexdigest(),per_update_loss_values='Finite/nonnegative checked; no intermediate logits are stored, so actual per-update loss/parameter gradient cannot be independently reconstructed without a forbidden forward.')
def main(fit_sha,baseline_sha):
 assert not(H/'REVIEW.json').exists(),'No overwrite'
 assert sha(TRAINER)==TRAINER_SHA and sha(BASECODE)==BASECODE_SHA
 assert sha(RUN/'FIT_COMPLETE.json')==fit_sha and sha(BASE/'BASELINE_COMPLETE.json')==baseline_sha,'Wait for root completed hashes'
 fit=read(RUN/'FIT_COMPLETE.json');base=read(BASE/'BASELINE_COMPLETE.json');assert fit['completed']and fit['PASS_execution']and fit['optimizer_updates']==2000 and base['completed']and base['PASS_execution']
 freeze=read(check(fit['MODEL_FREEZE']));entry=read(check(freeze['entry']));assert entry['approved_actual_source_fit']and entry['source_only']and not entry['current_allowed']
 for key in ['trainer','protocol','model_code','candidate_definition','input_pack_manifest','input_pack_seal','base_input_lineage','GT_ledger','generated_tests','independent_preparation_review','independent_input_audit']:check(entry[key])
 assert entry['trainer']['sha256']==TRAINER_SHA
 proto=read(entry['protocol']['path']);assert proto['settings']['quotas']==QUOTAS and proto['settings']['seed']==194 and proto['settings']['updates']==2000
 manifest=read(entry['input_pack_manifest']['path']);lineage=rows(check(manifest['input_lineage']));gt=rows(entry['GT_ledger']['path']);m=metadata(lineage,gt)
 equal(proto['expected_counts'],counts(m),'protocol/counts')
 c3,z,a=audit_scores(check(fit['scores']),lineage,m);pub=read(check(fit['source_results']));equal(pub['metrics'],c3['metrics'],'C3/metrics');equal(pub['counts'],counts(m),'C3/counts');equal(pub['VAL_cohorts'],c3['VAL_cohorts'],'C3/VAL_cohorts');equal(pub['selector_five_VAL_cohorts_BCE'],c3['criterion'],'C3/selector')
 gates=read(check(fit['source_gates']));equal(gates['gates'],c3['gates'],'C3/gates');assert gates['all_gates_pass']==gates['PASS']==c3['all_gates_pass']and not gates['current_inference_authorized']
 history=rows(check(fit['history']));assert [r['update']for r in history]==list(range(0,2001,100))
 for h in history:
  assert set(h['VAL_cohorts'])==set(NAMES)
  for n,mask in cohorts(m,'VALIDATION').items():assert h['VAL_cohorts'][n]['count']==int(mask.sum())and h['VAL_cohorts'][n]['unavailable']==int((mask&~a).sum())
  mean=sum(h['VAL_cohorts'][n]['mean_BCE']for n in NAMES)/5
  equal(h['criterion_five_VAL_cohorts_BCE'],mean,'history/criterion')
 earliest=min(history,key=lambda x:x['criterion_five_VAL_cohorts_BCE']);selected=earliest['update']
 assert selected==fit['selected_update']==freeze['selected_update']==pub['selected_update']and freeze['updates']==2000
 equal(earliest['criterion_five_VAL_cohorts_BCE'],c3['criterion'],'selected criterion')
 for k in ['checkpoint','initial_checkpoint','final_checkpoint']:check(freeze[k])
 draws=audit_sampler(check(fit['train_updates']),m,lineage)
 # C1 checkpoint is fixed, never chosen from the new target results.
 bentry=read(check(base['entry']));assert bentry['code']['sha256']==BASECODE_SHA and bentry['diagnostic_not_selector']and not bentry['training_allowed']and not bentry['current_allowed']
 for key in ['checkpoint','model_freeze','input_pack_manifest','input_pack_seal','base_input_lineage','GT_ledger']:check(bentry[key])
 assert bentry['GT_ledger']==entry['GT_ledger']and bentry['input_pack_manifest']==entry['input_pack_manifest']
 bfreeze=read(bentry['model_freeze']['path']);assert bfreeze['selected_update']==100 and bfreeze['checkpoint']==bentry['checkpoint']and base['model_unchanged']and base['diagnostic_not_selector']
 c1,bz,ba=audit_scores(check(base['scores']),lineage,m);bpub=read(check(base['results']));equal(bpub['metrics'],c1['metrics'],'C1/metrics');equal(bpub['counts'],counts(m),'C1/counts');assert np.array_equal(a,ba)
 comparison={}
 for split in ['TRAIN','VALIDATION']:
  comparison[split]={name:{'C1':c1['metrics'][split]['five_cohorts'][name]['matrix'],'C3':c3['metrics'][split]['five_cohorts'][name]['matrix']}for name in NAMES}
 report=dict(completed=True,PASS=True,discrepancies=0,fit_complete=bind(RUN/'FIT_COMPLETE.json'),baseline_complete=bind(BASE/'BASELINE_COMPLETE.json'),independent_code=bind(__file__),trainer=bind(TRAINER),baseline_code=bind(BASECODE),selected_update=selected,selector_criterion=c3['criterion'],checkpoint=freeze['checkpoint'],history_checkpoints=21,sampler=draws,counts=counts(m),C3=c3,C1_diagnostic=c1,cohort_CM_comparison=comparison,
  exact_all31680_ID_q_L_rawsplit_rank_availability_label_U_preservation=True,old_new_labels_separate=True,all7_gates_exact=True,
  limitations=['No model/checkpoint parameter deserialization or forward; checkpoint SHA and frozen metadata checked only.','All21 checkpoint criteria independently reconstructed from stored five-cohort history means; only selected logits exist for full independent BCE replay.','Per-update losses/parameter gradients cannot be replayed from saved indices alone; sampler/counts/finite values checked.','C1 on changed target is diagnostic, not a pure causal ablation or new selector.','Conditional AI cue labels, native expectedpart UNKNOWN, no human gold.'],new_models_forwards_optimizers_current_PnP=0)
 write(H/'REVIEW.json',report);write(H/'SEAL.json',dict(completed=True,PASS=True,outputs=[bind(p)for p in sorted(H.iterdir())if p.is_file()]))
 print(json.dumps(dict(PASS=True,report=bind(H/'REVIEW.json'),seal=bind(H/'SEAL.json'))))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--fit-complete-sha',required=True);ap.add_argument('--baseline-complete-sha',required=True);args=ap.parse_args();main(args.fit_complete_sha,args.baseline_complete_sha)
