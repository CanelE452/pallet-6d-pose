"""External-only metadata adapter; all raw D/E/S/U rows are preserved."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,importlib.util,json
import numpy as np
import torch
P=Path(__file__).resolve().parents[2]
C5=P/'query_visibility_local_residual_head_20261023_v101'
C5_ADAPTER=C5/'source_fit_preparation/data_adapter.py'
C5_ADAPTER_SHA='37aef6fc8987b98c373a93377c0e0643bff754e1ac41d73e5ef7ceb5a734187f'
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
assert hashlib.sha256(C5_ADAPTER.read_bytes()).hexdigest()==C5_ADAPTER_SHA
c5=module(C5_ADAPTER,'v102_immutable_c5_data')
sha=c5.sha;bind=c5.bind;check=c5.check;read=c5.read;rows=c5.rows;same=c5.same;old_helper=c5.old_helper
NOLD=c5.NOLD;NNEW=c5.NNEW;N=c5.N;FIELDS=c5.FIELDS;STAGES=c5.STAGES
VAL_NAMES=tuple(k for k in c5.VAL_NAMES if k!='new_ALL_S')

def load_metadata(bindings):
 m=c5.load_metadata(bindings);raw=m['typed_state'];m['raw_teacher_known']=raw!='U';m['known_geometric_S']=raw=='S';m['unresolved_teacher_U']=raw=='U';m['known']=np.isin(raw,['D','E'])
 # -1 below means no external target, not unknown teacher state. raw S remains S.
 m['external_target']=np.where(m['known'],m['labels'],-1).astype(np.int8)
 m['external_counts']={s:{'D':int(((m['split']==s)&(raw=='D')).sum()),'E':int(((m['split']==s)&(raw=='E')).sum()),'known_geometric_S_out_of_task':int(((m['split']==s)&(raw=='S')).sum()),'unresolved_U':int(((m['split']==s)&(raw=='U')).sum())}for s in('TRAIN','VALIDATION')}
 assert [int((raw==x).sum())for x in('D','E','S','U')]==[1215,303,396,32070]
 assert int(m['known'].sum())==1518
 return m

def val_masks(meta):
 r={k:v for k,v in c5.val_masks(meta).items()if k!='new_ALL_S'}
 assert tuple(r)==VAL_NAMES and all(meta['known'][v].all()for v in r.values())
 return r

def sampling_pools(meta):
 old=c5.old_masks(meta,'TRAIN');m=(meta['domain']=='V99_SYNTH')&(meta['split']=='TRAIN');t=meta['typed_state']
 r={'old_triplets':meta['old_pairs'][meta['split'][meta['old_pairs'][:,0]]=='TRAIN'],'new_triplets':meta['new_pairs'][meta['split'][meta['new_pairs'][:,0]]=='TRAIN']}
 for name in('LOCO_BEFORE_D','PALLOX_BEFORE_D','BEFORE_E'):r['old_'+name]=np.flatnonzero(old[name])
 for state in('D','E'):r['new_ALL_'+state]=np.flatnonzero(m&(t==state))
 assert all(len(v)>0 for v in r.values())and all(v.any()for v in val_masks(meta).values()),'Empty required cohort; no fallback'
 return r

def draw(pools,generator):
 oi=torch.randint(len(pools['old_triplets']),(4,),generator=generator).numpy();ni=torch.randint(len(pools['new_triplets']),(4,),generator=generator).numpy();triples=np.concatenate([pools['old_triplets'][oi],pools['new_triplets'][ni]])
 x=[triples.reshape(-1)];quotas=(('old_LOCO_BEFORE_D',4),('old_PALLOX_BEFORE_D',4),('old_BEFORE_E',8),('new_ALL_D',8),('new_ALL_E',16))
 for name,n in quotas:x.append(pools[name][torch.randint(len(pools[name]),(n,),generator=generator).numpy()])
 unshuffled=np.concatenate(x);assert len(unshuffled)==64;perm=torch.randperm(64,generator=generator).numpy();inv=np.argsort(perm)
 return unshuffled[perm],inv[np.arange(24).reshape(8,3)],dict(old_pair_pool_indices=oi.tolist(),new_pair_pool_indices=ni.tolist(),unshuffled_occurrence_indices=unshuffled.tolist(),shuffle_permutation=perm.tolist(),cohort_draw_quotas=dict(old_triplets=4,new_triplets=4,**dict(quotas)))

def selector(z,available,meta):
 safe=np.where(available,z,0).astype(np.float64);loss=np.logaddexp(0.,safe)-np.where(meta['known'],meta['labels'],0)*safe;parts={}
 for k,m in val_masks(meta).items():
  assert m.any();parts[k]=dict(count=int(m.sum()),mean_BCE=float(loss[m].mean()),unavailable=int((m&~available).sum()))
 return float(np.mean([v['mean_BCE']for v in parts.values()])),parts

def load_inputs(bindings,meta):return c5.load_inputs(bindings,meta)
def decode(z,a):return np.where(a,(z>=0).astype(np.int8),-1)
def external_cm(z,a,meta,mask):
 p=decode(z,a);known=meta['known'];y=meta['labels'];s=meta['known_geometric_S'];u=meta['unresolved_teacher_U']
 cm=np.array([[np.sum(mask&known&(y==k)&(p==j))for j in(0,1,-1)]for k in(0,1)],np.int64)
 ratio=lambda x,n:float(x/n)if n else None
 return dict(matrix=cm.tolist(),rows=['conditional_AUX_visible_D','conditional_AUX_external_E'],columns=['D','external_E','U'],D_recall=ratio(cm[0,0],cm[0].sum()),E_recall=ratio(cm[1,1],cm[1].sum()),D_precision=ratio(cm[0,0],cm[:,0].sum()),E_precision=ratio(cm[1,1],cm[:,1].sum()),known_external_U_rate=ratio(cm[:,2].sum(),cm.sum()),all_N=int(mask.sum()),known_external_N=int(cm.sum()),known_geometric_S_out_of_task_N=int((mask&s).sum()),unresolved_teacher_U_N=int((mask&u).sum()),S_out_of_task_raw_decisions={str(k):int((mask&s&(p==k)).sum())for k in(0,1,-1)},teacher_U_raw_decisions={str(k):int((mask&u&(p==k)).sum())for k in(0,1,-1)},learned_self_classification=False)
def new_pair_metrics(z,a,meta,mask=None):
 p=decode(z,a);pairs=meta['new_pairs'];m=np.ones(len(pairs),bool)if mask is None else mask;hit=(p[pairs]==np.array([0,1,0])).all(1);n=int(m.sum())
 return dict(selected_triplets=n,all3_correct=int((m&hit).sum()),all3_correct_rate=float(hit[m].mean())if n else None,before_D_correct=int((m&(p[pairs[:,0]]==0)).sum()),cover_E_correct=int((m&(p[pairs[:,1]]==1)).sum()),sham_D_correct=int((m&(p[pairs[:,2]]==0)).sum()),any_U=int((m&(p[pairs]<0).any(1)).sum()))
