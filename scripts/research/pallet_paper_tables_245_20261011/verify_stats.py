"""Independent saved-row statistical audit; no producer functions, model or PnP.

Moment calculations use NumPy, t p values scipy.ttest_1samp and Holm a
vectorized step-down implementation, independent of the producer arithmetic.
"""
from __future__ import annotations
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np
from scipy.stats import ttest_1samp

ROOT=Path(__file__).resolve().parents[3]
DOC=Path(os.environ.get('PALLET_PAPER_OUTPUT',ROOT/'_docs/experiments/pallet_paper_tables_245_20261011')).resolve()
METRICS=('T_cm','R_deg','ADDsym_m','ADDsym_normalized','IoU3D','success_rate','confusion_rate','rotation_gt45_rate','ADDsym_AUC')
FIELDS=('translation_cm','rotation_deg','ADDsym_m','ADDsym_normalized','IoU3D')
TEST_METRICS=('PCK10','T_cm','R_deg','success_rate','confusion_rate','ADDsym_AUC','IoU3D')
CHECKS=0
MAX_ERROR=0.

def read(path):return json.loads(Path(path).read_text())
def rows(path):
 with gzip.open(path,'rt') as f:return [json.loads(x) for x in f]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def check(actual,expected,name,tolerance=1e-9):
 global CHECKS,MAX_ERROR
 CHECKS+=1
 if actual is None or expected is None:assert actual is None and expected is None,(name,actual,expected);return
 if isinstance(actual,(list,tuple)) or isinstance(expected,(list,tuple)):
  assert len(actual)==len(expected),(name,'length')
  for j,(a,b) in enumerate(zip(actual,expected)):check(a,b,name+f'[{j}]',tolerance)
 elif isinstance(expected,(bool,str)):assert actual==expected,(name,actual,expected)
 else:
  difference=abs(float(actual)-float(expected));MAX_ERROR=max(MAX_ERROR,difference)
  assert difference<=tolerance,(name,actual,expected,difference)
def finite(value):return value is not None and np.isfinite(float(value))

class IndependentDraws:
 def __init__(self,ids,sessions,population):
  self.ids=list(ids);self.position={i:j for j,i in enumerate(ids)};self.population=population
  self.labels=sorted(set(sessions)) if population=='RECT' else list(ids)
  self.group=np.array([self.labels.index(s) for s in sessions]) if population=='RECT' else np.arange(len(ids))
  n=len(self.labels)
  self.counts=np.random.default_rng(20260917).multinomial(n,np.full(n,1/n),size=10000).astype('<u2')
  self.sha256=hashlib.sha256(self.counts.tobytes()).hexdigest();self.weight=self.counts.astype(float)
 def samples(self,selected,values,denominator=None):
  value=np.asarray(values,float);den=np.ones(len(value)) if denominator is None else np.asarray(denominator,float)
  valid=np.isfinite(value)&np.isfinite(den)&(den>0)
  group=self.group[[self.position[i] for i in selected]]
  sums=np.array([value[valid&(group==k)].sum() for k in range(len(self.labels))])
  counts=np.array([den[valid&(group==k)].sum() for k in range(len(self.labels))])
  total=self.weight@sums;size=self.weight@counts;keep=size>0
  return total[keep]/size[keep]

def row_vectors(rr,population):
 n=len(rr);a=np.full((n,13),np.nan)
 threshold=np.linspace(0,.1,1001)
 for j,r in enumerate(rr):
  p=r['pose'];ok=bool(p.get('available'))
  a[j,9]=float(ok);a[j,10]=r['corner']['pck10_denominator'];a[j,11]=r['corner']['pck10_hits'] if ok else 0.;a[j,12]=r['corner']['pck10_hits']
  for k,f in enumerate(FIELDS):a[j,k]=float(p[f]) if ok and finite(p.get(f)) else np.nan
  a[j,5]=float(ok and p['translation_cm']<5 and p['rotation_deg']<5)
  a[j,6]=float(ok and p['rotation_deg']>45 and abs(p['yaw_deg'])>=60) if population=='RECT' else np.nan
  a[j,7]=float(ok and p['rotation_deg']>45)
  error=float(p['ADDsym_normalized']) if ok else np.inf
  indicator=(error<=threshold).astype(float)
  a[j,8]=float(np.sum((indicator[:-1]+indicator[1:])*np.diff(threshold)*.5)/.1)
 return a

def combine(vectors):
 stack=np.stack(vectors);valid=np.isfinite(stack);count=valid.sum(axis=0)
 return np.divide(np.where(valid,stack,0).sum(axis=0),count,out=np.full(stack.shape[1:],np.nan),where=count>0)

def moments(values):
 v=np.asarray(values,float);v=v[np.isfinite(v)]
 if not len(v):return {k:None for k in ('mean','variance','std','median','P90','min','max')},0
 return dict(mean=float(v.mean()),variance=float(v.var(ddof=1)) if len(v)>1 else None,std=float(v.std(ddof=1)) if len(v)>1 else None,
  median=float(np.median(v)),P90=float(np.percentile(v,90)),min=float(v.min()),max=float(v.max())),len(v)

def summary_check(packet,matrix,ids,draw,name,seeds=1):
 check(packet['frames'],len(ids),name+'.frames')
 check(packet['available_frames_any_seed'],int((matrix[:,9]>0).sum()),name+'.available_any')
 check(packet['available_frames_all_seeds'],int((matrix[:,9]==1).sum()),name+'.available_all')
 for k,metric in enumerate(METRICS):
  expected,n=moments(matrix[:,k]);given=packet['metrics'][metric]
  check(given['n'],n,name+'.'+metric+'.n')
  for field,value in expected.items():check(given[field],value,name+'.'+metric+'.'+field)
  sampled=draw.samples(ids,matrix[:,k]);interval=np.quantile(sampled,[.025,.975]).tolist() if len(sampled) else None
  check(given['CI95'],interval,name+'.'+metric+'.CI95')
 p=packet['PCK10'];hit=matrix[:,11];den=matrix[:,10];sampled=draw.samples(ids,hit,den)
 for key,value in dict(rate=float(hit.sum()/den.sum()),numerator=float(hit.sum()),denominator=int(den.sum()),
  raw_2D_rate_ignoring_pose_availability=float(matrix[:,12].sum()/den.sum()),raw_2D_hits=float(matrix[:,12].sum())).items():check(p[key],value,name+'.PCK.'+key)
 check(p['CI95'],np.quantile(sampled,[.025,.975]).tolist(),name+'.PCK.CI95')

CONTRASTS1=[(scope,b,'N3_THEN_SUBPIX','BASE') for scope in ('RECT245','SQUARE118') for b in ('YOLO','DOPE','ResNet-18')]
CONTRASTS2=[('RECT245','YOLO','N3_THEN_SUBPIX',m) for m in ('BASE','SUBPIX','POSEFIX','POSEFIX_SUBPIX')]
ORDER3=('N0','N1','N2','N3_DIM_SYM','SUBPIX','N3_THEN_SUBPIX','JOINT_FIXED_ISOTROPIC','FG_JOINT_POSTERIOR')
CONTRASTS3=[('RECT245','YOLO',b,a) for a,b in zip(ORDER3,ORDER3[1:])]+[('RECT245','YOLO','N3_THEN_SUBPIX',m) for m in ('JOINT_FIXED_ISOTROPIC','FG_JOINT_POSTERIOR')]

def p_t(values):
 value=np.asarray(values,float);value=value[np.isfinite(value)]
 if len(value)<2:return None,len(value)-1
 if np.ptp(value)==0:return (1. if value[0]==0 else 0.),len(value)-1
 return float(ttest_1samp(value,0.).pvalue),len(value)-1

def verify_pairs(paired,data,draws):
 for family,contrasts in (('TABLE_1',CONTRASTS1),('TABLE_2',CONTRASTS2),('TABLE_3',CONTRASTS3)):
  given=paired['tables'][family];expectedrecords=[]
  assert len(given)==7*len(contrasts)
  for (scope,backbone,left,right) in contrasts:
   ga=data[scope][backbone+'::'+left];gb=data[scope][backbone+'::'+right];ids=ga['ids'];sessions=np.array(ga['sessions']);draw=draws[scope]
   a,b=ga['mean'],gb['mean']
   for metric in TEST_METRICS:
    if metric=='confusion_rate' and scope=='SQUARE118':expectedrecords.append(dict(p_t=None,p_bootstrap=None));continue
    if metric=='PCK10':
     num=a[:,11]-b[:,11];den=a[:,10];delta=float(num.sum()/den.sum());diff=num/den;valid=np.ones(len(ids),bool)
     sampled=draw.samples(ids,num,den)
    else:
     idx=METRICS.index(metric);diff=a[:,idx]-b[:,idx];valid=np.isfinite(diff);delta=float(diff[valid].mean()) if valid.any() else None
     num=np.where(valid,diff,0.);den=valid.astype(float);sampled=draw.samples(ids,diff)
    if scope!='SQUARE118':
     sessionmeans=[]
     for s in sorted(set(sessions)):
      mask=sessions==s
      if den[mask].sum()>0:sessionmeans.append(float(num[mask].sum()/den[mask].sum()))
     if len(sessionmeans)==13:pt,df=p_t(sessionmeans)
     else:pt,df=None,12
     pb=(1+np.count_nonzero(np.abs(sampled-delta)>=abs(delta)))/(len(sampled)+1) if delta is not None and len(sampled) else None
    else:pt,df=p_t(diff);pb=None
    expectedrecords.append(dict(delta=delta,CI95=np.quantile(sampled,[.025,.975]).tolist() if len(sampled) else None,
     paired_frames=int(valid.sum()),p_t=pt,p_bootstrap=pb,df=df))
  for j,(actual,expect) in enumerate(zip(given,expectedrecords)):
   for key,value in expect.items():
    if key=='df':check(actual['t_test']['df'],value,f'{family}[{j}].df')
    else:check(actual.get(key),value,f'{family}[{j}].'+key)
  for pfield in ('p_t','p_bootstrap'):
   ps=[r[pfield] for r in expectedrecords];mask=np.array([p is not None for p in ps]);positions=np.flatnonzero(mask);values=np.array([ps[i] for i in positions])
   order=np.argsort(values,kind='stable');adjusted=np.minimum(1.,np.maximum.accumulate(values[order]*(len(values)-np.arange(len(values)))))
   corrected={int(positions[i]):float(value) for i,value in zip(order,adjusted)}
   for j,actual in enumerate(given):
    check(actual[pfield+'_holm'],corrected.get(j),f'{family}[{j}].'+pfield+'_holm')
    check(actual[pfield+'_holm_family_size'],len(values),f'{family}[{j}].family_size')

def verify_failures(packet,data):
 for scope,groups in data.items():
  for group,g in groups.items():
   backbone,method=group.split('::');base=groups.get(backbone+'::BASE');actual=packet['scopes'][scope][group]
   for seed in g['seeds']:
    expected={key:[] for key in actual[str(seed)] if key!='counts'}
    for i in g['ids']:
     a=g['index'][seed,i];ok=bool(a['pose'].get('available'))
     if not ok:expected['unavailable_ids'].append(i)
     if not base or seed not in base['seeds']:continue
     b=base['index'][seed,i];bok=bool(b['pose'].get('available'))
     if ok and bok:
      for field,tag in (('translation_cm','T'),('rotation_deg','R')):
       if finite(a['pose'].get(field)) and finite(b['pose'].get(field)):
        delta=a['pose'][field]-b['pose'][field]
        if delta>0:expected[tag+'_increase_ids'].append(i)
        if delta<0:expected[tag+'_decrease_ids'].append(i)
     sa=ok and a['pose']['translation_cm']<5 and a['pose']['rotation_deg']<5
     sb=bok and b['pose']['translation_cm']<5 and b['pose']['rotation_deg']<5
     if sb and not sa:expected['success_to_failure_ids'].append(i)
     if sa and not sb:expected['failure_to_success_ids'].append(i)
     ha=a['corner']['pck10_hits'] if ok else 0;hb=b['corner']['pck10_hits'] if bok else 0
     if ha<hb:expected['PCK_hit_decrease_ids'].append(i)
     if ha>hb:expected['PCK_hit_increase_ids'].append(i)
     if a.get('final_hypothesis')!=b.get('final_hypothesis'):expected['hypothesis_change_ids'].append(i)
     if scope!='SQUARE118':
      ca=ok and a['pose']['rotation_deg']>45 and abs(a['pose']['yaw_deg'])>=60
      cb=bok and b['pose']['rotation_deg']>45 and abs(b['pose']['yaw_deg'])>=60
      if ca and not cb:expected['confusion_damage_ids'].append(i)
      if cb and not ca:expected['confusion_recovery_ids'].append(i)
    for key,value in expected.items():
     check(actual[str(seed)][key],value,f'failures.{scope}.{group}.{seed}.{key}')
     check(actual[str(seed)]['counts'][key],len(value),f'failures.{scope}.{group}.{seed}.{key}.count')

def verify_tables(metrics):
 mapping={'frames':None,'pose_available_n':'available_frames_any_seed','pose_all_seed_n':'available_frames_all_seeds'}
 fieldmap={'T_mean_cm':('T_cm','mean'),'T_SD_cm':('T_cm','std'),'R_mean_deg':('R_deg','mean'),'R_SD_deg':('R_deg','std'),
  'T_median_cm':('T_cm','median'),'T_P90_cm':('T_cm','P90'),'R_median_deg':('R_deg','median'),'R_P90_deg':('R_deg','P90'),
  'ADDsym_AUC':('ADDsym_AUC','mean'),'IoU3D_mean':('IoU3D','mean')}
 for stem in ('TABLE_1','TABLE_2','TABLE_3','SUPP','SUPPLEMENT_319_TABLE_1','SUPPLEMENT_319_TABLE_2'):
  path=DOC/(stem+'.csv')
  if not path.is_file():continue
  with path.open() as f:rr=list(csv.DictReader(f))
  for row in rr:
   p=metrics['scopes'][row['scope']]['groups'][row['backbone']+'::'+row['method']];m=p['seed_mean']
   values={'frames':p['frames'],'pose_available_n':m['available_frames_any_seed'],'pose_all_seed_n':m['available_frames_all_seeds'],
    'PCK10_percent':100*m['PCK10']['rate'],'success_percent':100*m['metrics']['success_rate']['mean'],
    'confusion_percent':100*m['metrics']['confusion_rate']['mean'] if m['metrics']['confusion_rate']['mean'] is not None else None}
   for field,(key,stat) in fieldmap.items():values[field]=m['metrics'][key][stat]
   for field,value in values.items():check(float(row[field]) if row[field] else None,value,stem+'.'+field)

def run():
 assert (DOC/'METHOD_LOCK.json').is_file();assert not (DOC/'STATISTICS_VERIFICATION.json').exists()
 metrics=read(DOC/'METRICS.json');paired=read(DOC/'PAIRED.json');raw={p:rows(DOC/(p+'_ROWS.jsonl.gz')) for p in ('RECT','SQUARE')}
 masterids=metrics['scopes']['RECT319']['ids'];mastermeta={r['id']:r for r in raw['RECT'] if r['backbone']=='YOLO' and r['method']=='BASE' and r['seed']==1}
 rectangle=IndependentDraws(masterids,[mastermeta[i]['session'] for i in masterids],'RECT')
 check(rectangle.sha256,'63e288a51d7b0612616beefac28fcc625e76e8c5d7b5a0ecc5e8b85c73048fa5','rectangle_drawSHA')
 squareids=metrics['scopes']['SQUARE118']['ids'];square=IndependentDraws(squareids,['0918_dataset']*118,'SQUARE')
 data={};draws={}
 for scope,packet in metrics['scopes'].items():
  population=packet['population'];ids=packet['ids'];draw=rectangle if population=='RECT' else square;draws[scope]=draw
  check(packet['bootstrap']['draws_sha256_uint16_le'],draw.sha256,scope+'.drawSHA')
  grouped=defaultdict(list)
  for r in raw[population]:
   if r['id'] in set(ids):grouped[r['backbone']+'::'+r['method']].append(r)
  data[scope]={}
  for name,rr in grouped.items():
   seeds=sorted({r['seed'] for r in rr});index={(r['seed'],r['id']):r for r in rr};assert len(index)==len(rr)
   mats={s:row_vectors([index[s,i] for i in ids],population) for s in seeds};average=combine(list(mats.values()))
   group=packet['groups'][name];summary_check(group['seed_mean'],average,ids,draw,scope+'.'+name)
   for s,matrix in mats.items():summary_check(group['per_seed'][str(s)],matrix,ids,draw,scope+'.'+name+'.seed'+str(s))
   check(group['seeds'],seeds,scope+'.'+name+'.seeds')
   for j,frame in enumerate(group['frame_values']):
    check(frame['id'],ids[j],scope+'.'+name+'.frameid')
    for k,metric in enumerate(METRICS):check(frame[metric],float(average[j,k]) if np.isfinite(average[j,k]) else None,scope+'.'+name+'.'+metric)
   data[scope][name]=dict(ids=ids,seeds=seeds,index=index,mean=average,mats=mats,sessions=[index[seeds[0],i]['session'] for i in ids])
 verify_pairs(paired,data,draws);verify_failures(read(DOC/'FAILURES.json'),data);verify_tables(metrics)
 receipt=dict(status='PASS',numerical_comparisons=CHECKS,max_absolute_difference=MAX_ERROR,tolerance=1e-9,
  independent=True,producer_functions_called=False,checks=['allscope/perseed/frame moments and95%CI','full-referencePCK conditionalposepolicy','official1001-pointADDsymAUC including failures','pairedt/null-centeredbootstrapp/within-original-tableHolm','allfailure/damage/recoveryIDs','CSV numeric cells'],
  inputs={p:dict(path=p+'_ROWS.jsonl.gz',sha256=sha(DOC/(p+'_ROWS.jsonl.gz'))) for p in raw},
  method_lock_sha256=sha(DOC/'METHOD_LOCK.json'),verifier_source_sha256=sha(Path(__file__)),additional_model_forwards=0,additional_PnP_calls=0)
 with (DOC/'STATISTICS_VERIFICATION.json').open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n')
 print('PAPER_STATISTICS_VERIFICATION_PASS',CHECKS,MAX_ERROR,flush=True)
if __name__=='__main__':run()
