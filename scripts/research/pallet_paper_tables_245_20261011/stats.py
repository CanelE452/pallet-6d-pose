"""Frozen saved-row aggregation: no image inference, correction or pose solver.

The full denominator is retained for binary rates, PCK and ADDsym AUC.
Continuous pose errors use available values and publish their sample size.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import numpy as np
from scipy.stats import t as student_t
from scripts.research.pallet_wd_hypothesis_diag_20261010.statistics import Draws

ROOT=Path(__file__).resolve().parents[3]
DOC=Path(os.environ.get('PALLET_PAPER_OUTPUT',ROOT/'_docs/experiments/pallet_paper_tables_245_20261011')).resolve()
SOURCE=Path(os.environ.get('PALLET_SOURCE_ROOT',ROOT))
NAME='pallet_paper_tables_245_20261011'
REAL_DRAW_SHA='63e288a51d7b0612616beefac28fcc625e76e8c5d7b5a0ecc5e8b85c73048fa5'
NUMERIC={'T_cm':'translation_cm','R_deg':'rotation_deg','ADDsym_m':'ADDsym_m','ADDsym_normalized':'ADDsym_normalized','IoU3D':'IoU3D'}
TEST_METRICS=('PCK10','T_cm','R_deg','success_rate','confusion_rate','ADDsym_AUC','IoU3D')
ENGLISH={'BASE':'Base','N3_DIM_SYM':'Stage 1 full','SUBPIX':'cornerSubPix only','N3_THEN_SUBPIX':'Two-stage (ours)',
 'POSEFIX':'PoseFix-style','POSEFIX_SUBPIX':'PoseFix-style + cornerSubPix','N0':'Image only','N1':'+ symmetry','N2':'+ dimensions',
 'JOINT_FIXED_ISOTROPIC':'Joint–isotropic','FG_JOINT_POSTERIOR':'Joint–learned covariance','P':'P','D':'D','L':'L'}
SEEDS=(1,2,3)

def read(path):return json.loads(Path(path).read_text())
def rows(path):
 with gzip.open(path,'rt') as f:return [json.loads(line) for line in f]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def clean(value):
 if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
 if isinstance(value,(tuple,list)):return [clean(v) for v in value]
 if hasattr(value,'tolist'):return clean(value.tolist())
 if isinstance(value,float) and not math.isfinite(value):return None
 return value
def write(name,value):
 with (DOC/name).open('x') as f:json.dump(clean(value),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def source(rel):
 for p in (ROOT/rel,SOURCE/rel):
  if p.is_file():return p
 raise FileNotFoundError(rel)

def finite(value):return value is not None and math.isfinite(float(value))
def moments(values,ci=None):
 a=[float(x) for x in values if finite(x)]
 if not a:return dict(n=0,mean=None,variance=None,std=None,median=None,P90=None,min=None,max=None,CI95=None,variance_ddof=1)
 var=statistics.variance(a) if len(a)>1 else None
 return dict(n=len(a),mean=statistics.fmean(a),variance=var,std=math.sqrt(var) if var is not None else None,
  median=statistics.median(a),P90=float(np.quantile(a,.9)),min=min(a),max=max(a),CI95=ci,variance_ddof=1,CI95_target='mean')

def auc_values(normalized):
 """Individual contributions to official pose_auc(normalized errors, 1.)."""
 threshold=np.linspace(0,.1,1001)
 weight=np.full(1001,1/1000);weight[[0,-1]]*=.5
 return (np.asarray(normalized,float)[:,None]<=threshold)@weight

class Bootstrap:
 def __init__(self,ids,sessions,population):
  self.ids=list(ids);self.population=population;self.position={i:j for j,i in enumerate(ids)}
  if population=='RECT':
   d=Draws(sessions,'paired cluster')
   assert d.n==13 and d.sha256==REAL_DRAW_SHA
   self.group=d.group;self.weights=d.weights;self.sha256=d.sha256;self.units=13
  else:
   self.group=np.arange(len(ids));self.units=len(ids)
   rng=np.random.default_rng(20260917)
   raw=np.vstack([rng.multinomial(self.units,np.full(self.units,1/self.units),size=100) for _ in range(100)])
   self.sha256=hashlib.sha256(raw.astype('<u2').tobytes()).hexdigest();self.weights=raw.astype(float)
  self.group_names=list(sorted(set(sessions))) if population=='RECT' else list(ids)
 def meta(self,selected):
  return dict(level='paired13sessioncluster' if self.population=='RECT' else 'pairedframe_descriptive',
   frames=len(selected),master_frames=len(self.ids),units=self.units,resamples=10000,seed=20260917,
   draws_sha256_uint16_le=self.sha256,scope_rule='same master draws, only fixed eligible IDs retained',
   interval='linear percentiles2.5/97.5',descriptive_only=self.population=='SQUARE')
 def sample_ratio(self,selected,numerator,denominator):
  group=self.group[[self.position[i] for i in selected]]
  numerator=np.asarray(numerator,float);denominator=np.asarray(denominator,float)
  ok=np.isfinite(numerator)&np.isfinite(denominator)&(denominator>0)
  totals=np.bincount(group,weights=np.where(ok,numerator,0),minlength=self.units)
  counts=np.bincount(group,weights=np.where(ok,denominator,0),minlength=self.units)
  denom=self.weights@counts
  samples=np.divide(self.weights@totals,denom,out=np.full(len(denom),np.nan),where=denom>0)
  return samples[np.isfinite(samples)]
 def sample(self,selected,values):return self.sample_ratio(selected,values,np.ones(len(values)))
 def ci(self,selected,values):
  s=self.sample(selected,values)
  return np.quantile(s,[.025,.975]).tolist() if len(s) else None

def seed_vectors(rr,population):
 out={k:np.asarray([float(r['pose'][field]) if r['pose'].get('available') and finite(r['pose'].get(field)) else np.nan for r in rr]) for k,field in NUMERIC.items()}
 out['available']=np.asarray([bool(r['pose'].get('available')) for r in rr])
 out['success_rate']=np.asarray([float(bool(r['pose'].get('available')) and float(r['pose']['translation_cm'])<5 and float(r['pose']['rotation_deg'])<5) for r in rr])
 out['confusion_rate']=np.asarray([float(bool(r['pose'].get('available')) and float(r['pose']['rotation_deg'])>45 and abs(float(r['pose']['yaw_deg']))>=60) for r in rr]) if population=='RECT' else np.full(len(rr),np.nan)
 out['rotation_gt45_rate']=np.asarray([float(bool(r['pose'].get('available')) and float(r['pose']['rotation_deg'])>45) for r in rr])
 out['PCK10_denominator']=np.asarray([int(r['corner']['pck10_denominator']) for r in rr],float)
 out['PCK10_raw_hits']=np.asarray([int(r['corner']['pck10_hits']) for r in rr],float)
 assert (out['PCK10_denominator']>0).all()
 assert (out['PCK10_raw_hits']<=out['PCK10_denominator']).all()
 out['PCK10_hits']=out['PCK10_raw_hits']*out['available']
 out['PCK10']=out['PCK10_hits']/out['PCK10_denominator']
 out['ADDsym_AUC']=auc_values(np.where(out['available'],out['ADDsym_normalized'],np.inf))
 return out

def seed_mean(vv):
 out={}
 for key in vv[0]:
  stack=np.stack([v[key] for v in vv])
  if key=='available':continue
  if key=='PCK10_denominator':assert (stack==stack[0]).all()
  valid=np.isfinite(stack);n=valid.sum(0)
  out[key]=np.divide(np.where(valid,stack,0).sum(0),n,out=np.full(stack.shape[1],np.nan),where=n>0)
 out['available_seed_counts']=np.stack([v['available'] for v in vv]).sum(0)
 out['available_all_seeds']=out['available_seed_counts']==len(vv)
 out['available']=out['available_seed_counts']>0
 return out

def summarize(vv,ids,boot):
 result=dict(frames=len(ids),available_frames_any_seed=int(vv['available'].sum()),
  available_frames_all_seeds=int(vv.get('available_all_seeds',vv['available']).sum()),
  unavailable_frames_all_seeds=int((~vv['available']).sum()),
  metrics={key:moments(vv[key],boot.ci(ids,vv[key])) for key in (*NUMERIC,'success_rate','confusion_rate','rotation_gt45_rate','ADDsym_AUC')},
  missing_policy='T/R/ADD/IoU available-only, binary and PCK full fixed denominator; PCK zero if pose unavailable')
 for metric in ('success_rate','confusion_rate','rotation_gt45_rate','ADDsym_AUC'):
  result['metrics'][metric]['full_frame_denominator']=len(ids)
 numerator=vv['PCK10_hits'];denominator=vv['PCK10_denominator'];samples=boot.sample_ratio(ids,numerator,denominator)
 raw=vv['PCK10_raw_hits']
 result['PCK10']=dict(rate=float(numerator.sum()/denominator.sum()),numerator=float(numerator.sum()),denominator=int(denominator.sum()),
  CI95=np.quantile(samples,[.025,.975]).tolist(),threshold_px=10,operator='<=',full_reference_corner_denominator=True,
  raw_2D_rate_ignoring_pose_availability=float(raw.sum()/denominator.sum()),raw_2D_hits=float(raw.sum()),
  per_frame_distribution=moments(vv['PCK10']),binary_before_seed_mean=True)
 result['ADDsym_AUC_definition']='proper-group corresponding8corner ADD normalized; official pose_auc thresholds0..0.1/1001 trapz; each seed threshold indicator first; failures=inf'
 return result

def paired_t(values):
 a=np.asarray(values,float);a=a[np.isfinite(a)];n=len(a)
 if n<2:return dict(p=None,t=None,df=n-1,n=n,mean=None)
 mean=float(a.mean());sd=float(a.std(ddof=1))
 if sd==0:p=1. if mean==0 else 0.;stat=0. if mean==0 else math.copysign(math.inf,mean)
 else:stat=mean/(sd/math.sqrt(n));p=float(2*student_t.sf(abs(stat),n-1))
 return dict(p=p,t=stat,df=n-1,n=n,mean=mean)

def holm(values):
 result=[None]*len(values);valid=sorted(((float(p),i) for i,p in enumerate(values) if p is not None and math.isfinite(p)),key=lambda x:(x[0],x[1]))
 maximum=0.
 for rank,(p,i) in enumerate(valid):maximum=max(maximum,min(1.,(len(valid)-rank)*p));result[i]=maximum
 return result

def contrast(left,right,ids,sessions,boot,population):
 output={}
 for metric in TEST_METRICS:
  if population=='SQUARE' and metric=='confusion_rate':
   output[metric]=dict(status='NOT_APPLICABLE_C4',delta=None,CI95=None,p_t=None,p_bootstrap=None);continue
  if metric=='PCK10':
   numerator=left['PCK10_hits']-right['PCK10_hits'];denominator=left['PCK10_denominator']
   assert (denominator==right['PCK10_denominator']).all()
   valid=np.ones(len(ids),bool);delta=float(numerator.sum()/denominator.sum());samples=boot.sample_ratio(ids,numerator,denominator)
   perframe=numerator/denominator
  else:
   perframe=left[metric]-right[metric];valid=np.isfinite(perframe);numerator=np.where(valid,perframe,0);denominator=valid.astype(float)
   delta=float(perframe[valid].mean()) if valid.any() else None;samples=boot.sample(ids,perframe)
  if population=='RECT':
   sessionmeans=[];sessiondata={}
   for session in sorted(set(sessions)):
    mask=np.asarray(sessions)==session;den=float(denominator[mask].sum())
    value=float(numerator[mask].sum()/den) if den>0 else None
    sessiondata[session]=dict(delta=value,paired_frames=int(valid[mask].sum()),denominator=den)
    if value is not None:sessionmeans.append(value)
   missing=[session for session,item in sessiondata.items() if item['delta'] is None]
   if len(sessiondata)!=13 or missing:
    tt=dict(p=None,t=None,df=12,n=len(sessionmeans),mean=None,status='NOT_ESTIMABLE_13_SESSION',absent_sessions=missing)
   else:
    tt=paired_t(sessionmeans);assert tt['df']==12
    tt.update(status='CALCULATED_13_SESSION',absent_sessions=[])
   tt['target']='equal-weight13session means; PCK uses each session pooled corner-hit difference/reference-corner denominator; differs from frame/corner-weighted bootstrap target'
   tt['sessions']=sessiondata
   if delta is not None and len(samples):
    pboot=float((1+int((np.abs(samples-delta)>=abs(delta)).sum()))/(len(samples)+1))
   else:pboot=None
  else:
   tt=paired_t(perframe);tt['target']='paired frame means; descriptive single-session comparison';pboot=None
  output[metric]=dict(status='CALCULATED',delta=delta,CI95=np.quantile(samples,[.025,.975]).tolist() if len(samples) else None,
   paired_frames=int(valid.sum()),full_frames=len(ids),p_t=tt['p'],t_test=tt,p_bootstrap=pboot,
   bootstrap_p_definition='(1+#|bootstrap_delta-observed_delta|>=|observed_delta|)/(valid_draws+1)' if population=='RECT' else None,
   bootstrap_p_target='paired frame-weighted delta; PCK paired reference-corner-weighted delta' if population=='RECT' else None,
   per_seed_delta=None,lower_is_better=metric in ('T_cm','R_deg','confusion_rate'))
 return output

def family_records(contrasts,groups,scope,boot):
 records=[]
 for backbone,left,right in contrasts:
  a=groups[backbone+'::'+left];b=groups[backbone+'::'+right]
  assert a['ids']==b['ids'];ids=a['ids'];sessions=a['sessions']
  result=contrast(a['vectors'],b['vectors'],ids,sessions,boot,'SQUARE' if scope=='SQUARE118' else 'RECT')
  for metric,packet in result.items():
   forseed=[]
   for seed in a['seeds']:
    assert seed in b['seeds']
    l,r=a['per_seed_vectors'][seed],b['per_seed_vectors'][seed]
    if metric=='confusion_rate' and scope=='SQUARE118':forseed.append(None)
    elif metric=='PCK10':forseed.append(float((l['PCK10_hits']-r['PCK10_hits']).sum()/l['PCK10_denominator'].sum()))
    else:
     diff=l[metric]-r[metric];forseed.append(float(diff[np.isfinite(diff)].mean()) if np.isfinite(diff).any() else None)
   packet['per_seed_delta']=forseed
   records.append(dict(scope=scope,backbone=backbone,left=left,right=right,contrast=f'{left} − {right}',metric=metric,**packet))
 return records

def adjust_family(records):
 for kind in ('p_t','p_bootstrap'):
  corrected=holm([r.get(kind) for r in records]);size=sum(r.get(kind) is not None for r in records)
  for r,value in zip(records,corrected):r[kind+'_holm']=value;r[kind+'_holm_family_size']=size
 return records

TABLE1=[(b,'N3_THEN_SUBPIX','BASE') for b in ('YOLO','DOPE','ResNet-18')]
TABLE2=[('YOLO','N3_THEN_SUBPIX',m) for m in ('BASE','SUBPIX','POSEFIX','POSEFIX_SUBPIX')]
TABLE3_ORDER=('N0','N1','N2','N3_DIM_SYM','SUBPIX','N3_THEN_SUBPIX','JOINT_FIXED_ISOTROPIC','FG_JOINT_POSTERIOR')
TABLE3=[('YOLO',b,a) for a,b in zip(TABLE3_ORDER,TABLE3_ORDER[1:])]+[('YOLO','N3_THEN_SUBPIX',m) for m in ('JOINT_FIXED_ISOTROPIC','FG_JOINT_POSTERIOR')]

def make_group(rr,ids,boot,population):
 seeds=sorted({r['seed'] for r in rr});index={(r['seed'],r['id']):r for r in rr};assert len(index)==len(rr)
 assert all((s,i) in index for s in seeds for i in ids),'Missing seed-frame row'
 vv={s:seed_vectors([index[s,i] for i in ids],population) for s in seeds}
 vm=seed_mean([vv[s] for s in seeds]);sessions=[index[seeds[0],i]['session'] for i in ids]
 packet=dict(frames=len(ids),seeds=seeds,seed_count=len(seeds),bootstrap=boot.meta(ids),
  seed_mean=summarize(vm,ids,boot),per_seed={str(s):summarize(vv[s],ids,boot) for s in seeds},
  frame_values=[dict(id=i,session=sessions[j],grade=index[seeds[0],i]['grade'],
   pallet_diagonal_px=index[seeds[0],i]['corner'].get('pallet_diagonal_px'),
   **{key:clean(value[j]) for key,value in vm.items()}) for j,i in enumerate(ids)])
 return packet,dict(ids=ids,sessions=sessions,seeds=seeds,vectors=vm,per_seed_vectors=vv,index=index)

def make_failures(internal):
 result={}
 for scope,groups in internal.items():
  result[scope]={}
  for name,g in groups.items():
   backbone,method=name.split('::');base=groups.get(backbone+'::BASE');byseed={}
   for seed in g['seeds']:
    index=g['index'];fields={key:[] for key in ('unavailable_ids','T_increase_ids','T_decrease_ids','R_increase_ids','R_decrease_ids',
     'success_to_failure_ids','failure_to_success_ids','PCK_hit_decrease_ids','PCK_hit_increase_ids','hypothesis_change_ids','confusion_damage_ids','confusion_recovery_ids')}
    for i in g['ids']:
     a=index[seed,i];av=a['pose'].get('available')
     if not av:fields['unavailable_ids'].append(i)
     if not base or seed not in base['seeds']:continue
     b=base['index'][seed,i];bv=b['pose'].get('available')
     for key,label in (('translation_cm','T'),('rotation_deg','R')):
      if av and bv and finite(a['pose'].get(key)) and finite(b['pose'].get(key)):
       diff=float(a['pose'][key])-float(b['pose'][key])
       if diff>0:fields[label+'_increase_ids'].append(i)
       elif diff<0:fields[label+'_decrease_ids'].append(i)
     suc=lambda r:bool(r['pose'].get('available')) and r['pose']['translation_cm']<5 and r['pose']['rotation_deg']<5
     conf=lambda r:bool(r['pose'].get('available')) and r['pose']['rotation_deg']>45 and abs(r['pose']['yaw_deg'])>=60
     if suc(b) and not suc(a):fields['success_to_failure_ids'].append(i)
     if not suc(b) and suc(a):fields['failure_to_success_ids'].append(i)
     ah=int(a['corner']['pck10_hits']) if av else 0;bh=int(b['corner']['pck10_hits']) if bv else 0
     if ah<bh:fields['PCK_hit_decrease_ids'].append(i)
     if ah>bh:fields['PCK_hit_increase_ids'].append(i)
     if a.get('final_hypothesis')!=b.get('final_hypothesis'):fields['hypothesis_change_ids'].append(i)
     if scope!='SQUARE118':
      if conf(a) and not conf(b):fields['confusion_damage_ids'].append(i)
      if conf(b) and not conf(a):fields['confusion_recovery_ids'].append(i)
    byseed[str(seed)]={**fields,'counts':{key:len(val) for key,val in fields.items()}}
   result[scope][name]=byseed
 return dict(status='COMPLETE',scopes=result,comparison='every method versus same-backbone BASE, seed by seed',
  damage_definition='any finite paired T/R increase; PCK hit decrease; success loss; confusion gain; no severity threshold',
  square_excluded_ids=['029844'])

def paper_numbers(metrics,rows_by_pop,internal):
 g=metrics['scopes']['RECT245']['groups'];sq=metrics['scopes']['SQUARE118']['groups']
 packet=dict(status='COMPLETE',scope='RECT245 development-selected clean153/moderate92; SQUARE118 separate manualPnP reference')
 packet['N3_PCK10_increase_percentage_points']={b:dict(value=100*(g[b+'::N3_DIM_SYM']['seed_mean']['PCK10']['rate']-g[b+'::BASE']['seed_mean']['PCK10']['rate']),
  source_cells=[f'METRICS.json:scopes.RECT245.groups.{b}::N3_DIM_SYM.seed_mean.PCK10.rate',f'METRICS.json:scopes.RECT245.groups.{b}::BASE.seed_mean.PCK10.rate']) for b in ('YOLO','DOPE','ResNet-18')}
 packet['YOLO_success_RECT245_percent']={m:dict(value=100*g['YOLO::'+m]['seed_mean']['metrics']['success_rate']['mean'],source_cell=f'TABLE_2.csv:YOLO/{m}/success_percent') for m in ('BASE','SUBPIX','N3_THEN_SUBPIX')}
 packet['square_success_Base_to_Two_stage_percent']={b:dict(BASE=100*sq[b+'::BASE']['seed_mean']['metrics']['success_rate']['mean'],N3_THEN_SUBPIX=100*sq[b+'::N3_THEN_SUBPIX']['seed_mean']['metrics']['success_rate']['mean'],
  source_cells=[f'TABLE_1.csv:SQUARE118/{b}/BASE/success_percent',f'TABLE_1.csv:SQUARE118/{b}/N3_THEN_SUBPIX/success_percent']) for b in ('YOLO','DOPE','ResNet-18')}
 fv=g['YOLO::BASE']['frame_values'];diag=[r['pallet_diagonal_px'] for r in fv];assert all(finite(x) and x>0 for x in diag)
 med=float(np.median(diag));packet['RECT245_pallet_diagonal_px']=dict(median_px=med,ten_px_percent_of_median=1000/med,frames=len(diag),
  definition='all8 finite reference corners0..7 axis-alignedbbox hypot(ptpX,ptpY), no clipping; includes old projected/extrapolated2D reference points',source_cell='METRICS.json:scopes.RECT245.groups.YOLO::BASE.frame_values.pallet_diagonal_px')
 packet['confusion_share_among_R_gt45_percent']={}
 for name,p in g.items():
  fv=p['frame_values'];c=sum(r['confusion_rate'] for r in fv);r45=sum(r['rotation_gt45_rate'] for r in fv)
  packet['confusion_share_among_R_gt45_percent'][name]=dict(value=100*c/r45 if r45 else None,confusion_seed_mean_sum=c,R_gt45_seed_mean_sum=r45,
   source_cells=[f'METRICS.json:scopes.RECT245.groups.{name}.frame_values.confusion_rate',f'METRICS.json:scopes.RECT245.groups.{name}.frame_values.rotation_gt45_rate'])
 stage1=rows(source('_docs/experiments/pallet_wd_hypothesis_diag_20261010/STAGE1_ROWS_REAL.jsonl.gz'))
 ids=set(internal['RECT245']['YOLO::N3_THEN_SUBPIX']['ids'])
 selected=[r for r in stage1 if r['method']=='N3_THEN_SUBPIX' and r['id'] in ids and r.get('backbone','YOLO')=='YOLO']
 assert len(selected)==735
 categories=('<5','5–10','10–20','20–30','≥30');bins=[]
 for k,label in enumerate(categories):
  bucket=[r for r in selected if int(np.digitize(r['elevation_deg'],[5,10,20,30]))==k]
  frameids=sorted({r['id'] for r in bucket});assert len(bucket)==3*len(frameids)
  arms={}
  for arm in ('S0','ORACLE'):
   values=[float(r['pose'][arm].get('available') and r['pose'][arm]['rotation_deg']>45 and abs(r['pose'][arm]['yaw_deg'])>=60) for r in bucket]
   arms[arm]=dict(confusion_percent=100*statistics.fmean(values) if values else None,confusions=sum(values),prediction_count=len(values))
  bins.append(dict(label=label,frames=len(frameids),frame_ids=frameids,**arms))
 packet['altitude_confusion_bins']=dict(method='YOLO/N3_THEN_SUBPIX',source='_docs/experiments/pallet_wd_hypothesis_diag_20261010/STAGE1_ROWS_REAL.jsonl.gz',
  oracle='GT-parity reference_long_axis; not minimum error choice',bins=bins)
 return packet

def run():
 assert (DOC/'METHOD_LOCK.json').is_file(),'No pre-results method lock'
 assert not (DOC/'METRICS.json').exists(),'Scientific aggregation outputs already exist'
 lock=read(DOC/'METHOD_LOCK.json')
 paths={pop:DOC/(pop+'_ROWS.jsonl.gz') for pop in ('RECT','SQUARE')}
 allrows={pop:rows(p) for pop,p in paths.items()}
 rr=allrows['RECT'];sr=allrows['SQUARE']
 index0={r['id']:r for r in rr if r['backbone']=='YOLO' and r['method']=='BASE' and r['seed']==1}
 assert len(index0)==319
 fullids=sorted(index0);rectids=[i for i in fullids if index0[i]['grade'] in ('clean','moderate')];assert len(rectids)==245
 master=Bootstrap(fullids,[index0[i]['session'] for i in fullids],'RECT')
 olds=read(source('_docs/experiments/pallet_square6d_manualpnp_20261010/METRICS.json'))
 squareids=olds['scopes_frame_ids']['ALL118'];assert len(squareids)==118
 square=Bootstrap(squareids,['0918_dataset']*118,'SQUARE');assert square.sha256==olds['bootstrap']['draws_sha256_uint16_le']
 scopes={'RECT245':('RECT',rectids,master),'SQUARE118':('SQUARE',squareids,square),'RECT319':('RECT',fullids,master),
  'CLEAN153':('RECT',[i for i in fullids if index0[i]['grade']=='clean'],master),'MODERATE92':('RECT',[i for i in fullids if index0[i]['grade']=='moderate'],master)}
 metrics=dict(status='COMPLETE',aggregation='continuous available seed means per ID then frame statistics; binary seed decisions then seed means; PCK pooled full reference corners',
  selection_is_post_development=True,grade_is_physical_occlusion_rate=False,scopes={})
 internal={}
 for scope,(population,ids,boot) in scopes.items():
  groups=defaultdict(list)
  for r in allrows[population]:
   if r['id'] in set(ids):groups[r['backbone']+'::'+r['method']].append(r)
  metrics['scopes'][scope]=dict(population=population,frames=len(ids),sessions=len({r['session'] for r in allrows[population] if r['id'] in set(ids)}),
   bootstrap=boot.meta(ids),ids=ids,groups={});internal[scope]={}
  for name,group in sorted(groups.items()):
   packet,ins=make_group(group,ids,boot,population);metrics['scopes'][scope]['groups'][name]=packet;internal[scope][name]=ins
 families={}
 table1=family_records(TABLE1,internal['RECT245'],'RECT245',master)+family_records(TABLE1,internal['SQUARE118'],'SQUARE118',square)
 families['TABLE_1']=adjust_family(table1)
 families['TABLE_2']=adjust_family(family_records(TABLE2,internal['RECT245'],'RECT245',master))
 families['TABLE_3']=adjust_family(family_records(TABLE3,internal['RECT245'],'RECT245',master))
 paired=dict(status='COMPLETE',tables=families,bootstrap_target='frame-weighted paired effect; PCK corner-weighted paired effect',
  t_test_target='RECT equal13session means(df12) when13common sessions; SQUARE pairedframes descriptive',
  multiplicity='Holm separately for t and bootstrap within each original table; Table1 combinesRECT+SQUARE; missing/non-applicable excluded',
  table3_duplicate_contrasts_retained=True)
 failure_packet=make_failures(internal);numbers=paper_numbers(metrics,allrows,internal)
 write('METRICS.json',metrics);write('PAIRED.json',paired);write('FAILURES.json',failure_packet)
 write('PAPER_NUMBERS.json',numbers)
 write('STATISTICS_EXECUTION.json',dict(status='COMPLETE',method_lock_sha256=sha(DOC/'METHOD_LOCK.json'),
  inputs={pop:dict(path=p.name,sha256=sha(p),rows=len(allrows[pop])) for pop,p in paths.items()},
  source_sha256=sha(Path(__file__)),additional_model_forwards=0,additional_PnP_calls=0,
  table1_tests_t=sum(r['p_t'] is not None for r in paired['tables']['TABLE_1']),table1_tests_bootstrap=sum(r['p_bootstrap'] is not None for r in paired['tables']['TABLE_1'])))
 print('PAPER_STATISTICS_COMPLETE',len(rr),len(sr),flush=True)
if __name__=='__main__':run()
