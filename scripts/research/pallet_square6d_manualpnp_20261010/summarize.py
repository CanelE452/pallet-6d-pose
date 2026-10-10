"""Descriptive paired-frame intervals for the fixed manual-PnP reference."""
from collections import Counter
import hashlib, math
import numpy as np
from . import common as C
NUMERIC={'T_cm':'translation_cm','R_deg':'rotation_deg','ADDsym_m':'ADDsym_m','IoU3D':'IoU3D'}
CONTRASTS={'N3_MINUS_BASE':('N3_DIM_SYM','BASE'),
 'SEQUENCE_MINUS_BASE':('N3_THEN_SUBPIX','BASE'),
 'SEQUENCE_MINUS_SUBPIX':('N3_THEN_SUBPIX','SUBPIX')}

def stats(values,CI=None):
 x=np.asarray([v for v in values if v is not None and math.isfinite(v)],float)
 if not len(x):return dict(n=0,mean=None,variance=None,std=None,median=None,P90=None,max=None,CI95=None,variance_ddof=1)
 return dict(n=len(x),mean=float(x.mean()),variance=float(x.var(ddof=1)) if len(x)>1 else None,
  std=float(x.std(ddof=1)) if len(x)>1 else None,median=float(np.median(x)),P90=float(np.quantile(x,.9)),
  max=float(x.max()),CI95=CI,variance_ddof=1)
def vectors(rr):
 v={name:[float(r['pose'][key]) if r['pose']['available'] else None for r in rr] for name,key in NUMERIC.items()}
 v['available']=[r['pose']['available'] for r in rr]
 v['success_rate']=[float(r['pose']['available'] and r['pose']['translation_cm']<5 and r['pose']['rotation_deg']<5) for r in rr]
 return v
def mean_vectors(vv):
 out={}
 for key in vv[0]:
  out[key]=[all(x) if key=='available' else float(np.mean(x)) if all(y is not None for y in x) else None for x in zip(*(v[key] for v in vv))]
 return out

def main():
 assert C.read(C.DOC/'EXECUTION.json')['status']=='COMPLETE'
 assert not (C.DOC/'METRICS.json').exists()
 rr=C.rows(C.DOC/'PREDICTIONS.jsonl.gz');index={(r['backbone'],r['seed'],r['method'],r['id']):r for r in rr}
 ids=[r['id'] for r in rr if r['backbone']=='yolo' and r['seed']==1 and r['method']=='BASE']
 assert len(ids)==118 and len(index)==len(rr)==4248
 assert set(index)=={(b,s,m,i) for b in C.BACKBONES for s in C.SEEDS for m in C.METHODS for i in ids}
 references={r['id']:r for r in C.read(C.DOC/'SQUARE_REFERENCE_POSES.json')['frames'] if r['eligible']}
 scopes={'ALL118':ids,'reference_residual_mean_le5':[i for i in ids if references[i]['available'] and references[i]['residual_mean_px']<=5],
  'manual4':[i for i in ids if references[i]['manual_count']==4],
  'manual_ge5':[i for i in ids if references[i]['manual_count']>=5]}
 rng=np.random.default_rng(20260917);draws=np.vstack([rng.multinomial(118,np.full(118,1/118),size=100) for _ in range(100)])
 draw_sha=hashlib.sha256(draws.astype('<u2').tobytes()).hexdigest();weights=draws.astype(float);position={i:j for j,i in enumerate(ids)}
 metrics=dict(status='COMPLETE',reference_type='manual_keypoint_PnP_reference',primary_frames=118,
  excluded_ids=['029844'],unique_capture_sessions=1,scopes_frame_ids=scopes,
  bootstrap=dict(level='paired frame',seed=20260917,resamples=10000,descriptive_only=True,draws_sha256_uint16_le=draw_sha,units=118,
   scopes_use_same_master_frame_draw=True,cluster_bootstrap=False),backbones={})
 paired=dict(status='DESCRIPTIVE_ONLY',primary_frames=118,bootstrap=metrics['bootstrap'],backbones={})
 failures=dict(status='COMPLETE',excluded_ids=['029844'],reference_unavailable_ids=[i for i in ids if not references[i]['available']],backbones={})
 for b in C.BACKBONES:
  metrics['backbones'][b]={};paired['backbones'][b]={};failures['backbones'][b]={}
  for scope,selected in scopes.items():
   w=weights[:,[position[i] for i in selected]];ci_cache={}
   def interval(values):
    key=tuple(values)
    if key in ci_cache:return ci_cache[key]
    ok=np.array([v is not None and math.isfinite(v) for v in values]);x=np.array([float(v) if k else 0. for v,k in zip(values,ok)])
    if not ok.any():return None
    denom=w@ok.astype(float);num=w@x;samples=num[denom>0]/denom[denom>0]
    result=np.quantile(samples,[.025,.975]).tolist();ci_cache[key]=result;return result
   def summary(v):
    return dict(frames=len(selected),available=sum(v['available']),unavailable=len(selected)-sum(v['available']),
     coverage=sum(v['available'])/len(selected),
     metrics={k:stats(v[k],interval(v[k])) for k in NUMERIC},
     success_rate=dict(rate=float(np.mean(v['success_rate'])),positive_count_sum=float(np.sum(v['success_rate'])),
      denominator=len(selected),CI95=interval(v['success_rate']),binary_before_seed_mean=True))
   vv={(s,m):vectors([index[(b,s,m,i)] for i in selected]) for s in C.SEEDS for m in C.METHODS}
   packet=dict(by_seed={str(s):{m:summary(vv[(s,m)]) for m in C.METHODS} for s in C.SEEDS},
    seed_mean={m:summary(mean_vectors([vv[(s,m)] for s in C.SEEDS])) for m in C.METHODS})
   metrics['backbones'][b][scope]=packet;paired['backbones'][b][scope]={};failures['backbones'][b][scope]={}
   for name,(left,right) in CONTRASTS.items():
    out=dict(by_seed={},seed_mean={},damage_by_seed={});failing={}
    for key in (*NUMERIC,'success_rate'):
     deltas=[[a-z if a is not None and z is not None else None for a,z in zip(vv[(s,left)][key],vv[(s,right)][key])] for s in C.SEEDS]
     dmean=[float(np.mean(x)) if all(v is not None for v in x) else None for x in zip(*deltas)]
     out['seed_mean'][key]=dict(delta=stats(dmean)['mean'],CI95=interval(dmean),paired_frames=sum(v is not None for v in dmean),
      per_seed_delta=[stats(x)['mean'] for x in deltas],lower_is_better=key not in ('success_rate','IoU3D'))
     for s,delta in zip(C.SEEDS,deltas):
      out['by_seed'].setdefault(str(s),{})[key]=dict(delta=stats(delta)['mean'],CI95=interval(delta),paired_frames=sum(v is not None for v in delta))
    for s in C.SEEDS:
     damage=[i for i,a,z in zip(selected,vv[(s,left)]['success_rate'],vv[(s,right)]['success_rate']) if z and not a]
     recovery=[i for i,a,z in zip(selected,vv[(s,left)]['success_rate'],vv[(s,right)]['success_rate']) if a and not z]
     missing_left=[i for i,a in zip(selected,vv[(s,left)]['available']) if not a]
     missing_right=[i for i,a in zip(selected,vv[(s,right)]['available']) if not a]
     out['damage_by_seed'][str(s)]=dict(success_to_failure=len(damage),failure_to_success=len(recovery),
      success_to_failure_ids=damage,failure_to_success_ids=recovery,
      unavailable_left_ids=missing_left,unavailable_right_ids=missing_right)
     failing[str(s)]=out['damage_by_seed'][str(s)]
    paired['backbones'][b][scope][name]=out;failures['backbones'][b][scope][name]=failing
  failures['backbones'][b]['all_method_unavailable_ids']={str(s):{m:[i for i in ids if not index[(b,s,m,i)]['pose']['available']] for m in C.METHODS} for s in C.SEEDS}
 C.write('METRICS.json',metrics);C.write('PAIRED.json',paired);C.write('FAILURES.json',failures)
 C.write('VERDICT.json',dict(status='FEASIBILITY_ONLY',reason='One capture session; manual-keypoint PnP reference is a proxy, frame bootstrap descriptive only',
  residual_gate='PASS',confirmatory_claim=False,rectangle_results_combined=False,square_90_degree_WD_swap_equivalent_under_C4=True))
 print('SQUARE_STATISTICS_COMPLETE',draw_sha,flush=True)
if __name__=='__main__':main()
