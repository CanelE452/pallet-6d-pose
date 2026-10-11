"""Independent exact rational audit of amended bootstrap p values.

The original frozen verifier checks all moments, CIs, PCK, AUC, failure IDs
and main CSV cells. Only its paired-p audit is replaced: discrete tail
membership is calculated with Python Fraction, independent of the producer's
int64 sign cases and integer cross products.
"""
from __future__ import annotations
from fractions import Fraction
import json
from pathlib import Path
import numpy as np
from . import verify_stats as V

DISCRETE=('PCK10','success_rate','confusion_rate','ADDsym_AUC')

def rational_inputs(left,right,metric):
 seeds=left['seeds'];assert seeds==[1,2,3] and right['seeds']==seeds
 if metric=='PCK10':
  numerator=np.rint(sum(left['mats'][s][:,11]-right['mats'][s][:,11] for s in seeds)).astype(np.int64)
  denominator=np.rint(left['mean'][:,10]*len(seeds)).astype(np.int64)
 else:
  j=V.METRICS.index(metric);scale=2000 if metric=='ADDsym_AUC' else 1
  a=np.stack([left['mats'][s][:,j] for s in seeds]);b=np.stack([right['mats'][s][:,j] for s in seeds])
  av=np.rint(a*scale).astype(np.int64);bv=np.rint(b*scale).astype(np.int64)
  assert np.allclose(av/scale,a,rtol=0,atol=1e-12) and np.allclose(bv/scale,b,rtol=0,atol=1e-12)
  numerator=(av-bv).sum(0);denominator=np.full(len(numerator),len(seeds)*scale,dtype=np.int64)
 return numerator,denominator

def fractional_p(numerator,denominator,sessions,draw):
 session=np.asarray(sessions);labels=draw.labels
 groups=[np.flatnonzero(session==s) for s in labels]
 total_n=[sum(int(numerator[j]) for j in indices) for indices in groups]
 total_d=[sum(int(denominator[j]) for j in indices) for indices in groups]
 full=Fraction(sum(total_n),sum(total_d));tail=0;valid=0
 for counts in draw.counts:
  n=sum(int(w)*v for w,v in zip(counts,total_n));d=sum(int(w)*v for w,v in zip(counts,total_d))
  if d<=0:continue
  value=Fraction(n,d);valid+=1
  if abs(value-full)>=abs(full):tail+=1
 return float(full),(tail+1)/(valid+1),tail,valid

def verify_pairs(paired,data,draws):
 for family,contrasts in (('TABLE_1',V.CONTRASTS1),('TABLE_2',V.CONTRASTS2),('TABLE_3',V.CONTRASTS3)):
  given=paired['tables'][family];expectedrecords=[]
  assert len(given)==7*len(contrasts)
  for (scope,backbone,left,right) in contrasts:
   ga=data[scope][backbone+'::'+left];gb=data[scope][backbone+'::'+right];ids=ga['ids'];sessions=np.array(ga['sessions']);draw=draws[scope]
   a,b=ga['mean'],gb['mean']
   for metric in V.TEST_METRICS:
    if metric=='confusion_rate' and scope=='SQUARE118':expectedrecords.append(dict(p_t=None,p_bootstrap=None));continue
    if metric=='PCK10':
     num=a[:,11]-b[:,11];den=a[:,10];delta=float(num.sum()/den.sum());diff=num/den;valid=np.ones(len(ids),bool)
     sampled=draw.samples(ids,num,den)
    else:
     idx=V.METRICS.index(metric);diff=a[:,idx]-b[:,idx];valid=np.isfinite(diff);delta=float(diff[valid].mean()) if valid.any() else None
     num=np.where(valid,diff,0.);den=valid.astype(float);sampled=draw.samples(ids,diff)
    if scope!='SQUARE118':
     sessionmeans=[]
     for s in sorted(set(sessions)):
      mask=sessions==s
      if den[mask].sum()>0:sessionmeans.append(float(num[mask].sum()/den[mask].sum()))
     pt,df=V.p_t(sessionmeans) if len(sessionmeans)==13 else (None,12)
     if metric in DISCRETE:
      exact_n,exact_d=rational_inputs(ga,gb,metric);delta,pb,tail,n_draws=fractional_p(exact_n,exact_d,sessions,draw)
     else:pb=(1+np.count_nonzero(np.abs(sampled-delta)>=abs(delta)))/(len(sampled)+1) if delta is not None and len(sampled) else None
    else:
     pt,df=V.p_t(diff);pb=None
     if metric in DISCRETE:
      exact_n,exact_d=rational_inputs(ga,gb,metric);delta=float(Fraction(int(exact_n.sum()),int(exact_d.sum())))
    expectedrecords.append(dict(delta=delta,CI95=np.quantile(sampled,[.025,.975]).tolist() if len(sampled) else None,
     paired_frames=int(valid.sum()),p_t=pt,p_bootstrap=pb,df=df))
  for j,(actual,expect) in enumerate(zip(given,expectedrecords)):
   for key,value in expect.items():
    if key=='df':V.check(actual['t_test']['df'],value,f'{family}[{j}].df')
    else:V.check(actual.get(key),value,f'{family}[{j}].'+key)
  for pfield in ('p_t','p_bootstrap'):
   ps=[r[pfield] for r in expectedrecords];mask=np.array([p is not None for p in ps]);positions=np.flatnonzero(mask);values=np.array([ps[i] for i in positions])
   order=np.argsort(values,kind='stable');adjusted=np.minimum(1.,np.maximum.accumulate(values[order]*(len(values)-np.arange(len(values)))))
   corrected={int(positions[i]):float(value) for i,value in zip(order,adjusted)}
   for j,actual in enumerate(given):
    V.check(actual[pfield+'_holm'],corrected.get(j),f'{family}[{j}].'+pfield+'_holm')
    V.check(actual[pfield+'_holm_family_size'],len(values),f'{family}[{j}].family_size')

def run():
 amendment=V.read(V.DOC/'NUMERICAL_AMENDMENT.json');assert amendment['status']=='COMPLETE_NUMERICAL_REPAIR'
 for binding in amendment['current_bindings']:assert V.sha(V.ROOT/binding['path'])==binding['sha256'],binding['path']
 assert V.sha(V.DOC/amendment['original_verifier_archived_path'])==amendment['original_verifier_sha256']
 assert V.sha(V.DOC/'PAIRED.json')==amendment['corrected_PAIRED_sha256']
 V.verify_pairs=verify_pairs
 V.run()
 path=V.DOC/'STATISTICS_VERIFICATION.json';receipt=V.read(path)
 receipt.update(amended_verifier_source_path=str(Path(__file__).relative_to(V.ROOT)),amended_verifier_source_sha256=V.sha(Path(__file__)),
  original_frozen_verifier_source_sha256=V.sha(Path(V.__file__)),numerical_amendment_sha256=V.sha(V.DOC/'NUMERICAL_AMENDMENT.json'),
  exact_discrete_p_implementation='Independent fractions.Fraction per draw, including exactzero and inclusive equality',
  unchanged_original_producer_sources=True)
 pending=path.with_suffix('.json.pending')
 with pending.open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n')
 pending.replace(path)
 print('EXACT_RATIONAL_AMENDED_VERIFICATION_PASS',V.CHECKS,V.MAX_ERROR,flush=True)
if __name__=='__main__':run()
