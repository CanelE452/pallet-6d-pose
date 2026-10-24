"""Independent saved-hidden arithmetic check. No producer import/model/weights/features.

Only completed diagnostic tensors, saved head outputs and JSON receipts are read.
"""
from pathlib import Path
import hashlib,json
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;OUT=V/'frozen_hidden_probe_v1'
NAMES=('OLD6144','NEW20480','PALLOX512','MR1840');SIZES=(6144,20480,512,1840)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def checked(b):
 p=Path(b['path']);assert '_v112' not in str(p).lower() and bind(p)==b;return p
def packet(b):
 with np.load(checked(b),allow_pickle=False)as f:
  assert 'features'not in f.files
  return{k:f[k].copy()for k in f.files}
def eq(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def near(a,b):assert np.allclose(a,b,rtol=1e-9,atol=1e-10,equal_nan=True)
def groupmask(name,a):
 bits=name.split('::');truth,scope=bits[-2:];base=bits[:-2]
 m=np.ones(28976,bool)if base[0]=='ALL28976'else a['dataset']==base[0]
 if len(base)==2:m&=a['split']==base[1]
 assert scope in('CONSTRUCTED','INPUTVALID')and truth in('ALL_STATES','KNOWN_D','KNOWN_E')
 if truth!='ALL_STATES':m&=a['visibility_truth']==int(truth=='KNOWN_E')
 return m&(a['feature_constructed']if scope=='CONSTRUCTED'else a['query_available'])
def stats(x):
 x=np.asarray(x,np.float64);n,d=x.shape
 if n<2:return dict(N=n,defined=False),np.full((d,d),np.nan),np.full(d,np.nan)
 cov=np.cov(x,rowvar=False,ddof=1);eig=np.maximum(np.linalg.eigvalsh(cov)[::-1],0)
 total=eig.sum();fra=eig/total if total else np.zeros(d);nz=fra[fra>0]
 return dict(N=n,defined=True,mean=x.mean(0),std=x.std(0,ddof=1),positive_fraction=(x>0).mean(0),exact_nonzero_fraction=(x!=0).mean(0),
  units_ever_nonzero=int(np.any(x!=0,0).sum()),units_nonconstant_exact=int(np.any(x!=x[0],0).sum()),total_variance=total,
  numerical_rank=int((eig>eig[0]*d*np.finfo(float).eps).sum()),
  entropy_effective_rank=float(np.exp(-sum(nz*np.log(nz))))if total else 0.,participation_effective_rank=float(total**2/(eig@eig))if total else 0.,
  top_eigenvalue_fractions={str(k):float(eig[:k].sum()/total)if total else None for k in(1,5,10,20)}),cov,eig
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def main():
 seal=read(OUT/'PROBE_COMPLETE.json');assert seal['completed']and seal['PASS']and seal['selected_model']is None
 protocol=read(checked(seal['protocol']));summary=read(checked(seal['summary']))
 assert summary['completed']and summary['PASS']and len(seal['outputs'])==3
 results={};baseline=None
 excluded={'raw_three_logits','primary_visibility_prediction','secondary_entity_gated_visibility'}
 for b,st in zip(seal['outputs'],protocol['stages']):
  a=packet(b);name=st['name'];s=summary['stages'][name]
  assert a['hidden64'].shape==(28976,64)and a['raw_three_logits'].shape==(28976,3)
  assert a['hidden64'].dtype==a['raw_three_logits'].dtype==np.float32
  co=a['feature_constructed'];av=a['query_available'];assert co.sum()==28840 and av.sum()==28441 and not(av&~co).any()
  assert np.isfinite(a['hidden64'][co]).all()and np.isnan(a['hidden64'][~co]).all()
  assert np.isfinite(a['raw_three_logits'][co]).all()and np.isnan(a['raw_three_logits'][~co]).all()
  local=0
  for ds,n,original in zip(NAMES,SIZES,st['saved_outputs']):
   z=packet(original);ix=slice(local,local+n);local+=n
   assert set(z)-excluded and(a['dataset'][ix]==ds).all()
   assert eq(a['raw_three_logits'][ix],z['raw_three_logits'])
   assert eq(a['query_available'][ix],z['query_available'])
   near(a['query_xy'][ix],z['query_xy']);near(a['L'][ix],z['L'])
   ids=z['query_id']if'query_id'in z else z['occurrence_id'];assert np.array_equal(a['query_id'][ix],ids)
   assert np.array_equal(a['dataset_local_global_index'][ix],np.arange(n))
  keys=('dataset','split','visibility_truth','query_available','feature_constructed','dataset_local_global_index','query_id','query_xy','L','group_names')
  if baseline is None:baseline={k:a[k].copy()for k in keys}
  else:
   for k in keys:assert eq(a[k],baseline[k]),k
  w=a['output_weights'];length=np.sqrt((w*w).sum(1));valid=(length[:,None]!=0)&(length[None,:]!=0)
  cosine=np.full((3,3),np.nan);np.divide(w@w.T,np.outer(length,length),out=cosine,where=valid)
  near(cosine,a['output_weight_cosine']);assert np.array_equal(valid,a['output_weight_cosine_defined'])
  near(length,s['output_head_norms']);near(a['output_bias'],s['output_head_biases'])
  near(cosine,np.array([[np.nan if v is None else v for v in row]for row in s['output_head_cosine']]))
  assert set(a['group_names'])==set(s['groups'])and len(a['group_names'])==42
  for j,g in enumerate(a['group_names']):
   m=groupmask(str(g),a)
   for what,array in(('hidden',a['hidden64']),('logit',a['raw_three_logits'])):
    calc,cov,eig=stats(array[m]);r=s['groups'][str(g)]['hidden64'if what=='hidden'else'logits3']
    assert calc['N']==r['N']and calc['defined']==r['defined']
    near(cov,a[f'g{j:03d}_{what}_covariance']);near(eig,a[f'g{j:03d}_{what}_eigenvalues'])
    for k in('mean','std','positive_fraction','exact_nonzero_fraction','total_variance','entropy_effective_rank','participation_effective_rank'):near(calc[k],r[k])
    for k in('numerical_rank','units_ever_nonzero','units_nonconstant_exact'):assert calc[k]==r[k],(name,g,k)
    for k in('1','5','10','20'):near(calc['top_eigenvalue_fractions'][k],r['top_eigenvalue_fractions'][k])
  selected={}
  for g in('ALL28976::ALL_STATES::CONSTRUCTED','ALL28976::ALL_STATES::INPUTVALID','MR1840::SOURCE_VALIDATION::ALL_STATES::INPUTVALID','MR1840::SOURCE_VALIDATION::KNOWN_D::INPUTVALID','MR1840::SOURCE_VALIDATION::KNOWN_E::INPUTVALID'):
   calc,_,_=stats(a['hidden64'][groupmask(g,a)])
   selected[g]={k:calc[k]for k in('N','entropy_effective_rank','participation_effective_rank','numerical_rank','units_nonconstant_exact','top_eigenvalue_fractions')}
  # Logit correlations are observational; E-positive vs entity/quality-positive have opposite intended semantics.
  mr=(a['dataset']=='MR1840')&(a['split']=='SOURCE_VALIDATION')&av
  if name=='C36_FINAL':logitcorr=np.corrcoef(a['raw_three_logits'][mr].astype(float),rowvar=False).tolist()
  else:logitcorr=None
  results[name]=dict(all42groups_two_covariances_recomputed=True,all_original_saved_logits_query_ids_q_L_flags_match=True,
   selected_groups=selected,readout_cosine=[[float(x)if np.isfinite(x)else None for x in row]for row in cosine],MRheld_inputvalid_logit_correlations=logitcorr)
 report=dict(completed=True,PASS=True,code=bind(__file__),probe_complete=bind(OUT/'PROBE_COMPLETE.json'),protocol=seal['protocol'],
  results=results,checked_stages=3,checked_groups_each=42,checked_centered_covariances=252,rows_each=28976,selected_model=None,
  source_only=True,new_model_forward_encoder_optimizer_training_private_calls=0,source_feature_checkpoint_RGB_body_reads=0,
  saved_hidden_and_saved_readout_vector_reads=True,
  limits=['The measured distributions have concentrated variance, not exact dimension1/2: numerical rank is59/60 for broad groups.',
   'C36 increases empirical effective rank relative to C35; these data do not show additional rank collapse caused by its auxiliary training.',
   'Readout anti-alignment partly matches opposite output meanings; it does not prove harmful gradient interference or a capacity ceiling.',
   'KnownD/E source validation has93/52 available rows. Weak labels, correlated frames, and source-specific data limit generalization claims.',
   'State/forward parity claims remain the sealed producer result; this review independently recalculates saved tensors without rerunning models.'])
 write(H/'SAVED_PROBE_REVIEW.json',report)
 print(json.dumps(dict(completed=True,PASS=True,receipt=bind(H/'SAVED_PROBE_REVIEW.json'),results=results)),flush=True)
if __name__=='__main__':main()
