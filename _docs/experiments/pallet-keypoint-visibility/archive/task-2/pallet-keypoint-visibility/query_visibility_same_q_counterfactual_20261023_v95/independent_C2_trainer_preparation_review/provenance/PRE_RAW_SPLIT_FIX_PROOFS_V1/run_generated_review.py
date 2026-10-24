"""Independent generated-only trainer math/control tests. No actual source IO."""
from pathlib import Path
import copy,hashlib,importlib.util,json,tempfile
import numpy as np
import torch
H=Path(__file__).resolve().parent;V=H.parent;T=V/'source_fit_preparation/train_source_once.py'
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
s=importlib.util.spec_from_file_location('v95_independent_trainer',T);t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
s=importlib.util.spec_from_file_location('v95_invented_fixture',V/'source_fit_preparation/test_generated_source_trainer.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g)

def run():
 torch.set_num_threads(1);lineage,ledger=g.metadata();meta=t.join_metadata(lineage,ledger);rng=np.random.default_rng(194)
 selector_checks=0
 for repeat in range(32):
  z=rng.normal(size=2880).astype(np.float32);z[repeat]=0
  actual,strata=t.selector(z,meta);means=[]
  for variant,k in [('BEFORE',0),('BEFORE',1),('CUE_PRESERVING_SHAM',0),('OPAQUE_COVER',1)]:
   values=[np.logaddexp(0.,float(z[i]))-int(meta['labels'][i])*float(z[i])for i in range(2880)if meta['split'][i]=='VALIDATION'and meta['variant'][i]==variant and meta['labels'][i]==k]
   means.append(sum(values)/len(values))
  assert abs(actual-sum(means)/4)<1e-14;selector_checks+=1
 empty=copy.deepcopy(meta);empty['variant']=meta['variant'].copy();empty['variant'][empty['variant']=='OPAQUE_COVER']='NO_SUCH_STRATUM'
 try:t.selector(np.zeros(2880,np.float32),empty)
 except AssertionError:pass
 else:raise AssertionError('Empty selector was silently omitted')
 # Direct loop 2x3 CMs and paired all3: unavailable stays in recall denominators.
 z=rng.normal(size=2880).astype(np.float32);available=rng.random(2880)>.3
 results,checks,passed=t.report(z,available,meta);pred=[(-1 if not a else int(x>=0))for x,a in zip(z,available)]
 checked=0
 for split in ['TRAIN','VALIDATION']:
  for variant in [None,*t.VARIANTS]:
   cm=[[0]*3 for _ in range(2)];N=unknown=0
   for i,(y,p)in enumerate(zip(meta['labels'],pred)):
    if meta['split'][i]!=split or(variant is not None and meta['variant'][i]!=variant):continue
    N+=1
    if y<0:unknown+=1;continue
    cm[int(y)][[0,1,-1].index(p)]+=1
   r=results[split]['all_known']if variant is None else results[split]['by_variant'][variant]
   assert r['matrix']==cm and r['known_N']==sum(map(sum,cm))and r['all_N']==N and r['unknown_GT_N']==unknown
   for y in [0,1]:
    if sum(cm[y]):assert r['D_recall'if y==0 else'E_recall']==cm[y][y]/sum(cm[y])
   checked+=1
  pairs=[i for i in range(960)if meta['pair_ready'][i]and meta['split'][3*i]==split]
  good=sum(pred[3*i:3*i+3]==[0,1,0]for i in pairs)
  assert results[split]['accepted_pairs']==len(pairs)and results[split]['all3_correct']==good
 # Execute both NO_FIT branches with invented metadata, blocking all tensor/model loaders.
 nofit=[]
 for missing_split in ['TRAIN','VALIDATION']:
  bad=copy.deepcopy(ledger)
  for i in range(960):
   if lineage[3*i]['split']==missing_split:
    original=bad[3*i]['original_natural_label_GT_only']
    for j in range(3):
     bad[3*i+j]['pair_ready']=False;bad[3*i+j]['candidate_aux_hidden_label_GT_only']=original if j==0 else -1;bad[3*i+j]['known_candidate_GT_only']=(original>=0 and j==0)
  mm=t.join_metadata(lineage,bad);assert mm['counts'][missing_split]['accepted_pairs']==0
  saved=(t.RUN,t.require_entry,t.load_metadata,t.load_inputs,t.operator,torch.cuda.is_available)
  with tempfile.TemporaryDirectory()as d:
   d=Path(d);entry=d/'entry.json';entry.write_text('{}');t.RUN=d/'generated_run'
   t.require_entry=lambda _:{};t.load_metadata=lambda *_:({},lineage,mm)
   def forbidden(*_):raise AssertionError('Tensor/model loader reached in NO_FIT')
   t.load_inputs=forbidden;t.operator=forbidden;torch.cuda.is_available=lambda:True
   try:
    t.run(entry);r=json.loads((t.RUN/'NO_FIT_FEASIBILITY_ONLY.json').read_text());assert r['fit_executed']is False and not(t.RUN/'FIT_COMPLETE.json').exists();nofit.append(missing_split)
   finally:t.RUN,t.require_entry,t.load_metadata,t.load_inputs,t.operator,torch.cuda.is_available=saved
 # Tiny-zero raw decisions, unknown supervision and score unit meanings are separate.
 assert t.decode(np.array([-1e-30,0.,1e-30],np.float32),np.array([True,True,False])).tolist()==[0,1,-1]
 proof=dict(completed=True,PASS=True,trainer=bind(T),code=bind(__file__),independent_equal4strata_scalar_replays=selector_checks,empty_stratum_rejected_not_renormalized=True,independent_loop_CM_cohorts=checked,accepted_pair_U_denominators_preserved=True,no_fit_run_branches_before_tensor_or_model_access=nofit,original_E20_BEFORE_labels_preserved=int(np.sum(meta['labels'][::3]==1))==20,raw_sign_and_U_decoding=True,actual_source_arrays_GT_weights_private_current_model_load_fit_PnP=0)
 p=H/'INDEPENDENT_GENERATED_CHECKS.json';p.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
if __name__=='__main__':run()
