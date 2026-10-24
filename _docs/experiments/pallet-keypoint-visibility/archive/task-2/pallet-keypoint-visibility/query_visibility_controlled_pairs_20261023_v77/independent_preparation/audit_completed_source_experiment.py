"""Post-freeze V77 saved-source audit, including entry/pools and V76 baseline.

No torch/model imports or forward. Frozen artifact data are read explicitly
only when this script is run after root notification of FIT_COMPLETE.
"""
from pathlib import Path
import argparse,ast,csv,hashlib,json
import numpy as np
import audit_frozen_pair_saved as A
from pair_audit_primitives import audit_batch96,binary_metric,paired_saved_metrics,stable_bce
R=Path(__file__).resolve().parent;V=R.parent;P=V.parent;SOURCES=('SYNTHETIC','YCB_PUBLIC_REAL')

def pure(path,name):
 fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)and n.name==name)
 scope=dict(np=np);exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),scope);return scope[name]

def bindings():
 inp=A.load(V/'COMPLETED_SOURCE_INPUT_BINDINGS.json')
 for b in inp['bindings'].values():A.check(b)
 assert inp['current_reads']==0 and inp['PnP']==0 and inp['frozen_choices_unchanged']
 assert Path(inp['bindings']['original_loader_code']['path']).resolve()==(P/'query_visibility_reference_local_support_20261023_v76/mixed_fit/train_mixed_once.py').resolve()
 assert Path(inp['bindings']['after_teacher_GT']['path']).resolve()==(V/'pair_generation/AFTER_GT_ONLY.npz').resolve()
 entry=A.load(V/'TRAINER_ENTRYLOCK.json');freeze=A.load(V/'runs/PAIRED_SUPPORT_V1/MODEL_FREEZE.json');decision=A.load(V/'EXPERIMENT_DECISIONS_BEFORE_PAIR_BUILD.json')
 assert entry['trainer']==freeze['trainer']==A.bind(V/'train_paired_once.py')
 assert entry['decisions']==freeze['decisions']==A.bind(V/'EXPERIMENT_DECISIONS_BEFORE_PAIR_BUILD.json')
 assert entry['source_inputs']==freeze['source_inputs']==A.bind(V/'COMPLETED_SOURCE_INPUT_BINDINGS.json')
 assert freeze['entry']==A.bind(V/'TRAINER_ENTRYLOCK.json')and freeze['model_code']==decision['bindings']['unchanged_model_code']
 assert entry['runtime']['torch']=='2.5.1+cu121'and entry['runtime']['device']=='cuda:0'
 assert entry['restart_after_first_update']is False and entry['current_read_or_PnP']is False
 for k in ['trainer','decisions','source_inputs','generated']:A.check(entry[k])
 for b in decision['bindings'].values():A.check(b)
 return dict(PASS=True,completed_source_inputs=A.bind(V/'COMPLETED_SOURCE_INPUT_BINDINGS.json'),entry=A.bind(V/'TRAINER_ENTRYLOCK.json'),trainer=freeze['trainer'],model_code=freeze['model_code'],original_loader=inp['bindings']['original_loader_code'],same_model_entry_trainer_decision_input_bindings=True)

def sampler(data,published):
 dr,g0,g1,f0,f1,p0,p1,*_=data;orig=g1['original_row_index'];ak=g1['anchor_k'].astype(int);n=len(orig);pairgood=data[-1]
 y=np.concatenate([g0['label8'],g1['label8']]).reshape(-1);pm=np.concatenate([p0,p1]).reshape(-1)
 source=np.repeat(np.concatenate([g0['source'],g1['source']]),8);split=np.repeat(np.concatenate([g0['split'],g1['split']]),8);domain=np.r_[np.zeros(41624,int),np.ones(n*8,int)]
 pairbefore=orig*8+ak;pairafter=(5203+np.arange(n))*8+ak
 is_anchor=np.zeros(len(y),bool);is_anchor[pairafter]=True
 natural={(s,c):np.flatnonzero((domain==0)&pm&(split=='train')&(source==SOURCES[s])&(y==c))for s in range(2)for c in range(2)}
 nonanchor={(s,c):np.flatnonzero((domain==1)&~is_anchor&pm&(split=='train')&(source==SOURCES[s])&(y==c))for s in range(2)for c in range(2)}
 pairs={s:np.flatnonzero(pairgood&(g1['split']=='train')&(g1['source']==SOURCES[s]))for s in range(2)}
 meta=dict(natural_pools=natural,nonanchor_pools=nonanchor,pair_train_pools=pairs,pair_before_query=pairbefore,pair_after_query=pairafter)
 counts=dict(natural={f'{SOURCES[s]}|{c}':len(v)for(s,c),v in natural.items()},after_nonanchor={f'{SOURCES[s]}|{c}':len(v)for(s,c),v in nonanchor.items()},pairs_train={SOURCES[s]:len(v)for s,v in pairs.items()})
 assert counts==published['pools']and all(v>0 for d in counts.values()for v in d.values())
 # Replay the exact frozen sampler AST, independently check each actual source ID.
 draw=pure(V/'train_paired_once.py','sample');rng=np.random.default_rng(20261023);h=hashlib.sha256();idsamples=0
 parent=np.repeat(np.r_[np.arange(5203),orig],8)
 components=np.array(['natural_original']*32+['after_nonanchor']*32+['matched_pair']*32)
 for step in range(1,2001):
  ids=draw(meta,rng);assert pm[ids].all()and(split[ids]=='train').all()
  audit_batch96(source[ids],y[ids],components,parent[ids])
  assert (domain[ids[:32]]==0).all()and(domain[ids[32:64]]==1).all()and not is_anchor[ids[32:64]].any()
  assert (domain[ids[64:][0::2]]==0).all()and(domain[ids[64:][1::2]]==1).all()
  h.update(ids.astype('<i8').tobytes());idsamples+=len(ids)
 return dict(PASS=True,source_pools=counts,reconstructed_optimizer_batches=2000,sampled_training_query_slots=idsamples,all_train_and_primary_known=True,exact32_32_32=True,all_source_class24=True,component_source_class8=True,matched_pair_same_original_beforeV_afterC=True,after_nonanchor_excludes_anchor=True,deterministic_draws_SHA256=h.hexdigest(),note='Replays the frozen sampler against actual pools; does not read parameter gradients or run training.')

def baseline(data):
 dr,g0,g1,f0,f1,p0,p1,sq0,sq1,av0,av1,good=data;out=V/'frozen_V76_source_baseline';seal=A.load(out/'BASELINE_SEAL.json');assert seal['completed']and seal['PASS']and seal['weights_unchanged']and seal['fit_updates']==seal['current_reads']==seal['PnP']==0
 for k in ['entry','scores','results']:A.check(seal[k])
 entry=A.load(V/'FROZEN_V76_BASELINE_ENTRY.json');[A.check(b)for b in entry['bindings'].values()]
 assert entry['bindings']['loader_code']==A.bind(V/'train_paired_once.py')and entry['bindings']['source_inputs']==A.bind(V/'COMPLETED_SOURCE_INPUT_BINDINGS.json')
 f76=A.load(P/'query_visibility_reference_local_support_20261023_v76/mixed_fit/runs/MIXED_SUPPORT_V1/MODEL_FREEZE.json')
 assert entry['bindings']['checkpoint']==f76['checkpoint']and entry['bindings']['model_code']==f76['model_code']
 y=np.concatenate([g0['label8'],g1['label8']]);primary=np.concatenate([p0,p1]);sq=np.concatenate([sq0,sq1]);available=np.concatenate([av0,av1])
 with np.load(out/'SOURCE_SCORES.npz',allow_pickle=False)as z:
  for k in ['id','source','split']:assert np.array_equal(z[k],np.concatenate([g0[k],g1[k]]))
  assert np.array_equal(z['primary_loss_mask'],primary)and np.array_equal(z['reference_in_local_square_GT_only'],sq)and np.array_equal(z['input_available'],available)
  assert np.array_equal(z['domain'],np.r_[np.zeros(5203,int),np.ones(len(g1['id']),int)])
  assert np.array_equal(z['after_original_row_index'],g1['original_row_index'])and np.array_equal(z['after_anchor_k'],g1['anchor_k'])and np.array_equal(z['pair_good'],good)
  logits=z['support_logit'];nll=z['NLL_FP32'];assert logits.shape==nll.shape==y.shape
 with np.load(entry['bindings']['old_original_scores']['path'],allow_pickle=False)as z:assert np.array_equal(logits[:5203],z['support_logit'])
 assert np.isfinite(logits).all()and np.isfinite(nll).all()
 diff=abs(stable_bce(logits[primary].astype(float),y[primary])-nll[primary]);assert diff.max()<1e-5
 pub=A.load(out/'SOURCE_RESULTS.json');metrics=[]
 for name,start,g,pm in [('NATURAL_ORIGINAL',0,g0,p0),('CONTROLLED_AFTER',5203,g1,p1)]:
  z=logits[start:start+len(g['id'])];nl=nll[start:start+len(g['id'])]
  for source in SOURCES:
   for split in ['train','val']:
    m=pm&(g['source']==source)[:,None]&(g['split']==split)[:,None];r=binary_metric(z,g['label8'],m);pp=pub['results']['natural_and_after'][f'{name}|{source}|{split}']['metrics']
    assert r['N']==pp['N']and r['CM_rows_CV_cols_CV']==pp['confusion_rows_GT_cols_pred_COVERED_VISIBLE']and float(nl[m].astype(float).mean())==pp['mean_BCE_FP32']
    for k,pk in [('V_precision','visible_precision'),('V_recall','visible_recall'),('C_precision','covered_precision'),('C_recall','covered_recall')]:assert r[k]==pp[pk]
    metrics.append(dict(domain=name,source=source,split=split,**r))
 paired=paired_saved_metrics(logits[g1['original_row_index']],logits[5203:],g0['label8'][g1['original_row_index']],g1['label8'],p0[g1['original_row_index']],p1,g1['anchor_k'],g1['source'],g1['split'],g1['pair_anchor_success'])
 for r in paired:
  pp=pub['results']['paired_anchors'][r['source']+'|'+r['split']]
  assert r['attempted_pairs']==pp['attempted_rows']and r['known_admitted_V_to_C_pairs']==pp['N']and r['both_correct_known_pair']==pp['both_correct']and r['after_C_correct_known_pair']==pp['after_COVERED_correct']
 return dict(PASS=True,checkpoint=entry['bindings']['checkpoint'],baseline_seal=A.bind(out/'BASELINE_SEAL.json'),original_source_logit_rows_bitexact=41624,NLL64_vs_saved32_maxdiff=float(diff.max()),source_metrics=metrics,paired_and_same_after_control_diagnostics=paired,fit_updates_model_forward_current_PnP=0)

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--include-baseline',action='store_true');a=ap.parse_args()
 assert (V/'runs/PAIRED_SUPPORT_V1/FIT_COMPLETE.json').exists(),'Wait for root FIT_COMPLETE notification before running'
 a.output.mkdir(parents=True,exist_ok=False);binding=bindings();data=A.data_audit();pub=A.load(V/'runs/PAIRED_SUPPORT_V1/SOURCE_RESULTS.json');pool=sampler(data,pub);saved=A.saved_audit(data,V/'runs/PAIRED_SUPPORT_V1')
 A.save(a.output/'INPUT_ENTRY_AND_SAMPLER_AUDIT.json',dict(**binding,actual_sampler=pool))
 A.save(a.output/'INDEPENDENT_PAIR_DATA_AND_INPUT_FACTS_AUDIT.json',data[0])
 A.save(a.output/'INDEPENDENT_SELECTED_SAVED_AUDIT.json',dict(**saved,code=A.bind(__file__)))
 if a.include_baseline:A.save(a.output/'INDEPENDENT_FROZEN_V76_BASELINE_SAVED_AUDIT.json',baseline(data))
 A.save(a.output/'AUDIT_SEAL.json',dict(PASS=True,bindings=[A.bind(p)for p in sorted(a.output.iterdir())if p.is_file()],source_only=True,new_model_forward_fit_current_PnP=0,original_files_modified=False))
 print(json.dumps(dict(PASS=True,output=str(a.output.resolve()),selected_report=A.bind(a.output/'INDEPENDENT_SELECTED_SAVED_AUDIT.json'))))
if __name__=='__main__':main()
