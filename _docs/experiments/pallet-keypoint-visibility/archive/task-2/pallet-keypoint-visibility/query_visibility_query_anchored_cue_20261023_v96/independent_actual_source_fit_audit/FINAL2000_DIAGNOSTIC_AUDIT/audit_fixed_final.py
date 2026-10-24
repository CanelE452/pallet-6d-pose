"""Saved FINAL2000 diagnosis only. Imports independent math, no model/trainer."""
from pathlib import Path
import importlib.util,json
import numpy as np
H=Path(__file__).resolve().parent;A=H.parent;V=A.parent;RUN=V/'runs/C3_QUERY_ANCHORED_SOURCE_V1';D=V/'source_diagnosis/FINAL2000_FIXED_NOT_SELECTOR'
s=importlib.util.spec_from_file_location('own_metrics_for_fixed_final',A/'audit_completed_source_fit.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
def main():
 assert not(H/'REVIEW.json').exists()
 assert a.sha(D/'DIAGNOSIS_COMPLETE.json')=='6f632bee8e1e50a61724fb2863878139aec7e6a45380670a740b892b58349955'
 done=a.read(D/'DIAGNOSIS_COMPLETE.json');assert done['completed']and done['PASS_execution']and done['diagnostic_only']and done['selected_update_remains']==0
 plan=a.read(a.check(done['plan']));assert plan['fixed_update']==2000 and plan['selected_update_remains']==0
 a.check(done['fixed_checkpoint']);freeze=a.read(a.check(done['authoritative_selected_source_freeze']));assert freeze['selected_update']==0 and a.same_binding(freeze['final_checkpoint'],done['fixed_checkpoint'])
 for name,b in plan['original_run_files_before'].items():a.check(b);assert (RUN/name).resolve()==Path(b['path']).resolve()
 assert done['all_original_run_files_unchanged']and done['model_unchanged']
 entry=a.read(a.check(freeze['entry']));manifest=a.read(a.check(entry['input_pack_manifest']));lineage=a.rows(a.check(manifest['input_lineage']));gt=a.rows(a.check(entry['GT_ledger']));m=a.metadata(lineage,gt)
 final,z,av=a.audit_scores(a.check(done['scores']),lineage,m);pub=a.read(a.check(done['results']))
 a.equal(pub['metrics'],final['metrics'],'final/metrics');a.equal(pub['counts'],a.counts(m),'final/counts');a.equal(pub['diagnostic_existing_seven_gate_checks'],final['gates'],'final/diagnostic_gates')
 assert pub['diagnostic_existing_seven_checks_allpass']==final['all_gates_pass']and pub['fixed_diagnostic_update']==2000 and pub['authoritative_selected_update']==0 and pub['diagnostic_not_selector']
 a.equal(pub['five_VAL_cohorts_BCE_diagnostic'],final['criterion']);a.equal(pub['VAL_strata'],final['VAL_cohorts'])
 history=a.rows(RUN/'HISTORY.jsonl');last=next(h for h in history if h['update']==2000);a.equal(last['criterion_five_VAL_cohorts_BCE'],final['criterion']);a.equal(pub['history_criterion_difference'],0.)
 clustering={}
 for split in ['TRAIN','VALIDATION']:
  base=(m['split'][::33]==split);ready=m['ready']&base[:,None]
  clustering[split]=dict(accepted_pairs=int(ready.sum()),distinct_base_queries_with_accepted_rank=int(ready.any(1).sum()),rank_pair_counts_per_base={str(i):int(ready[i].sum())for i in np.flatnonzero(ready.any(1))},families={source:int((ready&(m['source'][::33]==source)[:,None]).sum())for source in ['LOCO','PALLOX']})
 assert clustering['TRAIN']['accepted_pairs']==75 and clustering['TRAIN']['distinct_base_queries_with_accepted_rank']==10
 assert clustering['VALIDATION']['accepted_pairs']==119 and clustering['VALIDATION']['distinct_base_queries_with_accepted_rank']==18
 c3done=a.read(RUN/'FIT_COMPLETE.json');c1done=a.read(V/'source_baselines/C1_FROZEN_ON_V96/BASELINE_COMPLETE.json')
 zs={};available={}
 for name,path in [('C1',a.check(c1done['scores'])),('C3_SELECTED0',a.check(c3done['scores'])),('C3_FIXED_FINAL2000_DIAGNOSTIC',a.check(done['scores']))]:
  with np.load(path,allow_pickle=False)as f:zs[name]=np.array(f['query_logit']);available[name]=np.array(f['query_available'])
  assert np.array_equal(available[name],av)
 paired=[]
 for i in range(960):
  for k in range(16):
   if not m['ready'][i,k]:continue
   idx=[33*i,33*i+1+2*k,33*i+2+2*k];models={}
   for name,zz in zs.items():
    pred=a.predictions(zz[idx],av[idx]);models[name]=dict(logits_BEFORE_COVER_SHAM=zz[idx].tolist(),predictions_D0_E1_Uminus1=pred.tolist(),all3_correct=pred.tolist()==[0,1,0])
   paired.append(dict(base_row_index=i,donor_rank=k,source=m['source'][33*i],raw_split=m['raw'][33*i],original_q=lineage[33*i]['query_xy'],original_L=lineage[33*i]['L'],models=models))
 assert len(paired)==194
 with(H/'ALL194_PAIR_C1_SELECTED0_FIXEDFINAL_COMPARISON.jsonl').open('x')as f:
  for r in paired:f.write(json.dumps(r)+'\n')
 report=dict(completed=True,PASS=True,discrepancies=0,diagnosis_complete=a.bind(D/'DIAGNOSIS_COMPLETE.json'),selected_source_freeze=done['authoritative_selected_source_freeze'],code=a.bind(__file__),scope=a.bind(H/'SCOPE_BEFORE_FINAL_DIAGNOSTIC_BODIES.json'),fixed_diagnostic_update=2000,authoritative_selected_update_unchanged=0,all31680_scores_and_metrics_exact=True,criterion=final['criterion'],original2000_history_criterion=last['criterion_five_VAL_cohorts_BCE'],history_difference=0.,metrics=final['metrics'],diagnostic_seven_gates=final['gates'],pair_clustering=clustering,all194_models_pair_table=a.bind(H/'ALL194_PAIR_C1_SELECTED0_FIXEDFINAL_COMPARISON.jsonl'),
  limitations=['194 accepted rank pairs are clustered within28 basequeries: TRAIN75from10 andVAL119from18. They are not194 independent scenes or captures.','All controlled pairs are Pallox, one physical tall bin. LOCOzero controlled pairs remains untested, not successful flat-pallet E transfer.','FINAL2000 is a fixed postfit diagnosis of overfit/transfer failure; it never replaces selected0 or changes prospective gates.','Natural source cue annotations are conditional AI; native expectedpart UNKNOWN and human gold false.','No parameter reload/forward: freeze/checkpoint hashes and saved tensor outputs replayed only.'],new_model_optimizer_forward_current_PnP=0)
 a.write(H/'REVIEW.json',report);a.write(H/'SEAL.json',dict(completed=True,PASS=True,files=[a.bind(p)for p in sorted(H.iterdir())if p.is_file()]))
 print(json.dumps(dict(PASS=True,report=a.bind(H/'REVIEW.json'),seal=a.bind(H/'SEAL.json'),pair_clustering=clustering)))
if __name__=='__main__':main()
