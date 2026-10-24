"""Independent V78 saved-source audit. NumPy only, no model/torch/current IO.

Uses existing source GT and compact coordinate/validity metadata, never
RGB/P3/DINO vectors or model weights. Checkpoint files are hashed only.
"""
from pathlib import Path
import argparse,csv,hashlib,json,sys
import numpy as np
R=Path(__file__).resolve().parent;V=R.parent;P=V.parent;V77=P/'query_visibility_controlled_pairs_20261023_v77';RUN=V/'runs/SOURCE_DINO_V1'
sys.path.insert(0,str(V77/'independent_preparation'))
from pair_audit_primitives import binary_metric,stable_bce,audit_natural_selector,paired_saved_metrics
from audit_completed_source_experiment import sampler as replay_old_sampler
SOURCES=('SYNTHETIC','YCB_PUBLIC_REAL')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def load(p):return json.loads(Path(p).read_text())
def check(b):assert sha(b['path'])==b['sha256'],b['path']
def save(p,d):
 with p.open('x')as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def same(a,b):return np.array_equal(a,b,equal_nan=True)if a.dtype.kind in 'fc'else np.array_equal(a,b)
def npz(p,keys):
 with np.load(p,allow_pickle=False)as z:return {k:z[k]for k in keys}

def metric(z,nll,y,m):
 r=binary_metric(z,y,m)
 return dict(N=r['N'],confusion_rows_GT_cols_pred_COVERED_VISIBLE=r['CM_rows_CV_cols_CV'],visible_precision=r['V_precision'],visible_recall=r['V_recall'],covered_precision=r['C_precision'],covered_recall=r['C_recall'],mean_BCE_FP32=float(nll[m].astype(float).mean())if m.any()else None)

def actual(output):
 output.mkdir(parents=True,exist_ok=False)
 complete=load(RUN/'FIT_COMPLETE.json');freeze=load(RUN/'MODEL_FREEZE.json');entry=load(V/'TRAINER_ENTRYLOCK.json');inputs=load(V/'COMPLETED_DINO_INPUT_BINDINGS.json');dec=load(V/'EXPERIMENT_DECISIONS_BEFORE_FEATURES.json');published=load(RUN/'SOURCE_RESULTS.json');history=load(RUN/'HISTORY.json')
 assert sha(V/'EXPERIMENT_DECISIONS_BEFORE_FEATURES.json')=='a008fc7d35e9aa1d0d4b7d631a796f238aaba7f0aa052fe7da087080b70115e9'
 assert complete['completed']and freeze['completed']and complete['optimizer_updates']==freeze['optimizer_updates']==2000
 assert freeze['parameter_count']==48929 and freeze['threshold']==.5 and freeze['random_initialization']is True and freeze['current_used_for_fit_or_checkpoint_or_threshold_selection']is False and freeze['PnP']==0
 for k in ['checkpoint','model_code','decisions','trainer','entry','source_inputs']:check(freeze[k])
 for k in ['model_freeze','history','scores','source_results']:check(complete[k])
 assert freeze['entry']==bind(V/'TRAINER_ENTRYLOCK.json')and freeze['source_inputs']==bind(V/'COMPLETED_DINO_INPUT_BINDINGS.json')
 for k in ['trainer','model_code','decisions','source_inputs']:assert entry[k]==freeze[k]
 assert entry['approved']and not entry['restart_after_first_update']and not entry['current_read_or_PnP']
 assert entry['forward_fields']==['local_features','target_features','context42','validity_fractions']and entry['runtime']['torch']=='2.5.1+cu121'
 for k in ['generated','head_generated','authorization']:check(entry[k])
 auth=load(entry['authorization']['path']);assert auth['approved']and auth['allow_one_source_fit']and auth['current_reads_allowed']is False
 assert inputs['completed']and inputs['frame_count']==10002 and inputs['query_slots']==80016 and inputs['no_GT_forward']and inputs['current_reads']==inputs['PnP']==0
 for b in inputs['bindings'].values():check(b)
 for b in dec['bindings'].values():check(b)
 assert load(inputs['bindings']['feature_QA']['path'])['PASS']and load(V/'independent_feature_audit/INDEPENDENT_FEATURE_QA.json')['PASS']
 man=load(inputs['bindings']['feature_manifest']['path']);fields=['query8_xy','L','query_center_valid','ROI_anyvalid','localany','targetany','id','source','split','domain','original_row_index','frame_index'];parts={k:[]for k in fields}
 for chunk in man['chunks']:
  check(chunk['file'])
  with np.load(chunk['file']['path'],allow_pickle=False)as z:
   for k in fields:parts[k].append(z[k])
 d={k:np.concatenate(v)for k,v in parts.items()};assert np.array_equal(d['frame_index'],np.arange(10002))
 old_dec=load(dec['bindings']['V77_decisions']['path']);old_in=load(dec['bindings']['V77_source_inputs']['path']);check(old_dec['bindings']['teacher_GT']);check(old_in['bindings']['after_teacher_GT'])
 keys=['id','source','split','label8','native_GTref8','convex_self_proxy8_separate','near_boundary8','rule8']
 g0=npz(old_dec['bindings']['teacher_GT']['path'],keys);g1=npz(old_in['bindings']['after_teacher_GT']['path'],keys+['original_row_index','after_row_index','anchor_k','pair_anchor_success','anchor_training_target','original_primary8','query_valid8','reference_in_local_square8'])
 gt={k:np.concatenate([g0[k],g1[k]])for k in keys};orig=g1['original_row_index'];ak=g1['anchor_k'].astype(int);rr=np.arange(4799)
 assert len(g0['id'])==5203 and len(g1['id'])==4799 and len(np.unique(gt['id']))==10002
 for k in ['id','source','split']:assert np.array_equal(d[k],gt[k]),k
 assert np.array_equal(d['domain'],np.r_[np.zeros(5203,int),np.ones(4799,int)])and np.array_equal(d['original_row_index'],np.r_[np.arange(5203),orig])
 for k in ['query8_xy','L','query_center_valid','ROI_anyvalid','localany','targetany']:assert same(d[k][5203:],d[k][orig]),k
 for k in ['source','split','native_GTref8','convex_self_proxy8_separate']:assert same(g1[k],g0[k][orig]),k
 assert np.isfinite(d['L']).all()and(d['L']>0).all();offset=(gt['native_GTref8']-d['query8_xy'])/d['L'][:,None,None];sq=np.isfinite(offset).all(-1)&(offset>=-.5).all(-1)&(offset<.5).all(-1);available=d['ROI_anyvalid']&d['localany']&d['targetany'][:,None];known=np.isin(gt['label8'],[0,1]);primary=known&d['query_center_valid']&available&sq
 assert np.array_equal(g1['original_primary8'],primary[:5203][orig])and np.array_equal(g1['query_valid8'],d['query_center_valid'][5203:])and np.array_equal(g1['reference_in_local_square8'],sq[5203:])
 assert np.array_equal(orig,np.flatnonzero((primary[:5203]&(g0['label8']==1)).any(-1)))and np.all(primary[orig,ak]&(g0['label8'][orig,ak]==1))
 assert np.array_equal(g1['anchor_training_target'],np.where(g1['pair_anchor_success'],0,-1));good=g1['pair_anchor_success']&(g1['anchor_training_target']==0)&primary[5203:][rr,ak]&(g1['label8'][rr,ak]==0)
 data=(None,g0,g1,None,None,primary[:5203],primary[5203:],sq[:5203],sq[5203:],available[:5203],available[5203:],good);sampling=replay_old_sampler(data,published)
 oldsampler=load(V77/'independent_saved_audit/INPUT_ENTRY_AND_SAMPLER_AUDIT.json')['actual_sampler'];assert sampling['deterministic_draws_SHA256']==oldsampler['deterministic_draws_SHA256']
 with np.load(RUN/'SELECTED_SOURCE_SCORES.npz',allow_pickle=False)as z:
  for k in ['id','source','split']:assert np.array_equal(z[k],gt[k])
  for k,a in [('domain',d['domain']),('primary_loss_mask',primary),('reference_in_local_square_GT_only',sq),('input_available',available),('after_original_row_index',orig),('after_anchor_k',ak),('pair_good',good)]:assert np.array_equal(z[k],a),k
  logits=z['support_logit'];nll=z['NLL_FP32'];assert logits.shape==nll.shape==(10002,8)
 assert np.isfinite(logits).all()and np.isfinite(nll).all();err=abs(stable_bce(logits[primary].astype(float),gt['label8'][primary])-nll[primary]);assert err.max()<1e-5
 selector=audit_natural_selector(history,metric_key='naturalVAL4strata_macroBCE',strata_key='stratumBCE')
 assert selector['best_update']==freeze['best_update']==complete['best_update']==published['best_update']and selector['natural4strata_macroBCE']==freeze['best_naturalVAL4strata_macroBCE']==published['selection_naturalVALmacroBCE']
 rows=[];strata=[];counts=[]
 for domain,name in [(0,'NATURAL_ORIGINAL'),(1,'CONTROLLED_AFTER')]:
  for source in SOURCES:
   for split in ['train','val']:
    base=np.broadcast_to(((d['domain']==domain)&(gt['source']==source)&(gt['split']==split))[:,None],primary.shape);m=base&primary;met=metric(logits,nll,gt['label8'],m);pp=published['results']['natural_and_after'][f'{name}|{source}|{split}'];assert met==pp['metrics'],(name,source,split)
    c=dict(all_slots=int(base.sum()),known=int((base&known).sum()),admitted=int(m.sum()),U=int((base&(gt['label8']==-1)).sum()),known_ref_outside_square=int((base&known&~sq).sum()));assert all(pp[k]==v for k,v in c.items())
    conds=dict(strong=np.char.startswith(gt['rule8'].astype(str),'STRONG'),fallback=~np.char.startswith(gt['rule8'].astype(str),'STRONG'),convex_self_hidden=gt['convex_self_proxy8_separate']==0,convex_may_visible=gt['convex_self_proxy8_separate']==1)
    for k,cnd in conds.items():assert metric(logits,nll,gt['label8'],m&cnd)==pp['subgroups'][k]
    rows.append(dict(domain=name,source=source,split=split,metrics=met,counts=c))
    if domain==0 and split=='val':
     for cls in [0,1]:
      take=m&(gt['label8']==cls);strata.append(dict(source=source,label=cls,N=int(take.sum()),BCE_FP32=float(nll[take].astype(float).mean())))
 macro=float(np.mean([r['BCE_FP32']for r in strata]));assert abs(macro-selector['natural4strata_macroBCE'])<1e-6
 pairs=paired_saved_metrics(logits[orig],logits[5203:],g0['label8'][orig],g1['label8'],primary[:5203][orig],primary[5203:],ak,g1['source'],g1['split'],g1['pair_anchor_success'])
 for r in pairs:
  pp=published['results']['paired_anchors'][r['source']+'|'+r['split']]
  for k,pk in [('attempted_pairs','attempted_rows'),('failed_or_unknown_or_invalid_anchor_rows','failed_or_unknown_or_invalid_anchor_rows'),('known_admitted_V_to_C_pairs','N'),('both_correct_known_pair','both_correct'),('before_V_correct_known_pair','before_VISIBLE_correct'),('after_C_correct_known_pair','after_COVERED_correct')]:assert r[k]==pp[pk]
 with np.load(RUN/'BASELINE0_SOURCE_SCORES.npz',allow_pickle=False)as z:
  assert z['support_logit'].shape==(10002,8)and not z['support_logit'].any()and np.array_equal(z['primary_loss_mask'],primary)
  assert np.max(abs(z['NLL_FP32'][primary]-np.log(2)))<1e-7
 assert load(RUN/'BASELINE0_RESULTS.json')['selection_eligible']is False
 public=next(r['metrics']for r in rows if r['domain']=='NATURAL_ORIGINAL'and r['source']=='YCB_PUBLIC_REAL'and r['split']=='val');pair=next(r for r in pairs if r['source']=='YCB_PUBLIC_REAL'and r['split']=='val')
 criteria=dict(public_natural_C_precision_ge_0_50=dict(value=public['covered_precision'],minimum=.5),public_natural_C_recall_ge_0_80=dict(value=public['covered_recall'],minimum=.8),public_natural_V_recall_ge_0_80=dict(value=public['visible_recall'],minimum=.8),public_controlled_both_correct_ge_0_70=dict(value=pair['both_correct_known_pair']/pair['known_admitted_V_to_C_pairs'],minimum=.7))
 for c in criteria.values():c['PASS']=c['value']>=c['minimum']
 gate=V/'SOURCE_ADMISSION_DECISION.json';gate_compared=False
 if gate.exists():
  gp=load(gate);gate_compared=True
  assert gp['source_candidate_pass']==all(c['PASS']for c in criteria.values())and gp['threshold']==.5 and gp['threshold_or_checkpoint_reselected']is False and gp['current_reads']==gp['current_inference']==0
  for a,b in zip(criteria.values(),gp['checks'].values()):assert a==b
 report=dict(PASS=True,code=bind(__file__),scope='Saved source-only conditional reference-local support; no current data or new model execution.',checkpoint=freeze['checkpoint'],parameter_count=48929,random_initialization_lastzero_diagnostic=True,selection=selector,saved_allslot_natural4strata_BCE=macro,saved_minus_history=macro-selector['natural4strata_macroBCE'],NLL64_vs_saved_FP32_maxdiff_primary_only=float(err.max()),full_frames=10002,full_slots=80016,unchanged_original_frames=5203,unchanged_after_frames=4799,all_join_and_GT_admission_exact=True,all_published_confusion_precision_recall_NLL_subgroups_counts_exact=True,source_metrics=rows,natural_VAL_strata=strata,actual_sampler=sampling,paired_and_same_after_control_diagnostics=pairs,prospective_source_gate=criteria,prospective_source_gate_PASS=all(c['PASS']for c in criteria.values()),root_source_gate_reconciled=gate_compared,U_NLL_placeholder_never_used=True,feature_embeddings_RGB_P3_weights_decoded=0,model_forward_optimizer_current_PnP=0,bindings=[bind(p)for p in [RUN/'FIT_COMPLETE.json',RUN/'MODEL_FREEZE.json',RUN/'HISTORY.json',RUN/'SELECTED_SOURCE_SCORES.npz',RUN/'SOURCE_RESULTS.json',V/'TRAINER_ENTRYLOCK.json',V/'COMPLETED_DINO_INPUT_BINDINGS.json',V/'independent_feature_audit/INDEPENDENT_FEATURE_QA.json',V/'independent_preparation/CODE_AND_PROTOCOL_REVIEW.json']])
 save(output/'INDEPENDENT_SELECTED_SAVED_AUDIT.json',report)
 with (output/'SOURCE_METRICS.csv').open('x',newline='')as f:
  rr=[dict(domain=r['domain'],source=r['source'],split=r['split'],**r['metrics'])for r in rows];w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 save(output/'AUDIT_SEAL.json',dict(PASS=True,bindings=[bind(p)for p in sorted(output.iterdir())if p.is_file()],original_files_modified=False));print(json.dumps(dict(PASS=True,report=bind(output/'INDEPENDENT_SELECTED_SAVED_AUDIT.json'),source_gate=criteria,selection=selector)))

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();actual(a.output)
if __name__=='__main__':main()
