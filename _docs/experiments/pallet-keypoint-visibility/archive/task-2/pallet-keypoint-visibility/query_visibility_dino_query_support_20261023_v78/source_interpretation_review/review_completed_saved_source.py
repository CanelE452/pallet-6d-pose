"""Code prepared before V78 fit. Execute only after parent completion notification.

Reads sealed source scores/GT identities and already fixed V77 cohorts. Never imports
Torch/model/trainer or reads RGB/mask/latent feature/current evaluation files.
"""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;P=ROOT.parent
V77=P/'query_visibility_controlled_pairs_20261023_v77'
V76=P/'query_visibility_reference_local_support_20261023_v76'
RUN=ROOT/'runs/SOURCE_DINO_V1';DIAG=ROOT/'fixed_geometry_source_diagnosis'
COHORT='STRONG_AND_ERROR_LE3PX_AND_CENTER_CELL'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path']
def load(p):return json.loads(Path(p).read_text())
def npz(p,keys):
 with np.load(p,allow_pickle=False)as z:return{k:z[k]for k in keys}
def save(p,d):
 with open(p,'x')as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def csvsave(p,rows):
 with open(p,'x',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def div(n,d):return float(n/d)if d else None
def metrics(y,z,mask):
 pred=z>=0;cm=np.zeros((2,2),np.int64);np.add.at(cm,(y[mask],pred[mask].astype(int)),1)
 return dict(N=int(cm.sum()),C_support=int(cm[0].sum()),V_support=int(cm[1].sum()),C_pred_C_TN=int(cm[0,0]),C_pred_V_FP=int(cm[0,1]),V_pred_C_FN=int(cm[1,0]),V_pred_V_TP=int(cm[1,1]),V_precision_conditional_known_admitted=div(cm[1,1],cm[:,1].sum()),V_recall=div(cm[1,1],cm[1].sum()),C_precision_conditional_known_admitted=div(cm[0,0],cm[:,0].sum()),C_recall=div(cm[0,0],cm[0].sum()),accuracy=div(np.trace(cm),cm.sum()))
def main(fit_sha):
 assert fit_sha and sha(RUN/'FIT_COMPLETE.json')==fit_sha
 fit=load(RUN/'FIT_COMPLETE.json');assert fit['completed']and fit['optimizer_updates']==2000
 for k in ('model_freeze','history','scores','source_results'):check(fit[k])
 decisions=ROOT/'EXPERIMENT_DECISIONS_BEFORE_FEATURES.json';assert sha(decisions)=='a008fc7d35e9aa1d0d4b7d631a796f238aaba7f0aa052fe7da087080b70115e9'
 fixed_csv=V77/'source_failure_audit/FIXED_COHORT_IDS.csv';assert sha(fixed_csv)=='328caa6779d7dbdd4e715a64220332a0d0a78357fbc2a295265d37bf97b2e38d'
 old_report=V77/'source_failure_audit/PUBLIC_NATURAL_FIXED_GEOMETRY_SAVED_COMPARISON.json';assert sha(old_report)=='a21a97b781422809fea3ad20f7eeb1219fee91e28517b316ebf6ac6f9b5910df'
 old=load(old_report);assert old['fixed_threshold']==.5
 with open(fixed_csv,newline='')as f:fixed=list(csv.DictReader(f))
 paths=dict(V76=V77/'frozen_V76_source_baseline/SOURCE_SCORES.npz',V77=V77/'runs/PAIRED_SUPPORT_V1/SELECTED_SOURCE_SCORES.npz',V78=RUN/'SELECTED_SOURCE_SCORES.npz')
 assert sha(paths['V76'])=='f526afd4fa2ec9940a14e08f0d60ac945505d923f055bc90007e8848ed046268'
 assert sha(paths['V77'])=='c225ac0455ee6fb43aff05b21ef8686efc0a493dc436a927b5c0e5b345c43f7a'
 keys=['support_logit','NLL_FP32','input_available','primary_loss_mask','id','source','split','domain','after_original_row_index','after_anchor_k','pair_good']
 scores={k:npz(p,keys)for k,p in paths.items()};s=scores['V78']
 for z in scores.values():
  assert z['support_logit'].shape==(10002,8)and np.isfinite(z['support_logit']).all()
  for k in keys[2:]:assert np.array_equal(z[k],s[k]),k
 g0p=V76/'gt_corpus_build/GT_ONLY_TEACHER_CORPUS.npz';g1p=V77/'pair_generation/AFTER_GT_ONLY.npz'
 assert sha(g0p)=='626338ae0dee48f5c5d5f53930e05d02960225ee9d3972e751aeff4b25c30f45'
 assert sha(g1p)=='ccac3cfacfc53891aec3f9388ba54dfb8809071a27223263622d0478d0e80073'
 g0=npz(g0p,['id','source','split','label8']);g1=npz(g1p,['id','source','split','label8']);y=np.concatenate((g0['label8'],g1['label8']))
 for k in ('id','source','split'):assert np.array_equal(np.r_[g0[k],g1[k]],s[k])
 assert not(s['primary_loss_mask']&(y<0)).any()
 lookup={str(v):i for i,v in enumerate(s['id'])};assert len(lookup)==10002
 cohorts=[];cohort_csv=[];strict_mask=None
 for prior in old['cohorts']:
  name=prior['cohort'];rr=[r for r in fixed if r['cohort']==name];mask=np.zeros(y.shape,bool);ids=[]
  for row in rr:
   i=lookup[row['id']];k=int(row['native_k']);assert i<5203 and s['source'][i]=='YCB_PUBLIC_REAL'and s['split'][i]=='val'and s['primary_loss_mask'][i,k]and not mask[i,k]
   mask[i,k]=True;ids.append(row['id']+':k'+str(k))
  h=hashlib.sha256(('\n'.join(sorted(ids))+'\n').encode()).hexdigest();assert h==prior['cohort_IDs_SHA256']
  record=dict(cohort=name,cohort_IDs_SHA256=h)
  for model,z in scores.items():
   met=metrics(y,z['support_logit'],mask)
   if model in prior:assert met==prior[model],(name,model)
   record[model]=met;cohort_csv.append(dict(cohort=name,model=model,**met))
  transitions=[]
  for baseline in ('V76','V77'):
   for cls in (0,1):
    a=mask&(y==cls);cm=np.zeros((2,2),np.int64);np.add.at(cm,((scores[baseline]['support_logit'][a]>=0).astype(int),(s['support_logit'][a]>=0).astype(int)),1)
    transitions.append(dict(from_model=baseline,to_model='V78',GT=cls,N=int(a.sum()),CM_rows_old_CV_cols_V78_CV=cm.tolist()))
  record['prediction_transitions']=transitions;cohorts.append(record)
  if name==COHORT:strict_mask=mask
 assert strict_mask is not None and strict_mask.sum()==535 and (strict_mask&(y==1)).sum()==459 and (strict_mask&(y==0)).sum()==76
 details=[]
 for i,k in zip(*np.nonzero(strict_mask)):
  rr=dict(id=str(s['id'][i]),native_k=int(k),teacher_label=int(y[i,k]),query_geometry_and_cohort_unchanged=True)
  for model,z in scores.items():rr[model+'_support_logit']=float(z['support_logit'][i,k]);rr[model+'_pred_C0_V1']=int(z['support_logit'][i,k]>=0)
  details.append(rr)
 csvsave(DIAG/'FIXED_535_QUERY_COMPARISON.csv',details);csvsave(DIAG/'FIXED_459_VISIBLE_QUERY_COMPARISON.csv',[r for r in details if r['teacher_label']==1]);csvsave(DIAG/'COHORT_METRICS.csv',cohort_csv)
 geometry_report=dict(completed=True,PASS=True,fixed_threshold=.5,cohort_definition=old['definitions'],cohort_membership_reused_not_reselected=True,strict535_V459_C76=True,cohorts=cohorts,
  interpretation=['Same fixed source ref/query geometry and teacher cohorts; no new jitter/radius/threshold/checkpoint choices.','Any remaining strict-cohort visible false-covered cannot all be attributed to large query displacement, reference-MISS or footprint-fallback labels.','This new candidate changes encoder/pooling/head/initialization together. Saved-score differences do not isolate DINO as a cause.','Reference-local weak support is distinct from query-pixel occupancy and human semantic D/E/S.'],
  bindings=[bind(old_report),bind(fixed_csv),bind(RUN/'FIT_COMPLETE.json'),*[bind(p)for p in paths.values()],bind(g0p),bind(g1p),bind(Path(__file__))],RGB_mask_feature_weights_current_forward_fit_PnP=0)
 save(DIAG/'PUBLIC_NATURAL_FIXED_GEOMETRY_SAVED_COMPARISON.json',geometry_report)
 result_paths=dict(V76=V77/'frozen_V76_source_baseline/SOURCE_RESULTS.json',V77=V77/'runs/PAIRED_SUPPORT_V1/SOURCE_RESULTS.json',V78=RUN/'SOURCE_RESULTS.json')
 result={m:load(p)for m,p in result_paths.items()};natural=[];pairs=[];same=[];gaps=[]
 for model,d in result.items():
  for cohort,v in d['results']['natural_and_after'].items():natural.append(dict(model=model,cohort=cohort,**{k:v[k]for k in ('all_slots','known','admitted','U','known_ref_outside_square')},metrics=v['metrics']))
  for cohort,v in d['results']['paired_anchors'].items():pairs.append(dict(model=model,cohort=cohort,**v))
  for domain in ('NATURAL_ORIGINAL','CONTROLLED_AFTER'):
   for src in ('SYNTHETIC','YCB_PUBLIC_REAL'):
    tr=d['results']['natural_and_after'][f'{domain}|{src}|train']['metrics'];va=d['results']['natural_and_after'][f'{domain}|{src}|val']['metrics']
    gaps.append(dict(model=model,domain=domain,source=src,VAL_minus_TRAIN_V_recall=va['visible_recall']-tr['visible_recall'],VAL_minus_TRAIN_C_recall=va['covered_recall']-tr['covered_recall'],VAL_minus_TRAIN_BCE=va['mean_BCE_FP32']-tr['mean_BCE_FP32']))
  z=scores[model];za=z['support_logit'][5203:];ak=z['after_anchor_k'];orig=z['after_original_row_index'];idx=np.arange(4799);anchorC=za[idx,ak]<0;nv=z['primary_loss_mask'][5203:]&(y[5203:]==1);nv[idx,ak]=False
  for src in ('SYNTHETIC','YCB_PUBLIC_REAL'):
   for split in ('train','val'):
    rows=(z['source'][5203:]==src)&(z['split'][5203:]==split);v=nv&rows[:,None]
    same.append(dict(model=model,source=src,split=split,V_nonanchor_support=int(v.sum()),V_nonanchor_correct=int(((za>=0)&v).sum()),V_nonanchor_recall=div(((za>=0)&v).sum(),v.sum()),C_anchor_and_V_nonanchor_both_correct=int((anchorC[:,None]&(za>=0)&v).sum()),same_after_both_rate=div((anchorC[:,None]&(za>=0)&v).sum(),v.sum())))
 m=result['V78']['results']['natural_and_after']['NATURAL_ORIGINAL|YCB_PUBLIC_REAL|val']['metrics'];pair=result['V78']['results']['paired_anchors']['YCB_PUBLIC_REAL|val']
 gate=dict(public_natural_C_precision_ge_0_50=dict(value=m['covered_precision'],minimum=.5),public_natural_C_recall_ge_0_80=dict(value=m['covered_recall'],minimum=.8),public_natural_V_recall_ge_0_80=dict(value=m['visible_recall'],minimum=.8),public_controlled_both_correct_ge_0_70=dict(value=pair['both_correct_rate'],minimum=.7))
 for g in gate.values():g['PASS']=g['value']is not None and g['value']>=g['minimum']
 gate_pass=all(g['PASS']for g in gate.values())
 gate_path=ROOT/'SOURCE_ADMISSION_DECISION.json'
 if gate_path.exists():
  existing=load(gate_path);assert existing['source_candidate_pass']==gate_pass
  correspondence={'public_natural_C_precision_ge_0_50':'public_natural_C_precision_ge_0.50','public_natural_C_recall_ge_0_80':'public_natural_C_recall_ge_0.80','public_natural_V_recall_ge_0_80':'public_natural_V_recall_ge_0.80','public_controlled_both_correct_ge_0_70':'public_controlled_both_correct_ge_0.70'}
  for key,rootkey in correspondence.items():assert gate[key]==existing['checks'][rootkey],key
 review=dict(completed=True,PASS=True,PASS_meaning='Saved-source diagnostic completed; source gate is separate and never selects another step.',fully_solved=False,
  best_update=fit['best_update'],source_gate_checks=gate,source_candidate_gate_pass=gate_pass,current_stage_remains_forbidden=True,
  natural_TRAIN_VAL_and_after=natural,TRAIN_VAL_gaps=gaps,paired_TRAIN_VAL=pairs,same_after_CV_controls=same,
  fixed_strict_cohort=next(r for r in cohorts if r['cohort']==COHORT),
  interpretation_limits=['Natural VAL is the checkpoint selector, not an untouched test. Pair VAL is not the selector but shares the same composition algorithm.','Same q/L before-after evaluates RGB intervention response, not coordinate jitter invariance or human semantic reference association.','Visible preservation must accompany covered suppression; all-C output cannot be called successful target ownership.','Seam/cutout shortcut is not excluded by same-generator VAL. Same-image C/V controls challenge a global constant rule but do not identify the causal input branch.','DINO candidate differs in encoder/pooling/head/initialization; no pure backbone causal-effect claim.','Public queries are GT-derived source emulation; GT teacher native cuboids are not human semantic native8 gold.'],
  bindings=[bind(decisions),bind(RUN/'FIT_COMPLETE.json'),bind(RUN/'MODEL_FREEZE.json'),*[bind(p)for p in result_paths.values()],bind(DIAG/'PUBLIC_NATURAL_FIXED_GEOMETRY_SAVED_COMPARISON.json'),bind(Path(__file__))]+([bind(gate_path)]if gate_path.exists()else[]),
  threshold_checkpoint_data_selection_changes=False,RGB_mask_feature_weights_current_reads=0,new_forward_training_download_PnP=0)
 save(HERE/'SOURCE_INTERPRETATION_REVIEW.json',review)
 save(HERE/'REVIEW_SEAL.json',dict(completed=True,PASS=True,artifacts=[bind(HERE/'SOURCE_INTERPRETATION_REVIEW.json'),bind(DIAG/'PUBLIC_NATURAL_FIXED_GEOMETRY_SAVED_COMPARISON.json'),bind(DIAG/'FIXED_535_QUERY_COMPARISON.csv'),bind(DIAG/'FIXED_459_VISIBLE_QUERY_COMPARISON.csv'),bind(DIAG/'COHORT_METRICS.csv'),bind(Path(__file__))],current_forward_fit_PnP=0))
 print(json.dumps(dict(review=bind(HERE/'SOURCE_INTERPRETATION_REVIEW.json'),fixed=review['fixed_strict_cohort'],gate=gate)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fit-sha',required=True);a=p.parse_args();main(a.fit_sha)
