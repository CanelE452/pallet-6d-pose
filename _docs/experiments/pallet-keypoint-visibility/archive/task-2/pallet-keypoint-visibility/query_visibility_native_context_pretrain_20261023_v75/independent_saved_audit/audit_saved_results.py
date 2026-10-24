"""Read saved V75 source logits/attention and GT only. No torch/model imports."""
from pathlib import Path
from datetime import datetime,timezone
import ast,csv,hashlib,json
import numpy as np
R=Path(__file__).resolve().parent;V=R.parent;RUN=V/'runs/SOURCE_PRETRAIN_V1';P=V.parent/'query_visibility_loco_native_context_20261023_v74/native_context_cache'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def load(p):return json.loads(Path(p).read_text())
def save(name,d):
 p=R/name;assert not p.exists();p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n');return p
protocol=load(V/'PRETRAIN_PROTOCOL.json');freeze=load(RUN/'MODEL_FREEZE.json');history=load(RUN/'HISTORY.json');results=load(RUN/'SOURCE_RESULTS.json');basepub=load(RUN/'BASELINE_UPDATE0_RESULTS.json');completion=load(RUN/'FIT_COMPLETE.json')
bindings=list(protocol['bindings'].values())+[freeze[k]for k in ['checkpoint','model_code','trainer_code','protocol','entry']]+[results['source_freeze'],results['baseline0']]+[completion[k]for k in ['model_freeze','history','source_results','baseline0_scores','selected_scores']]
bad=[b['path']for b in bindings if sha(b['path'])!=b['sha256']];assert not bad
with np.load(P/'SYNTH5120_GT_ONLY.npz',allow_pickle=False)as z:
 gt={k:z[k]for k in ['frame_index','cache_row','sample_id','source','partition','proxy_state8','matched','query_valid','primary_DES_loss_mask','query_to_reference_offset_over_L8','reference_in_local_square8','human_semantic_visibility8','target_pixel_presence8']}
assert np.array_equal(gt['frame_index'],np.arange(5120));assert (gt['human_semantic_visibility8']==-1).all() and(gt['target_pixel_presence8']==-1).all()
y=gt['proxy_state8'].ravel();primary=gt['primary_DES_loss_mask'].ravel();src=np.repeat(gt['source'],8);partition=np.repeat(gt['partition'],8)
expected=(gt['matched'][:,None]&gt['query_valid']&(gt['proxy_state8']>=0)&(gt['proxy_state8']<3)).ravel();assert np.array_equal(primary,expected)
assert sum(primary&(partition=='train'))==29353 and sum(primary&(partition=='heldout'))==7358
lineage_path=P/'SYNTH5120_INPUT_PACK/INDEX_LINEAGE_INPUT_ONLY.jsonl';lineage=[json.loads(s)for s in lineage_path.read_text().splitlines()]
assert len(lineage)==5120 and all(r['frame_index']==i and r['cache_row']==gt['cache_row'][i] and r['id']==gt['sample_id'][i] and r['source']==gt['source'][i] and r['partition']==gt['partition'][i]for i,r in enumerate(lineage))
L=np.repeat(np.array([r['raw_L']for r in lineage]),8)
# Source selection replay from saved history only. Each arithmetic mean has exactly nine strata.
assert [h['update']for h in history]==list(range(50,1001,50));hrows=[];best=float('inf');bestupdate=None
for h in history:
 assert len(h['stratumNLL'])==9
 value=float(np.mean(list(h['stratumNLL'].values())));assert value==h['heldout9strata_macroNLL']
 saved=value<best
 if saved:best=value;bestupdate=h['update']
 assert saved==h['saved'] and bestupdate==h['best_update']
 hrows.append({'update':h['update'],'recomputed_macroNLL':value,'published_macroNLL':h['heldout9strata_macroNLL'],'strict_less_save_flag_exact':True})
assert bestupdate==750==freeze['best_update']==results['selected_update'] and best==freeze['best_heldout9strata_macroNLL']==results['selection_heldout_macroNLL']
assert basepub['update']==0 and not basepub['eligible_for_checkpoint_selection']
# Static input/GT boundary review; never import or call the trainer/model.
trainer=(V/'train_source_once.py').read_text();tree=ast.parse(trainer)
fields=next(ast.literal_eval(n.value)for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='FIELDS'for t in n.targets))
assert fields==('p3_roi','local_rgb96','local_valid96','target_rgb128','target_valid128','context42')
metrics=[];strata=[];comparisons=[];scores={};savednll={};probs={}

def metric(z,nll,p,m):
 yy=y[m].astype(int);pr=z[m].argmax(1);cm=np.zeros((3,3),np.int64);np.add.at(cm,(yy,pr),1)
 pp=[float(cm[c,c]/cm[:,c].sum()) if cm[:,c].sum() else None for c in range(3)];rr=[float(cm[c,c]/cm[c].sum()) if cm[c].sum() else None for c in range(3)]
 binary=p[m,0]>=.5;truth=yy==0;tp=int((binary&truth).sum());fn=int((~binary&truth).sum());fp=int((binary&~truth).sum());tn=int((~binary&~truth).sum())
 return {'N':int(m.sum()),'confusion_rows_GT_cols_pred_DES':cm.tolist(),'class_precision':pp,'class_recall':rr,'accuracy':float(np.trace(cm)/cm.sum()),'mean_NLL_FP32':float(nll[m].astype(np.float64).mean()),'binary_D_threshold':.5,'D_recall':tp/(tp+fn),'nonD_absence_recall':tn/(tn+fp),'binary_D_confusion':{'TP':tp,'FN':fn,'FP':fp,'TN':tn},'binary_D_precision':tp/(tp+fp) if tp+fp else None}

for name,file,published in [('baseline0','BASELINE_UPDATE0_SOURCE_SCORES.npz',basepub['metrics']),('selected','SELECTED_SOURCE_SCORES.npz',results['selected_metrics'])]:
 with np.load(RUN/file,allow_pickle=False)as z:
  logits=z['proxy_logits'].reshape(-1,3);nll=z['NLL_primary_FP32'].ravel()
 assert logits.shape==(40960,3) and np.isfinite(logits).all() and np.isfinite(nll).all()
 x=logits.astype(np.float64);lse=np.log(np.exp(x-x.max(1,keepdims=True)).sum(1))+x.max(1);prob=np.exp(x-lse[:,None]);newnll=lse-x[np.arange(len(y)),np.clip(y,0,2)]
 probs[name]=prob;scores[name]=logits;savednll[name]=nll
 diff=abs(newnll[primary]-nll[primary]);comparisons.append({'run':name,'FP64_softmax_vs_saved_FP32_NLL_max_abs_difference':float(diff.max()),'FP64_vs_FP32_mean_difference':float(newnll[primary].mean()-nll[primary].astype(np.float64).mean()),'argmax_softmax_vs_logits_mismatches':int((prob.argmax(1)!=logits.argmax(1)).sum())})
 assert diff.max()<1e-5 # numerical consistency check, not a performance/selection threshold
 for part in ['train','heldout']:
  for source in ['ALL','G38','P0','TEX']:
   m=primary&(partition==part)&((src==source)if source!='ALL'else True);v=metric(logits,nll,prob,m);pub=published[part]['overall']if source=='ALL'else published[part]['per_source'][source]
   for k in pub:
    if k=='confusion_rows_GT_cols_pred_DES':assert pub[k]==v[k]
    elif isinstance(pub[k],list):assert np.array_equal(pub[k],v[k])
    else:assert pub[k]==v[k],(name,part,source,k,pub[k],v[k])
   metrics.append({'run':name,'partition':part,'source':source,**v,'published_metrics_exact':True})
  for source in ['G38','P0','TEX']:
   for cl in range(3):
    m=primary&(partition==part)&(src==source)&(y==cl);strata.append({'run':name,'partition':part,'source':source,'class':cl,'N':int(m.sum()),'saved_FP32_meanNLL':float(nll[m].astype(np.float64).mean()),'independent_FP64_meanNLL':float(newnll[m].mean())})
  pubn=published[part];m=primary&(partition==part);assert pubn['primary_eligible_slots']==int(m.sum()) and pubn['all_retained_slots']==int((partition==part).sum())
macro={name:float(np.mean([r['saved_FP32_meanNLL']for r in strata if r['run']==name and r['partition']=='heldout']))for name in ['baseline0','selected']}
# Requested same-denominator location diagnostic, saved GT+attention+metadata L only.
with np.load(RUN/'SELECTED_SOURCE_SCORES.npz',allow_pickle=False)as z:attention=z['reference_attention_weights'].reshape(-1,9,9)
offset=gt['query_to_reference_offset_over_L8'].reshape(-1,2);square=gt['reference_in_local_square8'].ravel();aux=primary&(partition=='heldout')&(y==0)&square&np.isfinite(offset).all(1)
peak=attention.reshape(-1,81).argmax(1);peakxy=np.stack((peak%9-4,peak//9-4),1)/9
axis=(np.arange(9)-4)/9;yy,xx=np.meshgrid(axis,axis,indexing='ij');expectedxy=(attention[...,None]*np.stack((xx,yy),-1)).sum((1,2))
center=np.linalg.norm(offset[aux],axis=1);peakerr=np.linalg.norm(peakxy[aux]-offset[aux],axis=1);meanerr=np.linalg.norm(expectedxy[aux]-offset[aux],axis=1)
def dist(v):return {'median_over_L':float(np.median(v)),'p90_over_L':float(np.quantile(v,.9)),'median_native_px':float(np.median(v*L[aux])),'p90_native_px':float(np.quantile(v*L[aux],.9))}
def change(v):return {'improved_vs_query_center':int((v<center).sum()),'worsened_vs_query_center':int((v>center).sum()),'exactly_equal_vs_query_center':int((v==center).sum()),'numerically_within_1e-12_diagnostic':int((abs(v-center)<=1e-12).sum())}
location={'schema':'V75_SAVED_LOCATION_VS_QUERY_CENTER_DIAGNOSTIC','denominator':'Exactly selected heldout D proxy & primary & reference_in_local_square & finite GT offset. No source/model/quality based filtering.','N':int(aux.sum()),'query_center_offset0':dist(center),'attention_peak_native9':{**dist(peakerr),**change(peakerr)},'attention_expected_mean':{**dist(meanerr),**change(meanerr)},'coordinate_recipe':'Native9 axes (i-4)/9. Baseline reference estimate is unchanged query center, GT-only offset=(r-q)/L. raw_L is read from bound input-only JSONL; no feature arrays opened.','native9_quantization':'Peak is quantized to native9 cells; this comparison does not certify exact physical reference localization.','scope':'Virtual cuboid source proxy, not physical cue/human/native8 gold or current/public-real transfer.','inputs':[bind(RUN/'SELECTED_SOURCE_SCORES.npz'),bind(P/'SYNTH5120_GT_ONLY.npz'),bind(lineage_path)]}
locpath=save('LOCATION_VS_QUERY_CENTER_DIAGNOSTIC.json',location)
for name,rows in [('METRICS.csv',metrics),('STRATA_NLL.csv',strata),('HISTORY_SELECTION.csv',hrows)]:
 p=R/name;assert not p.exists()
 flat=[{k:json.dumps(v)if isinstance(v,(list,dict))else v for k,v in row.items()}for row in rows]
 with p.open('w')as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
report={'schema':'V75_INDEPENDENT_SAVED_SOURCE_RESULT_AUDIT','PASS':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'scope':'Saved source outputs and GT only; no torch, trainer or model imported; no new forward, fitting, public-real or current data access.',
 'bindings_checked':len(bindings),'binding_mismatches':bad,'checkpoint':freeze['checkpoint'],'primary_denominators':{'train':29353,'heldout':7358,'all_slots':40960},'GT_eligibility_formula_exact':True,'GT_semantic_and_pixel_UNKNOWN_preserved':True,
 'selection':{'history_evaluations':20,'eligible_updates':list(range(50,1001,50)),'all_nine_strata_arithmetic_means_exact':True,'strict_less_earliest_tie_flags_exact':True,'best_update':bestupdate,'history_best_macroNLL':best,'baseline0_excluded':True,'saved_allslot_macroNLL':macro,'selected_saved_minus_history_macroNLL':macro['selected']-best,'note':'Selection remains the original history criterion. All-slot saved inference can differ numerically from heldout-only batched inference; no reselection occurs.'},
 'softmax_NLL_reconciliation':comparisons,'metrics':metrics,'strata_NLL':strata,'published_confusion_precision_recall_NLL_all_exact':True,
 'GT_input_boundary_review':{'model_forward_fields':list(fields),'GT_forward_fields':0,'source_id_and_class_used_for_balanced_sampler_only':True,'GT_offset_and_square_used_for_D_location_loss_only':True,'forward_code_review':'Trainer builds GPU input dictionary from input cache fields only, then forward(model,gpu,ids) indexes those six fields; GT labels/reference/matched/source are separate sampling/loss/meta. No GT is passed to model.forward.','runtime_causal_claim':False,'source_Gold':'Conditional virtual-reference/ray/convex proxy only; human semantic/pixel fields remain UNKNOWN.'},
 'location_diagnostic':bind(locpath),'read_policy':{'checkpoint_bytes_hash_only':True,'feature_or_RGB_arrays_read':0,'current_or_public_real_data_read':0,'new_model_forward_training_PnP':0,'saved_attention_read':True,'GT_and_input_L_metadata_only':True},'audit_code':bind(Path(__file__))}
out=save('INDEPENDENT_SAVED_RESULT_AUDIT.json',report)
seal=save('AUDIT_SEAL.json',{'schema':'V75_INDEPENDENT_SAVED_AUDIT_SEAL','bindings':[bind(p)for p in sorted(R.iterdir())if p.is_file()],'original_run_files_modified':False})
print(json.dumps({'audit':bind(out),'seal':bind(seal),'selection':report['selection'],'location':location,'softmax_NLL':comparisons}))
