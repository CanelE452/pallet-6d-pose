"""Saved source predictions + metadata only, clustered rather than rank-IID."""
from pathlib import Path
from collections import defaultdict,Counter
import csv,json,hashlib
import numpy as np
H=Path(__file__).resolve().parent;P=H.parents[1];V=P/'query_visibility_query_anchored_cue_20261023_v96'
def rows(p):return [json.loads(s)for s in Path(p).read_text().splitlines()if s]
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def rate_rows(rs):
 n=len(rs)
 return dict(N=n,before_D=sum(r['p0']==0 for r in rs),cover_E=sum(r['pc']==1 for r in rs),sham_D=sum(r['ps']==0 for r in rs),both_after_correct=sum(r['pc']==1 and r['ps']==0 for r in rs),all3_correct=sum(r['p0']==0 and r['pc']==1 and r['ps']==0 for r in rs))
def aggregate(rs):
 raw=rate_rows(rs);n=len(rs);base=defaultdict(list);images=defaultdict(list)
 for r in rs:base[r['base']].append(r);images[r['image_sha']].append(r)
 keys=['before_D','cover_E','sham_D','both_after_correct','all3_correct']
 def macro(groups):return {k:float(np.mean([rate_rows(xs)[k]/len(xs)for xs in groups.values()]))if groups else None for k in keys}
 return dict(rank_counts=raw,rank_rates={k:raw[k]/n if n else None for k in keys},distinct_base_queries=len(base),distinct_original_images=len(images),base_macro_rates=macro(base),image_macro_rates=macro(images),
  bases_any_sham_error=sum(any(r['ps']!=0 for r in xs)for xs in base.values()),bases_all_sham_error=sum(all(r['ps']!=0 for r in xs)for xs in base.values()),bases_any_cover_error=sum(any(r['pc']!=1 for r in xs)for xs in base.values()),cover_prediction_counts=dict(Counter(str(r['pc'])for r in rs)),sham_prediction_counts=dict(Counter(str(r['ps'])for r in rs)))
def main():
 line=rows(V/'actual_source_pair_input_pack/ALL31680_INPUT_LINEAGE.jsonl');gt=rows(V/'actual_query_anchored_image_intake/OCCURRENCES31680_GT_ONLY.jsonl');assert len(line)==len(gt)==31680
 models={'C1_FROZEN':V/'source_baselines/C1_FROZEN_ON_V96/SELECTED_ALL31680_SOURCE_SCORES.npz','C3_SELECTED0':V/'runs/C3_QUERY_ANCHORED_SOURCE_V1/SELECTED_ALL31680_SOURCE_SCORES.npz','C3_FIXEDFINAL2000_DIAGNOSTIC':V/'source_diagnosis/FINAL2000_FIXED_NOT_SELECTOR/ALL31680_FINAL_SOURCE_SCORES.npz'}
 allresult={};basecsv=[];imagecsv=[];natural={}
 for model,path in models.items():
  with np.load(path,allow_pickle=False)as z:p=z['prediction'].copy();logits=z['query_logit'].copy();available=z['query_available'].copy();y=z['scalar_target_D0_E1_Uminus1'].copy()
  assert np.array_equal(y,[g['candidate_aux_hidden_label_GT_only']for g in gt])
  allresult[model]={};natural[model]={}
  for fold in ['TRAIN_LAB','VAL_HALL']:
   rs=[]
   for i in range(960):
    if line[33*i]['split']!=fold:continue
    for k in range(16):
     oi=33*i+1+2*k
     if not gt[oi]['pair_ready']:continue
     assert y[33*i]==0 and y[oi:oi+2].tolist()==[1,0]
     rs.append(dict(base=i,rank=k,image_sha=line[33*i]['original_image_sha256'],input_id=line[33*i]['input_id'],p0=int(p[33*i]),pc=int(p[oi]),ps=int(p[oi+1]),z0=float(logits[33*i]),zc=float(logits[oi]),zs=float(logits[oi+1])))
   result=dict(all_accepted=aggregate(rs),conditional_on_original_BEFORE={state:aggregate([r for r in rs if r['p0']==v])for state,v in [('correct_D',0),('wrong_E',1),('unavailable_U',-1)]},joint_pred_patterns={','.join(map(str,k)):v for k,v in Counter((r['p0'],r['pc'],r['ps'])for r in rs).items()})
   bases=defaultdict(list);images=defaultdict(list)
   for r in rs:bases[r['base']].append(r);images[r['image_sha']].append(r)
   for i,xs in bases.items():
    rec=dict(model=model,raw_split=fold,base_row_index=i,input_id=xs[0]['input_id'],image_sha256=xs[0]['image_sha'],native_slot=line[33*i]['native_slot_logical_only'],original_q=json.dumps(line[33*i]['query_xy']),original_L=line[33*i]['L'],original_BEFORE_prediction=xs[0]['p0'],original_BEFORE_logit=xs[0]['z0'],**rate_rows(xs),cover_minus_before_logit_mean=float(np.mean([r['zc']-r['z0']for r in xs])),sham_minus_before_logit_mean=float(np.mean([r['zs']-r['z0']for r in xs])))
    basecsv.append(rec)
   for image,xs in images.items():imagecsv.append(dict(model=model,raw_split=fold,image_sha256=image,input_id=xs[0]['input_id'],base_queries=len({r['base']for r in xs}),**rate_rows(xs)))
   allresult[model][fold]=result
  for source,fold in [('LOCO','TRAIN'),('LOCO','VALIDATION'),('PALLOX','TRAIN_LAB'),('PALLOX','VAL_HALL')]:
   ids=[33*i for i in range(960)if line[33*i]['source']==source and line[33*i]['split']==fold and y[33*i]>=0]
   cm=[[sum(y[i]==g and p[i]==d for i in ids)for d in [0,1,-1]]for g in [0,1]]
   im=defaultdict(list)
   for i in ids:
    if y[i]==0:im[line[i]['original_image_sha256']].append(int(p[i]==0))
   natural[model][source+'|'+fold]=dict(known_original_rows=len(ids),CM_D_E_by_D_E_U=cm,known_D_images=len(im),D_recall_image_macro=float(np.mean([np.mean(v)for v in im.values()]))if im else None)
 f=allresult['C3_FIXEDFINAL2000_DIAGNOSTIC']['VAL_HALL'];c=f['conditional_on_original_BEFORE']['correct_D'];w=f['conditional_on_original_BEFORE']['wrong_E']
 assert f['all_accepted']['rank_counts']==dict(N=119,before_D=40,cover_E=111,sham_D=49,both_after_correct=47,all3_correct=23)
 assert w['distinct_base_queries']==10 and w['rank_counts']['N']==79 and c['distinct_base_queries']==8 and c['rank_counts']['N']==40
 failure=dict(all3_failures=96,ranks_on_already_wrong_BEFORE_bases=79,already_wrong_BEFORE_distinct_bases=10,total_accepted_bases=18,remaining_failures_on_correct_BEFORE_bases=17,correct_BEFORE_sham_D_to_E=15,correct_BEFORE_cover_missed=3,
  interpretation='79/96 failed triples lie on bases already wrong on original RGB. This ordering is a descriptive overlap, not a causal attribution or79 independent scenes. Correct-BEFORE subset still has15/40 sham failures across4/8 basequeries, proving additional edit response instability.')
 report=dict(completed=True,source_only=True,method='Saved source logits/predictions+original metadata only. All accepted ranks retained; exact base and image clusters, no threshold selection/new inference.',models=allresult,natural_original_BEFORE=natural,dominance_diagnosis=failure,
  main_findings=['The trained fixedFINAL checkpoint already loses clean original-cue recognition on held-out source scenes: originalVAL PalloxD19/30 andLOCO2/3, while frozenC1 isPallox29/30 andLOCO3/3.','Within acceptedVAL, originalBEFORE wrongE affects10/18 basequeries carrying79/119 ranks; controlled donor response alone cannot explain the collapse.','Even holding originalBEFORE correct, SHAM retainsD only25/40 versusCOVER hiddenE37/40. Four of8 correct-BEFORE bases have sham errors, so originalscene failure is not the sole issue.','C1 keeps originalD18/18 andSHAM115/119 butCOVERonly2/119; trainedFINAL learns cover response111/119 while losing original/sham specificity. Selected0 is the authoritative untrained E-everywhere candidate and cannot be interpreted as a learned transfer result.'],
  interpretation_limits=['Scene/camera/view and donor-object partitions both change betweenTRAIN andVAL, so saved predictions do not isolate a causal domain factor or neural branch.','194 controls are28basequeries/15images: TRAIN75from10bases6images;VAL119from18bases9images, all onePalloxphysicalbin. Some conditional groups share images and are not disjoint capture families.','SourceSHAM keeps the complete protected cue raster and the positive bilinear centralstencil unchanged by audited construction; globalP3 receptive fields and peripheral RGB may still change.','BEFORE here means originalRGB of a controlled triplet; it may itself be the natural after capture. It does not mean unoccluded wholeframe.','LOCO haszero accepted controls. Sourcecue labels are weakconditionalAI; expectednativepart remainsUNKNOWN. No current/private evidence used.'],
  single_next_candidate=dict(name='Conservative C1-preserving source adaptation',recommendation='Warm-start the same scalar from immutableC1 rather than relearn from a tiny rank-heavy corpus; use a much smaller fixed learning rate and the same frozen source cohorts/selector/gates/qL. Check exactC1 update0 parity before any optimizer step. This is a single prospective candidate, not a claim of success.',evidence='C1 supplies the missing held-out BEFORE/SHAM specificity while the cover response is the required adaptation. More donor ranks alone do not fix pre-existing original-scene errors.',causal_scope='Initialization and learning-rate change jointly; any gain is not an isolated ablation. Sourcegate failure continues to block private inference.',parallel_expected_cue_data='An actual expected-cue RGB triplet can validate label/ownership construction, but one triplet oroneasset does not provide the scene diversity/generalization evidence missing here. It is not a prerequisite for this diagnosis.'),
  inputs={name:bind(path)for name,path in models.items()},lineage=bind(V/'actual_source_pair_input_pack/ALL31680_INPUT_LINEAGE.jsonl'),GT=bind(V/'actual_query_anchored_image_intake/OCCURRENCES31680_GT_ONLY.jsonl'),V96_root_seal=bind(V/'ROOT_PHASE_RESULT_SEAL.json'),code=bind(__file__),new_pixels_models_optimizer_current_PnP=0)
 for filename,data in [('BASE_QUERY_CLUSTERS.csv',basecsv),('ORIGINAL_IMAGE_CLUSTERS.csv',imagecsv)]:
  with(H/filename).open('x',newline='')as f:writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
 write(H/'SOURCE_TRANSFER_DIAGNOSIS.json',report)
 write(H/'SEAL.json',dict(completed=True,files=[bind(p)for p in sorted(H.iterdir())if p.is_file()],V96_modified=False,new_model_fit_current_PnP=0))
 print(json.dumps(dict(report=bind(H/'SOURCE_TRANSFER_DIAGNOSIS.json'),seal=bind(H/'SEAL.json'),finding=failure)))
if __name__=='__main__':main()
