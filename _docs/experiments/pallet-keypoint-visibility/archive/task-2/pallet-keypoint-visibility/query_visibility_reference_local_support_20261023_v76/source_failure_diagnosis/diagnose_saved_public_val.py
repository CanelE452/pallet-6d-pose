"""Saved source-only V76 public VAL error decomposition; RGB/masks/model unopened."""
from pathlib import Path
from datetime import datetime,timezone
import json,csv,hashlib,numpy as np
R=Path(__file__).resolve().parent;V=R.parent;RUN=V/'mixed_fit/runs/MIXED_SUPPORT_V1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def load(p):return json.loads(Path(p).read_text())
def write(name,x):
 p=R/name;assert not p.exists();p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');return p
freeze=load(RUN/'MODEL_FREEZE.json');assert freeze['best_update']==200 and freeze['current_used_for_fit_selection']is False and freeze['PnP']==0
inputb=load(V/'mixed_fit/SOURCE_INPUT_BINDINGS.json');pack=next(p for p in inputb['packs']if p['source']=='YCB_PUBLIC_REAL');man=load(pack['manifest']['path']);lineage=[json.loads(s)for s in Path(pack['lineage']['path']).read_text().splitlines()]
qparts=[];Lparts=[]
for chunk in man['chunks']:
 with np.load(chunk['file']['path'],allow_pickle=False)as z:qparts.append(z['query8_xy']);Lparts.append(z['L'])
q=np.concatenate(qparts);L=np.concatenate(Lparts)
with np.load(V/'gt_corpus_build/GT_ONLY_TEACHER_CORPUS.npz',allow_pickle=False)as z:
 keys=['id','source','split','label8','native_GTref8','rule8','near_boundary8','convex_self_proxy8_separate','center_M08','center_M48','rho_px','weighted_expected_mass_A8','weighted_visible_mass_V8','expected_support_pixels_A8','visible_over_expected_ratio8']
 gt={k:z[k]for k in keys};rows=np.flatnonzero(gt['source']=='YCB_PUBLIC_REAL');gt={k:x[rows]for k,x in gt.items()}
assert len(gt['id'])==len(q)==4200 and all(r['id']==gt['id'][i]for i,r in enumerate(lineage))
with np.load(RUN/'SELECTED_SOURCE_SCORES.npz',allow_pickle=False)as z:
 mask=z['source']=='YCB_PUBLIC_REAL';assert np.array_equal(z['id'][mask],gt['id'])and np.array_equal(z['split'][mask],gt['split']);logit=z['support_logit'][mask];primary=z['primary_loss_mask'][mask];sq=z['reference_in_local_square_GT_only'][mask]
y=gt['label8'];pred=(logit>=0).astype(np.int8);prob=np.exp(-np.logaddexp(0.,-logit.astype(float)));correct=pred==y
reference=gt['native_GTref8'];delta=reference-q;offset=delta/L[:,None,None];err=np.linalg.norm(delta,axis=-1);linf=np.max(abs(offset),axis=-1);l2=np.linalg.norm(offset,axis=-1)
val=np.broadcast_to((gt['split']=='val')[:,None],y.shape);eligible=val&primary;assert eligible.sum()==2018
branch=np.where(np.char.startswith(gt['rule8'],'STRONG'),'STRONG_3X3','FOOTPRINT_FALLBACK')
pix=np.where(err<=3,'px_0_to3_inclusive',np.where(err<=10,'px_gt3_to10','px_gt10'))
def normbin(a):return np.where(a<=1/18,'center_cell_le1_over18',np.where(a<=1/6,'center3x3_shell_gt1over18_le1over6','outside_center3x3_gt1over6'))
nb=normbin(linf);l2b=normbin(l2);selfv=gt['convex_self_proxy8_separate'];selfname=np.where(selfv==0,'CONVEX_SELF_HIDDEN',np.where(selfv==1,'CONVEX_MAY_VISIBLE','SELF_PROXY_UNKNOWN'))
center=np.char.add(np.char.add('r_M0',gt['center_M08'].astype(str)),np.char.add('_M4',gt['center_M48'].astype(str)))

def metrics(m):
 cm=np.zeros((2,2),np.int64);np.add.at(cm,(y[m],pred[m]),1)
 def d(a,b):return float(a/b)if b else None
 return {'N':int(m.sum()),'C_support':int(cm[0].sum()),'V_support':int(cm[1].sum()),'C_pred_C_TN':int(cm[0,0]),'C_pred_V_FP':int(cm[0,1]),'V_pred_C_FN':int(cm[1,0]),'V_pred_V_TP':int(cm[1,1]),'V_precision_conditional_known_admitted':d(cm[1,1],cm[:,1].sum()),'V_recall':d(cm[1,1],cm[1].sum()),'C_precision_conditional_known_admitted':d(cm[0,0],cm[:,0].sum()),'C_recall':d(cm[0,0],cm[0].sum()),'accuracy':d(np.trace(cm),cm.sum())}
records=[]
def group(dim,values):
 for value in sorted(np.unique(values[eligible])):records.append({'dimension':dim,'stratum':str(value),**metrics(eligible&(values==value))})
records.append({'dimension':'ALL','stratum':'public_VAL_known_admitted',**metrics(eligible)})
for dim,a in [('teacher_branch',branch),('q_r_error_px',pix),('reference_offset_Linf',nb),('reference_offset_L2_diagnostic',l2b),('convex_self_proxy',selfname),('reference_center_M0_M4',center),('image_boundary',np.where(gt['near_boundary8'],'near_image_boundary','not_near_image_boundary'))]:group(dim,a)
for b in ['STRONG_3X3','FOOTPRINT_FALLBACK']:
 for e in ['px_0_to3_inclusive','px_gt3_to10','px_gt10']:
  m=eligible&(branch==b)&(pix==e);records.append({'dimension':'branch_x_error','stratum':b+'|'+e,**metrics(m)})
  for n in ['center_cell_le1_over18','center3x3_shell_gt1over18_le1over6','outside_center3x3_gt1over6']:
   records.append({'dimension':'branch_x_error_x_offsetLinf','stratum':b+'|'+e+'|'+n,**metrics(m&(nb==n))})
 for s in ['CONVEX_SELF_HIDDEN','CONVEX_MAY_VISIBLE','SELF_PROXY_UNKNOWN']:records.append({'dimension':'branch_x_self','stratum':b+'|'+s,**metrics(eligible&(branch==b)&(selfname==s))})
# Decompositions are partitions of the same denominator; unknown/miss is never negative.
for dim in ['teacher_branch','q_r_error_px','reference_offset_Linf','reference_offset_L2_diagnostic','convex_self_proxy','reference_center_M0_M4','image_boundary','branch_x_error','branch_x_error_x_offsetLinf','branch_x_self']:
 rr=[r for r in records if r['dimension']==dim]
 for k in ['N','C_support','V_support','C_pred_C_TN','C_pred_V_FP','V_pred_C_FN','V_pred_V_TP']:assert sum(r[k]for r in rr)==records[0][k],(dim,k)
with (R/'PUBLIC_VAL_ERROR_DECOMPOSITION.csv').open('x')as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
# Freeze first lexicographic ID/native slot per teacher class/error/correctness.
metadata={r['id']:r for r in map(json.loads,(V/'gt_corpus_build/ORDERED_METADATA_INDEX_GT_ONLY.jsonl').read_text().splitlines())}
teacherrows={r['id']:r for r in map(json.loads,(V/'gt_corpus_build/PER_ROW_TEACHER_GT_ONLY.jsonl').read_text().splitlines())}
selected=[];empty=[]
for cls in [0,1]:
 for e in ['px_0_to3_inclusive','px_gt3_to10','px_gt10']:
  for iscorrect in [True,False]:
   cand=list(zip(*np.nonzero(eligible&(y==cls)&(pix==e)&(correct==iscorrect))))
   if not cand:empty.append({'GT_class':cls,'error_bin':e,'correct':iscorrect});continue
   i,k=min(cand,key=lambda a:(str(gt['id'][a[0]]),int(a[1])));ident=str(gt['id'][i]);meta=metadata[ident];t=teacherrows[ident]
   selected.append({'case_id':f'{ident}:k{k}','selection_stratum':{'teacher_GT':'VISIBLE'if cls else'COVERED','error_bin':e,'saved_correct':iscorrect},'selection_rule':'First lexicographic (source id,native slot) in fixed admitted publicVAL stratum, not confidence rank. Frozen before RGB/mask visual inspection.',
    'id':ident,'native_index':int(k),'object_id':meta['object_id'],'scene_id':meta['scene_id'],'frame_id':meta['frame_id'],'instance_id':meta['instance_id'],'image':meta['image_path_lineage_only_NOT_READ'],'image_sha256':meta['image_sha256_lineage_only'],'raw_hw':meta['raw_hw'],'q_xy':q[i,k].tolist(),'reference_r_xy':reference[i,k].tolist(),'original_L':float(L[i]),'q_crop_continuous_xyxy':(np.r_[q[i,k]-L[i]/2,q[i,k]+L[i]/2]).tolist(),'r96_diagnostic_crop_xyxy':(np.r_[reference[i,k]-48,reference[i,k]+48]).tolist(),
    'q_r_error_px_L2':float(err[i,k]),'reference_offset_over_L':offset[i,k].tolist(),'reference_offset_Linf_bin':str(nb[i,k]),'teacher_label':int(y[i,k]),'teacher_rule':str(gt['rule8'][i,k]),'convex_may_visible_flag':int(selfv[i,k]),'reference_center_M0':int(gt['center_M08'][i,k]),'reference_center_M4':int(gt['center_M48'][i,k]),'expected_mask_pixels':int(gt['expected_support_pixels_A8'][i,k]),'weighted_expected_mass':float(gt['weighted_expected_mass_A8'][i,k]),'weighted_visible_mass':float(gt['weighted_visible_mass_V8'][i,k]),'visible_expected_ratio':float(gt['visible_over_expected_ratio8'][i,k]),'rho_px':float(gt['rho_px'][i]),'saved_visible_probability':float(prob[i,k]),'saved_prediction':'VISIBLE'if pred[i,k]else'COVERED','M0':t['mask_pair']['M0'],'M4':t['mask_pair']['M4'],
    'RGB_or_mask_opened_for_this_selection':False,'human_semantic_or_native8_gold':False,'label_or_query_relocation':False})
casefile=write('FIXED_LEX_CASE_SELECTION_LOCK.json',{'schema':'V76_PUBLIC_VAL_FIXED_LEX_ERROR_CASE_METADATA','selected_cases':selected,'empty_strata':empty,'RGB_mask_body_read':0,'model_forward_fit_current_PnP':0,'scope':'Reviewable future source crop metadata, not visual proof, relabeling, hard-example sampling or a new fit.'})
report={'schema':'V76_SOURCE_ONLY_SAVED_PUBLIC_VAL_FAILURE_DECOMPOSITION','created_UTC':datetime.now(timezone.utc).isoformat(),'selected_source_update':200,'fixed_threshold':.5,'target':'Frozen reference-local expected target support V1/C0; semantic DES and q-pixel occupancy are separate/UNKNOWN.','row_scope':'Existing all800publicVAL instances6400slots retained; metrics only fixed primary2018known/admitted slots. No current scores/results read.',
 'overall':records[0],'definitions':{'pixel_error_bins':'Euclidean ||r-q|| nativepx: <=3,>3and<=10,>10.','normalized_primary':'Linf=max(abs((r-q)/L)); <=1/18 is central native9 cell square; (1/18,1/6] is central3x3 shell; >1/6 is outside center3x3. This is spatial geometry, not a tuned radius.','normalized_secondary':'L2 norm with the same fixed cutoffs is separately named as a diagnostic; it is not the square center-cell geometry.','teacher_branch':'STRONG_3X3_* vs FOOTPRINT_* from frozen teacher, on knownprimary subset only.','self_flag':'1=convex may-visible,0=convex self-hidden,-1=unknown; never relabeled into binary local support.','precision':'Conditioned on known/admitted source teacher V/C; unknown/miss is excluded, not negative. All labels/class counts and GT-positive supports shown.','q_and_r':'Label stays at source reference r; actual input crop stays at emulated q. Query-center M0/M4 is unavailable in typed corpus and is not invented from r-center fields.'},
 'marginal_and_cross_strata':records,'all_strata_reconcile_exact_to_overall':True,'fixed_case_metadata':bind(casefile),'RGB_P3_mask_arrays_decoded':0,'input_arrays_decoded':['query8_xy','L'],'model_current_forward_fit_PnP':0,'query_labels_thresholds_admission_changed':False,
 'bindings':[bind(p)for p in [RUN/'MODEL_FREEZE.json',RUN/'SELECTED_SOURCE_SCORES.npz',V/'gt_corpus_build/GT_ONLY_TEACHER_CORPUS.npz',V/'gt_corpus_build/ORDERED_METADATA_INDEX_GT_ONLY.jsonl',V/'gt_corpus_build/PER_ROW_TEACHER_GT_ONLY.jsonl',V/'mixed_fit/SOURCE_INPUT_BINDINGS.json',Path(pack['manifest']['path']),Path(pack['lineage']['path']),V/'mixed_fit_independent_saved_audit/INDEPENDENT_SAVED_RESULT_AUDIT.json',Path(__file__)]]}
out=write('PUBLIC_VAL_SAVED_FAILURE_DIAGNOSIS.json',report)
print(json.dumps({'report':bind(out),'overall':records[0],'key_strata':[r for r in records if r['dimension']in['teacher_branch','q_r_error_px','reference_offset_Linf','reference_center_M0_M4','convex_self_proxy']],'fixedcases':len(selected),'empty_strata':empty},ensure_ascii=False))
