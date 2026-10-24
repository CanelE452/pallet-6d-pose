"""Fixed earlier source VAL geometry cohorts, V76/V77 saved logits only."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
R=Path(__file__).resolve().parent;V=R.parent;P=V.parent;OLD=P/'query_visibility_reference_local_support_20261023_v76'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def load(p):return json.loads(Path(p).read_text())
def check(b):assert sha(b['path'])==b['sha256']
def save(p,d):
 with p.open('x')as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
old_report=load(OLD/'source_failure_diagnosis/PUBLIC_VAL_SAVED_FAILURE_DIAGNOSIS.json')
assert old_report['fixed_threshold']==.5
b=load(OLD/'mixed_fit/SOURCE_INPUT_BINDINGS.json');pack=next(x for x in b['packs']if x['source']=='YCB_PUBLIC_REAL')
for key in ['manifest','seal','lineage']:check(pack[key])
manifest=load(pack['manifest']['path']);qparts=[];Lparts=[]
for chunk in manifest['chunks']:
 check(chunk['file'])
 with np.load(chunk['file']['path'],allow_pickle=False)as z:qparts.append(z['query8_xy']);Lparts.append(z['L'])
q=np.concatenate(qparts);L=np.concatenate(Lparts)
with np.load(OLD/'gt_corpus_build/GT_ONLY_TEACHER_CORPUS.npz',allow_pickle=False)as z:
 ix=np.flatnonzero(z['source']=='YCB_PUBLIC_REAL');g={k:z[k][ix]for k in ['id','source','split','label8','native_GTref8','rule8','convex_self_proxy8_separate']}
assert len(ix)==4200 and np.array_equal(ix,np.arange(1003,5203))and len(q)==4200 and np.isfinite(L).all()and(L>0).all()
lineage=[json.loads(s)for s in Path(pack['lineage']['path']).read_text().splitlines()];assert all(x['id']==g['id'][i]for i,x in enumerate(lineage))
paths=dict(V76=OLD/'mixed_fit/runs/MIXED_SUPPORT_V1/SELECTED_SOURCE_SCORES.npz',V77=V/'runs/PAIRED_SUPPORT_V1/SELECTED_SOURCE_SCORES.npz')
zs={};pms={}
for model,path in paths.items():
 with np.load(path,allow_pickle=False)as z:
  assert np.array_equal(z['id'][ix],g['id'])and np.array_equal(z['source'][ix],g['source'])and np.array_equal(z['split'][ix],g['split'])
  if model=='V77':assert (z['domain'][ix]==0).all()
  zs[model]=z['support_logit'][ix];pms[model]=z['primary_loss_mask'][ix]
assert np.array_equal(pms['V76'],pms['V77'])
y=g['label8'];base=np.broadcast_to((g['split']=='val')[:,None],y.shape)&pms['V76'];assert base.sum()==2018
err=np.linalg.norm(g['native_GTref8']-q,axis=-1);offset=(g['native_GTref8']-q)/L[:,None,None];linf=np.max(abs(offset),axis=-1)
small=err<=3;strong=np.char.startswith(g['rule8'],'STRONG');center=linf<=1/18
cohorts=[('ALL_NATURAL_PUBLIC_VAL_PRIMARY',np.ones(y.shape,bool)),('ERROR_LE3PX',small),('STRONG_3X3',strong),('CENTER_NATIVE9_CELL_LINF_LE1_OVER18',center),('STRONG_AND_ERROR_LE3PX',strong&small),('STRONG_AND_CENTER_CELL',strong&center),('ERROR_LE3PX_AND_CENTER_CELL',small&center),('STRONG_AND_ERROR_LE3PX_AND_CENTER_CELL',strong&small&center)]

def metric(model,m):
 pred=zs[model]>=0;cm=np.zeros((2,2),int);np.add.at(cm,(y[m],pred[m].astype(int)),1)
 def div(a,b):return float(a/b)if b else None
 return dict(N=int(m.sum()),C_support=int(cm[0].sum()),V_support=int(cm[1].sum()),C_pred_C_TN=int(cm[0,0]),C_pred_V_FP=int(cm[0,1]),V_pred_C_FN=int(cm[1,0]),V_pred_V_TP=int(cm[1,1]),V_precision_conditional_known_admitted=div(cm[1,1],cm[:,1].sum()),V_recall=div(cm[1,1],cm[1].sum()),C_precision_conditional_known_admitted=div(cm[0,0],cm[:,0].sum()),C_recall=div(cm[0,0],cm[0].sum()),accuracy=div(np.trace(cm),cm.sum()))
lookup={('ALL','public_VAL_known_admitted'):'ALL_NATURAL_PUBLIC_VAL_PRIMARY',('q_r_error_px','px_0_to3_inclusive'):'ERROR_LE3PX',('teacher_branch','STRONG_3X3'):'STRONG_3X3',('reference_offset_Linf','center_cell_le1_over18'):'CENTER_NATIVE9_CELL_LINF_LE1_OVER18',('branch_x_error','STRONG_3X3|px_0_to3_inclusive'):'STRONG_AND_ERROR_LE3PX',('branch_x_error_x_offsetLinf','STRONG_3X3|px_0_to3_inclusive|center_cell_le1_over18'):'STRONG_AND_ERROR_LE3PX_AND_CENTER_CELL'}
prior={lookup[(x['dimension'],x['stratum'])]:x for x in old_report['marginal_and_cross_strata']if (x['dimension'],x['stratum'])in lookup}
records=[];csvrows=[];cohortrows=[]
for name,condition in cohorts:
 m=base&condition;old=metric('V76',m);new=metric('V77',m)
 if name in prior:assert all(old[k]==prior[name][k]for k in old),name
 ids=sorted(str(g['id'][i])+':k'+str(k)for i,k in zip(*np.nonzero(m)))
 cohort_hash=hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest();t=[]
 for cls in [0,1]:
  a=m&(y==cls);cm=np.zeros((2,2),int);np.add.at(cm,((zs['V76'][a]>=0).astype(int),(zs['V77'][a]>=0).astype(int)),1)
  t.append(dict(GT='VISIBLE'if cls else'COVERED',N=int(a.sum()),CM_rows_V76_pred_CV_cols_V77_pred_CV=cm.tolist()))
 row=dict(cohort=name,cohort_IDs_SHA256=cohort_hash,V76=old,V77=new,paired_prediction_transitions=t);records.append(row)
 for model,met in [('V76',old),('V77',new)]:csvrows.append(dict(cohort=name,model=model,**met))
 for i,k in zip(*np.nonzero(m)):cohortrows.append(dict(cohort=name,id=str(g['id'][i]),native_k=int(k)))
primary=next(x for x in records if x['cohort']=='STRONG_AND_ERROR_LE3PX_AND_CENTER_CELL');assert primary['V76']['N']==primary['V77']['N']==535 and primary['V76']['V_support']==459 and primary['V76']['C_support']==76
self_slices=[]
for v,name in [(0,'CONVEX_SELF_HIDDEN'),(1,'CONVEX_MAY_VISIBLE'),(-1,'SELF_PROXY_UNKNOWN')]:
 m=base&strong&small&center&(g['convex_self_proxy8_separate']==v)
 self_slices.append(dict(self_flag=name,V76=metric('V76',m),V77=metric('V77',m)))
for path,rows in [(R/'PUBLIC_NATURAL_FIXED_GEOMETRY_SAVED_COMPARISON.csv',csvrows),(R/'FIXED_COHORT_IDS.csv',cohortrows)]:
 with path.open('x',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
freeze=load(V/'runs/PAIRED_SUPPORT_V1/MODEL_FREEZE.json');assert freeze['best_update']==600 and freeze['threshold']==.5
report=dict(PASS=True,schema='V77_SAVED_ONLY_PUBLIC_NATURAL_FIXED_GEOMETRY_COMPARISON',fixed_threshold=.5,V76_update=200,V77_update=600,checkpoint=freeze['checkpoint'],same_primary_admission_and_IDs=True,old_V76_published_cohort_counts_metrics_reconciled=True,base_public_VAL_instances=800,base_all_slots=6400,base_known_primary_slots=2018,definitions=dict(error='Euclidean ||r-q|| native pixels <=3, exactly the previous V76 rule.',strong='Frozen teacher rule prefix STRONG_3X3; source silhouette support, not semantic human/native8 gold.',central_cell='Linf=max(abs((r-q)/L))<=1/18, exactly prior native9 center-cell square rule; not L2 radius or newly chosen tolerance.',labels='Teacher stays at reference r; model crop remains emulated q. Unknown/invalid/ref-MISS not negatives.',threshold='support_logit>=0 means VISIBLE at fixed p=.5; no threshold or checkpoint selection here.'),cohorts=records,strict_intersection_self_slices=self_slices,interpretation=['Errors remaining in the <=3px+strong3x3+central-cell intersection cannot all be attributed to large query displacement, crop-reference MISS or footprint-fallback labels.','This is a source-only controlled comparison of saved model outputs; it does not isolate encoder/identity/scale as a causal mechanism.','Known support at reference r is a conditional source proxy. Exact query-pixel occupancy and human semantic reference-corner D/E/S are not inferred.','V77 improves controlled visibility suppression while preserving substantial natural-visible false-covered errors; inspect the shown C/V tradeoff rather than declaring the full goal solved.'],query_labels_admission_threshold_checkpoint_changes=False,input_arrays_decoded=['query8_xy','L'],RGB_mask_P3_model_current_forward_fit_PnP=0,bindings=[bind(p)for p in [OLD/'source_failure_diagnosis/PUBLIC_VAL_SAVED_FAILURE_DIAGNOSIS.json',OLD/'gt_corpus_build/GT_ONLY_TEACHER_CORPUS.npz',*paths.values(),V/'runs/PAIRED_SUPPORT_V1/MODEL_FREEZE.json',Path(pack['manifest']['path']),Path(pack['lineage']['path']),R/'PUBLIC_NATURAL_FIXED_GEOMETRY_SAVED_COMPARISON.csv',R/'FIXED_COHORT_IDS.csv',Path(__file__)]])
p=R/'PUBLIC_NATURAL_FIXED_GEOMETRY_SAVED_COMPARISON.json';save(p,report);print(json.dumps(dict(PASS=True,report=bind(p),strict_intersection=primary)))
