"""Saved-only minSHA diagnostic selection. No RGB/mask/features/weights I/O."""
from pathlib import Path
import hashlib,json,csv,math,datetime
import numpy as np
R=Path(__file__).resolve().parents[1];O=Path(__file__).resolve().parent;RUN=R/'runs/YCB_EXTERNAL_RAY_V1'
def read(p):return json.loads(Path(p).read_text())
def bind(p):
 p=Path(p);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def check(b):assert bind(b['path'])['sha256']==b['sha256'],b['path']
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def plain(x):
 if isinstance(x,np.ndarray):return plain(x.tolist())
 if isinstance(x,np.generic):return plain(x.item())
 if isinstance(x,list):return [plain(v) for v in x]
 if isinstance(x,float) and not math.isfinite(x):return {'nonfinite':str(x)}
 return x
scope=read(O/'SCOPE_BEFORE_INDIVIDUAL_SCORES_AND_RGB.json');entry=read(O/'SELECTION_CODE_LOCK_BEFORE_SCORES.json');assert entry['code']==bind(__file__) and entry['scope']==bind(O/'SCOPE_BEFORE_INDIVIDUAL_SCORES_AND_RGB.json')
p=read(R/'FIT_PROTOCOL_BEFORE_SOURCE_ARRAYS.json');complete=read(RUN/'FIT_COMPLETE.json');assert complete['completed'] and complete['optimizer_updates']==2000
for k in ('model_freeze','scores','source_results'):check(complete[k])
freeze=read(complete['model_freeze']['path']);assert freeze['protocol']==bind(R/'FIT_PROTOCOL_BEFORE_SOURCE_ARRAYS.json') and freeze['external_probability_threshold']==.5
check(p['bindings']['teacher_GT']);gtpath=Path(p['bindings']['teacher_GT']['path'])
with np.load(gtpath,allow_pickle=False) as z:gt={k:z[k] for k in z.files}
with np.load(complete['scores']['path'],allow_pickle=False) as z:s={k:z[k] for k in z.files}
assert np.array_equal(s['id'],gt['id']) and np.array_equal(s['split'],gt['split']) and s['external_logit'].shape==(4200,8) and np.isfinite(s['external_logit']).all()
source=read(p['bindings']['source_input_bindings']['path']);pack=next(x for x in source['packs'] if x['source']=='YCB_PUBLIC_REAL')
for k in ('manifest','lineage','seal'):check(pack[k])
manifest=read(pack['manifest']['path']);check(manifest['input_query_manifest']);queries=[json.loads(x) for x in Path(manifest['input_query_manifest']['path']).read_text().splitlines()];lines=[json.loads(x) for x in Path(pack['lineage']['path']).read_text().splitlines()]
assert len(queries)==len(lines)==4200 and all(x['id']==str(gt['id'][i])==queries[i]['id'] and x['split']==str(gt['split'][i])==queries[i]['split'] for i,x in enumerate(lines))
q=np.asarray([x['initial_points'] for x in queries],np.float64);L=np.asarray([x['roi_side'] for x in queries],np.float64);r=gt['reference_xy8'];assert q.shape==r.shape==(4200,8,2) and np.isfinite(L).all() and (L>0).all()
offset=(r-q)/L[:,None,None];inside=np.isfinite(offset).all(-1)&(offset>=-.5).all(-1)&(offset<.5).all(-1);assert np.array_equal(inside,s['reference_inside_qcrop_GT_only'])
qvalid=s['query_center_valid'].astype(bool);available=s['input_available'].astype(bool);assert np.array_equal(qvalid,np.asarray([x['query_center_valid8'] for x in lines],bool))
known=np.isin(gt['label8'],[0,1]);assert np.array_equal(known&qvalid&available&inside,s['loss_admitted_GT_only'])
dist=np.linalg.norm(q-r,axis=-1);near=(gt['split']=='val')[:,None]&known&qvalid&available&inside&np.isfinite(dist)&(dist<=3.)
pred=(s['external_logit']>=0).astype(np.int8);selfproxy=gt['convex_self_visible_proxy8_separate'];assert np.isin(selfproxy[near],[0,1]).all()
cohort=[]
for i,k in np.argwhere(near):
 digest=hashlib.sha256((str(gt['id'][i])+'|'+str(int(k))).encode('utf-8')).hexdigest();logit=float(s['external_logit'][i,k]);prob=1/(1+math.exp(-logit))
 cohort.append(dict(frame_index=int(i),native_k=int(k),id=str(gt['id'][i]),split=str(gt['split'][i]),GT=int(gt['label8'][i,k]),pred=int(pred[i,k]),convexselfproxy=int(selfproxy[i,k]),q_x=float(q[i,k,0]),q_y=float(q[i,k,1]),r_x=float(r[i,k,0]),r_y=float(r[i,k,1]),L=float(L[i]),Euclidean_q_r_native_px=float(dist[i,k]),external_logit_FP32=logit,p_external_derived_display=prob,selector_sha256=digest))
with (O/'ALL_NEAR3_COHORT_ROWS.csv').open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(cohort[0]));w.writeheader();w.writerows(cohort)
selected=[];strata=[]
for rule in scope['selection']['error_strata']+scope['selection']['correct_controls']:
 eligible=[x for x in cohort if x['GT']==rule['GT'] and x['pred']==rule['pred'] and (rule['convexselfproxy']=='either' or x['convexselfproxy']==rule['convexselfproxy'])]
 item={'name':rule['name'],'rule':rule,'eligible_N':len(eligible),'empty':not bool(eligible)}
 if eligible:
  chosen=min(eligible,key=lambda x:x['selector_sha256']);item['selected']=chosen;selected.append(dict(case_number=len(selected),stratum=rule['name'],**chosen))
 strata.append(item)
# Select metadata and exact numerical teacher fields only; no RGB/mask/feature reading.
metapath=gtpath.parent/'ORDERED_METADATA_INDEX_GT_ONLY.jsonl';metadata={}
want={x['frame_index'] for x in selected}
for i,line in enumerate(metapath.open()):
 if i in want:metadata[i]=json.loads(line)
case_meta=[]
for row in selected:
 i,k=row['frame_index'],row['native_k'];teacher={}
 for key,a in gt.items():
  if a.ndim>=2 and a.shape[:2]==(4200,8):teacher[key]=plain(a[i,k])
  elif a.shape[0]==4200:teacher[key]=plain(a[i])
  else:raise AssertionError((key,a.shape))
 assert metadata[i]['id']==row['id'] and metadata[i]['row_index']==i and metadata[i]['metadata']['image']==queries[i]['image']
 case_meta.append(dict(selection=row,teacher_original_numeric_values=teacher,teacher_array_dtypes={k:str(v.dtype) for k,v in gt.items()},source_metadata_original=metadata[i],input_query_metadata=queries[i],input_lineage_metadata=lines[i]))
summary=dict(completed=True,scope=bind(O/'SCOPE_BEFORE_INDIVIDUAL_SCORES_AND_RGB.json'),selection_code=bind(__file__),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),FIT_COMPLETE=bind(RUN/'FIT_COMPLETE.json'),model_freeze=complete['model_freeze'],scores=complete['scores'],teacher=p['bindings']['teacher_GT'],query_manifest=manifest['input_query_manifest'],input_pack=pack,selection_rule_exact=scope['selection'],cohort_N=int(near.sum()),cohort_GT_clear_N=int(np.sum(near&(gt['label8']==0))),cohort_GT_external_N=int(np.sum(near&(gt['label8']==1))),cohort_FP_N=int(np.sum(near&(gt['label8']==0)&(pred==1))),cohort_FN_N=int(np.sum(near&(gt['label8']==1)&(pred==0))),strata=strata,selected_case_N=len(selected),selected_cases=selected,no_individual_RGB_mask_feature_weights_reads_yet=True,new_inference_fit_current_PnP=0,coords_L_labels_order_unchanged=True)
write(O/'SELECTION_AND_COHORT_LOCK_BEFORE_RGB.json',summary);write(O/'SELECTED_CASE_METADATA_GT_ONLY.json',dict(completed=True,cases=case_meta,metadata_source=bind(metapath),meaning='Allreference/depth/owner/labels GT-only; no physicalcue/humanD/E/S equivalence.'))
write(O/'SELECTION_SEAL_BEFORE_RGB.json',dict(completed=True,PASS=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=bind(O/'SCOPE_BEFORE_INDIVIDUAL_SCORES_AND_RGB.json'),code=bind(__file__),files=[bind(O/x) for x in ('ALL_NEAR3_COHORT_ROWS.csv','SELECTION_AND_COHORT_LOCK_BEFORE_RGB.json','SELECTED_CASE_METADATA_GT_ONLY.json','SELECTION_CODE_LOCK_BEFORE_SCORES.json')],source_RGB_mask_features_weights_current_reads=0,new_forward_fit_PnP=0,no_favorable_replacement=True))
print(json.dumps({'selection_seal':bind(O/'SELECTION_SEAL_BEFORE_RGB.json'),'cohort_N':len(cohort),'strata':strata,'selected_case_N':len(selected)},ensure_ascii=False))
