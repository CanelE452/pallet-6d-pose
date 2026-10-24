"""Natural YCB class/split breakdown; GT class is analysis only, never forward."""
from pathlib import Path
import json,hashlib
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent
V76=P/'query_visibility_reference_local_support_20261023_v76';V77=P/'query_visibility_controlled_pairs_20261023_v77';V78=P/'query_visibility_dino_query_support_20261023_v78'
def b(p):
 p=Path(p).resolve();return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def ck(x):assert b(x['path'])==x
def write(p,x):
 with p.open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def metrics(y,z,mask):
 y=y[mask];p=z[mask]>=0;cm=[[int(np.sum((y==g)&(p==j)))for j in [0,1]]for g in [0,1]]
 cc,cv=cm[0];vc,vv=cm[1]
 ratio=lambda a,c:None if c==0 else a/c
 return dict(N=len(y),CM_rows_CV_cols_CV=cm,C_recall=ratio(cc,cc+cv),V_recall=ratio(vv,vc+vv),C_precision=ratio(cc,cc+vc),V_precision=ratio(vv,vv+cv))
def main():
 out=R/'object_split_diagnosis';out.mkdir(exist_ok=False)
 index=V76/'gt_corpus_build/ORDERED_METADATA_INDEX_GT_ONLY.jsonl';gtnpz=V76/'gt_corpus_build/GT_ONLY_TEACHER_CORPUS.npz'
 source_files={'V76':V77/'frozen_V76_source_baseline/SOURCE_SCORES.npz','V77':V77/'runs/PAIRED_SUPPORT_V1/SELECTED_SOURCE_SCORES.npz','V78':V78/'runs/SOURCE_DINO_V1/SELECTED_SOURCE_SCORES.npz'}
 # Fix all categories and seen/unseen membership using metadata before reading scores.
 rows=[json.loads(l)for l in index.read_text().splitlines()];assert len(rows)==5203
 ids=np.array([r['id']for r in rows]);part=np.array([r['split']for r in rows]);obj=np.array([r.get('object_id',-1)for r in rows]);public=np.array([r['source']=='YCB_PUBLIC_REAL'for r in rows])
 train_ids=sorted(set(obj[public&(part=='train')].tolist()));val_ids=sorted(set(obj[public&(part=='val')].tolist()))
 proto=dict(frozen=True,code=b(Path(__file__)),metadata=b(index),source_GT=b(gtnpz),source_scores={k:b(v)for k,v in source_files.items()},
  rule='All publicYCB natural rows, each objectID and train/VAL; plus fixed metadata seenTRAIN vs unseenTRAIN grouping. Identical original primary mask, no row filtering/reselection/prototype construction.',
  train_objectIDs=train_ids,val_objectIDs=val_ids,VAL_only_objectIDs=sorted(set(val_ids)-set(train_ids)),
  interpretation='GTcategory is stratification only, never model input/prompt or test-template lookup. Capturing scene/category co-vary, so no causal split attribution. Global source gate unchanged.',current_forward_fit_PnP=0)
 write(out/'PROTOCOL_BEFORE_CATEGORY_SCORES.json',proto)
 with np.load(gtnpz,allow_pickle=False)as z:y=z['label8'];assert np.array_equal(z['id'],ids)
 scores={}
 for k,f in source_files.items():
  ck(proto['source_scores'][k])
  with np.load(f,allow_pickle=False)as z:
   assert np.array_equal(z['id'][:5203],ids)
   scores[k]=dict(z=z['support_logit'][:5203],primary=z['primary_loss_mask'][:5203])
 for v in scores.values():assert np.array_equal(v['primary'],scores['V78']['primary'])
 result=[]
 for name,s in scores.items():
  for split in ['train','val']:
   for category in sorted(set(obj[public].tolist())):
    frame=public&(part==split)&(obj==category);mask=np.broadcast_to(frame[:,None],y.shape)&s['primary']
    result.append(dict(model=name,split=split,objectID=category,category_present_in_TRAIN=category in train_ids,frames=int(frame.sum()),metrics=metrics(y,s['z'],mask)))
 groups=[]
 for name,s in scores.items():
  for seen in [True,False]:
   frame=public&(part=='val')&(np.isin(obj,train_ids)==seen);mask=np.broadcast_to(frame[:,None],y.shape)&s['primary']
   groups.append(dict(model=name,VAL_group='TRAIN_SEEN_CATEGORIES'if seen else'TRAIN_UNSEEN_CATEGORIES',frames=int(frame.sum()),metrics=metrics(y,s['z'],mask)))
  m=np.broadcast_to((public&(part=='val'))[:,None],y.shape)&s['primary'];assert metrics(y,s['z'],m)['N']==2018
 write(out/'OBJECT_SPLIT_SAVED_RESULTS.json',dict(completed=True,protocol=b(out/'PROTOCOL_BEFORE_CATEGORY_SCORES.json'),natural_class_metrics=result,VAL_seen_unseen=groups,GTcategory_used_for_inference=False,model_threshold_checkpoint_changes=False,current_forward_fit_PnP=0))
 write(out/'SEAL.json',dict(completed=True,files=[b(f)for f in sorted(out.iterdir())if f.is_file()]))
 print(json.dumps(dict(VAL_only_objectIDs=proto['VAL_only_objectIDs'],group_results=groups),ensure_ascii=False))
if __name__=='__main__':main()
