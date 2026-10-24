from pathlib import Path
import sys,csv,json,time
import numpy as np
R=Path(__file__).resolve().parent;OLD=R.parent/'external_ray_proxy_head_20261022_v60';sys.path.insert(0,str(OLD));from metrics import classification
from common import sha,read,utc
start=time.monotonic();rr=list(csv.DictReader(open(R/'PRESENCE_QUERY_LABELS.csv')));z=np.load(R/'SYNTH_PRESENCE_OUTPUTS.npz');inputs=read(OLD/'PREPARED_INPUTS_SYNTH.json.gz');index={u['id']:i for i,u in enumerate(inputs)};valid=[]
for r in rr:
 i=index[r['id']];k=int(r['native_corner']);r['presence_probability']=float(z['probability'][i,k]);r['predicted_present']=r['presence_probability']>=.5;r['false_presence']=r['valid']=='True'and r['predicted_present']and r['presence_target']=='0';r['false_absence']=r['valid']=='True'and not r['predicted_present']and r['presence_target']=='1';r['prediction_valid']=r['valid']=='True'
 if r['prediction_valid']:valid.append(r)
with(R/'SYNTH_QUERY_PREDICTIONS.csv').open('x')as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
ss=[]
for split in ['train','validation']:
 a=[r for r in valid if r['split']==split]
 for attr in ['ALL','boundary','pixel_attribute','initial_geometric_role','initial_convex_self_hidden']:
  for key in ['ALL']if attr=='ALL'else sorted({r[attr]for r in a}):
   rows=a if attr=='ALL'else[r for r in a if r[attr]==key];s=classification([int(r['presence_target'])for r in rows],[r['presence_probability']for r in rows]);s['balanced_accuracy']=.5*(s['recall']+1-s['FPR'])if s['recall']is not None and s['FPR']is not None else None;ss.append(dict(split=split,stratification=attr,stratum=key,**s))
with(R/'STRATIFIED_METRICS.csv').open('x')as f:w=csv.DictWriter(f,fieldnames=list(ss[0]));w.writeheader();w.writerows(ss)
# Distinct objective: characterize disagreement with historical ray targets, without relabeling either task.
old={(r['id'],r['native_corner']):r for r in csv.DictReader(open(OLD/'SYNTH_QUERY_OUTPUTS.csv'))};dis=[]
for split in ['train','validation']:
 a=[r for r in valid if r['split']==split];dis.append(dict(split=split,n=len(a),presence_and_ray_blocked=sum(r['presence_target']=='1'and old[(r['id'],r['native_corner'])]['target']=='1'for r in a),absence_and_ray_not_blocked=sum(r['presence_target']=='0'and old[(r['id'],r['native_corner'])]['target']=='0'for r in a),note='Different pixel-presence and virtual-ray targets; no ray-head accuracy claim for presence'))
with(R/'TARGET_DISTINCTION.csv').open('x')as f:w=csv.DictWriter(f,fieldnames=list(dis[0]));w.writeheader();w.writerows(dis)
with(R/'EVALUATION_COMPLETE.json').open('x')as f:json.dump(dict(UTC=utc(),seconds=time.monotonic()-start,valid_queries=len(valid),invalid_queries=len(rr)-len(valid),no_pose_calls=True,no_real_labels=True,head_sha256=sha(R/'BEST_PRESENCE_HEAD.pt')),f,indent=2)
print('PRESENCE_STRATA_COMPLETE',len(ss),flush=True)
