"""One TRAIN-only noGT pixel-scale eligibility rule, stored whole poses."""
from pathlib import Path
import json,csv,gzip,hashlib,sys,time
import numpy as np
R=Path(__file__).resolve().parent;T=R.parent;sys.path[:0]=[str(T/'geometric_influence_20261021_v19')];import experiment as G
p=G.read(R/'PROTOCOL.json')
for k,v in p['bindings'].items():assert G.sha(v['path'])==v['sha256']
start=time.monotonic();box=G.index(G.rows(p['bindings']['boxes']['path']));up=G.rows(p['bindings']['TRAIN_membership']['path']);sizes={u['id']:float(np.max(np.asarray(box[u['id']]['detector_box_xyxy_raw'])[2:]-np.asarray(box[u['id']]['detector_box_xyxy_raw'])[:2])*640/max(np.asarray(u['raw_hw'])+200))for u in up};train=[sizes[u['id']]for u in up if u['split']=='train'];threshold=float(np.quantile(train,.25));G.write(R/'THRESHOLD_TRAIN_ONLY_NO_GT.json',dict(scope='TRAIN768 boxes and rawHW only',n=len(train),quantile=.25,threshold_network_px=threshold,GT_reads=0,validation_members_used_for_threshold=0))
source=G.rows(p['bindings']['old_policy']['path']);out=R/'SYNTHETIC_VAL';out.mkdir(exist_ok=False);ps=[]
for s in source:
 arm='JOINT_EVIDENCE'if sizes[s['id']]<=threshold else'JK_CONTROL';q=dict(id=s['id'],recording=s['recording'],methods=dict(s['methods']),bank_ids=s['bank_ids'],scale_net_px=sizes[s['id']],small=sizes[s['id']]<=threshold,selected_source=arm);q['methods']['SCALE_POLICY']=s['methods'][arm];ps.append(q)
G.save(out/'PREDICTIONS.jsonl.gz',ps);G.write(out/'SEAL.json',dict(status='SEALED_BEFORE_SCORING',prediction_sha256=G.sha(out/'PREDICTIONS.jsonl.gz'),n=len(ps),small=sum(q['small']for q in ps),seconds=time.monotonic()-start,new_training_forward_RGB_PnP=0))
# Labels read only after complete frozen prediction seal.
rows={z['id']:z for z in csv.DictReader(Path(p['bindings']['old_metrics']['path']).open())};groups={a:[]for a in ps[0]['methods']};paired=[]
for q in ps:
 r=rows[q['id']];z=dict(id=q['id'],small=q['small'],scale_net_px=q['scale_net_px'],selected_source=q['selected_source'])
 for a in groups:
  sourcearm=q['selected_source']if a=='SCALE_POLICY'else a;m=dict(pose_available=True,translation_cm=float(r[sourcearm+'_translation_cm']),rotation_deg=float(r[sourcearm+'_rotation_deg']),success=r[sourcearm+'_success']=='True');groups[a].append(m);z[a+'_id']=q['methods'][a]['candidate_id'];z.update({a+'_'+k:v for k,v in m.items()})
 paired.append(z)
st={a:G.stats(x)for a,x in groups.items()};why={a:G.gate(st['SCALE_POLICY'],s)for a,s in st.items()if a!='SCALE_POLICY'};rd={a:dict(recover=sum(not x['success']and y['success']for x,y in zip(groups[a],groups['SCALE_POLICY'])),damage=sum(x['success']and not y['success']for x,y in zip(groups[a],groups['SCALE_POLICY'])))for a in why};result=dict(methods=st,gate_passed=not any(why.values()),gate_reasons=why,recover_damage=rd,threshold_from_TRAIN_only=True,exposed_development=True,new_independent_test=0,new_training_forward_RGB_PnP=0);G.csvwrite(out/'PAIRED_ROWS.csv',paired);G.write(out/'EVALUATION.json',result);print(json.dumps(result,indent=2))
