"""Retain original+JK bank and add single frozen ROI pose, own H1 and norm."""
import pathlib,sys,importlib.util,json,time,argparse,copy,os
import numpy as np
import torch
import cv2
R=pathlib.Path(__file__).resolve().parent;T=R.parent
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
V=load('roi30',T/'frozen_roi_reinference_20261021_v30/experiment.py');J=load('jk_export24',T/'jackknife_bank_20261021_v24/experiment_v3.py');G=J.G;L=load('unionlearn24',T/'jackknife_bank_20261021_v24/learn.py');U=R/'SYNTHETIC';OLD=J.R/'SYNTHETIC_v3'
def bound():
 p=G.read(R/'PROTOCOL.json')
 for k,v in p['bindings'].items():assert G.sha(v['path'])==v['sha256'],k
 return p
def generate():
 bound();V.bound();V.firewall();out=R/'TRAIN_ROI';out.mkdir(exist_ok=False);start=time.monotonic();tools=V.init();fn=V.functions();fs=[u for u in V.rows(V.SAN)if u['split']=='train'];box=V.index(V.rows(V.BOX));ps=[];tt=[]
 for i,u in enumerate(fs):
  if i%50==0:V.gpu()
  tick=time.monotonic();assert V.sha(u['image'])==u['image_sha256'];image=cv2.imread(u['image']);h,w=u['raw_hw'];b=np.asarray(box[u['id']]['detector_box_xyxy_raw'],float);center=(b[:2]+b[2:])/2;radius=(b[2:]-b[:2])*.75;x0,y0=np.maximum(np.floor(center-radius),0).astype(int);x1,y1=np.minimum(np.ceil(center+radius),[w,h]).astype(int);assert x1>x0 and y1>y0;p=V.infer_one(u,image[y0:y1,x0:x1],np.array([x0,y0]),tools,fn);ps.append(dict(id=u['id'],image_sha256=u['image_sha256'],ROI=p));tt.append(dict(id=u['id'],seconds=time.monotonic()-tick,proposal_available=p['pose']['available']))
  if(i+1)%100==0:print(json.dumps(dict(stage='TRAIN_FROZEN_ROI',done=i+1,total=len(fs),seconds=time.monotonic()-start)),flush=True)
  assert time.monotonic()-start<600
 G.save(out/'PREDICTIONS.jsonl.gz',ps);G.csvwrite(out/'TIMING.csv',tt);G.write(out/'SEAL.json',dict(status='SEALED_BEFORE_TARGETS',n=len(ps),seconds=time.monotonic()-start,sha256=G.sha(out/'PREDICTIONS.jsonl.gz'),new_detector_N3_pairs=len(ps),new_training=0,new_SAM=0,max_allocated=torch.cuda.max_memory_allocated(),max_reserved=torch.cuda.max_memory_reserved()));print(G.read(out/'SEAL.json'))
def features():
 bound();V.firewall();cv2.setNumThreads(1);U.mkdir(exist_ok=False);start=time.monotonic();fs=G.rows(OLD/'HEAD_FEATURES.jsonl.gz');original=G.index(G.rows(G.B/'d1_synthetic_candidates_v1/HEAD_FEATURES.jsonl.gz'));pred=G.index(G.rows(R/'TRAIN_ROI/PREDICTIONS.jsonl.gz')+G.rows(V.R/'SYNTHETIC_VAL/PREDICTIONS.jsonl.gz'));root=G.B/'d1_synthetic_candidates_v1';result=[];tt=[]
 for i,f in enumerate(fs):
  tick=time.monotonic();p=pred[f['id']]['ROI']['pose'];new=[dict(candidate_id='ROI_FROZEN_SINGLE',source='FROZEN_ROI_N3_SUBPIX',R_physical=p['R_physical'],centroid=p['centroid'])]if p['available']else[]
  if new:
   inp,models=J.packet(i,original[f['id']],root);nf,added=J.export(f,inp,models,new)
  else:nf=copy.deepcopy(f);added=[]
  # Only bank-context scoregaps/pooling may change; preserve all incumbent poses/local26.
  before={c['candidate_id']:c for c in f['candidates']}
  for c in nf['candidates']:
   if c['candidate_id']in before:
    o=before[c['candidate_id']];assert c['actual_pose']==o['actual_pose']and c['eligible']==o['eligible']and c['features'][:26]==o['features'][:26]and c['features'][31:57]==o['features'][31:57]
  result.append(nf);tt.append(dict(id=f['id'],split=f['split'],original_actions=len(G.A.actions(original[f['id']])),JK_actions=len(G.A.actions(f)),union_actions=len(G.A.actions(nf)),ROI_available=p['available'],new_eligible=sum(c['eligible']for c in added),seconds=time.monotonic()-tick))
  if(i+1)%100==0:print(json.dumps(dict(stage='ROI_UNION_EXPORT',done=i+1,total=len(fs),seconds=time.monotonic()-start)),flush=True)
  assert time.monotonic()-start<900
 G.save(U/'HEAD_FEATURES.jsonl.gz',result);G.csvwrite(U/'TIMING.csv',tt);G.write(U/'NORMALIZER.json',G.A.fit_normalizer([f for f in result if f['split']=='train']));G.write(U/'SEAL.json',dict(status='SEALED_FEATURES_BEFORE_TARGETS',n=len(result),seconds=time.monotonic()-start,features_sha256=G.sha(U/'HEAD_FEATURES.jsonl.gz'),TRAIN_only_normalizer=True,all_incumbent_pose_eligibility_local26_bitwise_preserved=True,pooling_count_scoregap_recomputed=True,exact_minimum_fallbacks=J.RECONSTRUCTION_FALLBACKS));print(G.read(U/'SEAL.json'))
def targets():
 bound();fs=G.rows(U/'HEAD_FEATURES.jsonl.gz');gt=G.index(G.rows(G.B.parent/'pallet-statewise-end-to-end-recovery-v1/synthetic_v1/SYNTH_GT_LABELS.jsonl.gz'));old=G.index(G.rows(OLD/'C2_TARGETS.jsonl.gz'));labs=[];coverage=[]
 for f in fs:
  g=gt[f['id']];ref=dict(R=g['R_fixed'],t=g['t'],order=2);ms=[]
  for e in G.A.actions(f):
   m=G.metric(e['actual_pose'],ref);ms.append(dict(candidate_id=e['candidate_id'],translation_cm=m['translation_cm'],rotation_deg=m['rotation_deg'],success_5cm5deg=m['success']))
  labs.append(dict(id=f['id'],split=f['split'],action_metrics=ms));good=any(a['success_5cm5deg']for a in old[f['id']]['action_metrics']);ng=any(a['success_5cm5deg']for a in ms);coverage.append(dict(id=f['id'],split=f['split'],old_JK_good=good,union_good=ng,new_good=ng and not good))
 G.save(U/'C2_TARGETS.jsonl.gz',labs);G.csvwrite(U/'COVERAGE.csv',coverage);counts={s:dict(n=sum(c['split']==s for c in coverage),old_good=sum(c['old_JK_good']for c in coverage if c['split']==s),union_good=sum(c['union_good']for c in coverage if c['split']==s),new_good=sum(c['new_good']for c in coverage if c['split']==s))for s in ['train','validation']};G.write(U/'TARGET_SEAL.json',dict(counts=counts,GT_for_target_only=True,sha256=G.sha(U/'C2_TARGETS.jsonl.gz')));print(json.dumps(counts,indent=2))
def train():
 L.R=R;L.U=U;L.bound=bound;L.train(20261018)
def predict():
 bound();G.setup();V.firewall();out=R/'SYNTHETIC_VAL_s20261018';out.mkdir(exist_ok=False);start=time.monotonic();models={}
 for arm,path in [('BASELINE',G.V5/'s20261018/best.pt'),('CONTROL',J.R/'s20261018/best.pt'),('ROI_UNION',R/'s20261018/best.pt')]:
  c=torch.load(path,weights_only=False,map_location='cpu');m=G.M.make_model('MLP');m.load_state_dict(c['model_state_dict']);m.eval().requires_grad_(False);models[arm]=(m,c)
 old=G.index(G.rows(G.B/'d1_synthetic_candidates_v1/HEAD_FEATURES.jsonl.gz'));jk=G.index(G.rows(OLD/'HEAD_FEATURES.jsonl.gz'));ps=[]
 with torch.inference_mode():
  for f in G.rows(U/'HEAD_FEATURES.jsonl.gz'):
   if f['split']!='validation':continue
   p=dict(id=f['id'],recording=f['cluster_id'],methods={},bank_ids=[e['candidate_id']for e in G.A.actions(f)])
   for arm,(m,c)in models.items():
    ff=old[f['id']]if arm=='BASELINE'else jk[f['id']]if arm=='CONTROL'else f;x,es=G.A.inputs(ff,c['normalizer'],'H1');k=int(m(torch.from_numpy(x)).flatten().argmax());p['methods'][arm]=dict(candidate_id=es[k]['candidate_id'],pose=es[k]['actual_pose'])
   ps.append(p)
 G.save(out/'PREDICTIONS.jsonl.gz',ps);G.write(out/'SEAL.json',dict(status='SEALED_BEFORE_SCORING',prediction_sha256=G.sha(out/'PREDICTIONS.jsonl.gz'),n=len(ps),seconds=time.monotonic()-start,GT_inputs=0,new_RGB_PnP=0));print(dict(stage='SEALED_ROI_UNION_SELECTION',n=len(ps)))
def evaluate():
 bound();out=R/'SYNTHETIC_VAL_s20261018';ps=G.rows(out/'PREDICTIONS.jsonl.gz');assert G.sha(out/'PREDICTIONS.jsonl.gz')==G.read(out/'SEAL.json')['prediction_sha256'];labs=G.index(G.rows(U/'C2_TARGETS.jsonl.gz'));groups={a:[]for a in ['BASELINE','CONTROL','ROI_UNION']};paired=[]
 for p in ps:
  by={a['candidate_id']:a for a in labs[p['id']]['action_metrics']};r=dict(id=p['id'],candidate_n=len(p['bank_ids']),bank_good=any(by[k]['success_5cm5deg']for k in p['bank_ids']))
  for a,q in p['methods'].items():
   y=by[q['candidate_id']];m=dict(pose_available=True,translation_cm=y['translation_cm'],rotation_deg=y['rotation_deg'],success=y['success_5cm5deg']);groups[a].append(m);r[a+'_id']=q['candidate_id'];r.update({a+'_'+k:v for k,v in m.items()})
  paired.append(r)
 st={a:G.stats(z)for a,z in groups.items()};why={a:G.gate(st['ROI_UNION'],st[a])for a in ['BASELINE','CONTROL']};rd={a:dict(recover=sum(not x['success']and y['success']for x,y in zip(groups[a],groups['ROI_UNION'])),damage=sum(x['success']and not y['success']for x,y in zip(groups[a],groups['ROI_UNION'])))for a in ['BASELINE','CONTROL']};ev=dict(methods=st,gate_passed=not any(why.values()),gate_reasons=why,recover_damage=rd,bank_good=sum(r['bank_good']for r in paired),selection_failures=sum(r['bank_good']and not r['ROI_UNION_success']for r in paired),new_independent_test=0,adapted_after_negative_ROI_replacement=True);G.csvwrite(out/'PAIRED_ROWS.csv',paired);G.write(out/'EVALUATION.json',ev);print(json.dumps(ev,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['generate','features','targets','train','predict','evaluate']);a=p.parse_args();globals()[a.stage]()
