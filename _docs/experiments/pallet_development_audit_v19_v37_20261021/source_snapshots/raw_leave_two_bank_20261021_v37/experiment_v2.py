"""Same old observations; fixed28 raw leave2 proposals, actual bank retraining."""
import pathlib,sys,importlib.util,itertools,time,json,argparse,copy
import numpy as np
import torch
import cv2
R=pathlib.Path(__file__).resolve().parent;T=R.parent
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
J=load('j24',T/'jackknife_bank_20261021_v24/experiment_v3.py');G=J.G;L=load('l24',J.R/'learn.py');OLD=J.R/'SYNTHETIC_v3';U=R/'SYNTHETIC_v2'
def bound():
 p=G.read(R/'PROTOCOL_V2.json')
 for k,v in p['bindings'].items():assert G.sha(v['path'])==v['sha256'],k
 return p
def newposes(u,fn):
 X=np.asarray(u['native_xyz']);xy=np.asarray(u['initial_points'])[:8];K=np.asarray(u['K']);Q=np.asarray(u['initial_pose']['R_cf']).T@u['initial_pose']['R_physical'];valid=np.isfinite(xy).all(-1)&~np.all(xy==-1,axis=1);new=[];calls=0
 for dropped in itertools.combinations(np.flatnonzero(valid),2):
  take=valid.copy();take[list(dropped)]=False
  if take.sum()<6:continue
  calls+=1
  try:z=fn['solve'](X,xy,K,take)
  except cv2.error:continue
  if z is None:continue
  rot,t,_=z
  if not np.isfinite(rot).all()or not np.isfinite(t).all()or((X@rot.T+t)[:,2]<=0).any():continue
  new.append(dict(candidate_id='JK2_RAW/'+','.join(map(str,dropped)),source='TWO_FORECAST_OUTLIER',R_physical=(rot@Q).tolist(),centroid=(rot@X.mean(0)+t).tolist()))
 return new,calls
def features():
 bound();G.firewall();cv2.setNumThreads(1);U.mkdir(exist_ok=False);start=time.monotonic();fs=G.rows(OLD/'HEAD_FEATURES.jsonl.gz');up=G.index(G.rows(G.B/'d1_upstream_v1/SYNTHETIC1003_UPSTREAM.jsonl.gz'));pilot=G.index(G.rows(T/'raw_two_outlier_probe_20261021_v36/SEALED_TRAIN_PILOT_PREDICTIONS.jsonl.gz'));fn=J.P.load_functions();rr=[];tt=[];calls=0;preserved=0
 for i,f in enumerate(fs):
  tick=time.monotonic()
  if f['id']in pilot:
   new=[dict(candidate_id=c['candidate_id'],source='TWO_FORECAST_OUTLIER',R_physical=c['actual_pose']['R_physical'],centroid=c['actual_pose']['centroid'])for c in pilot[f['id']]['new_candidates']];n=0
  else:new,n=newposes(up[f['id']],fn)
  inp,m=J.packet(i,f,G.B/'d1_synthetic_candidates_v1');nf,added=J.export(f,inp,m,new);old={c['candidate_id']:c for c in f['candidates']}
  for c in nf['candidates']:
   if c['candidate_id']in old:
    a=old[c['candidate_id']];assert c['actual_pose']==a['actual_pose']and c['eligible']==a['eligible']and c['features'][:26]==a['features'][:26]and c['features'][31:57]==a['features'][31:57];preserved+=1
  calls+=n;rr.append(nf);tt.append(dict(id=f['id'],split=f['split'],old_actions=len(G.A.actions(f)),new_poses=len(new),new_eligible=sum(c['eligible']for c in added),union_actions=len(G.A.actions(nf)),PnP_pairs=n,seconds=time.monotonic()-tick));(U/'PROGRESS.jsonl').open('a').write(json.dumps(dict(done=i+1,total=len(fs),seconds=time.monotonic()-start,new_PnP_pairs=calls,last_id=f['id']))+'\n')
  if(i+1)%50==0:print(json.dumps(dict(stage='LEAVE2_BANK_EXPORT',done=i+1,total=len(fs),seconds=time.monotonic()-start)),flush=True)
  assert time.monotonic()-start<900,'900second fixed exportcap; do not autoextend'
 G.save(U/'HEAD_FEATURES.jsonl.gz',rr);G.csvwrite(U/'TIMING.csv',tt);G.write(U/'NORMALIZER.json',G.A.fit_normalizer([f for f in rr if f['split']=='train']));G.write(U/'SEAL.json',dict(status='SEALED_BEFORE_TARGETS',n=len(rr),seconds=time.monotonic()-start,sha256=G.sha(U/'HEAD_FEATURES.jsonl.gz'),new_PnP_pairs=calls,old_pose_eligibility_local26_preserved=preserved,new_RGB_SAM=0,TRAIN_only_normalizer=True,bank_scoregap_mean_max_count_recomputed=True,exact_minimum_fallbacks=J.RECONSTRUCTION_FALLBACKS));print(G.read(U/'SEAL.json'))
def targets():
 bound();fs=G.rows(U/'HEAD_FEATURES.jsonl.gz');gt=G.index(G.rows(G.B.parent/'pallet-statewise-end-to-end-recovery-v1/synthetic_v1/SYNTH_GT_LABELS.jsonl.gz'));old=G.index(G.rows(OLD/'C2_TARGETS.jsonl.gz'));labs=[];cv=[]
 for f in fs:
  g=gt[f['id']];ref=dict(R=g['R_fixed'],t=g['t'],order=2);ms=[]
  for e in G.A.actions(f):
   m=G.metric(e['actual_pose'],ref);ms.append(dict(candidate_id=e['candidate_id'],translation_cm=m['translation_cm'],rotation_deg=m['rotation_deg'],success_5cm5deg=m['success']))
  labs.append(dict(id=f['id'],split=f['split'],action_metrics=ms));before=any(a['success_5cm5deg']for a in old[f['id']]['action_metrics']);after=any(a['success_5cm5deg']for a in ms);cv.append(dict(id=f['id'],split=f['split'],before_good=before,after_good=after,new_good=after and not before))
 G.save(U/'C2_TARGETS.jsonl.gz',labs);G.csvwrite(U/'COVERAGE.csv',cv);counts={s:dict(n=sum(c['split']==s for c in cv),before_good=sum(c['before_good']for c in cv if c['split']==s),union_good=sum(c['after_good']for c in cv if c['split']==s),new_good=sum(c['new_good']for c in cv if c['split']==s))for s in ['train','validation']};G.write(U/'TARGET_SEAL.json',dict(counts=counts,sha256=G.sha(U/'C2_TARGETS.jsonl.gz'),GT_target_only=True));print(json.dumps(counts,indent=2))
def train():
 bound();cs=G.read(U/'TARGET_SEAL.json')['counts'];assert cs['train']['new_good']>=5 and cs['validation']['new_good']>0;L.R=R;L.U=U;L.bound=bound;L.train(20261018)
def predict():
 bound();G.setup();G.firewall();out=R/'SYNTHETIC_VAL_s20261018';out.mkdir(exist_ok=False);start=time.monotonic();models={}
 for arm,path in [('BASELINE',G.V5/'s20261018/best.pt'),('CONTROL',J.R/'s20261018/best.pt'),('LEAVE2',R/'s20261018/best.pt')]:
  c=torch.load(path,weights_only=False,map_location='cpu');m=G.M.make_model('MLP');m.load_state_dict(c['model_state_dict']);m.eval().requires_grad_(False);models[arm]=(m,c)
 old=G.index(G.rows(G.B/'d1_synthetic_candidates_v1/HEAD_FEATURES.jsonl.gz'));jk=G.index(G.rows(OLD/'HEAD_FEATURES.jsonl.gz'));ps=[]
 with torch.inference_mode():
  for f in G.rows(U/'HEAD_FEATURES.jsonl.gz'):
   if f['split']!='validation':continue
   p=dict(id=f['id'],recording=f['cluster_id'],methods={},bank_ids=[e['candidate_id']for e in G.A.actions(f)])
   for arm,(m,c)in models.items():
    ff=old[f['id']]if arm=='BASELINE'else jk[f['id']]if arm=='CONTROL'else f;x,es=G.A.inputs(ff,c['normalizer'],'H1');k=int(m(torch.from_numpy(x)).flatten().argmax());p['methods'][arm]=dict(candidate_id=es[k]['candidate_id'],pose=es[k]['actual_pose'])
   ps.append(p)
 G.save(out/'PREDICTIONS.jsonl.gz',ps);G.write(out/'SEAL.json',dict(status='SEALED_BEFORE_SCORING',prediction_sha256=G.sha(out/'PREDICTIONS.jsonl.gz'),n=len(ps),seconds=time.monotonic()-start,GT_inputs=0,new_RGB_PnP=0));print(dict(stage='SEALED_LEAVE2_SELECTION',n=len(ps)))
def evaluate():
 bound();out=R/'SYNTHETIC_VAL_s20261018';ps=G.rows(out/'PREDICTIONS.jsonl.gz');assert G.sha(out/'PREDICTIONS.jsonl.gz')==G.read(out/'SEAL.json')['prediction_sha256'];labs=G.index(G.rows(U/'C2_TARGETS.jsonl.gz'));groups={a:[]for a in ['BASELINE','CONTROL','LEAVE2']};paired=[]
 for p in ps:
  by={a['candidate_id']:a for a in labs[p['id']]['action_metrics']};r=dict(id=p['id'],candidate_n=len(p['bank_ids']),bank_good=any(by[k]['success_5cm5deg']for k in p['bank_ids']))
  for a,q in p['methods'].items():
   y=by[q['candidate_id']];m=dict(pose_available=True,translation_cm=y['translation_cm'],rotation_deg=y['rotation_deg'],success=y['success_5cm5deg']);groups[a].append(m);r[a+'_id']=q['candidate_id'];r.update({a+'_'+k:v for k,v in m.items()})
  paired.append(r)
 st={a:G.stats(z)for a,z in groups.items()};previous=G.read(T/'roi_box_joint_evidence_20261021_v33/SYNTHETIC_VAL_s20261018/EVALUATION.json')['methods'];comps=dict(BASELINE=st['BASELINE'],CONTROL=st['CONTROL'],V31=previous['ROI_CONTROL'],V32=previous['EVIDENCE_CONTROL'],V33=previous['JOINT_EVIDENCE']);why={a:G.gate(st['LEAVE2'],s)for a,s in comps.items()};rd={a:dict(recover=sum(not x['success']and y['success']for x,y in zip(groups[a],groups['LEAVE2'])),damage=sum(x['success']and not y['success']for x,y in zip(groups[a],groups['LEAVE2'])))for a in ['BASELINE','CONTROL']};ev=dict(methods=st,gate_passed=not any(why.values()),gate_reasons=why,recover_damage=rd,bank_good=sum(r['bank_good']for r in paired),selection_failures=sum(r['bank_good']and not r['LEAVE2_success']for r in paired),new_independent_test=0,adaptive_TRAIN_diagnostic_after_v35=True);G.csvwrite(out/'PAIRED_ROWS.csv',paired);G.write(out/'EVALUATION.json',ev);print(json.dumps(ev,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['features','targets','train','predict','evaluate']);a=p.parse_args();globals()[a.stage]()
