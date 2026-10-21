"""Same v31 poses, restore actual new ROI corner evidence to candidate selector."""
import pathlib,sys,importlib.util,json,time,argparse
import numpy as np
import torch
R=pathlib.Path(__file__).resolve().parent;T=R.parent
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
P=load('roi_union31',T/'roi_union_selector_20261021_v31/experiment.py');G=P.G;U=P.U;B=load('boxtrain29',T/'detector_box_compatibility_20261021_v29/experiment_v3.py')
def bound():
 p=G.read(R/'PROTOCOL.json')
 for k,v in p['bindings'].items():assert G.sha(v['path'])==v['sha256'],k
 return p
def features():
 bound();P.V.firewall();out=R/'SYNTHETIC_v3';out.mkdir(exist_ok=False);start=time.monotonic();fn=P.V.functions();up=G.index(G.rows(P.V.SAN));roi=G.index(G.rows(P.R/'TRAIN_ROI/PREDICTIONS.jsonl.gz')+G.rows(P.V.R/'SYNTHETIC_VAL/PREDICTIONS.jsonl.gz'));fs=G.rows(U/'HEAD_FEATURES.jsonl.gz');zs=[];nvalid=0
 for f in fs:
  u=up[f['id']];p=roi[f['id']]['ROI'];rr=[];valid=p['pose']['available']and p['q']is not None;K=np.asarray(u['K']);diag=float(np.hypot(*u['raw_hw']))
  if valid:
   native=fn['cuboid'](*p['pose']['cf_extents']);native-=native.mean(0);Q=np.asarray(p['pose']['R_cf']).T@p['pose']['R_physical'];xy=np.asarray(p['q'])[:8];support=np.isfinite(xy).all(-1)&~np.all(xy==-1,axis=1);nvalid+=1
  for e in G.A.actions(f):
   vals=np.zeros((8,2));ok=np.zeros((8,2),bool)
   if valid:
    pose=e['actual_pose'];cam=native@(np.asarray(pose['R_physical'])@Q.T).T+pose['centroid'];h=cam@K.T;pix=h[:,:2]/h[:,2:];good=support&np.isfinite(pix).all(-1)&(cam[:,2]>0);vals[good]=(pix[good]-xy[good])/diag;ok[good]=True
   rr.append(dict(candidate_id=e['candidate_id'],values=vals.ravel().tolist(),validity=ok.ravel().tolist()))
  zs.append(dict(id=f['id'],split=f['split'],actions=rr,ROI_evidence_available=bool(valid)))
 G.save(out/'FEATURES.jsonl.gz',zs);a=np.array([e['values']for f in zs if f['split']=='train'for e in f['actions']]);valid=np.array([e['validity']for f in zs if f['split']=='train'for e in f['actions']],bool);a=np.where(valid,np.sign(a)*np.log1p(abs(a)),np.nan);mu=np.nanmean(a,axis=0);sd=np.nanstd(a,axis=0);seen=np.isfinite(mu);mu=np.nan_to_num(mu);sd=np.where(np.isfinite(sd)&(sd>1e-12),sd,1.);G.write(out/'NORMALIZER.json',dict(mean=mu.tolist(),scale=sd.tolist(),seen=seen.tolist(),scope='TRAIN eligible actions only',units='signed dx/dy to actual ROI N3+SubPix points / original image diagonal; candidate projection uses ROI own prediction-only physical registration; source_basis fixed, not GT-derived correspondence'));G.write(out/'SEAL.json',dict(n=len(fs),valid_ROI_frames=nvalid,seconds=time.monotonic()-start,sha256=G.sha(out/'FEATURES.jsonl.gz'),new_forward_PnP=0,new_owner_GT=0));print(G.read(out/'SEAL.json'))
def make(initial=False,seed=20261018):
 torch.manual_seed(seed);base=G.M.make_model('MLP');st={k:v.clone()for k,v in base.state_dict().items()};m=torch.nn.Sequential(torch.nn.Linear(220,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,1))
 if initial:st['0.weight']=torch.cat([st['0.weight'],torch.zeros(64,32)],dim=1);m.load_state_dict(st)
 return m
def train():
 B.R=R;B.U=U;B.bound=bound;B.make=make;B.train(20261018);G.write(R/'ACTUAL_PARAMETER_COUNT.json',dict(parameters=sum(x.numel()for x in make().parameters()),input_dim=220,receipt_literal_from_reused_trainer=17089,correction='metadata only; actual loaded state18369, training unchanged'))
def predict():
 bound();G.setup();P.V.firewall();out=R/'SYNTHETIC_VAL_s20261018';out.mkdir(exist_ok=False);start=time.monotonic();models={}
 for arm,path in [('BASELINE',G.V5/'s20261018/best.pt'),('JK_CONTROL',P.J.R/'s20261018/best.pt'),('ROI_CONTROL',P.R/'s20261018/best.pt'),('ROI_EVIDENCE',R/'s20261018/best.pt')]:
  c=torch.load(path,weights_only=False,map_location='cpu');m=make(False)if arm=='ROI_EVIDENCE'else G.M.make_model('MLP');m.load_state_dict(c['model_state_dict']);m.eval().requires_grad_(False);models[arm]=(m,c)
 old=G.index(G.rows(G.B/'d1_synthetic_candidates_v1/HEAD_FEATURES.jsonl.gz'));jk=G.index(G.rows(P.OLD/'HEAD_FEATURES.jsonl.gz'));extra=G.index(G.rows(R/'SYNTHETIC_v3/FEATURES.jsonl.gz'));ps=[]
 with torch.inference_mode():
  for f in G.rows(U/'HEAD_FEATURES.jsonl.gz'):
   if f['split']!='validation':continue
   p=dict(id=f['id'],recording=f['cluster_id'],methods={},bank_ids=[e['candidate_id']for e in G.A.actions(f)])
   for arm,(m,c)in models.items():
    ff=old[f['id']]if arm=='BASELINE'else jk[f['id']]if arm=='JK_CONTROL'else f
    x,es=B.extend(ff,c['normalizer'],extra[f['id']],c['extra_normalizer'])if arm=='ROI_EVIDENCE'else G.A.inputs(ff,c['normalizer'],'H1');k=int(m(torch.from_numpy(x)).flatten().argmax());p['methods'][arm]=dict(candidate_id=es[k]['candidate_id'],pose=es[k]['actual_pose'])
   if not extra[f['id']]['ROI_evidence_available']:p['methods']['ROI_EVIDENCE']=p['methods']['ROI_CONTROL']
   ps.append(p)
 G.save(out/'PREDICTIONS.jsonl.gz',ps);G.write(out/'SEAL.json',dict(status='SEALED_BEFORE_SCORING',prediction_sha256=G.sha(out/'PREDICTIONS.jsonl.gz'),n=len(ps),seconds=time.monotonic()-start,GT_inputs=0,new_RGB_PnP=0));print(dict(stage='SEALED_ROI_EVIDENCE_SELECTION',n=len(ps)))
def evaluate():
 bound();out=R/'SYNTHETIC_VAL_s20261018';ps=G.rows(out/'PREDICTIONS.jsonl.gz');assert G.sha(out/'PREDICTIONS.jsonl.gz')==G.read(out/'SEAL.json')['prediction_sha256'];labs=G.index(G.rows(U/'C2_TARGETS.jsonl.gz'));groups={a:[]for a in ['BASELINE','JK_CONTROL','ROI_CONTROL','ROI_EVIDENCE']};paired=[]
 for p in ps:
  by={a['candidate_id']:a for a in labs[p['id']]['action_metrics']};r=dict(id=p['id'],candidate_n=len(p['bank_ids']),bank_good=any(by[k]['success_5cm5deg']for k in p['bank_ids']))
  for a,q in p['methods'].items():
   y=by[q['candidate_id']];m=dict(pose_available=True,translation_cm=y['translation_cm'],rotation_deg=y['rotation_deg'],success=y['success_5cm5deg']);groups[a].append(m);r[a+'_id']=q['candidate_id'];r.update({a+'_'+k:v for k,v in m.items()})
  paired.append(r)
 st={a:G.stats(z)for a,z in groups.items()};why={a:G.gate(st['ROI_EVIDENCE'],st[a])for a in groups if a!='ROI_EVIDENCE'};rd={a:dict(recover=sum(not x['success']and y['success']for x,y in zip(groups[a],groups['ROI_EVIDENCE'])),damage=sum(x['success']and not y['success']for x,y in zip(groups[a],groups['ROI_EVIDENCE'])))for a in why};ev=dict(methods=st,gate_passed=not any(why.values()),gate_reasons=why,recover_damage=rd,bank_good=sum(r['bank_good']for r in paired),selection_failures=sum(r['bank_good']and not r['ROI_EVIDENCE_success']for r in paired),new_independent_test=0,adapted_after_negative_v31=True);G.csvwrite(out/'PAIRED_ROWS.csv',paired);G.write(out/'EVALUATION.json',ev);print(json.dumps(ev,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['features','train','predict','evaluate']);a=p.parse_args();globals()[a.stage]()
