"""Matched same-bank conjunction of ROI corner and coarse box evidence."""
import pathlib,importlib.util,argparse,json,time
import numpy as np
import torch
R=pathlib.Path(__file__).resolve().parent;T=R.parent
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
E=load('roi_evidence32',T/'roi_evidence_selector_20261021_v32/experiment.py');P=E.P;G=E.G;B=E.B;U=E.U
def bound():
 p=G.read(R/'PROTOCOL.json')
 for k,v in p['bindings'].items():assert G.sha(v['path'])==v['sha256'],k
 return p
def features():
 bound();P.V.firewall();out=R/'SYNTHETIC_v3';out.mkdir(exist_ok=False);start=time.monotonic();old=G.index(G.rows(E.R/'SYNTHETIC_v3/FEATURES.jsonl.gz'));up=G.index(G.rows(G.B/'d1_upstream_v1/SYNTHETIC1003_UPSTREAM.jsonl.gz'));fs=G.rows(U/'HEAD_FEATURES.jsonl.gz');zs=[]
 for f in fs:
  box=B.extra_features(f,up[f['id']]);by={e['candidate_id']:e for e in box['actions']};z=old[f['id']];rr=[]
  for a in z['actions']:
   b=by[a['candidate_id']];rr.append(dict(candidate_id=a['candidate_id'],values=a['values']+b['values'],validity=a['validity']+b['validity']))
  zs.append(dict(id=f['id'],split=f['split'],actions=rr,ROI_evidence_available=z['ROI_evidence_available']))
 G.save(out/'FEATURES.jsonl.gz',zs);a=np.array([e['values']for f in zs if f['split']=='train'for e in f['actions']]);valid=np.array([e['validity']for f in zs if f['split']=='train'for e in f['actions']],bool);a=np.where(valid,np.sign(a)*np.log1p(abs(a)),np.nan);mu=np.nanmean(a,axis=0);sd=np.nanstd(a,axis=0);seen=np.isfinite(mu);mu=np.nan_to_num(mu);sd=np.where(np.isfinite(sd)&(sd>1e-12),sd,1.);G.write(out/'NORMALIZER.json',dict(mean=mu.tolist(),scale=sd.tolist(),seen=seen.tolist(),scope='TRAIN eligible actions only',units='v32 sixteen ROI signed residual scalars + v29 six detector-box compatibility scalars; existing same values, new own unionTRAIN norm'));G.write(out/'SEAL.json',dict(n=len(fs),seconds=time.monotonic()-start,sha256=G.sha(out/'FEATURES.jsonl.gz'),new_RGB_PnP=0,new_owner_GT=0));print(G.read(out/'SEAL.json'))
def make(initial=False,seed=20261018):
 torch.manual_seed(seed);base=G.M.make_model('MLP');st={k:v.clone()for k,v in base.state_dict().items()};m=torch.nn.Sequential(torch.nn.Linear(232,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,1))
 if initial:st['0.weight']=torch.cat([st['0.weight'],torch.zeros(64,44)],dim=1);m.load_state_dict(st)
 return m
def train():
 B.R=R;B.U=U;B.bound=bound;B.make=make;B.train(20261018);G.write(R/'ACTUAL_PARAMETER_COUNT.json',dict(parameters=sum(x.numel()for x in make().parameters()),input_dim=232,receipt_literal_from_reused_trainer=17089,correction='metadata only, actual232input/19137params unchanged training'))
def predict():
 bound();G.setup();P.V.firewall();out=R/'SYNTHETIC_VAL_s20261018';out.mkdir(exist_ok=False);start=time.monotonic();models={}
 for arm,path in [('BASELINE',G.V5/'s20261018/best.pt'),('JK_CONTROL',P.J.R/'s20261018/best.pt'),('ROI_CONTROL',P.R/'s20261018/best.pt'),('EVIDENCE_CONTROL',E.R/'s20261018/best.pt'),('JOINT_EVIDENCE',R/'s20261018/best.pt')]:
  c=torch.load(path,weights_only=False,map_location='cpu');m=make(False)if arm=='JOINT_EVIDENCE'else E.make(False)if arm=='EVIDENCE_CONTROL'else G.M.make_model('MLP');m.load_state_dict(c['model_state_dict']);m.eval().requires_grad_(False);models[arm]=(m,c)
 old=G.index(G.rows(G.B/'d1_synthetic_candidates_v1/HEAD_FEATURES.jsonl.gz'));jk=G.index(G.rows(P.OLD/'HEAD_FEATURES.jsonl.gz'));extra=G.index(G.rows(R/'SYNTHETIC_v3/FEATURES.jsonl.gz'));roiextra=G.index(G.rows(E.R/'SYNTHETIC_v3/FEATURES.jsonl.gz'));ps=[]
 with torch.inference_mode():
  for f in G.rows(U/'HEAD_FEATURES.jsonl.gz'):
   if f['split']!='validation':continue
   p=dict(id=f['id'],recording=f['cluster_id'],methods={},bank_ids=[e['candidate_id']for e in G.A.actions(f)])
   for arm,(m,c)in models.items():
    ff=old[f['id']]if arm=='BASELINE'else jk[f['id']]if arm=='JK_CONTROL'else f
    x,es=B.extend(ff,c['normalizer'],extra[f['id']]if arm=='JOINT_EVIDENCE'else roiextra[f['id']],c['extra_normalizer'])if arm in ['JOINT_EVIDENCE','EVIDENCE_CONTROL']else G.A.inputs(ff,c['normalizer'],'H1');k=int(m(torch.from_numpy(x)).flatten().argmax());p['methods'][arm]=dict(candidate_id=es[k]['candidate_id'],pose=es[k]['actual_pose'])
   if not extra[f['id']]['ROI_evidence_available']:
    p['methods']['EVIDENCE_CONTROL']=p['methods']['ROI_CONTROL'];p['methods']['JOINT_EVIDENCE']=p['methods']['ROI_CONTROL']
   ps.append(p)
 G.save(out/'PREDICTIONS.jsonl.gz',ps);G.write(out/'SEAL.json',dict(status='SEALED_BEFORE_SCORING',prediction_sha256=G.sha(out/'PREDICTIONS.jsonl.gz'),n=len(ps),seconds=time.monotonic()-start,GT_inputs=0,new_RGB_PnP=0));print(dict(stage='SEALED_JOINT_EVIDENCE',n=len(ps)))
def evaluate():
 bound();out=R/'SYNTHETIC_VAL_s20261018';ps=G.rows(out/'PREDICTIONS.jsonl.gz');assert G.sha(out/'PREDICTIONS.jsonl.gz')==G.read(out/'SEAL.json')['prediction_sha256'];labs=G.index(G.rows(U/'C2_TARGETS.jsonl.gz'));groups={a:[]for a in ['BASELINE','JK_CONTROL','ROI_CONTROL','EVIDENCE_CONTROL','JOINT_EVIDENCE']};paired=[]
 for p in ps:
  by={a['candidate_id']:a for a in labs[p['id']]['action_metrics']};r=dict(id=p['id'],candidate_n=len(p['bank_ids']),bank_good=any(by[k]['success_5cm5deg']for k in p['bank_ids']))
  for a,q in p['methods'].items():
   y=by[q['candidate_id']];m=dict(pose_available=True,translation_cm=y['translation_cm'],rotation_deg=y['rotation_deg'],success=y['success_5cm5deg']);groups[a].append(m);r[a+'_id']=q['candidate_id'];r.update({a+'_'+k:v for k,v in m.items()})
  paired.append(r)
 st={a:G.stats(z)for a,z in groups.items()};why={a:G.gate(st['JOINT_EVIDENCE'],st[a])for a in groups if a!='JOINT_EVIDENCE'};rd={a:dict(recover=sum(not x['success']and y['success']for x,y in zip(groups[a],groups['JOINT_EVIDENCE'])),damage=sum(x['success']and not y['success']for x,y in zip(groups[a],groups['JOINT_EVIDENCE'])))for a in why};ev=dict(methods=st,gate_passed=not any(why.values()),gate_reasons=why,recover_damage=rd,bank_good=sum(r['bank_good']for r in paired),selection_failures=sum(r['bank_good']and not r['JOINT_EVIDENCE_success']for r in paired),new_independent_test=0,adapted_after_negative_v32=True);G.csvwrite(out/'PAIRED_ROWS.csv',paired);G.write(out/'EVALUATION.json',ev);print(json.dumps(ev,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['features','train','predict','evaluate']);a=p.parse_args();globals()[a.stage]()
