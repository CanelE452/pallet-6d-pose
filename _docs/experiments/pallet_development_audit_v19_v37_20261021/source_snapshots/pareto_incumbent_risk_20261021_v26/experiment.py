"""Within-success Pareto damage supervision on frozen v24 bank/head, synthetic only."""
import pathlib,sys,importlib.util,argparse,json,time,math
import numpy as np
import torch
R=pathlib.Path(__file__).resolve().parent;T=R.parent
spec=importlib.util.spec_from_file_location('forecast_runtime',T/'forecast_residual_structure_20261021_v25/experiment.py');F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F);G=F.G;G.R=R;F.R=R;U=F.U;V24=F.V24;EXTRA=T/'forecast_residual_structure_20261021_v25';OLD={};ON=None
def inputs(f,norm,extra,en,anchor):
 x,es=F.extend(f,norm,extra,en);old=OLD[f['id']];ox,oe=G.A.inputs(old,ON,'H1')
 with torch.inference_mode():aID=oe[int(anchor(torch.from_numpy(ox)).flatten().argmax())]['candidate_id']
 a=next(i for i,e in enumerate(es)if e['candidate_id']==aID);z=np.concatenate([x[:,:188],np.repeat(x[a:a+1,:62],len(x),axis=0),x[:,188:]-x[a,188:]],axis=1).astype(np.float32);assert z.shape[1]==282;return x[:,:188],z,es,a
def setup(seed,syn):
 global OLD,ON
 G.setup();G.bound();cp=torch.load(G.V5/f's{seed}/best.pt',weights_only=False,map_location='cpu');anchor=G.M.make_model('MLP');anchor.load_state_dict(cp['model_state_dict']);anchor.eval().requires_grad_(False);ON=cp['normalizer'];OLD=G.index(G.rows(G.B/('d1_synthetic_candidates_v1'if syn else'd1_real_candidates_v1')/'HEAD_FEATURES.jsonl.gz'));return anchor
def train(seed):
 anchor=setup(seed,True);start=time.monotonic();folder=R/f's{seed}';folder.mkdir(exist_ok=False);fs=G.rows(U/'HEAD_FEATURES.jsonl.gz');norm=G.read(U/'NORMALIZER.json');en=G.read(EXTRA/'SYNTHETIC/NORMALIZER.json');extra=G.index(G.rows(EXTRA/'SYNTHETIC/FEATURES.jsonl.gz'));labs=G.index(G.rows(U/'C2_TARGETS.jsonl.gz'));items={'train':[],'validation':[]};counts={s:dict(frames=0,actions=0,pareto_damage_actions=0,nonanchor_safe_frames=0,within_joint_success_damage_actions=0)for s in items}
 for f in fs:
  x,z,es,a=inputs(f,norm,extra[f['id']],en,anchor);by={m['candidate_id']:m for m in labs[f['id']]['action_metrics']};ms=[by[e['candidate_id']]for e in es];b=ms[a];y=np.array([float(m['translation_cm']>b['translation_cm']+1e-7 or m['rotation_deg']>b['rotation_deg']+1e-7)for m in ms],np.float32);assert y[a]==0;items[f['split']].append(dict(z=torch.from_numpy(z),y=torch.from_numpy(y)));c=counts[f['split']];c['frames']+=1;c['actions']+=len(es);c['pareto_damage_actions']+=int(y.sum());c['nonanchor_safe_frames']+=int(sum(y==0)>1);c['within_joint_success_damage_actions']+=sum(bool(y[i])and m['success_5cm5deg']and b['success_5cm5deg']for i,m in enumerate(ms))
 G.write(folder/'SUPERVISION_COUNTS.json',counts);assert counts['train']['nonanchor_safe_frames']>=20;ledger=R/'TRAINING_ATTEMPTS.jsonl';prev=ledger.read_text().splitlines()if ledger.exists()else[];assert len(prev)<3
 with ledger.open('a')as h:h.write(json.dumps(dict(seed=seed,state='RESERVED_BEFORE_OPTIMIZER'))+'\n')
 torch.manual_seed(seed);m=torch.nn.Linear(282,1);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);rng=np.random.default_rng(seed);best=math.inf;wait=0;hist=[];fit=time.monotonic()
 def loss(group):return torch.stack([torch.nn.functional.binary_cross_entropy_with_logits(m(r['z']).flatten(),r['y'])for r in group]).mean()
 def ev(items):
  m.eval()
  with torch.inference_mode():return sum(float(loss(items[i:i+32]))*len(items[i:i+32])for i in range(0,len(items),32))/len(items)
 for ep in range(50):
  m.train();order=rng.permutation(len(items['train']))
  for i in range(0,len(order),32):
   group=[items['train'][int(k)]for k in order[i:i+32]];opt.zero_grad(set_to_none=True);l=loss(group);assert torch.isfinite(l);l.backward();opt.step()
  tr=ev(items['train']);vl=ev(items['validation']);hist.append(dict(epoch=ep+1,train_BCE=tr,validation_BCE=vl))
  if vl<best:best=vl;wait=0;bestep=ep+1;torch.save(dict(model_state_dict=m.state_dict(),normalizer=norm,extra_normalizer=en,seed=seed,epoch=bestep),folder/'best.pt')
  else:wait+=1
  if wait>=5:break
  assert time.monotonic()-fit<180
 G.write(folder/'HISTORY.json',hist);G.write(folder/'RECEIPT.json',dict(seed=seed,epochs=len(hist),best_epoch=bestep,parameters=283,fit_seconds=time.monotonic()-fit,total_seconds=time.monotonic()-start,checkpoint_sha256=G.sha(folder/'best.pt'),CPU_threads=1,GPU=0,new_risk_fit=1,new_success_fit=0));print(dict(stage='TRAINED_PARETO_RISK',epochs=len(hist),best_epoch=bestep,seconds=time.monotonic()-fit))
def predict(pop,seed):
 syn=pop=='SYNTHETIC_VAL';anchor=setup(seed,syn);G.firewall()
 if not syn:assert G.read(R/f'SYNTHETIC_VAL_s{seed}/EVALUATION.json')['gate_passed']
 out=R/f'{pop}_s{seed}';out.mkdir(exist_ok=False);start=time.monotonic();ck=torch.load(R/f's{seed}/best.pt',weights_only=False,map_location='cpu');m=torch.nn.Linear(282,1);m.load_state_dict(ck['model_state_dict']);m.eval().requires_grad_(False);sc=torch.load(V24/f's{seed}/best.pt',weights_only=False,map_location='cpu');success=G.M.make_model('MLP');success.load_state_dict(sc['model_state_dict']);success.eval().requires_grad_(False);fs=G.rows((U if syn else V24/'CAL46_v3')/'HEAD_FEATURES.jsonl.gz');fs=[f for f in fs if f['split']=='validation']if syn else fs;extra=G.index(G.rows(EXTRA/('SYNTHETIC'if syn else'CAL46')/'FEATURES.jsonl.gz'));ps=[];diag=[]
 with torch.inference_mode():
  for f in fs:
   x,z,es,a=inputs(f,ck['normalizer'],extra[f['id']],ck['extra_normalizer'],anchor);v=success(torch.from_numpy(x)).flatten().numpy();rv=m(torch.from_numpy(z)).flatten().numpy();safe=rv<=0;safe[a]=True;new=int(np.argmax(np.where(safe,v,-np.inf)));before=int(np.argmax(v));p=dict(id=f['id'],partition=f['partition'],recording=f['cluster_id'],bank_ids=[e['candidate_id']for e in es],methods={})
   for arm,k in [('BASELINE',a),('CONTROL',before),('INFLUENCE',new)]:p['methods'][arm]=dict(candidate_id=es[k]['candidate_id'],pose=es[k]['actual_pose'])
   ps.append(p);diag.append(dict(id=f['id'],veto_n=int((~safe).sum()),anchor_id=es[a]['candidate_id'],before_id=es[before]['candidate_id'],after_id=es[new]['candidate_id'],selected_risk_logit=float(rv[new])))
 G.save(out/'PREDICTIONS.jsonl.gz',ps);G.csvwrite(out/'RISK_DIAGNOSTICS.csv',diag);G.write(out/'SEAL.json',dict(prediction_sha256=G.sha(out/'PREDICTIONS.jsonl.gz'),status='SEALED_BEFORE_EVALUATION',n=len(ps),reference_inputs=0,seconds=time.monotonic()-start,new_PnP=0));print(dict(stage='SEALED',pop=pop,n=len(ps)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','predict','evaluate']);p.add_argument('--population',default='SYNTHETIC_VAL');p.add_argument('--seed',type=int,default=20261018);a=p.parse_args();train(a.seed)if a.stage=='train'else predict(a.population,a.seed)if a.stage=='predict'else F.evaluate(a.population,a.seed)
