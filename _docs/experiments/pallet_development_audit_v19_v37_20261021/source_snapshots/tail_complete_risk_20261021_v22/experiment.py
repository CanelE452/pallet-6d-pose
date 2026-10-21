"""Gate-aligned severe-damage target with new profile/geometry evidence, incumbent safe."""
import pathlib,sys,importlib.util,json,gzip,time,math,argparse,os
import numpy as np
import torch
R=pathlib.Path(__file__).resolve().parent;T=R.parent
spec=importlib.util.spec_from_file_location('joint_runtime',T/'profile_geometry_interaction_20261021_v21/experiment.py');J=importlib.util.module_from_spec(spec);spec.loader.exec_module(J);G=J.G;G.R=R;J.S.R=R;PROFILE=T/'structured_profiles_20261021_v20';SUCCESS=T/'profile_geometry_interaction_20261021_v21'
def inputs(f,norm,extra,en,anchor_model):
 x,es=J.extend(f,norm,extra,en,'INFLUENCE')
 with torch.inference_mode():anchor=int(anchor_model(torch.from_numpy(x[:,:188])).flatten().argmax())
 z=np.concatenate([x[:,:188],np.repeat(x[anchor:anchor+1,:62],len(x),axis=0),x[:,188:]-x[anchor,188:]],axis=1).astype(np.float32);assert z.shape[1]==2218;return x,z,es,anchor
def models(seed):
 cp=torch.load(G.V5/f's{seed}/best.pt',weights_only=False,map_location='cpu');a=J.model('CONTROL');a.load_state_dict(cp['model_state_dict']);a.eval().requires_grad_(False);return a,cp
def train(seed):
 G.setup();G.bound();folder=R/f's{seed}';folder.mkdir(exist_ok=False);start=time.monotonic();anchor,cp=models(seed);features=G.index(G.rows(PROFILE/'SYNTHETIC/FEATURES.jsonl.gz'));en=G.read(PROFILE/'SYNTHETIC/NORMALIZER.json');labels=G.index(G.rows(G.V5/'DERIVED_C2_ACTION_TARGETS.jsonl.gz'));fs=G.rows(G.B/'d1_synthetic_candidates_v1/HEAD_FEATURES.jsonl.gz');items={'train':[],'validation':[]};counts={k:dict(frames=0,actions=0,positive_actions=0,positive_frames=0,new_T100_only_actions=0,new_T100_only_frames=0)for k in items}
 for f in fs:
  x,z,es,a=inputs(f,cp['normalizer'],features[f['id']],en,anchor);by={m['candidate_id']:m for m in labels[f['id']]['action_metrics']};ms=[by[e['candidate_id']]for e in es];b=ms[a];old=np.array([float((b['success_5cm5deg']and not m['success_5cm5deg'])or(b['translation_cm']<=50<m['translation_cm'])or(b['rotation_deg']<=45<m['rotation_deg']))for m in ms],np.float32);new=np.array([float(b['translation_cm']<=100<m['translation_cm'])for m in ms],np.float32);y=np.maximum(old,new);assert y[a]==0;items[f['split']].append(dict(id=f['id'],z=torch.from_numpy(z),y=torch.from_numpy(y)));c=counts[f['split']];c['frames']+=1;c['actions']+=len(es);c['positive_actions']+=int(y.sum());c['positive_frames']+=int(y.any());c['new_T100_only_actions']+=int(((old==0)&(new==1)).sum());c['new_T100_only_frames']+=int(((old==0)&(new==1)).any())
 G.write(folder/'SUPERVISION_COUNTS.json',counts);assert counts['train']['positive_actions']>=20 and counts['train']['positive_frames']>=20
 ledger=R/'TRAINING_ATTEMPTS.jsonl';old=ledger.read_text().splitlines()if ledger.exists()else[];assert len(old)<3
 with ledger.open('a')as h:h.write(json.dumps(dict(seed=seed,state='RESERVED_BEFORE_OPTIMIZER',time=time.time()))+'\n')
 torch.manual_seed(seed);model=torch.nn.Linear(2218,1);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);rng=np.random.default_rng(seed);best=math.inf;wait=0;hist=[];fitstart=time.monotonic()
 def loss(group):return torch.stack([torch.nn.functional.binary_cross_entropy_with_logits(model(r['z']).flatten(),r['y'])for r in group]).mean()
 def ev(items):
  model.eval()
  with torch.inference_mode():return sum(float(loss(items[i:i+32]))*len(items[i:i+32])for i in range(0,len(items),32))/len(items)
 for ep in range(50):
  model.train();order=rng.permutation(len(items['train']))
  for i in range(0,len(order),32):
   group=[items['train'][int(k)]for k in order[i:i+32]];opt.zero_grad(set_to_none=True);l=loss(group);assert torch.isfinite(l);l.backward();opt.step()
  tr=ev(items['train']);val=ev(items['validation']);hist.append(dict(epoch=ep+1,train_BCE=tr,validation_BCE=val))
  if val<best:best=val;wait=0;bestep=ep+1;torch.save(dict(model_state_dict=model.state_dict(),normalizer=cp['normalizer'],extra_normalizer=en,seed=seed,epoch=bestep),folder/'best.pt')
  else:wait+=1
  if wait>=5:break
  assert time.monotonic()-fitstart<180
 G.write(folder/'HISTORY.json',hist);G.write(folder/'RECEIPT.json',dict(seed=seed,epochs=len(hist),best_epoch=bestep,parameters=2219,fit_seconds=time.monotonic()-fitstart,total_seconds=time.monotonic()-start,checkpoint_sha256=G.sha(folder/'best.pt'),new_risk_fit=1,new_success_fit=0,CPU_threads=1,GPU=0,anchor_is_training_in_sample=True));print(dict(stage='TRAINED_RISK',epochs=len(hist),best_epoch=bestep,seconds=time.monotonic()-fitstart))
def predict(pop,seed):
 G.setup();G.bound();G.firewall();syn=pop=='SYNTHETIC_VAL'
 if not syn:assert G.read(R/f'SYNTHETIC_VAL_s{seed}/EVALUATION.json')['gate_passed']
 out=R/f'{pop}_s{seed}';out.mkdir(exist_ok=False);start=time.monotonic();anchor,basecp=models(seed);sc=torch.load(SUCCESS/f'INFLUENCE_s{seed}/best.pt',weights_only=False,map_location='cpu');success=J.model('INFLUENCE');success.load_state_dict(sc['model_state_dict']);success.eval().requires_grad_(False);rc=torch.load(R/f's{seed}/best.pt',weights_only=False,map_location='cpu');risk=torch.nn.Linear(2218,1);risk.load_state_dict(rc['model_state_dict']);risk.eval().requires_grad_(False);cc=torch.load(T/'geometric_influence_20261021_v19'/f'CONTROL_s{seed}/best.pt',weights_only=False,map_location='cpu');control=J.model('CONTROL');control.load_state_dict(cc['model_state_dict']);control.eval().requires_grad_(False);extra=G.index(G.rows((PROFILE/'SYNTHETIC'if syn else R/'CAL46')/'FEATURES.jsonl.gz'));fs=G.rows(G.B/('d1_synthetic_candidates_v1'if syn else'd1_real_candidates_v1')/'HEAD_FEATURES.jsonl.gz');fs=[f for f in fs if(f['split']=='validation'if syn else f['partition']=='CAL')];ps=[];diag=[]
 with torch.inference_mode():
  for f in fs:
   x,z,es,a=inputs(f,basecp['normalizer'],extra[f['id']],rc['extra_normalizer'],anchor);v=success(torch.from_numpy(x)).flatten().numpy();rv=risk(torch.from_numpy(z)).flatten().numpy();safe=rv<=0;safe[a]=True;k=int(np.argmax(np.where(safe,v,-np.inf)));c=int(control(torch.from_numpy(x[:,:188])).flatten().argmax());p=dict(id=f['id'],partition=f['partition'],recording=f['cluster_id'],bank_ids=[e['candidate_id']for e in es],methods={})
   for name,j in [('BASELINE',a),('CONTROL',c),('INFLUENCE',k)]:p['methods'][name]=dict(candidate_id=es[j]['candidate_id'],pose=es[j]['actual_pose'])
   ps.append(p);diag.append(dict(id=f['id'],before_joint_head=es[int(np.argmax(v))]['candidate_id'],after_risk=es[k]['candidate_id'],anchor=es[a]['candidate_id'],risk_vetoed=int((~safe).sum()),bank_n=len(es),selected_risk_logit=float(rv[k]),anchor_risk_logit=float(rv[a])))
 G.save(out/'PREDICTIONS.jsonl.gz',ps);G.csvwrite(out/'RISK_DIAGNOSTICS.csv',diag);G.write(out/'SEAL.json',dict(status='SEALED_BEFORE_SCORING',prediction_sha256=G.sha(out/'PREDICTIONS.jsonl.gz'),rows=len(ps),reference_inputs=0,seconds=time.monotonic()-start,new_PnP=0,new_observation=0,head_frame_forward_calls=4*len(ps)));print(dict(stage='PREDICTED_RISK',population=pop,n=len(ps)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','predict','evaluate','CALfeatures']);p.add_argument('--population',default='SYNTHETIC_VAL');p.add_argument('--seed',type=int,default=20261018);a=p.parse_args();train(a.seed)if a.stage=='train'else predict(a.population,a.seed)if a.stage=='predict'else G.evaluate(a.population,a.seed)if a.stage=='evaluate'else J.S.features('CAL46')
