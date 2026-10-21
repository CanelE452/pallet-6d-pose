"""Ordered raw forecast residuals on the sealed v24 union bank. Forecasts are not owners."""
import pathlib,sys,json,argparse,time
import numpy as np
import torch
R=pathlib.Path(__file__).resolve().parent;T=R.parent;sys.path.insert(0,str(T/'geometric_influence_20261021_v19'));import experiment as G
V24=T/'jackknife_bank_20261021_v24';U=V24/'SYNTHETIC_v3';D=16
def bound():
 p=G.read(R/'PROTOCOL.json')
 for k,v in p['bindings'].items():assert G.sha(v['path'])==v['sha256'],k
 return p
def extra_features(f,u):
 X=np.asarray(u['native_xyz']);X=X-X.mean(0);Q=np.asarray(u['initial_pose']['R_cf']).T@np.asarray(u['initial_pose']['R_physical']);K=np.asarray(u['K']);q=np.asarray(u['initial_points'])[:8];valid=np.isfinite(q).all(-1)&~np.all(q==-1,axis=1);diag=float(np.linalg.norm(u['raw_hw']));rr=[]
 for e in G.A.actions(f):
  p=e['actual_pose'];rot=np.asarray(p['R_physical'])@Q.T;camera=X@rot.T+p['centroid'];h=camera@K.T;pred=h[:,:2]/h[:,2:];raw=((pred-q)/diag).ravel();ok=np.repeat(valid,2)&np.isfinite(raw);raw=np.where(ok,raw,0.);rr.append(dict(candidate_id=e['candidate_id'],values=raw.tolist(),validity=ok.tolist()))
 return dict(id=f['id'],actions=rr,split=f['split'])
def features(pop):
 bound();G.firewall();out=R/pop;out.mkdir(exist_ok=False);start=time.monotonic();syn=pop=='SYNTHETIC';fs=G.rows((U if syn else V24/'CAL46_v3')/'HEAD_FEATURES.jsonl.gz');up=G.index(G.rows(G.B/'d1_upstream_v1'/('SYNTHETIC1003_UPSTREAM.jsonl.gz'if syn else'REAL238_UPSTREAM.jsonl.gz')));z=[extra_features(f,up[f['id']])for f in fs];G.save(out/'FEATURES.jsonl.gz',z)
 if syn:
  a=np.array([c['values']for f in z if f['split']=='train'for c in f['actions']]);valid=np.array([c['validity']for f in z if f['split']=='train'for c in f['actions']],bool);a=np.where(valid,np.sign(a)*np.log1p(abs(a)),np.nan);mu=np.nanmean(a,axis=0);sd=np.nanstd(a,axis=0);seen=np.isfinite(mu);mu=np.nan_to_num(mu);sd=np.where(np.isfinite(sd)&(sd>1e-12),sd,1.);G.write(out/'NORMALIZER.json',dict(mean=mu.tolist(),scale=sd.tolist(),seen=seen.tolist(),scope='own syntheticTRAIN eligible union actions only',units='raw forecast residual dx/dy divided by image diagonal, signedlog; no image visibility/owner assumption'))
 G.write(out/'SEAL.json',dict(rows=len(z),features_sha256=G.sha(out/'FEATURES.jsonl.gz'),seconds=time.monotonic()-start,new_PnP=0,new_RGB_queries=0));print(dict(stage='FORECAST_RESIDUAL_FEATURES',pop=pop,n=len(z),seconds=time.monotonic()-start))
def extend(f,norm,extra,en):
 x,es=G.A.inputs(f,norm,'H1');by={c['candidate_id']:c for c in extra['actions']};v=np.array([by[e['candidate_id']]['values']for e in es]);ok=np.array([by[e['candidate_id']]['validity']for e in es],bool);v=np.sign(v)*np.log1p(abs(v));v=np.where(ok&np.array(en['seen'])[None,:],(v-en['mean'])/en['scale'],0.);return np.concatenate([x,v,ok],axis=1).astype(np.float32),es
def make(initial=False,seed=20261018):
 torch.manual_seed(seed);base=G.M.make_model('MLP');state={k:v.clone()for k,v in base.state_dict().items()};m=torch.nn.Sequential(torch.nn.Linear(220,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,1))
 if initial:state['0.weight']=torch.cat([state['0.weight'],torch.zeros((64,32))],dim=1);m.load_state_dict(state)
 return m
def train(seed):
 G.setup();bound();out=R/f's{seed}';out.mkdir(exist_ok=False);start=time.monotonic();fs=G.rows(U/'HEAD_FEATURES.jsonl.gz');labels=G.index(G.rows(U/'C2_TARGETS.jsonl.gz'));norm=G.read(U/'NORMALIZER.json');en=G.read(R/'SYNTHETIC/NORMALIZER.json');extra=G.index(G.rows(R/'SYNTHETIC/FEATURES.jsonl.gz'));items={'train':[],'validation':[]}
 for f in fs:
  x,es=extend(f,norm,extra[f['id']],en);by={c['candidate_id']:c for c in labels[f['id']]['action_metrics']};items[f['split']].append(dict(x=torch.from_numpy(x),y=torch.tensor([float(by[e['candidate_id']]['success_5cm5deg'])for e in es])))
 ledger=R/'TRAINING_ATTEMPTS.jsonl';old=ledger.read_text().splitlines()if ledger.exists()else[];assert len(old)<3
 with ledger.open('a')as h:h.write(json.dumps(dict(seed=seed,state='RESERVED_BEFORE_OPTIMIZER'))+'\n')
 m=make(True,seed);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);rng=np.random.default_rng(seed);hist=[];best=float('inf');wait=0;fit=time.monotonic()
 def loss(group):return torch.stack([torch.nn.functional.binary_cross_entropy_with_logits(m(r['x']).flatten(),r['y'])for r in group]).mean()
 def ev(items):
  m.eval()
  with torch.inference_mode():return sum(float(loss(items[i:i+32]))*len(items[i:i+32])for i in range(0,len(items),32))/len(items)
 for ep in range(50):
  m.train();order=rng.permutation(len(items['train']))
  for i in range(0,len(order),32):
   group=[items['train'][int(k)]for k in order[i:i+32]];opt.zero_grad(set_to_none=True);l=loss(group);assert torch.isfinite(l);l.backward();opt.step()
  tr=ev(items['train']);vl=ev(items['validation']);hist.append(dict(epoch=ep+1,train_BCE=tr,validation_BCE=vl))
  if vl<best:best=vl;wait=0;bestep=ep+1;torch.save(dict(model_state_dict=m.state_dict(),normalizer=norm,extra_normalizer=en,seed=seed,epoch=bestep),out/'best.pt')
  else:wait+=1
  if wait>=5:break
  assert time.monotonic()-fit<180
 G.write(out/'HISTORY.json',hist);G.write(out/'RECEIPT.json',dict(seed=seed,epochs=len(hist),best_epoch=bestep,parameters=18369,fit_seconds=time.monotonic()-fit,total_seconds=time.monotonic()-start,checkpoint_sha256=G.sha(out/'best.pt'),initial188andhidden_weights_exact_same_random_as_v24=True,extra_columns_initial0=True,CPU_threads=1,GPU=0));print(dict(stage='TRAINED',epochs=len(hist),best_epoch=bestep,seconds=time.monotonic()-fit))
def predict(pop,seed):
 G.setup();bound();G.firewall();syn=pop=='SYNTHETIC_VAL'
 if not syn:assert G.read(R/f'SYNTHETIC_VAL_s{seed}/EVALUATION.json')['gate_passed']
 out=R/f'{pop}_s{seed}';out.mkdir(exist_ok=False);start=time.monotonic();ck=torch.load(R/f's{seed}/best.pt',weights_only=False,map_location='cpu');m=make(False,seed);m.load_state_dict(ck['model_state_dict']);m.eval().requires_grad_(False);models={}
 for arm,path in [('BASELINE',G.V5/f's{seed}/best.pt'),('UNION_CONTROL',V24/f's{seed}/best.pt')]:
  c=torch.load(path,weights_only=False,map_location='cpu');m0=G.M.make_model('MLP');m0.load_state_dict(c['model_state_dict']);m0.eval().requires_grad_(False);models[arm]=(m0,c)
 union=G.rows((U if syn else V24/'CAL46_v3')/'HEAD_FEATURES.jsonl.gz');union=[f for f in union if f['split']=='validation']if syn else union;old=G.index(G.rows(G.B/('d1_synthetic_candidates_v1'if syn else'd1_real_candidates_v1')/'HEAD_FEATURES.jsonl.gz'));extra=G.index(G.rows(R/('SYNTHETIC'if syn else'CAL46')/'FEATURES.jsonl.gz'));ps=[]
 with torch.inference_mode():
  for f in union:
   x,es=extend(f,ck['normalizer'],extra[f['id']],ck['extra_normalizer']);v=m(torch.from_numpy(x)).flatten().numpy();k=int(np.argmax(v));p=dict(id=f['id'],partition=f['partition'],recording=f['cluster_id'],bank_ids=[e['candidate_id']for e in es],methods={'INFLUENCE':dict(candidate_id=es[k]['candidate_id'],pose=es[k]['actual_pose'])})
   for arm,(mm,c)in models.items():
    xx,ee=G.A.inputs(old[f['id']]if arm=='BASELINE'else f,c['normalizer'],'H1');vv=mm(torch.from_numpy(xx)).flatten().numpy();kk=int(np.argmax(vv));p['methods'][arm]=dict(candidate_id=ee[kk]['candidate_id'],pose=ee[kk]['actual_pose'])
   p['methods']['CONTROL']=p['methods'].pop('UNION_CONTROL');ps.append(p)
 G.save(out/'PREDICTIONS.jsonl.gz',ps);G.write(out/'SEAL.json',dict(prediction_sha256=G.sha(out/'PREDICTIONS.jsonl.gz'),reference_inputs=0,status='SEALED_BEFORE_SCORING',n=len(ps),seconds=time.monotonic()-start));print(dict(stage='SEALED',pop=pop,n=len(ps)))
def evaluate(pop,seed):
 G.R=R
 # The generic evaluator's synthetic labels must be the actual expanded bank.
 labels=G.index(G.rows(U/'C2_TARGETS.jsonl.gz'));out=R/f'{pop}_s{seed}';syn=pop=='SYNTHETIC_VAL';refs=None if syn else G.read(G.B.parent/'pallet-statewise-correspondence-diagnosis-20261016-v1/score_v2/FIXED_REFERENCES.json')['references'];ps=G.rows(out/'PREDICTIONS.jsonl.gz');assert G.sha(out/'PREDICTIONS.jsonl.gz')==G.read(out/'SEAL.json')['prediction_sha256'];groups={k:[]for k in ['BASELINE','CONTROL','INFLUENCE']};paired=[]
 for p in ps:
  by={c['candidate_id']:c for c in labels[p['id']]['action_metrics']}if syn else None;r=dict(id=p['id'],recording=p['recording'],candidate_n=len(p['bank_ids']))
  for arm,q in p['methods'].items():
   if syn:y=by[q['candidate_id']];z=dict(pose_available=True,translation_cm=y['translation_cm'],rotation_deg=y['rotation_deg'],success=y['success_5cm5deg'])
   else:z=G.metric(q['pose'],refs[p['partition']][p['id']])
   groups[arm].append(z);r[arm+'_id']=q['candidate_id'];r.update({arm+'_'+k:v for k,v in z.items()})
  paired.append(r)
 st={k:G.stats(v)for k,v in groups.items()};why={k:G.gate(st['INFLUENCE'],st[k])for k in ['BASELINE','CONTROL']};rd={k:dict(recover=sum(not a['success']and b['success']for a,b in zip(groups[k],groups['INFLUENCE'])),damage=sum(a['success']and not b['success']for a,b in zip(groups[k],groups['INFLUENCE'])))for k in ['BASELINE','CONTROL']};ev=dict(population=pop,seed=seed,methods=st,gate_passed=not any(why.values()),gate_reasons=why,recover_damage=rd,new_independent_test=0);G.csvwrite(out/'PAIRED_ROWS.csv',paired);G.write(out/'EVALUATION.json',ev);print(json.dumps(ev,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['features','train','predict','evaluate']);p.add_argument('--population',default='SYNTHETIC');p.add_argument('--seed',type=int,default=20261018);a=p.parse_args();features(a.population)if a.stage=='features'else train(a.seed)if a.stage=='train'else(predict if a.stage=='predict'else evaluate)(a.population,a.seed)
