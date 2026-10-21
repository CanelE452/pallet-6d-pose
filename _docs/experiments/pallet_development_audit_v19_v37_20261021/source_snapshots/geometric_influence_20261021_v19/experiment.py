"""Saved-observation local information and delete-corner influence. Never a physical GT."""
import sys,pathlib,json,gzip,time,math,argparse,os,hashlib
import numpy as np
import torch
R=pathlib.Path(__file__).resolve().parent;T=R.parent;SRC=pathlib.Path('${SOURCE_REPO}');sys.path[:0]=[str(SRC),str(T/'mean_gap_audit_20261020_v1'),str(T/'minimal_improvement_20261020_v5')]
from analyze import read,rows,index,write,csvwrite,sha,metric
from kernel_probe import stats
from scripts.research.pallet_boundary_action_repair_v1 import action_data as A,action_models as M
B=pathlib.Path('${DATA_ROOT}/pallet-boundary-action-repair-20261018-v1');V5=T/'minimal_improvement_20261020_v5/H1_DEPLOYMENT_C2_TARGET_V1';FAMILIES=['ORIGINAL_WD','SWAPPED_WD']
NAMES=['constraint_rows','rank','condition','unit_noise_T_m','unit_noise_R_rad','linear_step_T_m','linear_step_R_rad','delete_corner_change_T_m','delete_corner_change_R_rad','max_leverage','residual_RMS','rank_deficient_deletions']
def save(p,z):
 with gzip.open(p,'xt')as h:
  for x in z:h.write(json.dumps(x,allow_nan=False)+'\n')
def setup():torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
def bound():
 p=read(R/'PROTOCOL.json')
 for k,v in p['bindings'].items():assert sha(v['path'])==v['sha256'],k
 return p
def firewall():
 def hook(e,a):
  if e=='open'and isinstance(a[0],(str,bytes))and pathlib.Path(os.fsdecode(a[0])).name in ['FIXED_REFERENCES.json','SYNTH_GT_LABELS.jsonl.gz','ACTION_TARGETS.jsonl.gz','DERIVED_C2_ACTION_TARGETS.jsonl.gz']:raise PermissionError('Reference prohibited in feature/prediction')
 sys.addaudithook(hook)
def project(pose,family,K,scale):
 X=np.asarray(family['native_xyz'],float);Q=np.asarray(family['physical_registration_Q']);Y=(X-X.mean(0))@Q;z=Y@np.asarray(pose['R_physical']).T;t=np.asarray(pose['centroid']);v=z+t;h=v@K.T;q=h[:,:2]/h[:,2:];P=(K[None,:2,:]*h[:,2,None,None]-h[:,:2,None]*K[None,2:3,:])/h[:,2,None,None]**2
 S=np.zeros((len(X),3,3));S[:,0,1]=-z[:,2];S[:,0,2]=z[:,1];S[:,1,0]=z[:,2];S[:,1,2]=-z[:,0];S[:,2,0]=-z[:,1];S[:,2,1]=z[:,0]
 J=np.concatenate([P@(-S),P*scale],axis=2);return q,J,v[:,2]
def system(pose,family,spec,scale):
 q,J,depth=project(pose,family,np.asarray(spec['K'],float),scale);assert np.all(depth>0)
 boundary=[b for b in spec['boundary_observations']if b.get('association_provenance',{}).get('actual_RGB_line_ids')];rev={b['native_corner']for b in spec['boundary_observations']};blocks=[]
 for m in spec['visible']:
  k=m['native_corner']
  if k in rev or m.get('image_measurement')is not True or m.get('measurement_is_pose_projection')is True:continue
  C=np.asarray(m.get('fit_covariance_px2',m['covariance_px2']),float);w=float(m.get('fit_reliability_weight',1.));ev,U=np.linalg.eigh(C);assert np.min(ev)>0 and w>=0
  W=(U/np.sqrt(ev))@U.T*np.sqrt(w);blocks.append((k,W@J[k],W@(q[k]-m['xy'])))
 for b in boundary:
  k=b['native_corner'];w=float(b.get('fit_reliability_weight',1.));assert w>=0
  for line in b['lines']:
   n=np.asarray(line['normal_unit_xy']);sig=float(line['normal_variance_px2']);assert sig>0
   f=np.sqrt(w/sig);blocks.append((k,(n@J[k])[None,:]*f,np.array([n@q[k]+line['offset_px']])*f))
 if not blocks:return np.empty((0,6)),np.empty(0),np.empty(0,int)
 return np.concatenate([x[1]for x in blocks]),np.concatenate([x[2]for x in blocks]),np.concatenate([np.repeat(x[0],len(x[2]))for x in blocks])
def inverse(J):
 if len(J)==0:return 0,None,None,None
 U,s,V=np.linalg.svd(J,full_matrices=False);rank=int(sum(s>s[0]*1e-10));ok=rank==6
 return rank,(V.T/s)@U.T if ok else None,(V.T/s**2)@V if ok else None,float(s[0]/s[-1])if ok else None
def values(pose,family,spec,scale):
 J,r,corner=system(pose,family,spec,scale);rank,P,C,condition=inverse(J);v=[float(len(r)),float(rank),condition]+[None]*7+[float(np.sqrt(np.mean(r*r)))if len(r)else None,0.]
 if P is not None:
  step=-P@r;v[3:7]=[scale*np.sqrt(np.trace(C[3:,3:])),np.sqrt(np.trace(C[:3,:3])),scale*np.linalg.norm(step[3:]),np.linalg.norm(step[:3])];dt=[];dr=[];bad=0
  for k in np.unique(corner):
   take=corner!=k;_,pp,_,_=inverse(J[take])
   if pp is None:bad+=1;continue
   delta=-pp@r[take]-step;dt.append(scale*np.linalg.norm(delta[3:]));dr.append(np.linalg.norm(delta[:3]))
  v[7:10]=[max(dt)if dt else None,max(dr)if dr else None,float(np.max(np.sum(J*P.T,axis=1)))];v[11]=float(bad)
 else:v[11]=float(len(np.unique(corner)))
 assert len(v)==12 and all(x is None or np.isfinite(x)for x in v);return v
def jacobian_check(fs):
 import cv2
 errs=[]
 for i in np.linspace(0,len(fs)-1,6,dtype=int):
  f=fs[i];run=rows(B/'d1_synthetic_candidates_v1/frames'/f'{i:04d}'/'E_RUN.jsonl.gz')[0];assert run['id']==f['id'];p=A.actions(f)[0]['actual_pose'];s=run['packet']['solver_input'];scale=f['object_scale_m']
  for name in FAMILIES:
   fam=run['families'][name];q,J,_=project(p,fam,np.asarray(s['K']),scale)
   for j in range(6):
    d=np.eye(6)[j]*1e-6;a=dict(p);b=dict(p)
    if j<3:a['R_physical']=(cv2.Rodrigues(d[:3])[0]@p['R_physical']).tolist();b['R_physical']=(cv2.Rodrigues(-d[:3])[0]@p['R_physical']).tolist()
    else:a['centroid']=(np.asarray(p['centroid'])+scale*d[3:]).tolist();b['centroid']=(np.asarray(p['centroid'])-scale*d[3:]).tolist()
    numeric=(project(a,fam,np.asarray(s['K']),scale)[0]-project(b,fam,np.asarray(s['K']),scale)[0])/2e-6;errs.append(float(np.max(np.abs(numeric-J[:,:,j]))))
 assert max(errs)<1e-4;return dict(checks=len(errs),max_abs_error=max(errs),epsilon=1e-6,status='PASS')
def features(pop):
 bound();firewall();start=time.monotonic();syn=pop=='SYNTHETIC';root=B/('d1_synthetic_candidates_v1'if syn else'd1_real_candidates_v1');fs=rows(root/'HEAD_FEATURES.jsonl.gz');out=R/pop;out.mkdir(exist_ok=False);allrows=[];actionn=0;timings=[]
 if syn:write(out/'JACOBIAN_CHECK.json',jacobian_check(fs))
 for i,f in enumerate(fs):
  if not syn and f['partition']!='CAL':continue
  tick=time.monotonic();fp=root/'frames'/f'{i:04d}'/'E_RUN.jsonl.gz';seal=read(fp.parent/'FRAME_SEAL.json');assert sha(fp)==seal['e_run_sha256'];run=rows(fp)[0];assert run['id']==f['id'];es=A.actions(f);rr=[]
  for e in es:
   vv=[]
   for family in FAMILIES:vv+=values(e['actual_pose'],run['families'][family],run['packet']['solver_input'],f['object_scale_m'])
   rr.append(dict(candidate_id=e['candidate_id'],values=vv))
  allrows.append(dict(id=f['id'],split=f['split'],partition=f['partition'],actions=rr,e_run_sha256=seal['e_run_sha256']));actionn+=len(rr);timings.append(dict(id=f['id'],candidates=len(rr),seconds=time.monotonic()-tick))
  if(i+1)%100==0:print(json.dumps(dict(stage='LOCAL_INFLUENCE_FEATURES',processed=i+1,total=len(fs),seconds=time.monotonic()-start)),flush=True)
 save(out/'FEATURES.jsonl.gz',allrows);csvwrite(out/'TIMING.csv',timings)
 if syn:
  z=np.array([a['values']for f in allrows if f['split']=='train'for a in f['actions']],float);z=np.sign(z)*np.log1p(np.abs(z));mu=np.nanmean(z,axis=0);sd=np.nanstd(z,axis=0);sd=np.where(sd>1e-12,sd,1.);seen=np.isfinite(mu);mu=np.nan_to_num(mu);sd=np.nan_to_num(sd,nan=1.);write(out/'NORMALIZER.json',dict(mean=mu.tolist(),scale=sd.tolist(),seen=seen.tolist(),scope='synthetic TRAIN eligible actions only; no CAL/VAL normalizer fit',names=[f+'/'+n for f in FAMILIES for n in NAMES]))
 write(out/'SEAL.json',dict(rows=len(allrows),actions=actionn,seconds=time.monotonic()-start,features_sha256=sha(out/'FEATURES.jsonl.gz'),source_sha256=sha(__file__),new_PnP=0,new_observation=0,new_model_forward=0,physical_uncertainty_calibration_claim=False,ignored_prior_outline_and_nonRGB_constraints=True));print(dict(stage='FEATURES_DONE',population=pop,frames=len(allrows),actions=actionn,seconds=time.monotonic()-start))
def extend(frame,norm,extra,extra_norm,arm):
 x,es=A.inputs(frame,norm,'H1')
 if arm=='CONTROL':return x,es
 by={a['candidate_id']:a['values']for a in extra['actions']};raw=np.array([by[e['candidate_id']]for e in es],float);valid=np.isfinite(raw);z=np.sign(raw)*np.log1p(abs(raw));v=np.where(valid&(np.array(extra_norm['seen'])[None,:]),(z-extra_norm['mean'])/extra_norm['scale'],0.);return np.concatenate([x,v,valid.astype(float)],axis=1).astype(np.float32),es
def model(arm,initial=None):
 n=188 if arm=='CONTROL'else 236;m=torch.nn.Sequential(torch.nn.Linear(n,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,1))
 if initial is not None:
  state={k:v.clone()for k,v in initial.items()}
  if arm!='CONTROL':state['0.weight']=torch.cat([state['0.weight'],torch.zeros((64,48))],dim=1)
  m.load_state_dict(state)
 return m
def train(seed,arm):
 setup();bound();out=R/f'{arm}_s{seed}';out.mkdir(exist_ok=False);ledger=R/'TRAINING_ATTEMPTS.jsonl';old=ledger.read_text().splitlines()if ledger.exists()else[];assert len(old)<6
 with ledger.open('a')as h:h.write(json.dumps(dict(seed=seed,arm=arm,status='RESERVED_BEFORE_OPTIMIZER',time=time.time()))+'\n')
 start=time.monotonic();features=index(rows(R/'SYNTHETIC/FEATURES.jsonl.gz'));en=read(R/'SYNTHETIC/NORMALIZER.json');cp=torch.load(V5/f's{seed}/best.pt',weights_only=False,map_location='cpu');labels=index(rows(V5/'DERIVED_C2_ACTION_TARGETS.jsonl.gz'));fs=rows(B/'d1_synthetic_candidates_v1/HEAD_FEATURES.jsonl.gz');data={'train':[],'validation':[]}
 for f in fs:
  x,es=extend(f,cp['normalizer'],features[f['id']],en,arm);by={a['candidate_id']:a for a in labels[f['id']]['action_metrics']};y=np.array([float(by[e['candidate_id']]['success_5cm5deg'])for e in es],np.float32);data[f['split']].append(dict(x=torch.from_numpy(x),y=torch.from_numpy(y)))
 torch.manual_seed(seed);m=model(arm,cp['model_state_dict']);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);rng=np.random.default_rng(seed);hist=[];best=math.inf;wait=0
 def loss(group):return torch.stack([torch.nn.functional.binary_cross_entropy_with_logits(m(r['x']).flatten(),r['y'])for r in group]).mean()
 def ev(items):
  m.eval()
  with torch.inference_mode():return sum(float(loss(items[i:i+32]))*len(items[i:i+32])for i in range(0,len(items),32))/len(items)
 fitstart=time.monotonic()
 for ep in range(50):
  m.train();order=rng.permutation(len(data['train']))
  for i in range(0,len(order),32):
   g=[data['train'][int(k)]for k in order[i:i+32]];opt.zero_grad(set_to_none=True);l=loss(g);assert torch.isfinite(l);l.backward();opt.step()
  tr=ev(data['train']);val=ev(data['validation']);hist.append(dict(epoch=ep+1,train_BCE=tr,validation_BCE=val))
  if val<best:best=val;wait=0;bestep=ep+1;torch.save(dict(model_state_dict=m.state_dict(),normalizer=cp['normalizer'],extra_normalizer=en,seed=seed,arm=arm,epoch=bestep),out/'best.pt')
  else:wait+=1
  if wait>=5:break
  assert time.monotonic()-fitstart<180
 write(out/'HISTORY.json',hist);write(out/'RECEIPT.json',dict(seed=seed,arm=arm,epochs=len(hist),best_epoch=bestep,parameters=sum(p.numel()for p in m.parameters()),fit_seconds=time.monotonic()-fitstart,total_seconds=time.monotonic()-start,new_fit=1,checkpoint_sha256=sha(out/'best.pt'),initializer_sha256=sha(V5/f's{seed}/best.pt'),new_pose=0,CPU_threads=1,GPU=0));print(dict(stage='TRAINED',seed=seed,arm=arm,epochs=len(hist),best_epoch=bestep,seconds=time.monotonic()-fitstart))
def predict(pop,seed):
 setup();bound();firewall();syn=pop=='SYNTHETIC_VAL'
 if not syn:assert read(R/f'SYNTHETIC_VAL_s{seed}/EVALUATION.json')['gate_passed']
 out=R/f'{pop}_s{seed}';out.mkdir(exist_ok=False);start=time.monotonic();extra=index(rows(R/('SYNTHETIC'if syn else'CAL46')/'FEATURES.jsonl.gz'));fs=rows(B/('d1_synthetic_candidates_v1'if syn else'd1_real_candidates_v1')/'HEAD_FEATURES.jsonl.gz');fs=[f for f in fs if(f['split']=='validation'if syn else f['partition']=='CAL')];models={};ck={}
 for arm in ['CONTROL','INFLUENCE','BASELINE']:
  cp=V5/f's{seed}/best.pt'if arm=='BASELINE'else R/f'{arm}_s{seed}/best.pt';c=torch.load(cp,weights_only=False,map_location='cpu');m=model('CONTROL'if arm=='BASELINE'else arm);m.load_state_dict(c['model_state_dict']);m.eval().requires_grad_(False);models[arm]=m;ck[arm]=c
 ps=[]
 with torch.inference_mode():
  for f in fs:
   p=dict(id=f['id'],partition=f['partition'],recording=f['cluster_id'],methods={})
   for arm,m in models.items():
    c=ck[arm];x,es=extend(f,c['normalizer'],extra[f['id']],c.get('extra_normalizer'), 'CONTROL'if arm=='BASELINE'else arm);v=m(torch.from_numpy(x)).flatten().numpy();k=int(np.argmax(v));p['methods'][arm]=dict(candidate_id=es[k]['candidate_id'],pose=es[k]['actual_pose'])
   p['bank_ids']=[e['candidate_id']for e in es];ps.append(p)
 save(out/'PREDICTIONS.jsonl.gz',ps);write(out/'SEAL.json',dict(prediction_sha256=sha(out/'PREDICTIONS.jsonl.gz'),status='SEALED_BEFORE_EVALUATION',rows=len(ps),reference_inputs=0,seconds=time.monotonic()-start,new_PnP=0,new_observation=0,head_frame_forward_calls=3*len(ps)));print(dict(stage='PREDICTED',population=pop,n=len(ps)))
def gate(a,b):
 why=[]
 if a['successes']<b['successes']:why.append('joint_success_decreased')
 for k in ['T_mean','R_mean','T_P90','T_P95','R_P90','R_P95','T_max','R_max','no_pose','T_gt50','T_gt100','R_gt45']:
  if a[k]>b[k]+1e-7:why.append(k+'_increased')
 if not(a['T_mean']<b['T_mean']-1e-7 or a['R_mean']<b['R_mean']-1e-7):why.append('no_mean_improvement')
 return why
def evaluate(pop,seed):
 out=R/f'{pop}_s{seed}';assert sha(out/'PREDICTIONS.jsonl.gz')==read(out/'SEAL.json')['prediction_sha256'];ps=rows(out/'PREDICTIONS.jsonl.gz');syn=pop=='SYNTHETIC_VAL';labels=index(rows(V5/'DERIVED_C2_ACTION_TARGETS.jsonl.gz'))if syn else None;refs=read(B.parent/'pallet-statewise-correspondence-diagnosis-20261016-v1/score_v2/FIXED_REFERENCES.json')['references']if not syn else None;ms={k:[]for k in ['BASELINE','CONTROL','INFLUENCE']};pairs=[]
 for p in ps:
  by={a['candidate_id']:a for a in labels[p['id']]['action_metrics']}if syn else None;r=dict(id=p['id'],recording=p['recording'],candidate_n=len(p['bank_ids']))
  for arm,pred in p['methods'].items():
   if syn:
    y=by[pred['candidate_id']];m=dict(pose_available=y['translation_cm']is not None,translation_cm=y['translation_cm'],rotation_deg=y['rotation_deg'],success=y['success_5cm5deg'])
   else:m=metric(pred['pose'],refs[p['partition']][p['id']])
   ms[arm].append(m);r[arm+'_id']=pred['candidate_id'];r.update({arm+'_'+k:v for k,v in m.items()})
  r['bank_good']=any(by[i]['success_5cm5deg']for i in p['bank_ids'])if syn else None;pairs.append(r)
 st={k:stats(v)for k,v in ms.items()};reasons={k:gate(st['INFLUENCE'],st[k])for k in ['BASELINE','CONTROL']};rec={k:dict(recover=sum(not a['success']and b['success']for a,b in zip(ms[k],ms['INFLUENCE'])),damage=sum(a['success']and not b['success']for a,b in zip(ms[k],ms['INFLUENCE'])),changed=sum(p['methods'][k]['candidate_id']!=p['methods']['INFLUENCE']['candidate_id']for p in ps))for k in ['BASELINE','CONTROL']};result=dict(population=pop,seed=seed,methods=st,gate_passed=not any(reasons.values()),gate_reasons=reasons,recover_damage=rec,control_gate_vs_original=gate(st['CONTROL'],st['BASELINE']),bank_good=sum(bool(p['bank_good'])for p in pairs)if syn else None,exposed_development=True,new_independent_real=0)
 csvwrite(out/'PAIRED_ROWS.csv',pairs);write(out/'EVALUATION.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['features','train','predict','evaluate']);p.add_argument('--population',default='SYNTHETIC');p.add_argument('--seed',type=int,choices=[20261018,20261019,20261020],default=20261018);p.add_argument('--arm',choices=['CONTROL','INFLUENCE']);a=p.parse_args();features(a.population)if a.stage=='features'else train(a.seed,a.arm)if a.stage=='train'else(predict if a.stage=='predict'else evaluate)(a.population,a.seed)
