"""Original188 H1/C2 recipe, own properly re-exported union train normalizer."""
import pathlib,sys,argparse,time,math,json,os
import numpy as np
import torch
R=pathlib.Path(__file__).resolve().parent;T=R.parent;sys.path.insert(0,str(T/'geometric_influence_20261021_v19'));import experiment as G
U=R/'SYNTHETIC_v3'
def bound():
 p=G.read(R/'LEARNING_PROTOCOL.json')
 for k,v in p['bindings'].items():assert G.sha(v['path'])==v['sha256'],k
 return p
def guard():
 def f(e,a):
  if e=='open'and isinstance(a[0],(str,bytes))and pathlib.Path(os.fsdecode(a[0])).name in ['FIXED_REFERENCES.json','SYNTH_GT_LABELS.jsonl.gz','C2_TARGETS.jsonl.gz','ACTION_TARGETS.jsonl.gz','DERIVED_C2_ACTION_TARGETS.jsonl.gz']:raise PermissionError('References forbidden in inference')
 sys.addaudithook(f)
def train(seed):
 G.setup();bound();seal=G.read(U/'TARGET_SEAL.json');assert seal['counts']['train']['new_good']>=5
 if seed!=20261018:assert G.read(R/'CAL46_s20261018/EVALUATION.json')['gate_passed']
 out=R/f's{seed}';out.mkdir(exist_ok=False);ledger=R/'TRAINING_ATTEMPTS.jsonl';prev=ledger.read_text().splitlines()if ledger.exists()else[];assert len(prev)<3
 with ledger.open('a')as h:h.write(json.dumps(dict(seed=seed,state='RESERVED_BEFORE_OPTIMIZER',time=time.time()))+'\n')
 start=time.monotonic();fs=G.rows(U/'HEAD_FEATURES.jsonl.gz');labs=G.index(G.rows(U/'C2_TARGETS.jsonl.gz'));norm=G.read(U/'NORMALIZER.json');items={s:[G.M.prepared(f,labs[f['id']],norm,'H1')for f in fs if f['split']==s]for s in ['train','validation']};torch.manual_seed(seed);rng=np.random.default_rng(seed);model=G.M.make_model('MLP');opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);best=math.inf;wait=0;hist=[];fit=time.monotonic()
 def ev(items):
  model.eval();n=0.;total=0
  with torch.inference_mode():
   for i in range(0,len(items),32):g=items[i:i+32];n+=float(G.M.loss_value(model,G.M.collate(g),'H1'))*len(g);total+=len(g)
  return n/total
 for ep in range(50):
  model.train();order=rng.permutation(len(items['train']))
  for i in range(0,len(order),32):
   group=[items['train'][int(k)]for k in order[i:i+32]];opt.zero_grad(set_to_none=True);l=G.M.loss_value(model,G.M.collate(group),'H1');assert torch.isfinite(l);l.backward();opt.step()
  tr=ev(items['train']);vl=ev(items['validation']);hist.append(dict(epoch=ep+1,train_BCE=tr,validation_BCE=vl))
  if vl<best:best=vl;wait=0;bestep=ep+1;torch.save(dict(model_state_dict=model.state_dict(),normalizer=norm,seed=seed,epoch=bestep,objective='unchangedC2binary_joint5cm5deg',feature_sha256=G.sha(U/'HEAD_FEATURES.jsonl.gz')),out/'best.pt')
  else:wait+=1
  if wait>=5:break
  assert time.monotonic()-fit<180
 G.write(out/'HISTORY.json',hist);G.write(out/'RECEIPT.json',dict(seed=seed,epochs=len(hist),best_epoch=bestep,parameters=16321,fit_seconds=time.monotonic()-fit,total_seconds=time.monotonic()-start,checkpoint_sha256=G.sha(out/'best.pt'),CPU_threads=1,GPU=0,new_success_fit=1,initialization='fresh same torch seed as originalv5',normalizer_sha256=G.sha(U/'NORMALIZER.json')));print(dict(stage='TRAINED_UNION_H1',seed=seed,epochs=len(hist),best_epoch=bestep,seconds=time.monotonic()-fit))
def predict(pop,seed):
 G.setup();bound();guard();syn=pop=='SYNTHETIC_VAL'
 if not syn:assert G.read(R/f'SYNTHETIC_VAL_s{seed}/EVALUATION.json')['gate_passed']
 out=R/f'{pop}_s{seed}';out.mkdir(exist_ok=False);start=time.monotonic();c=torch.load(R/f's{seed}/best.pt',map_location='cpu',weights_only=False);model=G.M.make_model('MLP');model.load_state_dict(c['model_state_dict']);model.eval().requires_grad_(False);bc=torch.load(G.V5/f's{seed}/best.pt',weights_only=False,map_location='cpu');base=G.M.make_model('MLP');base.load_state_dict(bc['model_state_dict']);base.eval().requires_grad_(False);union=G.rows((U if syn else R/'CAL46_v3')/'HEAD_FEATURES.jsonl.gz');old=G.index(G.rows(G.B/('d1_synthetic_candidates_v1'if syn else'd1_real_candidates_v1')/'HEAD_FEATURES.jsonl.gz'));union=[f for f in union if f['split']=='validation']if syn else union;ps=[];times=[]
 with torch.inference_mode():
  for f in union:
   tick=time.monotonic();x,es=G.A.inputs(f,c['normalizer'],'H1');v=model(torch.from_numpy(x)).flatten().numpy();new=int(np.argmax(v));bx,bes=G.A.inputs(old[f['id']],bc['normalizer'],'H1');bv=base(torch.from_numpy(bx)).flatten().numpy();b=int(np.argmax(bv));ps.append(dict(id=f['id'],partition=f['partition'],recording=f['cluster_id'],before_id=bes[b]['candidate_id'],before_pose=bes[b]['actual_pose'],after_id=es[new]['candidate_id'],after_pose=es[new]['actual_pose'],bank_ids=[e['candidate_id']for e in es],original_bank_ids=[e['candidate_id']for e in bes]));times.append(dict(id=f['id'],old_actions=len(bes),union_actions=len(es),input_and_two_heads_seconds=time.monotonic()-tick))
 G.save(out/'PREDICTIONS.jsonl.gz',ps);G.csvwrite(out/'TIMING.csv',times);G.write(out/'SEAL.json',dict(status='SEALED_BEFORE_SCORING',prediction_sha256=G.sha(out/'PREDICTIONS.jsonl.gz'),n=len(ps),reference_inputs=0,seconds=time.monotonic()-start,new_head_frame_forwards=2*len(ps),new_PnP=0,new_RGB=0));print(dict(stage='SEALED_UNION',pop=pop,n=len(ps)))
def evaluate(pop,seed):
 out=R/f'{pop}_s{seed}';assert G.sha(out/'PREDICTIONS.jsonl.gz')==G.read(out/'SEAL.json')['prediction_sha256'];ps=G.rows(out/'PREDICTIONS.jsonl.gz');syn=pop=='SYNTHETIC_VAL';labs=G.index(G.rows(U/'C2_TARGETS.jsonl.gz'))if syn else None;refs=G.read(G.B.parent/'pallet-statewise-correspondence-diagnosis-20261016-v1/score_v2/FIXED_REFERENCES.json')['references']if not syn else None;aa=[];bb=[];paired=[]
 for p in ps:
  by={m['candidate_id']:m for m in labs[p['id']]['action_metrics']}if syn else None
  def met(cid,pose):
   if not syn:return G.metric(pose,refs[p['partition']][p['id']])
   m=by[cid];return dict(pose_available=m['translation_cm']is not None,translation_cm=m['translation_cm'],rotation_deg=m['rotation_deg'],success=m['success_5cm5deg'])
  a=met(p['after_id'],p['after_pose']);b=met(p['before_id'],p['before_pose']);aa.append(a);bb.append(b);r=dict(id=p['id'],recording=p['recording'],before_id=p['before_id'],after_id=p['after_id'],old_bank_n=len(p['original_bank_ids']),union_bank_n=len(p['bank_ids']))
  for k,m in [('before',b),('after',a)]:r.update({k+'_'+j:v for j,v in m.items()})
  if syn:r['union_bank_good']=any(by[i]['success_5cm5deg']for i in p['bank_ids']);r['selection_failure']=r['union_bank_good']and not a['success']
  paired.append(r)
 a,b=G.stats(aa),G.stats(bb);why=G.gate(a,b);result=dict(population=pop,seed=seed,before=b,after=a,gate_passed=not why,gate_reasons=why,recover=sum(not x['success']and y['success']for x,y in zip(bb,aa)),damage=sum(x['success']and not y['success']for x,y in zip(bb,aa)),changed=sum(p['before_id']!=p['after_id']for p in ps),union_good=sum(p.get('union_bank_good',False)for p in paired)if syn else None,selection_failures=sum(p.get('selection_failure',False)for p in paired)if syn else None,new_independent_test=0,exposed_development=True)
 G.csvwrite(out/'PAIRED_ROWS.csv',paired);G.write(out/'EVALUATION.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','predict','evaluate']);p.add_argument('--population',default='SYNTHETIC_VAL');p.add_argument('--seed',type=int,default=20261018);a=p.parse_args();train(a.seed)if a.stage=='train'else(predict if a.stage=='predict'else evaluate)(a.population,a.seed)
