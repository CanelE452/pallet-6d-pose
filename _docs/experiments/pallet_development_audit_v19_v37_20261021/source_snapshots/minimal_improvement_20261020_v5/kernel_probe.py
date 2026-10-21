"""Fixed pose-basin mass selector on original H1 logits. No fit or new poses."""
import argparse,csv,gzip,json,math,sys,time,resource
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parent;TASK=ROOT.parent;SRC=Path('${SOURCE_REPO}');sys.path.insert(0,str(SRC));sys.path.insert(0,str(TASK/'mean_gap_audit_20261020_v1'))
from analyze import rows,index,read,metric,sha,write,csvwrite
from scripts.research.pallet_boundary_action_repair_v1 import action_data as A,action_models as M
B=Path('${DATA_ROOT}/pallet-boundary-action-repair-20261018-v1')
NAME='H1_POSE_BASIN_MASS_KERNEL_V1'
def log_kernel(es):
 R=np.asarray([e['actual_pose']['R_physical']for e in es]);t=np.asarray([e['actual_pose']['centroid']for e in es]);S=np.diag([-1,1,-1]);trace=np.maximum(np.einsum('iab,jab->ij',R,R),np.einsum('iab,jab->ij',R@S,R));a=np.arccos(np.clip((trace-1)/2,-1,1));d=np.linalg.norm(t[:,None]-t[None,:],axis=2);return -.5*((d/.05)**2+(a/np.deg2rad(5))**2)
def scores(es,logits):return logsumexp(log_kernel(es)+np.asarray(logits)[None,:],axis=1)-np.log(len(es))
def folder(pop,seed):return ROOT/NAME/f'{pop}_s{seed}'
def predict(pop,seed):
 start=time.monotonic();torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 protocol=read(ROOT/'KERNEL_PROTOCOL.json');assert protocol['variant']==NAME
 if pop=='PRIMARY192':assert read(folder('SYNTHETIC_VAL',20261018)/'EVALUATION.json')['expand_gate_passed']
 if seed!=20261018:assert read(folder('PRIMARY192',20261018)/'EVALUATION.json')['expand_gate_passed']
 out=folder(pop,seed);out.mkdir(parents=True,exist_ok=False)
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes))and Path(args[0]).name in ['FIXED_REFERENCES.json','SYNTH_GT_LABELS.jsonl.gz','ACTION_TARGETS.jsonl.gz','DENSITY_PRECHECK_FRAMES.csv','FOUR_CASE_DENSITY.csv']:raise PermissionError('Prediction firewall blocks labels/diagnostic results')
 sys.addaudithook(guard)
 frames=rows(B/('d1_synthetic_candidates_v1'if pop=='SYNTHETIC_VAL'else'd1_real_candidates_v1')/'HEAD_FEATURES.jsonl.gz');frames=[f for f in frames if f['split']=='validation']if pop=='SYNTHETIC_VAL'else[f for f in frames if f['partition']!='CAL'];assert len(frames)==(235 if pop=='SYNTHETIC_VAL' else 192)
 norm=read(B/'d1_action_labels_v1/TRAIN_ONLY_NORMALIZER.json');prepared=[];forward=0;cached=index(rows(B/'action_predictions_v1/PREDICTIONS.jsonl.gz'))if pop=='PRIMARY192'else None
 if cached is None:
  cp=B/'action_training_v1'/f'D1_H1_MLP_s{seed}'/'best.pt';ck=torch.load(cp,map_location='cpu',weights_only=False);model=M.make_model('MLP');model.load_state_dict(ck['model_state_dict']);model.eval().requires_grad_(False);assert ck['normalizer']==norm
 for f in frames:
  x,es=A.inputs(f,norm,'H1');prepared.append((f,x,es))
 result=[]
 for off in range(0,len(prepared),32):
  group=prepared[off:off+32]
  if cached is None:
   batch=torch.zeros((len(group),max(len(es)for f,x,es in group),188))
   for j,(f,x,es)in enumerate(group):batch[j,:len(es)]=torch.from_numpy(x)
   with torch.inference_mode():logits=model(batch).squeeze(-1).numpy();forward+=1
  for j,(f,x,es)in enumerate(group):
   if cached is None:ls=logits[j,:len(es)].astype(float);before=int(np.argmax(ls))
   else:
    p=cached[f['id']]['predictions'][f'D1_H1_MLP_s{seed}'];by={a['candidate_id']:a['score']for a in p['actions']};ls=np.array([by[e['candidate_id']]for e in es]);before=next(k for k,e in enumerate(es)if e['candidate_id']==p['selected_candidate_id'])
   s=scores(es,ls);after=int(np.argmax(s));a,b=es[before],es[after];result.append(dict(id=f['id'],partition=f['partition'],recording=f['cluster_id'],before=dict(candidate_id=a['candidate_id'],actual_pose=a['actual_pose'],original_logit=float(ls[before])),after=dict(candidate_id=b['candidate_id'],actual_pose=b['actual_pose'],kernel_mass_score=float(s[after]),original_logit=float(ls[after])),action_n=len(es),reference_inputs=False))
 with gzip.open(out/'PREDICTIONS.jsonl.gz','xt')as h:
  for r in result:h.write(json.dumps(r,allow_nan=False)+'\n')
 write(out/'PREDICTION_SEAL.json',dict(status='SEALED_BEFORE_SCORING',protocol_sha256=sha(ROOT/'KERNEL_PROTOCOL.json'),source_sha256=sha(__file__),prediction_sha256=sha(out/'PREDICTIONS.jsonl.gz'),rows=len(result),seed=seed,population=pop,original_candidate_bank=True,new_pose=0,new_training=0,new_observation=0,GT_reads=0,H1_batched_forward_calls=forward,CPU_threads=1,GPU=0,wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time(),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));print(dict(status='PREDICTIONS_SEALED',population=pop,seed=seed,n=len(result),forward=forward))
def stats(ms):
 r=dict(denominator=len(ms),successes=sum(m['success']for m in ms),no_pose=sum(not m['pose_available']for m in ms))
 for p,k in [('T','translation_cm'),('R','rotation_deg')]:
  v=[m[k]for m in ms if m[k]is not None];r[p+'_n']=len(v)
  for n,z in [('mean',np.mean(v)),('median',np.median(v)),('P90',np.quantile(v,.9)),('P95',np.quantile(v,.95)),('max',max(v))]:r[p+'_'+n]=float(z)
 r.update(T_gt50=sum(m['translation_cm']is not None and m['translation_cm']>50 for m in ms),T_gt100=sum(m['translation_cm']is not None and m['translation_cm']>100 for m in ms),R_gt45=sum(m['rotation_deg']is not None and m['rotation_deg']>45 for m in ms));return r
def evaluate(pop,seed):
 out=folder(pop,seed);s=read(out/'PREDICTION_SEAL.json');assert sha(out/'PREDICTIONS.jsonl.gz')==s['prediction_sha256'];pred=rows(out/'PREDICTIONS.jsonl.gz');labs=index(rows(B/'d1_action_labels_v1/ACTION_TARGETS.jsonl.gz'))if pop=='SYNTHETIC_VAL'else None;refs=read(B.parent/'pallet-statewise-correspondence-diagnosis-20261016-v1/score_v2/FIXED_REFERENCES.json')['references']if labs is None else None
 paired=[];before=[];after=[]
 for p in pred:
  mm=[]
  for kind in ['before','after']:
   if labs is not None:
    m=next(r for r in labs[p['id']]['action_metrics']if r['candidate_id']==p[kind]['candidate_id']);m=dict(pose_available=m['translation_cm']is not None,translation_cm=m['translation_cm'],rotation_deg=m['rotation_deg'],success=m['success_5cm5deg'])
   else:m=metric(p[kind]['actual_pose'],refs[p['partition']][p['id']])
   mm.append(m)
  a,b=mm;before.append(a);after.append(b);paired.append(dict(id=p['id'],recording=p['recording'],before_id=p['before']['candidate_id'],after_id=p['after']['candidate_id'],before_success=a['success'],after_success=b['success'],before_T_cm=a['translation_cm'],after_T_cm=b['translation_cm'],before_R_deg=a['rotation_deg'],after_R_deg=b['rotation_deg'],delta_T_cm=b['translation_cm']-a['translation_cm'],delta_R_deg=b['rotation_deg']-a['rotation_deg'],selection_changed=p['before']['candidate_id']!=p['after']['candidate_id']))
 a,b=stats(before),stats(after);reasons=[]
 if not b['T_mean']<a['T_mean']:reasons.append('T_mean_not_lower')
 if not b['R_mean']<a['R_mean']:reasons.append('R_mean_not_lower')
 if b['successes']<a['successes']:reasons.append('joint_success_decreased')
 for k in ['no_pose','T_gt50','T_gt100','R_gt45']:
  if b[k]>a[k]:reasons.append(k+'_increased')
 if pop=='SYNTHETIC_VAL':
  old=read(B/'action_training_v1'/f'D1_H1_MLP_s{seed}'/'RUN_RECEIPT.json')['best_metrics']['validation'];assert a['successes']==old['success_5cm5deg']and abs(a['T_mean']-old['translation_cm']['mean'])<1e-8 and abs(a['R_mean']-old['rotation_deg']['mean'])<1e-8
 result=dict(status='COMPLETE',variant=NAME,population=pop,seed=seed,before=a,after=b,delta_T_mean=b['T_mean']-a['T_mean'],delta_R_mean=b['R_mean']-a['R_mean'],recover=sum(not x['success']and y['success']for x,y in zip(before,after)),damage=sum(x['success']and not y['success']for x,y in zip(before,after)),common_available_n=sum(x['pose_available']and y['pose_available']for x,y in zip(before,after)),selection_changes=sum(r['selection_changed']for r in paired),expand_gate_passed=not reasons,gate_failure_reasons=reasons,new_training=0,new_pose=0,new_independent_real=0,scope='exploratory reused synthetic/development results; fixed rule, no threshold fitting')
 csvwrite(out/'PAIRED_ROWS.csv',paired);write(out/'EVALUATION.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['predict','evaluate']);p.add_argument('--population',choices=['SYNTHETIC_VAL','PRIMARY192'],required=True);p.add_argument('--seed',type=int,choices=[20261018,20261019,20261020],default=20261018);a=p.parse_args();(predict if a.stage=='predict'else evaluate)(a.population,a.seed)
