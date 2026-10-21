"""Controlled second SubPix proposal on fixed N0-family geometry; no model/GT inputs."""
import ast,sys,pathlib,json,gzip,time,math,hashlib,argparse,os
from collections import Counter
import numpy as np
import cv2
R=pathlib.Path(__file__).resolve().parent/'track2';TASK=R.parent.parent;SRC=pathlib.Path('${SOURCE_REPO}');sys.path.insert(0,str(TASK/'mean_gap_audit_20261020_v1'));sys.path.insert(0,str(TASK/'minimal_improvement_20261020_v5'))
from analyze import read,rows,index,write,csvwrite,sha,metric
from kernel_probe import stats
B=pathlib.Path('${DATA_ROOT}/pallet-boundary-action-repair-20261018-v1')
def saved(p,rs):
 with gzip.open(p,'xt')as h:
  for x in rs:h.write(json.dumps(x,allow_nan=False)+'\n')
def load_functions():
 cfg=dict(SUBPIX=dict(minimum_image_extent=15));ns=dict(np=np,cv2=cv2,Counter=Counter,METHODS=('SUBPIX',),_configuration=lambda:cfg)
 for rel,names in [('scripts/research/pallet_training_free_compare_20261007_v1/methods.py',['_inputs','cap_points','correct']),('scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py',['solve'])]:
  tree=ast.parse((SRC/rel).read_text());body=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names];assert len(body)==len(names);exec(compile(ast.Module(body=body,type_ignores=[]),rel,'exec'),ns)
 return ns
def bound():
 p=read(R/'PROTOCOL.json')
 for k,v in p['bindings'].items():assert sha(v['path'])==v['sha256'],k
 return p
def frames(pop):
 domain='SYNTHETIC1003'if pop in ['PARITY','SYNTHETIC_VAL']else'REAL238';fs=rows(B/f'd1_upstream_v1/{domain}_UPSTREAM.jsonl.gz')
 return[f for f in fs if f['split']=='validation']if pop=='SYNTHETIC_VAL'else[f for f in fs if (f['partition']=='CAL')==(pop=='CAL46')]if pop!='PARITY'else fs
def solve_pose(u,points,fn):
 q=np.asarray(points,float)[:8];valid=np.isfinite(q).all(-1);X=np.asarray(u['native_xyz'],float);Q=np.asarray(u['initial_pose']['R_cf']).T@np.asarray(u['initial_pose']['R_physical'])
 if valid.sum()<6:return dict(available=False,reason='less_than6finite_native_points')
 try:
  sol=fn['solve'](X,q,np.asarray(u['K'],float),valid)
  if sol is None:return dict(available=False,reason='SQPnP_failed')
  Rcf,t,residual=sol;Rphys=Rcf@Q;depth=(X@Rcf.T+t)[:,2]
  if not np.isfinite(Rphys).all()or not np.isfinite(t).all()or np.min(depth)<=0 or abs(np.linalg.det(Rphys)-1)>1e-6:return dict(available=False,reason='invalid_or_nonpositive_vertex_depth')
  return dict(available=True,R_cf=Rcf.tolist(),R_physical=Rphys.tolist(),centroid=t.tolist(),cf_extents=u['initial_pose']['cf_extents'],selected_hypothesis=u['initial_pose']['selected_hypothesis'],reprojection_px=float(residual))
 except cv2.error as e:return dict(available=False,reason='OpenCV_error',error=str(e))
def parity():
 cv2.setNumThreads(1);bound();fn=load_functions();fs=frames('PARITY');ids=np.linspace(0,len(fs)-1,12,dtype=int).tolist();reals=rows(B/'d1_upstream_v1/REAL238_UPSTREAM.jsonl.gz');cal=[u for u in reals if u['partition']=='CAL'];selected=[fs[i]for i in ids]+[cal[i]for i in [0,15,30,45]];rr=[];start=time.monotonic()
 for u in selected:
  p=solve_pose(u,u['initial_points'],fn);b=u['initial_pose'];assert p['available'];t=100*np.linalg.norm(np.asarray(p['centroid'])-b['centroid']);q=np.asarray(p['R_physical']).T@np.asarray(b['R_physical']);a=math.degrees(math.acos(float(np.clip((np.trace(q)-1)/2,-1,1))));rr.append(dict(id=u['id'],partition=u['partition'],delta_to_saved_N0_T_cm=float(t),delta_to_saved_N0_R_deg=a,pass_tolerance=t<=.001 and a<=.001))
 csvwrite(R/'TECHNICAL_PARITY16.csv',rr);write(R/'TECHNICAL_PARITY.json',dict(status='PASS'if all(x['pass_tolerance']for x in rr)else'BLOCKED_N0_PHYSICAL_REPLAY_MISMATCH',new_PnP=len(rr),new_SubPix=0,no_reference_reads=True,max_delta_T_cm=max(x['delta_to_saved_N0_T_cm']for x in rr),max_delta_R_deg=max(x['delta_to_saved_N0_R_deg']for x in rr),wall_seconds=time.monotonic()-start));print(read(R/'TECHNICAL_PARITY.json'))
def generate(pop):
 cv2.setNumThreads(1);bound();assert read(R/'TECHNICAL_PARITY.json')['status']=='PASS'
 if pop=='CAL46':assert read(R/'SYNTHETIC_VAL/EVALUATION.json')['gamma_safe_gate_passed']
 if pop=='PRIMARY192':assert read(R/'DECISION.json')['allow_PRIMARY']
 def hook(event,args):
  if event=='open'and isinstance(args[0],(str,bytes))and pathlib.Path(os.fsdecode(args[0])).name in ['FIXED_REFERENCES.json','SYNTH_GT_LABELS.jsonl.gz','DERIVED_C2_ACTION_TARGETS.jsonl.gz','ACTION_TARGETS.jsonl.gz']:raise PermissionError('No reference in proposal generation')
 sys.addaudithook(hook)
 out=R/pop;out.mkdir(exist_ok=False);fn=load_functions();lut=np.rint(255*(np.arange(256)/255.)**(1/1.2)).clip(0,255).astype(np.uint8);start=time.monotonic();cpu=time.process_time();ps=[];times=[];pn=0;sn=0
 for u in frames(pop):
  t=time.monotonic();image=pathlib.Path(u['image']);assert sha(image)==u['image_sha256'];rgb=cv2.imread(str(image));assert rgb is not None and list(rgb.shape[:2])==u['raw_hw'];gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY);gamma=cv2.LUT(gray,lut);q=np.asarray(u['initial_points'],float);support=np.isfinite(q).all(-1);native={};props={};diagnostics={}
  for name,img in [('RAW_REPEAT',gray),('GAMMA_REPEAT',gamma)]:
   n,diag=fn['correct'](img,q,support,'SUBPIX');sn+=diag['algorithm_corner_calls'];h,w=img.shape;n=fn['cap_points'](q,n,w,h,support);p=solve_pose(u,n,fn);pn+=1;native[name]=n.tolist();props[name]=p;diagnostics[name]=diag
  ps.append(dict(id=u['id'],partition=u['partition'],recording=u['cluster_id'],image_sha256=u['image_sha256'],N0=u['initial_pose'],proposals=props,native_points=native,SubPix=diagnostics,physical_transform='fixed Q0=Rcf_N0.T@Rphysical_N0; same actual native_xyz and camera K; no GT',GT_inputs=False));times.append(dict(id=u['id'],wall_seconds=time.monotonic()-t,SubPix_corner_calls=sum(d['algorithm_corner_calls']for d in diagnostics.values()),PnP_proposals=2))
 saved(out/'PREDICTIONS.jsonl.gz',ps);csvwrite(out/'TIMING.csv',times);write(out/'SEAL.json',dict(status='SEALED_BEFORE_SCORING',rows=len(ps),prediction_sha256=sha(out/'PREDICTIONS.jsonl.gz'),source_sha256=sha(__file__),protocol_sha256=sha(R/'PROTOCOL.json'),new_PnP=pn,new_SubPix_corner_calls=sn,new_neural_forward=0,new_training=0,new_SAM=0,no_reference_inputs=True,wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,timing_scope='read/hash RGB, grayscale/gamma LUT, second-pass SubPix, cap and same-family SQPnP/RefineLM, serialization; cached N0/SAM/detector excluded'));print(dict(status='SEALED',population=pop,n=len(ps),PnP=pn,SubPix_calls=sn))
def evaluate(pop):
 out=R/pop;assert sha(out/'PREDICTIONS.jsonl.gz')==read(out/'SEAL.json')['prediction_sha256'];ps=rows(out/'PREDICTIONS.jsonl.gz');syn=pop=='SYNTHETIC_VAL';gt=index(rows(B.parent/'pallet-statewise-end-to-end-recovery-v1/synthetic_v1/SYNTH_GT_LABELS.jsonl.gz'))if syn else None;refs=read(B.parent/'pallet-statewise-correspondence-diagnosis-20261016-v1/score_v2/FIXED_REFERENCES.json')['references']if not syn else None;labels=index(rows(TASK/'minimal_improvement_20261020_v5/H1_DEPLOYMENT_C2_TARGET_V1/DERIVED_C2_ACTION_TARGETS.jsonl.gz'))if syn else None;bank={r['id']:r for r in csv.DictReader((TASK/'mean_gap_audit_20261020_v1/results_attempt2/exact_bank_coverage.csv').open())}if not syn else None;groups={k:[]for k in ['N0','RAW_REPEAT','GAMMA_REPEAT']};pairs=[];coverage=[]
 for p in ps:
  g=gt[p['id']]if syn else None;ref=dict(R=g['R_fixed'],t=g['t'],order=2)if syn else refs[p['partition']][p['id']];ms={'N0':metric(p['N0'],ref)}
  r=dict(id=p['id'],recording=p['recording'])
  for name in ['RAW_REPEAT','GAMMA_REPEAT']:
   raw=p['proposals'][name];actual=raw if raw['available']else p['N0'];ms[name]=metric(actual,ref);r[name+'_proposal_available']=raw['available']
  for name,m in ms.items():groups[name].append(m);r.update({name+'_'+k:v for k,v in m.items()})
  pairs.append(r);orig_good=any(x['success_5cm5deg']for x in labels[p['id']]['action_metrics'])if syn else bank[p['id']]['HEAD_eligible_good']=='True';coverage.append(dict(id=p['id'],original_full_bank_good=orig_good,N0_good=ms['N0']['success'],N0_raw_union_good=ms['N0']['success']or ms['RAW_REPEAT']['success'],N0_gamma_union_good=ms['N0']['success']or ms['GAMMA_REPEAT']['success'],full_bank_gamma_union_good=orig_good or ms['GAMMA_REPEAT']['success'],full_bank_raw_union_good=orig_good or ms['RAW_REPEAT']['success']))
 st={k:stats(v)for k,v in groups.items()};why=[]
 for comparator in ['N0','RAW_REPEAT']:
  a,b=st['GAMMA_REPEAT'],st[comparator]
  if a['successes']<b['successes']:why.append(comparator+'_success_decreased')
  for k in ['T_mean','R_mean','T_P90','T_P95','R_P90','R_P95','T_max','R_max','no_pose','T_gt50','T_gt100','R_gt45']:
   if a[k]>b[k]+1e-7:why.append(comparator+'_'+k+'_increased')
  if not(a['T_mean']<b['T_mean']-1e-7 or a['R_mean']<b['R_mean']-1e-7):why.append(comparator+'_no_mean_improvement')
 result=dict(population=pop,methods=st,gamma_safe_gate_passed=not why,reasons=why,proposal_unavailable={k:sum(not r[k+'_proposal_available']for r in pairs)for k in ['RAW_REPEAT','GAMMA_REPEAT']},recovery_damage={k:dict(recover=sum(not x['success']and y['success']for x,y in zip(groups['N0'],groups[k])),damage=sum(x['success']and not y['success']for x,y in zip(groups['N0'],groups[k])))for k in ['RAW_REPEAT','GAMMA_REPEAT']},bank_coverage_diagnostic={k:sum(r[k]for r in coverage)for k in coverage[0]if k!='id'},oracle_not_method=True,actual_fixed_rule='always gamma proposal if available, otherwise original N0; no GT choice among union',bank_sizes='N0=1, plus raw repeat/gamma at most3 actual poses before strict dedup; original full bank extension diagnostic only',new_H1_fit=0,exposed_development=True)
 csvwrite(out/'PAIRED_ROWS.csv',pairs);csvwrite(out/'BANK_COVERAGE.csv',coverage);write(out/'EVALUATION.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['parity','generate','evaluate']);p.add_argument('--population',choices=['SYNTHETIC_VAL','CAL46','PRIMARY192'],default='SYNTHETIC_VAL');a=p.parse_args();parity()if a.stage=='parity'else(generate if a.stage=='generate'else evaluate)(a.population)
