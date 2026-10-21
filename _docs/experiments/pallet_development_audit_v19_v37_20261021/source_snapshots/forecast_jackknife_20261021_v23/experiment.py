"""Deterministic forecast-corner jackknife PnP. Original RGB forecasts are not visibility GT."""
import sys,pathlib,argparse,json,time,gzip
import numpy as np
import cv2
R=pathlib.Path(__file__).resolve().parent;T=R.parent;sys.path.insert(0,str(T/'bottleneck_repair_20261021_v14'));import track2_gamma_proposal as P
from analyze import read,rows,index,write,csvwrite,sha,metric
from kernel_probe import stats
def bound():
 p=read(R/'PROTOCOL.json')
 for k,v in p['bindings'].items():assert sha(v['path'])==v['sha256'],k
 return p
def save(p,z):
 with gzip.open(p,'xt')as h:
  for x in z:h.write(json.dumps(x,allow_nan=False)+'\n')
def generate(pop):
 cv2.setNumThreads(1);bound();assert pop=='SYNTHETIC_VAL';out=R/pop;out.mkdir(exist_ok=False);start=time.monotonic();fs=P.frames(pop);inc=index(rows(T/'geometric_influence_20261021_v19/SYNTHETIC_VAL_s20261018/PREDICTIONS.jsonl.gz'));fn=P.load_functions();ps=[];parity=[];calls=0;timings=[]
 def hook(e,a):
  if e=='open'and isinstance(a[0],str)and pathlib.Path(a[0]).name in ['FIXED_REFERENCES.json','SYNTH_GT_LABELS.jsonl.gz','ACTION_TARGETS.jsonl.gz','DERIVED_C2_ACTION_TARGETS.jsonl.gz']:raise PermissionError('Reference forbidden in jackknife generation')
 sys.addaudithook(hook)
 for u in fs:
  tick=time.monotonic();X=np.asarray(u['native_xyz'],float);K=np.asarray(u['K'],float);points=np.asarray(u['initial_points'],float)[:8];valid=np.isfinite(points).all(-1)&~np.all(points==-1,axis=1);Q=np.asarray(u['initial_pose']['R_cf']).T@u['initial_pose']['R_physical'];mean=X.mean(0)
  def solve(mask):
   nonlocal calls
   calls+=1
   if mask.sum()<6:return dict(available=False,reason='less_than6_forecast_corners')
   try:
    z=fn['solve'](X,points,K,mask)
    if z is None:return dict(available=False,reason='NO_SOLUTION')
    rot,t,res=z;depth=(X@rot.T+t)[:,2]
    if not np.isfinite(rot).all()or not np.isfinite(t).all()or depth.min()<=0:return dict(available=False,reason='NONFINITE_OR_NONPOSITIVE_DEPTH')
    return dict(available=True,R_physical=(rot@Q).tolist(),centroid=(rot@mean+t).tolist())
   except cv2.error:return dict(available=False,reason='OpenCV_error')
  p=solve(valid);n0=u['initial_pose'];assert p['available'];d=100*np.linalg.norm(np.asarray(p['centroid'])-n0['centroid']);tr=np.trace(np.asarray(p['R_physical']).T@n0['R_physical']);a=float(np.degrees(np.arccos(np.clip((tr-1)/2,-1,1))));parity.append(dict(id=u['id'],T_cm=float(d),R_deg=a,pass_tolerance=d<=.001 and a<=.001));assert parity[-1]['pass_tolerance'],'ALL_FRAME_N0_REPLAY_PARITY_FAILED'
  candidates=[dict(candidate_id='INCUMBENT_V5',pose=inc[u['id']]['methods']['BASELINE']['pose']),dict(candidate_id='N0',pose=n0)]
  for k in np.flatnonzero(valid):
   take=valid.copy();take[k]=False;new=solve(take)
   if new['available']:candidates.append(dict(candidate_id='LEAVE_FORECAST_'+str(int(k)),pose=new))
  for c in candidates:
   pose=c['pose'];rot=np.asarray(pose['R_physical'])@Q.T;t=np.asarray(pose['centroid'])-rot@mean;camera=X@rot.T+t;proj=camera@K.T;pixel=proj[:,:2]/proj[:,2:];er=np.linalg.norm(pixel[valid]-points[valid],axis=1);c['trimmed6_MSE_px2']=float(np.mean(np.sort(er)[:6]**2));c['all_forecast_RMS_px']=float(np.sqrt(np.mean(er**2)))
  selected=min(range(len(candidates)),key=lambda j:candidates[j]['trimmed6_MSE_px2']);ps.append(dict(id=u['id'],partition=u['partition'],recording=u['cluster_id'],candidates=candidates,selected=selected,reference_inputs=False,forecast_points_not_material_visibility_GT=True));timings.append(dict(id=u['id'],actual_candidates=len(candidates),seconds=time.monotonic()-tick))
 save(out/'PREDICTIONS.jsonl.gz',ps);csvwrite(out/'TECHNICAL_N0_PARITY235.csv',parity);csvwrite(out/'TIMING.csv',timings);write(out/'SEAL.json',dict(status='SEALED_BEFORE_SCORING',rows=len(ps),prediction_sha256=sha(out/'PREDICTIONS.jsonl.gz'),actual_SQPnP_LM_pairs=calls,parity_replay_pairs=len(parity),max_parity_T_cm=max(p['T_cm']for p in parity),max_parity_R_deg=max(p['R_deg']for p in parity),seconds=time.monotonic()-start,new_RGB=0,new_SAM=0,new_training=0));print(dict(stage='GENERATED',n=len(ps),PnP_pairs=calls,seconds=time.monotonic()-start))
def evaluate(pop):
 from geometric_influence_20261021_v19.experiment import gate
 out=R/pop;assert sha(out/'PREDICTIONS.jsonl.gz')==read(out/'SEAL.json')['prediction_sha256'];ps=rows(out/'PREDICTIONS.jsonl.gz');gt=index(rows(P.B.parent/'pallet-statewise-end-to-end-recovery-v1/synthetic_v1/SYNTH_GT_LABELS.jsonl.gz'));original=index(rows(T/'minimal_improvement_20261020_v5/H1_DEPLOYMENT_C2_TARGET_V1/DERIVED_C2_ACTION_TARGETS.jsonl.gz'));groups={k:[]for k in ['BASELINE','N0','JACKKNIFE']};pairs=[];coverage=[]
 for p in ps:
  g=gt[p['id']];ref=dict(R=g['R_fixed'],t=g['t'],order=2);ms=[metric(c['pose'],ref)for c in p['candidates']];ix={'BASELINE':0,'N0':1,'JACKKNIFE':p['selected']};r=dict(id=p['id'],recording=p['recording'],candidate_n=len(ms),selected_id=p['candidates'][p['selected']]['candidate_id'])
  for name,i in ix.items():groups[name].append(ms[i]);r.update({name+'_'+k:v for k,v in ms[i].items()})
  pairs.append(r);oldgood=any(x['success_5cm5deg']for x in original[p['id']]['action_metrics']);coverage.append(dict(id=p['id'],full_bank_good=oldgood,new_proposal_good=any(m['success']for m in ms[2:]),full_plus_proposal_good=oldgood or any(m['success']for m in ms[2:]),proposal_bank_good=any(m['success']for m in ms),actual_selected_good=ms[p['selected']]['success']))
 st={k:stats(v)for k,v in groups.items()};why={k:gate(st['JACKKNIFE'],st[k])for k in ['BASELINE','N0']};rd={k:dict(recover=sum(not a['success']and b['success']for a,b in zip(groups[k],groups['JACKKNIFE'])),damage=sum(a['success']and not b['success']for a,b in zip(groups[k],groups['JACKKNIFE'])))for k in ['BASELINE','N0']};ev=dict(population=pop,methods=st,gate_passed=not any(why.values()),gate_reasons=why,recover_damage=rd,coverage_diagnostic={k:sum(r[k]for r in coverage)for k in coverage[0]if k!='id'},oracle_not_method=True,independent_test_n=0)
 csvwrite(out/'PAIRED_ROWS.csv',pairs);csvwrite(out/'COVERAGE.csv',coverage);write(out/'EVALUATION.json',ev);print(json.dumps(ev,indent=2))
if __name__=='__main__':
 sys.path.insert(0,str(T));p=argparse.ArgumentParser();p.add_argument('stage',choices=['generate','evaluate']);a=p.parse_args();(generate if a.stage=='generate'else evaluate)('SYNTHETIC_VAL')
