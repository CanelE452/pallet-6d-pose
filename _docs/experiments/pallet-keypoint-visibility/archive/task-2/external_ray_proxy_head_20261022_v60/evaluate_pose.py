"""Frozen proxy policy; predictions sealed before reference joins. No fit."""
from common import *
import sys,time,argparse,math
import numpy as np
sys.path.insert(0,str(SRC));sys.path.insert(0,str(V58))
from masked_backend import infer_masked

def angle(A,B):
 return float(np.degrees(np.arccos(np.clip((np.trace(np.array(A).T@np.array(B))-1)/2,-1,1))))
def rotations(n):
 return [np.array([[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]])for a in np.arange(n)*2*np.pi/n]
def metric(p,g):
 if not p.get('available'):return dict(available=False,success=False)
 T=float(np.linalg.norm(np.array(p['centroid'])-g['t'])*100);R=min(angle(np.array(g['R'])@Q,p['R_physical'])for Q in rotations(g['order']));C=angle(g['R'],p['R_physical'])
 return dict(available=True,T_cm=T,R_deg=R,R_C1_deg=C,success=T<5 and R<5,T_gt50=T>50,R_gt45=R>45,canonical_near90=80<=C<=100)
def summarize(rr):
 out=dict(n=len(rr),available=sum(r['available']for r in rr),success=sum(r['success']for r in rr),fallback=sum(r.get('fallback',False)for r in rr),T_gt50=sum(r.get('T_gt50',False)for r in rr),R_gt45=sum(r.get('R_gt45',False)for r in rr),canonical_near90=sum(r.get('canonical_near90',False)for r in rr))
 for k in ['T_cm','R_deg','R_C1_deg']:
  v=[r[k]for r in rr if r['available']]
  for name,fn in [('mean',np.mean),('median',np.median),('P90',lambda x:np.percentile(x,90)),('P95',lambda x:np.percentile(x,95)),('max',np.max)]:out[k+'_'+name]=float(fn(v))if v else None
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--scope',choices=['SYNTH','REAL'],required=True);a=ap.parse_args();scope=a.scope;start=time.monotonic();inputs=read(ROOT/('PREPARED_INPUTS_REAL_VERIFIED_V2.json.gz'if scope=='REAL'else'PREPARED_INPUTS_SYNTH.json.gz'));z=np.load(ROOT/f'{scope}_HEAD_OUTPUTS.npz');assert z['probability'].shape==(len(inputs),8);assert sha(ROOT/'BEST_HEAD.pt')==read(ROOT/'MODEL_FREEZE.json')['checkpoint_sha256'];pred=[]
 if scope=='REAL':assert read(ROOT/'REAL_VERIFIED_PARITY_V2.json')['PASS']
 for i,u in enumerate(inputs):
  valid=np.ones(9,bool);valid[:8]&=~(z['pixel_valid'][i]&(z['probability'][i]>=.5));p=infer_masked(u['initial_points'],u['K'],u['known_physical_dimensions_xyz_m'],valid,source=scope=='SYNTH');fallback=not p['available'];final=u['initial_pose']if fallback else p
  pred.append(dict(id=u['id'],group=u['split']if scope=='SYNTH'else u['group'],original_pose=u['initial_pose'],final_pose=final,raw_masked_pose=p,fallback=fallback,valid9=valid.tolist(),excluded=int(8-valid[:8].sum()),probability=z['probability'][i].tolist(),pixel_valid=z['pixel_valid'][i].tolist(),coordinates_sha256=hashlib.sha256(np.asarray(u['initial_points'],np.float64).tobytes()).hexdigest()))
 save(f'{scope}_POSE_PREDICTIONS_SEALED.json.gz',pred);save(f'{scope}_POSE_SEAL.json',dict(UTC=utc(),frames=len(pred),head_sha256=sha(ROOT/'BEST_HEAD.pt'),predictions_sha256=sha(ROOT/f'{scope}_POSE_PREDICTIONS_SEALED.json.gz'),reference_reads_before_seal=0,seconds=time.monotonic()-start))
 # Only now open reference sidecars.
 if scope=='SYNTH':gt={g['id']:dict(R=g['R_fixed'],t=g['t'],order=g['symmetry_order'])for g in rows(B/'pallet-statewise-end-to-end-recovery-v1/synthetic_v1/SYNTH_GT_LABELS.jsonl.gz')}
 else:
  wood=read(B/'pallet-statewise-correspondence-diagnosis-20261016-v1/score_v2/FIXED_REFERENCES.json')['references'];sq={g['id']:g['reference']for g in read(ROOT.parent/'ocpr_wood_square_transfer_20261022_v50/EVALUATION_FIXED/PORTABLE_POSE_AUDIT_BUNDLE.json.gz')['frames']if g['group']=='SQUARE_YOLO'};gt={}
  for u in inputs:
   g=wood[u['partition']][u['id']]if u['group'].startswith('WOOD')else sq[u['id'].split(':')[-1]];gt[u['id']]=dict(R=g['R'],t=g['t'],order=u['symmetry_order'])
 rr=[];paired=[]
 for u,p in zip(inputs,pred):
  mm={}
  for arm,pose in [('N0',p['original_pose']),('EXTERNAL_RAY_HEAD',p['final_pose'])]:
   m=metric(pose,gt[p['id']]);rr.append(dict(id=p['id'],group=p['group'],arm=arm,fallback=p['fallback']if arm!='N0'else False,excluded=p['excluded'],**m));mm[arm]=m
  b,c=mm['N0'],mm['EXTERNAL_RAY_HEAD'];paired.append(dict(id=p['id'],group=p['group'],recovery=not b['success']and c['success'],damage=b['success']and not c['success'],N0_success=b['success'],HEAD_success=c['success'],excluded=p['excluded'],fallback=p['fallback'],T_change=c.get('T_cm',0)-b.get('T_cm',0),R_change=c.get('R_deg',0)-b.get('R_deg',0)))
 summary=[];gates=[]
 for group in sorted({r['group']for r in rr}):
  ss={}
  pp=[r for r in paired if r['group']==group];recover=sum(r['recovery']for r in pp);damage=sum(r['damage']for r in pp)
  for arm in ['N0','EXTERNAL_RAY_HEAD']:
   s=summarize([r for r in rr if r['group']==group and r['arm']==arm]);ss[arm]=s;summary.append(dict(group=group,arm=arm,**s))
  b,c=ss['N0'],ss['EXTERNAL_RAY_HEAD'];tests=dict(net_recovery_positive=recover>damage,no_Tgt50_increase=c['T_gt50']<=b['T_gt50'],no_meanT_worse=c['T_cm_mean']<=b['T_cm_mean'],no_meanR_worse=c['R_deg_mean']<=b['R_deg_mean'],no_P95T_worse=c['T_cm_P95']<=b['T_cm_P95'],no_P95R_worse=c['R_deg_P95']<=b['R_deg_P95']);gates.append(dict(group=group,recovery=recover,damage=damage,PASS=all(tests.values()),tests=tests))
 csvsave(f'{scope}_POSE_PER_FRAME.csv',rr);csvsave(f'{scope}_POSE_PAIRED.csv',paired);csvsave(f'{scope}_POSE_SUMMARY.csv',summary);save(f'{scope}_POSE_GATE.json',gates);save(f'{scope}_POSE_COMPLETE.json',dict(UTC=utc(),seconds=time.monotonic()-start,frames=len(inputs),new_fit=0,reference_only_after_prediction_seal=True));print(scope,'POSE_COMPLETE',summary,gates,flush=True)
if __name__=='__main__':main()
