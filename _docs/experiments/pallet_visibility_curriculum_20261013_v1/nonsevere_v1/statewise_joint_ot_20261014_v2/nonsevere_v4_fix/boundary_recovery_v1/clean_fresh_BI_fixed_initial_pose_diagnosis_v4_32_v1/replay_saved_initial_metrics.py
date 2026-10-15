from pathlib import Path
import json, hashlib, math
import numpy as np
B=Path(__file__).resolve().parent.parent
INPUT=B/'clean_fresh_BI_full_result_process_v4_32_r2_v1'
OUTPUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def metric(center,R,ref):
 t=100.*float(np.linalg.norm(np.asarray(center)-ref['t']))
 if ref['R'] is None:return dict(translation_cm=t,rotation_deg=None)
 G=np.asarray(ref['R']);angles=[]
 for k in range(ref['order']):
  a=k*2.*math.pi/ref['order'];c,s=math.cos(a),math.sin(a)
  target=G@np.array([[c,0.,s],[0.,1.,0.],[-s,0.,c]])
  angles.append(math.degrees(math.acos(float(np.clip((np.trace(target.T@R)-1.)/2.,-1.,1.)))))
 return dict(translation_cm=t,rotation_deg=min(angles))
def main():
 plan=json.loads((OUTPUT/'PLAN.json').read_text())
 for p,s in plan['input_sha256'].items():assert sha(Path(p))==s,p
 records=[];checks=0;diffs=[]
 for folder in sorted(INPUT.glob('case*')):
  d=json.loads((folder/'PROCESS_DATA.json').read_text());ref=d['fixed_reference'];p=d['own_independent_coarse']
  R=np.asarray(p['R']);t=np.asarray(p['t']);xyz=np.asarray(d['original_native_xyz']);Q=np.asarray(d['original_Q'])
  center=R@xyz.mean(axis=0)+t
  row=dict(id=d['id'],partition=d['partition'],initial_coarse=metric(center,R@Q,ref),initial_native_t=t.tolist(),initial_model_center=center.tolist(),states=d['states'],finals={})
  for arm,pred in d['final_predictions'].items():
   if not pred['pose_available']:row['finals'][arm]=dict(pose_available=False);continue
   pose=pred['actual_pose'];m=metric(pose['centroid'],np.asarray(pose['R_physical']),ref);saved=d['saved_metrics'][arm]
   for key in ['translation_cm','rotation_deg']:
    if m[key] is None:assert saved[key] is None
    else:delta=abs(m[key]-saved[key]);assert delta<1e-8,(d['id'],arm,key,delta);diffs.append(delta)
    checks+=1
   row['finals'][arm]=m
  row['active_lines']={a:len(p['active_groups'])for a,p in d['active_cost_plans'].items()}
  records.append(row)
 for p,s in plan['input_sha256'].items():assert sha(Path(p))==s,p
 report=dict(schema='SAVED_PROCESS_INITIAL_AND_FINAL_METRIC_REPLAY_V32',records=records,checks=checks,failures=0,maximum_saved_metric_difference=max(diffs),reference_or_prediction_source_reopened=False,new_PnP_NN_TLS_optimizer_or_query_calls=0,method_or_parameter_changes=0,diagnostic_only_no_pose_routing=True)
 (OUTPUT/'RESULT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 (OUTPUT/'SEAL.json').write_text(json.dumps(dict(result_sha256=sha(OUTPUT/'RESULT.json'),plan_sha256=sha(OUTPUT/'PLAN.json'),source_sha256=sha(Path(__file__))),indent=2)+'\n')
 print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
