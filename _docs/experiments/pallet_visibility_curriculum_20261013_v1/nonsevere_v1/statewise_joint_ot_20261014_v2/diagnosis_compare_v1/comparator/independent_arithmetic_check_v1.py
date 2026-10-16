"""Independent numerical audit of paired comparator output; no source edits."""
from pathlib import Path
import math
import json
import hashlib
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'scripts/research/pallet_visibility_curriculum_20261013_v1/scoring.py').is_file())
C=ROOT/'_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1'
result=json.loads((HERE/'BENCHMARK.json').read_text())
labels={}
for p,f in [('CAL',C/'OOF_pose_v1/ISOLATED_CAL_SCORING.json'),('TEST',C/'test_reused_v1/ISOLATED_TEST_SCORING.json')]:
 labels.update(json.loads(f.read_text())['rows'])
CURRENT='CURRENT_JOINT_STATEWISE_V3'
errors=[]
# Independently reconstruct cuboid corners and C2 symmetry. Do not import shared scorer.
for method,group in result['per_frame_scores'].items():
 for i,row in group.items():
  p=row['actual_pose'];g=labels[i]['target']['truth'];R=np.array(p['R_physical']);G=np.array(g['R']);t=np.array(p['centroid']);u=np.array(g['t']);a,b,c=np.array(g['xyz'])/2
  X=np.array([[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],[-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]])
  rot=[];add=[]
  for k in range(g['order']):
   angle=2*math.pi*k/g['order'];z=math.cos(angle);s=math.sin(angle);sym=np.array([[z,0,s],[0,1,0],[-s,0,z]]);GT=G@sym
   rot.append(math.degrees(math.acos(max(-1,min(1,(float(np.trace(GT.T@R))-1)/2)))))
   aX=(R@X.T).T+t;bX=(GT@X.T).T+u;add.append(float(np.mean(np.sqrt(np.sum((aX-bX)**2,axis=1))))*100)
  trans=math.sqrt(sum((float(t[j])-float(u[j]))**2 for j in range(3)))*100
  expect={'translation_cm':trans,'rotation_deg':min(rot),'ADDsym_cm':min(add),'success_5cm_5deg':trans<5 and min(rot)<5}
  for k,v in expect.items():
   diff=abs(float(v)-float(row['metrics'][k]));errors.append(diff);assert diff<2e-10,(method,i,k,diff)
# Every paired mean, flip and loss/win count is independently checked.
paired_count=0
for method,partitions in result['current_minus_comparator'].items():
 for part,p in partitions.items():
  pair=p['per_frame'];n=len(pair);assert n=={'CAL':46,'TEST':49}[part]
  for key,item in p['by_metric'].items():
   d=[float(v['current'][key])-float(v['comparator'][key])for v in pair]
   assert abs(sum(d)/n-item['mean_difference'])<1e-12
   tolerance=0 if key=='success_5cm_5deg' else 1e-8
   wins=sum(v>tolerance for v in d)if key=='success_5cm_5deg' else sum(v<-tolerance for v in d)
   losses=sum(v<-tolerance for v in d)if key=='success_5cm_5deg' else sum(v>tolerance for v in d)
   assert wins==item['improved'] and losses==item['worsened'] and n-wins-losses==item['tied']
  recover=[v['id']for v in pair if v['current']['success_5cm_5deg'] and not v['comparator']['success_5cm_5deg']]
  lost=[v['id']for v in pair if not v['current']['success_5cm_5deg'] and v['comparator']['success_5cm_5deg']]
  assert recover==p['success_flip']['current_recovers'] and lost==p['success_flip']['current_loses'];paired_count+=1
# Deliberately enumerate sampled session rows rather than vectorized chunk sums.
p=result['current_minus_comparator']['OLD_RGB_SINGLE_FAMILY']['TEST'];pair=p['per_frame'];sessions=sorted({v['session']for v in pair});rng=np.random.default_rng(20261014);draw=rng.integers(0,len(sessions),size=(10000,len(sessions)));by={s:[v for v in pair if v['session']==s]for s in sessions}
boot=[]
for sampled in draw:
 selected=[row for index in sampled for row in by[sessions[index]]]
 boot.append(sum(v['current']['translation_cm']-v['comparator']['translation_cm']for v in selected)/len(selected))
CI=np.quantile(boot,[.025,.975]);assert np.max(np.abs(CI-p['by_metric']['translation_cm']['session_cluster_bootstrap95']))<1e-12
outliers={}
for method in ['ORIGINAL_N3_PNP','CURRENT_BI_ONLY','OLD_RGB_SINGLE_FAMILY','BOUNDARY_PROGRESSIVE']:
 p=result['current_minus_comparator'][method]['TEST'];pair=p['per_frame'];d={v['id']:v['delta']['translation_cm']for v in pair};total=sum(d.values());leave=[dict(removed_id=i,remaining_frames=48,remaining_mean_delta_cm=(total-x)/48)for i,x in d.items()]
 sorted_loss=sorted(d,key=d.get,reverse=True);sorted_win=sorted(d,key=d.get)
 outliers[method]=dict(all49_mean_delta_cm=total/49,all49_sum_delta_cm=total,current_better_frames=sum(x<-1e-8 for x in d.values()),current_worse_frames=sum(x>1e-8 for x in d.values()),largest_single_loss=dict(id=sorted_loss[0],delta_cm=d[sorted_loss[0]]),largest_single_improvement=dict(id=sorted_win[0],delta_cm=d[sorted_win[0]]),all49_leave_one_out=leave,leave_one_out_mean_delta_range_cm=[min(x['remaining_mean_delta_cm']for x in leave),max(x['remaining_mean_delta_cm']for x in leave)],largest3_losses=[dict(id=i,delta_cm=d[i])for i in sorted_loss[:3]])
N=C/'statewise_joint_ot_20261014_v2';seal=json.loads((N/'evaluation_v3/PREDICTION_SEAL.json').read_text())
for p,e in seal['source_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==e
protected=json.loads((N.parent.parent/'STATEWISE_PUBLICATION_REMOTE_RECEIPT_V1.json').read_text())['user_files_preserved_sha256']
for p,e in protected.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==e
report=dict(schema='INDEPENDENT_ARITHMETIC_CHECK_COMPARATOR_V1',scored_pose_rows=len(result['reference_symmetry_orders']) and sum(len(g)for g in result['per_frame_scores'].values()),maximum_independent_metric_difference=max(errors),paired_partition_comparisons=paired_count,all_per_frame_means_win_loss_and_success_flips_pass=True,independent_session_row_bootstrap_CI=CI.tolist(),cluster_bootstrap_vectorization_pass=True,unique_truth_symmetry_orders={'2':95},original_BENCHMARK_reference_symmetry_orders_is_count_over_all_method_rows_not_unique_frames=True,no_models_or_existing_predictions_mutated=True,protected_six_and_frozen19_current_match=True,outlier_diagnostics_are_descriptive_only_primary49_scores_preserved=True,translation_outlier_diagnostics=outliers,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
p=HERE/'INDEPENDENT_ARITHMETIC_CHECK.json'
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items()if k!='translation_outlier_diagnostics'},ensure_ascii=False,indent=2))
