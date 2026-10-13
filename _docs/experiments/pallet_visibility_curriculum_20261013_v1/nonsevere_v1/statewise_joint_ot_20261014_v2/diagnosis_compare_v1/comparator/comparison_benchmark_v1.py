"""Re-score sealed comparator poses on the exact current CAL46/TEST49.

No model forward, pose fit, threshold selection, reference remapping, or existing
artifact mutation. Physical rotations and the current fixed symmetry metric are
used for every arm. All labels are opened only after all source prediction seals
and the complete input identity audit have passed.
"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import gzip
import hashlib
import json
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'scripts/research/pallet_visibility_curriculum_20261013_v1/scoring.py').is_file())
sys.path.insert(0,str(ROOT))
from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C
from scripts.research.pallet_visibility_curriculum_20261013_v1.scoring import pose_metric, distribution, rotation_y

COHORT=C.DOC/'nonsevere_v1'
STUDY=COHORT/'statewise_joint_ot_20261014_v2'
HERE=Path(__file__).resolve().parent
METRICS=('translation_cm','rotation_deg','ADDsym_cm','success_5cm_5deg')
REPEATS=10000
SEED=20261014
CURRENT='CURRENT_JOINT_STATEWISE_V3'
LABELS={'ORIGINAL_N3_PNP':'원 N3+SubPix PnP','CURRENT_BI_ONLY':'현재 보정 B/I만 PnP',CURRENT:'현재 모든 O/T 공동 v3',
 'OLD_RGB_SINGLE_FAMILY':'기존 RGB 신뢰도·보정, family 유지','BOUNDARY_PROGRESSIVE':'기존 경계 융합·진행 보정',
 'COUNT_RAW':'N3 count rule, raw','COUNT_FUSED':'N3 count rule, fused','OLD_FINITE_BASE':'이전 보호 completion BASE',
 'OLD_FINITE_VISIBLE':'이전 유한 visible 후보','OLD_FINITE_REGION':'이전 유한 region 후보',
 'YOLO_NATIVE_BASE':'원 YOLO+F','POSEFIX_RAW':'별도 PoseFix RAW+F','POSEFIX_SUB_PF_CAP':'PoseFix→SubPix, PF cap',
 'POSEFIX_SUB_YOLO_CAP':'PoseFix→SubPix, YOLO cap','OLD_RGB_WD_BRANCH':'기존 RGB 보정+W/D branch',
 'OLD_RGB_TOGETHER':'기존 RGB visible together','OLD_SEGMENTED_RGB_CURRICULUM':'기존 segmented RGB curriculum',
 'POSEFIX_COUNT_RAW':'PoseFix-SubPix count raw','POSEFIX_COUNT_FUSED':'PoseFix-SubPix count fused'}

def read(p):return json.loads(Path(p).read_text())
def sha(p):return C.sha(p)
def relative(p):return str(Path(p).resolve().relative_to(ROOT))
def binding(p):return dict(path=relative(p),sha256=sha(p),bytes=Path(p).stat().st_size)
def write_new(p,o):
 if p.exists():raise FileExistsError('Preserve existing comparison output: '+str(p))
 C.write(p,o)
def rows(p):return list(C.rows(p))
def maxdiff(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float)
 return float(np.max(np.abs(a-b))) if a.shape==b.shape else None

def verify_run(relative_dir):
 d=COHORT/relative_dir;p=d/'PREDICTIONS.jsonl.gz';s=d/'PREDICTION_SEAL.json'
 seal=read(s);expected=seal.get('predictions_sha256',seal.get('sha256'))
 assert seal.get('before_scoring') and expected==sha(p),('Prediction seal mismatch',d)
 protocol=d/'PROTOCOL.json'
 if not protocol.is_file():protocol=d.parent/'PROTOCOL.json'
 if seal.get('protocol_sha256'):assert sha(protocol)==seal['protocol_sha256'],('Protocol seal mismatch',d)
 data=rows(p)
 assert len(data)==seal['rows'],('Sealed row count mismatch',d)
 return data,dict(predictions=binding(p),seal=binding(s),protocol=binding(protocol),rows=len(data),before_original_scoring=True)

def select(data,method,part=None):
 group=[r for r in data if (r.get('method') or r.get('arm'))==method and (part is None or r.get('partition')==part)]
 assert len(group)==len({r['id']for r in group}),('Duplicate selected method IDs',method,part)
 return {r['id']:r for r in group}

def pose_audit(pose,coarse):
 R=np.asarray(pose['R_cf'],float);P=np.asarray(pose['R_physical'],float);t=np.asarray(pose['centroid'],float)
 assert R.shape==P.shape==(3,3) and t.shape==(3,) and np.isfinite([*R.flat,*P.flat,*t]).all()
 assert abs(np.linalg.det(R)-1)<1e-7 and abs(np.linalg.det(P)-1)<1e-7
 assert np.linalg.norm(R.T@R-np.eye(3))<1e-7 and np.linalg.norm(P.T@P-np.eye(3))<1e-7
 Q=R.T@P;Q0=np.asarray(coarse['R_cf']).T@np.asarray(coarse['R_physical'])
 X=C.cuboid(*pose['cf_extents'])@Q;X0=C.cuboid(*coarse['cf_extents'])@Q0
 cloud_gap=float(np.max(np.min(np.linalg.norm(X[:,None]-X0[None,:],axis=2),axis=1)))
 return dict(proper_rotation=True,finite_translation=True,original_native_Q_max_abs=maxdiff(Q,Q0),
  original_native_extents_equal=pose['cf_extents']==coarse['cf_extents'],registered_physical_corner_cloud_gap_m=cloud_gap,
  ordered_native_model_gap_m=float(np.linalg.norm(X-X0,axis=1).max()))

def paired(group,current,repeat=REPEATS):
 ids=list(group);sessions=sorted({current[i]['session']for i in ids});by={s:[i for i in ids if current[i]['session']==s]for s in sessions}
 rng=np.random.default_rng(SEED);draw=rng.integers(0,len(sessions),size=(repeat,len(sessions)))
 result={}
 for metric in METRICS:
  chunks=[np.array([float(current[i]['metrics'][metric])-float(group[i]['metrics'][metric])for i in by[s]])for s in sessions]
  sums=np.array([v.sum()for v in chunks]);counts=np.array([len(v)for v in chunks]);boot=sums[draw].sum(1)/counts[draw].sum(1);x=np.concatenate(chunks)
  tol=1e-8 if metric!='success_5cm_5deg' else 0
  improve=[i for i in ids if (float(current[i]['metrics'][metric])-float(group[i]['metrics'][metric])>tol if metric=='success_5cm_5deg' else float(current[i]['metrics'][metric])-float(group[i]['metrics'][metric]) < -tol)]
  worsen=[i for i in ids if (float(current[i]['metrics'][metric])-float(group[i]['metrics'][metric]) < -tol if metric=='success_5cm_5deg' else float(current[i]['metrics'][metric])-float(group[i]['metrics'][metric]) >tol)]
  result[metric]=dict(delta_definition='current_minus_comparator',mean_difference=float(x.mean()),median_difference=float(np.median(x)),session_cluster_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),improved=len(improve),worsened=len(worsen),tied=len(ids)-len(improve)-len(worsen),improved_ids=improve,worsened_ids=worsen)
 perframe=[]
 for i in ids:
  delta={k:float(current[i]['metrics'][k])-float(group[i]['metrics'][k])for k in METRICS}
  perframe.append(dict(id=i,session=current[i]['session'],current=current[i]['metrics'],comparator=group[i]['metrics'],delta=delta,current_pose_source=current[i].get('pose_source'),current_accepted=current[i].get('accepted'),current_status=current[i].get('status'),current_states=current[i].get('states')))
 recover=[v['id']for v in perframe if v['current']['success_5cm_5deg'] and not v['comparator']['success_5cm_5deg']]
 lost=[v['id']for v in perframe if not v['current']['success_5cm_5deg'] and v['comparator']['success_5cm_5deg']]
 session=[]
 for s in sessions:
  sr=[v for v in perframe if v['session']==s]
  session.append(dict(session=s,n=len(sr),mean_difference={k:float(np.mean([v['delta'][k]for v in sr]))for k in METRICS},summed_translation_delta_cm=sum(v['delta']['translation_cm']for v in sr),current_success=sum(v['current']['success_5cm_5deg']for v in sr),comparator_success=sum(v['comparator']['success_5cm_5deg']for v in sr)))
 bothwin=[v['id']for v in perframe if v['delta']['translation_cm'] < -1e-8 and v['delta']['rotation_deg'] < -1e-8]
 bothloss=[v['id']for v in perframe if v['delta']['translation_cm'] >1e-8 and v['delta']['rotation_deg'] >1e-8]
 return dict(frames=len(ids),sessions=len(sessions),cluster_unit='session, sampled uniformly with replacement; pooled sampled frame mean',bootstrap_repeats=repeat,bootstrap_seed=SEED,exploratory_only=True,not_independent_confirmation=True,by_metric=result,success_flip=dict(current_recovers=recover,current_loses=lost,both_success=sum(v['current']['success_5cm_5deg']and v['comparator']['success_5cm_5deg']for v in perframe),both_failure=sum(not v['current']['success_5cm_5deg']and not v['comparator']['success_5cm_5deg']for v in perframe)),both_translation_and_rotation_improved=bothwin,both_translation_and_rotation_worsened=bothloss,session_contributions=session,largest_translation_losses=sorted(perframe,key=lambda v:v['delta']['translation_cm'],reverse=True)[:10],largest_translation_wins=sorted(perframe,key=lambda v:v['delta']['translation_cm'])[:10],largest_rotation_losses=sorted(perframe,key=lambda v:v['delta']['rotation_deg'],reverse=True)[:10],per_frame=perframe)

def summary(group):
 return dict(frames=len(group),sessions=len({r['session']for r in group.values()}),metrics={k:distribution([float(r['metrics'][k])for r in group.values()])for k in METRICS},success_count=sum(r['metrics']['success_5cm_5deg']for r in group.values()),all_frames_have_pose=True,success_thresholds_strict=dict(translation_cm=5,rotation_deg=5))

def main(output):
 output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
 if (output/'BENCHMARK.json').exists():raise FileExistsError('Preserve completed comparison')
 receipt_path=C.DOC/'STATEWISE_PUBLICATION_REMOTE_RECEIPT_V1.json';protected=read(receipt_path)['user_files_preserved_sha256']
 assert len(protected)==6 and all(sha(ROOT/p)==e for p,e in protected.items())
 canonical_path=COHORT/'PREDICTION_INPUTS.json';canonical={r['id']:r for r in read(canonical_path)['frames']}
 ids={p:[r['id']for r in canonical.values()if r['partition']==p]for p in ('CAL','TEST')}
 assert {p:len(x)for p,x in ids.items()}=={'CAL':46,'TEST':49}
 cohort=read(COHORT/'COHORT_PROTOCOL.json');assert not(set(canonical)&set(cohort['excluded_severe_ids']))
 sources={};run={}
 dirs=['statewise_joint_ot_20261014_v2/evaluation_v3','posefix_sub_comparison_v1','boundary_progressive_replay_v1','boundary_count_rules_v1','boundary_rule_replay_v1','test_reused_v1','region_completion_v1/CAL_N3','region_completion_v1/TEST_N3']
 for d in dirs:run[d],sources[d]=verify_run(d)
 current_seal=read(STUDY/'evaluation_v3/PREDICTION_SEAL.json')
 assert all(sha(ROOT/p)==e for p,e in current_seal['source_sha256'].items())
 full_current={r['id']:r for r in run[dirs[0]] if r['cohort']=='PRIMARY_NONSEVERE'};assert set(full_current)==set(canonical)
 arm_rows={};audit={};descriptors={}
 def add(name,records,level,same_input,notes):
  arm_rows[name]=records;descriptors[name]=dict(label=LABELS[name],comparison_level=level,same_upstream_N3_SUBPIX=same_input,notes=notes,available_partitions=sorted({canonical[i]['partition']for i in records}))
 def converted(pose,r):return dict(id=r['id'],session=r.get('session',r['id'].split(':')[0]),actual_pose=pose,source_row=r)
 for name,path in [('ORIGINAL_N3_PNP','initial_pose'),('CURRENT_BI_ONLY','BI_only'),(CURRENT,'runtime')]:
  data={}
  for i,r in full_current.items():
   p=r[path] if path=='initial_pose' else r[path]['actual_pose'];value=converted(p,r)
   if path=='runtime':value.update(pose_source=p['source'],accepted=r['runtime']['accepted'],status=r['runtime']['joint_pose']['status'],states=r['runtime']['states'],material_ambiguity=r['runtime']['ambiguity_audit']['material_ambiguity'])
   data[i]=value
  add(name,data,'current_same_frozen_observations' if name!='ORIGINAL_N3_PNP' else 'same_upstream_N3',True,'Exact current v3 sealed result including declared fallback; crops excluded.')
 combo=run['posefix_sub_comparison_v1'];combo_sel={m:select(combo,m)for m in ['NATIVE_BASE','INITIAL_N3_SUBPIX','POSEFIX_RAW','POSEFIX_SUBPIX_PF_ANCHOR','POSEFIX_SUBPIX_YOLO_ANCHOR','FUSED_ACTIVE','JOINT_SINGLE_FAMILY']}
 baseline_input_path=COHORT/'posefix_sub_comparison_v1/BASELINE_INPUTS.json';baseline={r['frame']['id']:r['frame']for r in read(baseline_input_path)['frames']};assert set(baseline)==set(canonical)
 baseline_audit=[]
 for i,f in canonical.items():
  b=baseline[i];r=full_current[i]
  assert b['K']==f['K']==r['K'] and b['raw_hw']==f['raw_hw']==r['raw_hw'] and b['image']==f['image']==r['image']
  assert maxdiff(b['points']['N3_SUBPIX'],f['points']['N3_SUBPIX'])==0
  assert maxdiff(r['runtime']['original_raw_points'],f['points']['N3_SUBPIX'])==0
  initial=combo_sel['INITIAL_N3_SUBPIX'][i]['actual_pose'];orig=f['initial_poses']['N3_SUBPIX']
  assert initial==orig==r['initial_pose']
  baseline_audit.append(dict(id=i,image_K_canvas_N3_points_and_initial_pose_exact=True))
 for method,name,same in [('JOINT_SINGLE_FAMILY','OLD_RGB_SINGLE_FAMILY',True),('FUSED_ACTIVE','BOUNDARY_PROGRESSIVE',True),('NATIVE_BASE','YOLO_NATIVE_BASE',False),('POSEFIX_RAW','POSEFIX_RAW',False),('POSEFIX_SUBPIX_PF_ANCHOR','POSEFIX_SUB_PF_CAP',False),('POSEFIX_SUBPIX_YOLO_ANCHOR','POSEFIX_SUB_YOLO_CAP',False)]:
  add(name,{i:converted(r['actual_pose'],r)for i,r in combo_sel[method].items()},'same_upstream_N3' if same else 'matched_full_pipeline_different_upstream',same,'CAL old joint is full-CAL fitted in-sample; TEST already reused. PoseFix is separate RGB ResNet152 head; own-q F can change family.'if name=='OLD_RGB_SINGLE_FAMILY' or not same else 'Same N3 upstream/family; different image observer/solver rules.')
 controls={}
 controls['BOUNDARY_PROGRESSIVE_TEST']=select(run['boundary_progressive_replay_v1'],'FUSED_ACTIVE')
 controls['OLD_RGB_TEST']=select(run['test_reused_v1'],'JOINT_SINGLE_FAMILY')
 for control,name in [('BOUNDARY_PROGRESSIVE_TEST','BOUNDARY_PROGRESSIVE'),('OLD_RGB_TEST','OLD_RGB_SINGLE_FAMILY')]:
  for i,r in controls[control].items():
   assert r['actual_pose']==arm_rows[name][i]['actual_pose']
   q=r['original_points'];assert maxdiff(np.asarray(q)[:8],np.asarray(canonical[i]['points']['N3_SUBPIX'])[:8])==0
 # CAL progressive copied control also has raw q; cross-check entire95 copied poses.
 controls['BOUNDARY_PROGRESSIVE_CAL']=select(run['boundary_count_rules_v1'],'FUSED_SIGMA2_PROGRESSIVE_ACTIVE_SUPPORT')
 for i,r in controls['BOUNDARY_PROGRESSIVE_CAL'].items():
  assert r['actual_pose']==arm_rows['BOUNDARY_PROGRESSIVE'][i]['actual_pose']
  assert maxdiff(np.asarray(r['raw_points'])[:8],np.asarray(canonical[i]['points']['N3_SUBPIX'])[:8])==0
 for name,calmethod,testmethod in [('COUNT_RAW','RAW_BOUNDARY_COUNT_RULE','N3_SUB_COUNT_RAW'),('COUNT_FUSED','FUSED_SIGMA2_BOUNDARY_COUNT_RULE','N3_SUB_COUNT_FUSED')]:
  selected={**select(run['boundary_count_rules_v1'],calmethod),**select(run['boundary_rule_replay_v1'],testmethod)}
  for i,r in selected.items():assert maxdiff(np.asarray(r.get('original_points',r.get('raw_points')))[:8],np.asarray(canonical[i]['points']['N3_SUBPIX'])[:8])==0
  add(name,{i:converted(r['actual_pose'],r)for i,r in selected.items()},'same_upstream_N3',True,'Sealed CAL and TEST count arms paired by RAW/FUSED policy; historical helper amendments are preserved.')
 for name,method in [('OLD_FINITE_BASE','BASE'),('OLD_FINITE_VISIBLE','FINITE_VISIBLE'),('OLD_FINITE_REGION','FINITE_REGION')]:
  selected={**select(run['region_completion_v1/CAL_N3'],method),**select(run['region_completion_v1/TEST_N3'],method)}
  for i,r in selected.items():assert maxdiff(r['original_points'],canonical[i]['points']['N3_SUBPIX'])==0 and r['initial_pose']==canonical[i]['initial_poses']['N3_SUBPIX']
  add(name,{i:converted(r['actual_pose'],r)for i,r in selected.items()},'same_upstream_N3',True,'Historical finite proposal method is a distinct comparator; S region policy was superseded. Not current joint continuous solver.')
 for name,method in [('OLD_RGB_WD_BRANCH','JOINT_WD_BRANCH'),('OLD_RGB_TOGETHER','JOINT_VISIBLE_TOGETHER'),('OLD_SEGMENTED_RGB_CURRICULUM','SEGMENTED_RGB_CURRICULUM')]:
  selected=select(run['test_reused_v1'],method)
  for i,r in selected.items():assert maxdiff(np.asarray(r['original_points'])[:8],np.asarray(canonical[i]['points']['N3_SUBPIX'])[:8])==0
  add(name,{i:converted(r['actual_pose'],r)for i,r in selected.items()},'same_upstream_N3_optional_family_search',True,'TEST49 only; original N3 q is shared but W/D family can change; retained physical registration used.')
 for name,method in [('POSEFIX_COUNT_RAW','PF_SUB_COUNT_RAW'),('POSEFIX_COUNT_FUSED','PF_SUB_COUNT_FUSED')]:
  selected=select(run['boundary_rule_replay_v1'],method)
  add(name,{i:converted(r['actual_pose'],r)for i,r in selected.items()},'matched_full_pipeline_different_upstream',False,'TEST49 only; PF-SubPix q differs from N3; not isolated postprocessor effect.')
 for name,data in arm_rows.items():
  parts=descriptors[name]['available_partitions']
  expected=set().union(*(set(ids[p])for p in parts))
  assert set(data)==expected,(name,'Nonidentical IDs')
  checks={i:pose_audit(r['actual_pose'],canonical[i]['initial_poses']['N3_SUBPIX'])for i,r in data.items()}
  assert all(v['registered_physical_corner_cloud_gap_m']<1e-7 for v in checks.values()),(name,'Wrong registered object model')
  if descriptors[name]['comparison_level'] in ('same_upstream_N3','current_same_frozen_observations'):
   assert all(v['original_native_extents_equal'] and v['original_native_Q_max_abs']<1e-7 for v in checks.values())
  audit[name]=dict(frames=len(data),exact_IDs_per_available_partition=True,all_valid_physical_pose_and_model=True,max_Q_difference_from_original=max(v['original_native_Q_max_abs']for v in checks.values()),native_family_changed_ids=[i for i,v in checks.items()if not v['original_native_extents_equal']or v['original_native_Q_max_abs']>1e-7],per_frame=checks)
 inventory=dict(schema='SEALED_COMPARATOR_INVENTORY_COMMON95_V1',before_comparison_label_load=True,created_UTC=datetime.now(timezone.utc).isoformat(),source_bindings=sources,input=binding(canonical_path),baseline_input=binding(baseline_input_path),same_image_K_canvas_N3_initial_identity=baseline_audit,methods=descriptors,compatibility=audit,exact_ids=ids,severe_frames=0,crop_frames=0,new_inference_or_training=0,physical_metric_source=binding(ROOT/'scripts/research/pallet_visibility_curriculum_20261013_v1/scoring.py'),bootstrap_fixed_before_new_comparison=dict(repeats=REPEATS,seed=SEED,cluster='session'),missing_unavailable_methods=[dict(method='DOPE or external paper SOTA',reason='No same-image prediction artifacts identified; no invented or unmatched score comparison')],limitations=['TEST is reused exploratory, not new independent test','CAL old learned heads are in-sample or development; not independent generalization','Raw-input identities shared do not imply all masks/corrections/fit rules identical','PoseFix/YOLO alternate-upstream arms are matched full pipelines'],protected_user_SHA=protected,current_frozen_source_SHA=current_seal['source_sha256'])
 write_new(output/'COMPARATOR_INVENTORY.json',inventory)
 # Label access starts only here: all prediction files, protocols and population/input audits have passed.
 label_paths={'CAL':COHORT/'OOF_pose_v1/ISOLATED_CAL_SCORING.json','TEST':COHORT/'test_reused_v1/ISOLATED_TEST_SCORING.json'}
 labels={p:read(path)['rows']for p,path in label_paths.items()}
 for p in labels:assert set(labels[p])==set(ids[p])
 order_counts=Counter();scored={};summary_all={};pairwise={}
 for name,data in arm_rows.items():
  scored[name]={}
  for i,r in data.items():
   part=canonical[i]['partition'];truth=labels[part][i]['target']['truth'];order_counts[truth['order']]+=1
   record={k:v for k,v in r.items()if k not in ('source_row','actual_pose')};record.update(partition=part,metrics=pose_metric(r['actual_pose'],truth),actual_pose=r['actual_pose'])
   scored[name][i]=record
  summary_all[name]={p:summary({i:r for i,r in scored[name].items()if r['partition']==p})for p in descriptors[name]['available_partitions']}
 for name in scored:
  if name==CURRENT:continue
  pairwise[name]={p:paired({i:r for i,r in scored[name].items()if r['partition']==p},scored[CURRENT])for p in descriptors[name]['available_partitions']}
 # Validate current three-arm metrics against already sealed/postseal scorer output, without borrowing summary values.
 existing=read(STUDY/'evaluation_v3/POSTSEAL_SCORING.json');reference={r['id']:r for r in existing['rows']if r['cohort']=='PRIMARY_NONSEVERE'}
 parity={}
 for name,oldname in [('ORIGINAL_N3_PNP','ORIGINAL_PNP'),('CURRENT_BI_ONLY','FROZEN_BI_ONLY_PNP'),(CURRENT,'ALL_OT_JOINT_SE3')]:
  maximum=max(abs(float(scored[name][i]['metrics'][k])-float(reference[i]['metrics'][oldname][k]))for i in scored[name]for k in METRICS)
  assert maximum==0
  parity[name]=dict(frames=95,maximum_metric_difference=maximum)
 equal_pairs=[]
 names=list(scored)
 for aidx,a in enumerate(names):
  for b in names[aidx+1:]:
   common=set(scored[a])&set(scored[b])
   if common and all(maxdiff(scored[a][i]['actual_pose']['R_physical'],scored[b][i]['actual_pose']['R_physical'])<1e-8 and maxdiff(scored[a][i]['actual_pose']['centroid'],scored[b][i]['actual_pose']['centroid'])<1e-8 for i in common):equal_pairs.append(dict(a=a,b=b,common_frames=len(common),same_physical_poses_to_1e_8=True))
 assert all(sha(ROOT/p)==e for p,e in current_seal['source_sha256'].items())
 assert all(sha(ROOT/p)==e for p,e in protected.items())
 for d,bindings in sources.items():assert sha(ROOT/bindings['predictions']['path'])==bindings['predictions']['sha256'] and sha(ROOT/bindings['seal']['path'])==bindings['seal']['sha256'] and sha(ROOT/bindings['protocol']['path'])==bindings['protocol']['sha256']
 result=dict(schema='COMMON95_PHYSICAL_METRIC_COMPARATOR_BENCHMARK_V1',created_UTC=datetime.now(timezone.utc).isoformat(),method_labels=LABELS,method_inventory=binding(output/'COMPARATOR_INVENTORY.json'),script=binding(__file__),label_bindings={p:binding(path)for p,path in label_paths.items()},summary=summary_all,current_minus_comparator=pairwise,per_frame_scores=scored,current_existing_scorer_parity=parity,reference_symmetry_orders=dict(order_counts),physically_identical_arm_pairs=equal_pairs,severe_frames=0,crops_excluded=3,all_pose_fallbacks_included=True,source_and_primary_prediction_mutations=0,protected_six_user_files_unchanged=True,bootstrap_interpretation='Only four TEST sessions; correlated frames and reused selection. Intervals are descriptive exploratory uncertainty, not proof of generalization.',comparison_levels=descriptors)
 write_new(output/'BENCHMARK.json',result)
 C.save_rows(output/'PER_FRAME_METRICS.jsonl.gz',[dict(method=name,**row)for name,group in scored.items()for row in group.values()])
 table=['| TEST49 방법 | 위치 mean cm | 회전 mean ° | ADDsym mean cm | 성공 /49 |','|---|---:|---:|---:|---:|']
 for name in names:
  if 'TEST'not in summary_all[name]:continue
  s=summary_all[name]['TEST'];m=s['metrics'];table.append(f"| {LABELS[name]} | {m['translation_cm']['mean']:.4f} | {m['rotation_deg']['mean']:.4f} | {m['ADDsym_cm']['mean']:.4f} | {s['success_count']} |")
 pairs=[]
 for name in ['ORIGINAL_N3_PNP','CURRENT_BI_ONLY','OLD_RGB_SINGLE_FAMILY','BOUNDARY_PROGRESSIVE','COUNT_RAW','COUNT_FUSED','POSEFIX_SUB_PF_CAP','POSEFIX_SUB_YOLO_CAP']:
  p=pairwise[name]['TEST'];t=p['by_metric']['translation_cm'];rot=p['by_metric']['rotation_deg'];flip=p['success_flip']
  pairs.append(f"- 현재−{LABELS[name]}: 위치 {t['mean_difference']:+.4f}cm, 회전 {rot['mean_difference']:+.4f}°. 위치 세션 bootstrap95% [{t['session_cluster_bootstrap95'][0]:+.4f}, {t['session_cluster_bootstrap95'][1]:+.4f}]cm. 성공 복구 {len(flip['current_recovers'])} / 손실 {len(flip['current_loses'])}.")
 sameinput=[name for name in names if descriptors[name]['same_upstream_N3_SUBPIX']]
 text='''# 동일 CAL46/TEST49의 봉인 자세 비교\n\n모든 방법의 저장된 pose를 다시 읽고 **현재 scorer의 동일 reference, physical R, 고정 대칭, 위치/회전/ADDsym/5cm·5° 기준**으로 재계산했다. 이전 표 숫자를 합치지 않았다. prediction/protocol seal과 exact IDs, 원 입력, 등록 모델을 먼저 검증한 뒤 격리된 CAL/TEST target을 열었다. 새 영상 추론·자세 fitting·모델 학습·튜닝은 0회다. crop과 severe는 제외하며 fallback을 포함한 전체49장을 유지했다.\n\n'''+ '\n'.join(table)+'''\n\n## 같은 입력과 다른 파이프라인\n\n원 N3, 기존 RGB family 유지, 경계 progressive/count, 이전 finite 및 현재 v3는 동일 N3_SUBPIX native 입력을 사용한다. K·원 이미지·canvas와 초기 pose가 모두 같은95장임을 확인했다. same-family arm들은 cf_extents 및 physical registration Q도 유지한다. W/D branch arm은 같은 원 N3에서 family를 바꾸므로 별도로 표시했다.\n\nPoseFix RAW/SubPix와 YOLO arm은 같은 사진·물체·평가 target을 공유하지만 원 image head 좌표가 다르다. PoseFix는 별도 RGB ResNet152 head이며 N3와 같지 않다. 자체 q로 F의 W/D family를 다시 고르므로 전체 pipeline 비교다. R_cf를 직접 비교하지 않고 R_physical의 고정 대칭 지표를 썼다.\n\nCAL46도 전부 재계산한 수치가 JSON에 있다. 기존 RGB confidence의 CAL은 full-CAL fit의 in-sample 결과다. TEST49는 이미 반복 관찰한 탐색 평가이며 독립 일반화 검증은 아니다.\n\n## 현재−비교법의 짝비교\n\n'''+ '\n'.join(pairs)+'''\n\n모든 comparator에 대해 프레임별 승/패/동률, 성공 flip ID, 위치·회전 동시 개선/악화, 큰 손실/개선 10건과 세션별 평균·총 기여도를 `BENCHMARK.json`에 저장했다. 부호는 현재−비교법이며 오차 지표의 양수는 현재가 나쁘다는 뜻이다. TEST는 4개 세션 cluster를 10,000회 bootstrap했고 모든 프레임을 유지했다. 표본 세션 수가 작고 TEST를 재사용했으므로 통계적 우수성·새 일반화의 증명이 아니다.\n\n현재3조건을 기존 v3 postseal scorer의 개별95장 값과 대조했으며 모든 metric 차이가 정확히0이었다. 물리 pose가 같은 arm도 JSON에 표시했으므로 여러 이름을 독립적인 근거 수로 세면 안 된다.\n\n`COMPARATOR_INVENTORY.json`: 방법별 출처·seal·exactIDs·원 KP/K/model 호환성. `PER_FRAME_METRICS.jsonl.gz`: 전체 개별 점수. `BENCHMARK.json`: 전체 CAL/TEST 분포와 현재 기준 짝비교. 현재 source19개·모델·예측과 보호 사용자파일6개는 바뀌지 않았다. 실제 DOPE/SOTA의 동일사진 예측 artifact는 없으므로 해당 비교를 만들지 않았다.\n'''
 target=output/'BENCHMARK_KO.md'
 if target.exists():raise FileExistsError(target)
 target.write_text(text)
 write_new(output/'COMPARISON_RECEIPT.json',dict(comparator_artifacts={p.name:binding(p)for p in [output/'COMPARATOR_INVENTORY.json',output/'BENCHMARK.json',output/'PER_FRAME_METRICS.jsonl.gz',target]},source_and_prediction_bindings_verified_unchanged=True,protected_six_user_SHA_verified_unchanged=True,model_or_solver_reexecution_count=0,new_RGB_or_model_forward_count=0,shared_root_protocol_written=False))
 print(json.dumps({name:{p:dict(mean_translation=s['metrics']['translation_cm']['mean'],mean_rotation=s['metrics']['rotation_deg']['mean'],mean_ADDsym=s['metrics']['ADDsym_cm']['mean'],success=s['success_count'])for p,s in value.items()}for name,value in summary_all.items()},ensure_ascii=False,indent=2))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=HERE);args=parser.parse_args();main(args.output)
