"""Postseal pure arithmetic/constraint verification; no GT/RGB/TEST file reads."""
from pathlib import Path
import sys
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p/'scripts/research').is_dir())))
import gzip,json
import numpy as np
from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C
from scripts.research.pallet_visibility_curriculum_20261013_v1 import joint_ot_pnp as P
from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_uncertain_solver_v4 as S
from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_selection_v4 as D
O=Path(__file__).resolve().parent
protocol=C.read(O/'PROTOCOL.json');D.verify_bindings(protocol)
read=lambda name:D.read_lines(O/name)
pred=read('CANDIDATE_PREDICTIONS.jsonl.gz');inputs=read('CANDIDATE_INPUTS.jsonl.gz')
controls=read('CONTROL_PREDICTIONS.jsonl.gz');ci=read('CONTROL_INPUTS.jsonl.gz')
assert len(pred)==len(inputs)==184 and len(controls)==len(ci)==92
failures=[];counts=dict(rows=0,accepted=0,rejected=0,domains_checked=0,BI_observations_checked=0,two_BI_families=0)
maximum_BI_soft_maha=0.;minimum_slack=1e9
for row,entry in zip(pred+controls,inputs+ci):
 assert row['id']==entry['id'] and row['arm']==entry['arm']
 assert row['solver_input_sha256']==entry['solver_input_sha256']==D.physics_key(entry['solver_input'])
 spec=entry['solver_input'];rt=row['runtime'];s=rt['joint_pose'];counts['rows']+=1
 assert rt['calibrated_visible_measurements']==spec['visible']
 for source,fit,observed in zip(rt['frozen_visible_measurements'],spec['visible'],s['visible_observations']):
  assert source['xy']==fit['xy']==observed['xy']
  assert source['covariance_px2']==fit['covariance_px2']==observed['covariance_px2']
  assert rt['keypoint_readout']['reported_points'][source['native_corner']]==source['xy']
  counts['BI_observations_checked']+=1
 assert s['provenance']['S_image_terms']==s['provenance']['S_region_terms']==0
 assert not s['provenance']['B_I_absolute_covariance_residual_constraints']
 assert not s['provenance']['seed_projection_protection_enabled']
 assert np.allclose(np.asarray(row['actual_pose']['R_physical']),np.asarray(row['actual_pose']['R_cf'])@np.asarray(spec['coarse_pose']['physical_registration_Q']))
 if s['accepted']:
  counts['accepted']+=1
  cfg=S._configuration(spec['config'])
  xyz,K,states,m,domains,hw=P._inputs(spec['xyz'],spec['K'],spec['states'],spec['visible'],spec['domains'],spec['image_hw'],S._helper_config(cfg))
  sides=tuple(d['outside_canvas_side']for d in s['joint_constraint_audit'])
  scale=max(float(np.linalg.norm(np.ptp(xyz,axis=0))),1e-3)
  R,t=np.asarray(s['R']),np.asarray(s['t']);p=P._pose_vector(R,t,scale)
  values=S._constraints(p,xyz,K,m,domains,sides,hw,scale,cfg,analytic=False)
  minimum_slack=min(minimum_slack,float(min(values)))
  assert np.isfinite(values).all() and min(values)>=-cfg['feasibility_tolerance']
  q,_=P.project_native(xyz,R,t,K)
  assert np.allclose(q,s['rigid_projection8'],atol=1e-9)
  assert len(s['joint_constraint_audit'])==len(domains) and all(d['feasible']for d in s['joint_constraint_audit'])
  counts['domains_checked']+=len(domains)
  maximum_BI_soft_maha=max(maximum_BI_soft_maha,max(d['measurement_mahalanobis']for d in s['visible_protection_audit']))
  if len(m)==2:
   counts['two_BI_families']+=1
   assert s['ambiguity']['detected'] and s['status']=='joint_feasible_pose_family'
   assert not s['OT_solved']and not s['OT_unique_solution_claimed']
 else:
  counts['rejected']+=1
  assert not rt['keypoint_readout']['OT_solved']
  for i,state in enumerate(rt['states']):
   if state in('O','T'):assert rt['keypoint_readout']['reported_points'][i]is None
  seed=s.get('BI_prior_seed')or{}
  if seed.get('contains_underconstrained_pose_family'):assert 'COARSE_FALLBACK' in row['actual_pose']['source']
selected=C.read(O/'SELECTED_CONFIG.json');selection_seal=C.read(O/'SELECTION_SEAL.json')
assert C.sha(O/'SELECTED_CONFIG.json')==selection_seal['selected_config_sha256']
assert selected['selected_prior_weight']==1.
scored=read('CANDIDATE_SCORED_ROWS.jsonl.gz');score=C.read(O/'CAL_SELECTION_SCORE.json')
for weight in D.WEIGHTS:
 rows=[r for r in scored if r['prior_weight']==weight];assert len(rows)==46
 reported=score['summary'][str(weight)]
 assert sum(r['metrics']['success_5cm_5deg']for r in rows)==reported['success_count']
 for metric in D.METRICS:assert np.isclose(np.mean([r['metrics'][metric]for r in rows]),reported['metrics'][metric]['mean'],atol=1e-12)
chosen={r['id']:r for r in pred if r['prior_weight']==selected['selected_prior_weight']}
zero={r['id']:r for r in pred if r['prior_weight']==0.}
paired_changes=[]
for row in controls:
 base=chosen[row['id']]
 assert row['calibrated_BI_sha256']==base['calibrated_BI_sha256']
 assert row['nonself_prediction_scale_px']==base['nonself_prediction_scale_px']
 if row['arm']=='SELECTED_V4_NO_PRIOR':
  assert row['actual_pose']==zero[row['id']]['actual_pose']
 else:
  assert not row['runtime']['ot_domains']['domains']
  assert not row['runtime']['keypoint_readout']['OT_solved']
  a,b=np.asarray(base['runtime']['keypoint_readout']['rigid_model_corners']),np.asarray(row['runtime']['keypoint_readout']['rigid_model_corners'])
  paired_changes.append(dict(id=row['id'],BI_count=len(base['runtime']['frozen_visible_measurements']),
   joint_accepted=base['runtime']['accepted'],no_OT_accepted=row['runtime']['accepted'],
   model_projection_RMS_delta_px=float(np.sqrt(np.mean(np.sum((a-b)**2,axis=1)))),
   center_delta_m=float(np.linalg.norm(np.asarray(base['actual_pose']['centroid'])-row['actual_pose']['centroid']))))
D.verify_bindings(protocol)
for path,expected in protocol['source_sha256'].items():assert C.sha(O/'code_snapshot'/path)==expected
for name,binding in C.read(O/'COMPLETION_RECEIPT.json')['artifacts'].items():assert C.sha(C.ROOT/binding['path'])==binding['sha256']
report=dict(schema='POSTSEAL_CAL_SELECTION_CONTRACT_AUDIT_V4',all_pass=True,counts=counts,
 minimum_scalar_guard_slack=minimum_slack,maximum_accepted_soft_BI_mahalanobis=maximum_BI_soft_maha,
 original_BI_xy_covariance_preserved=True,all_individual_OT_and_physical_guards_pass=True,
 physical_registration_Q_preserved=True,selection_arithmetic_recomputed_from_sealed_scores=True,
 NO_PRIOR_exactly_matches_grid_zero_candidates=True,NO_OT_same_fit_covariance_and_scale=True,
 source27_snapshots_and_current_and_input_model_bindings_pass=True,protected6_pass=True,
 pose_solver_calls=0,RGB_reads=0,GT_target_reads=0,TEST_files_opened=False,
 paired_NO_OT_pose_changes=paired_changes)
D.write(O/'CONTRACT_AUDIT.json',report)
print(json.dumps({k:v for k,v in report.items()if k!='paired_NO_OT_pose_changes'}))
