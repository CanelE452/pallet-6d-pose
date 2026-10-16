"""Post-choice CAL-only same-Sigma/prior NO_OT control, never changes choice."""
from pathlib import Path
import sys,os
ROOT=next(p for p in Path(__file__).resolve().parents if(p/'scripts/research').is_dir())
sys.path.insert(0,str(ROOT))
for key in('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import time,json
import numpy as np
from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C
from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_selection_v4 as D
from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_bias_selection_v4_1 as B
from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_uncertain_solver_v4_bias as P
from scripts.research.pallet_visibility_curriculum_20261013_v1.statewise_joint_api_v4 import finish_observations
from scripts.research.pallet_visibility_curriculum_20261013_v1.scoring import pose_metric
O=Path(__file__).resolve().parent;A=O.parent

def main():
 choice=C.read(A/'SELECTED_CONFIG.json');seal=C.read(A/'SELECTION_SEAL.json');protocol=C.read(A/'PROTOCOL.json')
 assert C.sha(A/'SELECTED_CONFIG.json')==seal['selected_config_sha256']
 assert choice['selected_bias_strength']==1. and choice['selected_prior_weight']==1.
 D.verify_bindings(protocol)
 rows,folds,_,_=D.load_inputs();tasks=[]
 for row in rows:
  fold=folds[row['id']]
  t=D.build_spec(row,fold['model'],1.,'SELECTED_V4_NO_OT')
  t['arm']='POST_CHOICE_SELECTED_BIAS_NO_OT';t['bias_strength']=choice['selected_bias_strength']
  t['solver_input']['config'].update(prior_bias6=deepcopy(fold['model']['prior_fit']['median_bias6']),bias_strength=t['bias_strength'])
  t['solver_input_sha256']=D.physics_key(t['solver_input']);tasks.append(t)
 base_inputs={r['id']:r for r in D.read_lines(A/'CANDIDATE_INPUTS.jsonl.gz')if r['bias_strength']==1.}
 for t in tasks:
  base=base_inputs[t['id']]['solver_input'];s=t['solver_input']
  assert s['visible']==base['visible']and s['config']==base['config']and s['coarse_pose']==base['coarse_pose']
  assert s['domains']==[]and all(b=='U'if a in('O','T')else b==a for a,b in zip(base['states'],s['states']))
 controlproto=dict(schema='POST_CHOICE_CAL_MEDIAN_BIAS_NO_OT_V1',created_UTC=D.now(),CAL_ids=[r['id']for r in rows],
  after_selection_sealed=True,does_not_change_selection=True,selected_config_sha256=C.sha(A/'SELECTED_CONFIG.json'),
  selected_seal_sha256=C.sha(A/'SELECTION_SEAL.json'),source_sha256={**protocol['source_sha256'],str(Path(__file__).relative_to(C.ROOT)):C.sha(__file__)},
  input_bindings=protocol['input_bindings'],preserved_model_bindings=protocol['preserved_model_bindings'],
  protected_user_sha256=protocol['protected_user_sha256'],TEST_files_opened=False,RGB_reads=0,
  mutation='only O/T states become U and domains removed; fold Sigma/weight/nonSscale/prior median/lambda unchanged')
 if(O/'PROTOCOL.json').exists():assert C.read(O/'PROTOCOL.json')['selected_config_sha256']==controlproto['selected_config_sha256']
 else:
  D.write(O/'PROTOCOL.json',controlproto)
  D.write_lines(O/'INPUTS.jsonl.gz',[{k:t[k]for k in('id','solver_input','solver_input_sha256','original_BI_sha256','calibrated_BI_sha256','nonself_prediction_scale_px')}for t in tasks])
  D.write(O/'INPUT_SEAL.json',dict(before_control_fitting=True,rows=46,inputs_sha256=C.sha(O/'INPUTS.jsonl.gz'),protocol_sha256=C.sha(O/'PROTOCOL.json')))
 D.verify_bindings(controlproto)
 cache={r['solver_input_sha256']:r for r in D.read_lines(A/'SOLVER_CHECKPOINTS.jsonl.gz')}
 checkpoint=O/'SOLVER_CHECKPOINTS.jsonl.gz'
 if checkpoint.exists():cache.update({r['solver_input_sha256']:r for r in D.read_lines(checkpoint)})
 pending=[t for t in tasks if t['solver_input_sha256']not in cache];started=time.perf_counter()
 print(dict(rows=46,cached=46-len(pending),new=len(pending)),flush=True)
 if not(O/'PREDICTION_SEAL.json').exists():
  with ProcessPoolExecutor(max_workers=3,mp_context=get_context('spawn'))as executor:
   futures={executor.submit(B.worker,t):t for t in pending}
   for i,f in enumerate(as_completed(futures),1):
    result=f.result();D.append_checkpoint(checkpoint,result);cache[result['solver_input_sha256']]=result
    if i%5==0 or i==len(pending):print(dict(completed_new=i,new=len(pending),elapsed_seconds=time.perf_counter()-started),flush=True)
  pred=[]
  for t in tasks:
   solved=cache[t['solver_input_sha256']];rt=finish_observations(t['observation'],deepcopy(solved['solution']),t['solver_input']['visible'],t['model'])
   pred.append(dict(id=t['id'],partition='CAL',session=t['session'],arm=t['arm'],runtime=rt,actual_pose=rt['actual_pose'],solver_input_sha256=t['solver_input_sha256']))
  D.write_lines(O/'PREDICTIONS.jsonl.gz',pred);D.verify_bindings(controlproto)
  D.write(O/'PREDICTION_SEAL.json',dict(after_choice=True,before_control_CAL_target_read=True,rows=46,predictions_sha256=C.sha(O/'PREDICTIONS.jsonl.gz'),protocol_sha256=C.sha(O/'PROTOCOL.json'),cached_inputs=46-len(pending),new_solves=len(pending)))
 else:pred=D.read_lines(O/'PREDICTIONS.jsonl.gz')
 assert C.sha(O/'PREDICTIONS.jsonl.gz')==C.read(O/'PREDICTION_SEAL.json')['predictions_sha256']
 # These are CAL-only targets, opened after the control prediction seal.
 target=protocol['CAL_target'];assert C.sha(C.ROOT/target['path'])==target['sha256']
 labels=C.read(C.ROOT/target['path'])['rows'];biasrows=D.read_lines(A/'CANDIDATE_SCORED_ROWS.jsonl.gz')
 zero={r['id']:r for r in biasrows if r['bias_strength']==0.};chosen={r['id']:r for r in biasrows if r['bias_strength']==1.}
 raw={r['id']:r for r in rows};pairs=[];flips=[]
 for r in pred:
  rid=r['id'];r['metrics']=pose_metric(r['actual_pose'],labels[rid]['target']['truth']);b=chosen[rid];z=zero[rid]
  original=pose_metric(raw[rid]['initial_pose'],labels[rid]['target']['truth'])
  qa=np.asarray(b['runtime']['keypoint_readout']['rigid_model_corners']);qb=np.asarray(r['runtime']['keypoint_readout']['rigid_model_corners'])
  joint=b['runtime']['joint_pose'];individual=joint['joint_constraint_audit'];minimum=min((d['minimum_slack']for d in individual),default=None)
  pairs.append(dict(id=rid,BI_count=len(b['runtime']['frozen_visible_measurements']),model_projection_RMS_delta_px=float(np.sqrt(np.mean(np.sum((qa-qb)**2,axis=1)))),
   center_delta_m=float(np.linalg.norm(np.asarray(b['actual_pose']['centroid'])-r['actual_pose']['centroid'])),joint_accepted=b['runtime']['accepted'],NO_OT_accepted=r['runtime']['accepted'],
   OT_domains=len(individual),minimum_OT_slack=minimum,active_domain_constraints=[dict(native_corner=d['native_corner'],inequality=v['name'],slack=v['slack'])for d in individual for v in d['inequalities']if abs(v['slack'])<=1e-5],
   joint_metrics=b['metrics'],NO_OT_metrics=r['metrics']))
  status=[bool(original['success_5cm_5deg']),bool(z['metrics']['success_5cm_5deg']),bool(b['metrics']['success_5cm_5deg'])]
  if len(set(status))>1:
   visible=b['runtime']['frozen_visible_measurements'];fit=b['runtime']['calibrated_visible_measurements'];audits=b['runtime']['joint_pose'].get('visible_protection_audit',[])
   flips.append(dict(id=rid,original_metrics=original,bias0_metrics=z['metrics'],bias1_metrics=b['metrics'],NO_OT_metrics=r['metrics'],success_original_bias0_bias1=status,
    states=b['runtime']['states'],BI_by_state={s:sum(m['state']==s for m in visible)for s in('B','I')},
    BI_groups=[dict(native_corner=m['native_corner'],state=m['state'],group=m.get('uncertainty_calibration',{}).get('group'),xy=m['xy'],observed_covariance_px2=m['covariance_px2'],fit_covariance_px2=m['fit_covariance_px2'],fit_reliability_weight=m['fit_reliability_weight'])for m in fit],
    visible_model_residual_audit=audits,OT_constraints=individual))
 D.write_lines(O/'SCORED_ROWS.jsonl.gz',pred)
 report=dict(schema='POST_CHOICE_CAL_BIAS_NO_OT_INFORMATION_AND_SUCCESS_FLIP_AUDIT_V1',after_choice=True,selection_unchanged=True,
  selected_joint_summary=D.summarize(list(chosen.values())),NO_OT_summary=D.summarize(pred),
  BI_ge3_pairs=[r for r in pairs if r['BI_count']>=3],BI_lt3_pairs=[r for r in pairs if r['BI_count']<3],
  changed_success_cases=flips,bias0_to_bias1_success_flips=[r for r in flips if r['success_original_bias0_bias1'][1]!=r['success_original_bias0_bias1'][2]],
  same_fold_Sigma_nonSscale_bias_lambda=True,TEST_files_opened=False,RGB_reads=0,no_parameter_choices_from_control=True,
  reference_is_historical_annotation_derived_not_independent_metrology=True)
 D.write(O/'CAL_INFORMATION_AUDIT.json',report);D.verify_bindings(controlproto)
 assert C.sha(A/'SELECTED_CONFIG.json')==seal['selected_config_sha256']
 D.write(O/'COMPLETION_RECEIPT.json',dict(all_source29_and_original_selection_preserved=True,TEST_files_opened=False,
  artifacts={n:D.binding(O/n)for n in('PROTOCOL.json','INPUT_SEAL.json','PREDICTIONS.jsonl.gz','PREDICTION_SEAL.json','SCORED_ROWS.jsonl.gz','CAL_INFORMATION_AUDIT.json')}))
 print(dict(summary=report['NO_OT_summary'],BI_ge3_changed=sum(r['model_projection_RMS_delta_px']>1e-3 for r in report['BI_ge3_pairs']),
  bias0_to_bias1_success_flips=len(report['bias0_to_bias1_success_flips'])),flush=True)

if __name__=='__main__':main()
