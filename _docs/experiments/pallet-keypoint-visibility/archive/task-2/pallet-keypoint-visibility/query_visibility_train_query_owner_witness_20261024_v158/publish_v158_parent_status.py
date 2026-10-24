"""Publish completed source audit while preserving current evaluation and source fits."""
from pathlib import Path
import datetime,hashlib,json
V=Path(__file__).resolve().parent;P=V.parent
def read(p):return json.loads(Path(p).read_text())
def bind(p):
 p=Path(p).resolve();assert '_v112' not in str(p).lower()
 return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main():
 result=read(V/'ROOT_RESULTS_KO.json')
 independent=read(V/'independent_join_review/REVIEW.json')
 assert independent.get('PASS') is True
 final_review=read(V/'ROOT_FINAL_INDEPENDENT_REPORT_REVIEW.json');assert final_review['PASS']
 visual=read(V/'ROOT_REPORT_VISUAL_QA.json');assert visual['PASS']
 assert result['stage_complete'] and not result['fully_solved'] and result['selected_model']is None
 phase=dict(schema='V158_COMPLETE_SOURCE_OWNER_WITNESS_AUDIT',completed=True,stage_complete=True,goal_status='active_unsolved',fully_solved=False,selected_model=None,
   UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),results=bind(V/'ROOT_RESULTS_KO.json'),report=bind(V/'ROOT_READOUT_KO.html'),
   join=bind(V/'joined_owner_witness_v1/OWNER_WITNESS_JOIN_COMPLETE.json'),independent_join=bind(V/'independent_join_review/REVIEW.json'),
   actual_visual_QA=bind(V/'ROOT_REPORT_VISUAL_QA.json'),gallery=bind(V/'source_gallery_v1/GALLERY_COMPLETE.json'),final_report_review=bind(V/'ROOT_FINAL_INDEPENDENT_REPORT_REVIEW.json'),
   original_rows=390,source_frames=88,original_D_E=[341,49],all_q_L_native_roles_splits_unchanged=True,
   final_center_owner_counts=result['final_counts']['ALL'],new_gold=0,training_admitted=False,
   new_training_model_encoder_R0_private_input_prediction_V112_PnP_qL_calls=0,
   latest_actual_current_evaluation_preserved='V156',latest_actual_source_training_preserved='V155',
   next_source_action=bind(V/'ROOT_NEXT_SOURCE_ACTION_AFTER_OWNER_WITNESS.json'),live_processes=[])
 write(V/'ROOT_PHASE_RESULT.json',phase)
 phase_b=bind(V/'ROOT_PHASE_RESULT.json');result_b=bind(V/'ROOT_RESULTS_KO.json');report_b=bind(V/'ROOT_READOUT_KO.html')
 folder=V/'provenance/PARENT_POINTERS_BEFORE_V158_PUBLICATION';folder.mkdir()
 qa=[]
 for name in ('LATEST_STATUS_KO.json','ACTIVE_HANDOFF_KO.json'):
  dest=P/name;old_bytes=dest.read_bytes();(folder/name).write_bytes(old_bytes);old=json.loads(old_bytes);d=json.loads(old_bytes)
  preserve_keys=['latest_current_evaluation','latest_source_training','latest_actual_current','latest_phase_result_V156']
  preserved={k:old.get(k)for k in preserve_keys}
  d.update(schema='PALLET_QUERY_VISIBILITY_V158_SOURCE_OWNER_WITNESS_COMPLETE_UNSOLVED',updated_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),goal_status='active_unsolved',fully_solved=False,selected_model=None,
   active_work=str(V),active_work_state='V158_COMPLETE_SOURCE_OWNER_WITNESS_NEXT_NATIVE_SUPPORT_FEASIBILITY',
   latest_completed_work=str(V),latest_report=str(V/'ROOT_READOUT_KO.html'),latest_report_kind='V158_PUBLIC_TRAIN_QUERY_OWNER_AUDIT_NOT_MODEL_SUCCESS',
   latest_source_report=report_b,latest_source_report_QA=bind(V/'ROOT_REPORT_VISUAL_QA.json'),latest_report_QA=bind(V/'ROOT_REPORT_VISUAL_QA.json'),
   latest_results=result_b,latest_index=result_b,latest_source_diagnostic=result_b,
   latest_phase_result=phase_b,latest_phase_seal=phase_b,latest_source_phase_seal=phase_b,
   latest_source_owner_witness=bind(V/'joined_owner_witness_v1/OWNER_WITNESS_JOIN_COMPLETE.json'),
   V158_state='COMPLETE_SOURCE_TRAIN390_BLIND_OWNER_REVIEW_AND_SEPARATE_JOIN',V158_results=result_b,V158_phase_result=phase_b,
   V158_new_training_model_private_PnP_qL_calls=0,V158_selected_model=None,
   latest_progress=dict(full_goal_solved=False,source_diagnostic_complete=True,phase=phase_b,source_available_weak_role_queries_retained=390,source_frames=88,
    owner_counts=result['final_counts']['ALL'],original_D_E_unchanged=[341,49],cross40_two_reviewer_agree=34,cross40_three_observations_agree=30,
    new_training_model_encoder_R0_private_PnP_qL_calls=0,training_admitted=False,next_source_action=phase['next_source_action']),live_processes=[])
  assert all(d.get(k)==v for k,v in preserved.items())
  temp=dest.with_suffix('.v158.tmp');temp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(dest)
  after=read(dest);assert all(after.get(k)==v for k,v in preserved.items())
  qa.append(dict(parent=bind(dest),snapshot=bind(folder/name),preserved_keys=preserve_keys,latest_actual_current_V156_unchanged=True,latest_actual_source_fit_V155_unchanged=True))
 write(V/'ROOT_PARENT_POINTER_PUBLICATION_QA.json',dict(completed=True,PASS=True,parent_status=qa,phase=phase_b,goal_status='active_unsolved',selected_model=None))
 print(json.dumps(dict(completed=True,phase=phase_b,publication=bind(V/'ROOT_PARENT_POINTER_PUBLICATION_QA.json')),ensure_ascii=False))
if __name__=='__main__':main()
