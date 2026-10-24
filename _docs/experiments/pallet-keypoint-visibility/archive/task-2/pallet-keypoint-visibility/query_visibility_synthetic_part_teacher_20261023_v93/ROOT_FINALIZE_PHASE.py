from pathlib import Path
import json,hashlib,datetime,os
P=Path(__file__).resolve().parent;BASE=P.parent
V94=BASE/'query_visibility_matched_query_scalar_20261023_v94'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):p=Path(p);return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def read(p):return json.loads(Path(p).read_text())
def write(p,o):
 assert not p.exists(),str(p);p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 report=P/'ROOT_READOUT_KO.html';results=P/'ROOT_PHASE_RESULTS_KO.json';qa=P/'source_report_review/root_visual_qa/ROOT_VISUAL_QA.json'
 assert sha(report)=='d64af3011b25e69ef65fcda7aa436d40c10e22298b22161fec2abb1c0b70ff6f'
 assert sha(results)=='9f8989591a13a3df1d89edddc66e750e082d12d47914d5bb3643f5dfe262addd'
 assert read(P/'source_report_review/REVIEW.json')['PASS'] and read(qa)['PASS']
 assert read(P/'independent_actual_raster_audit/REVIEW.json')['PASS']
 prior=BASE/'query_visibility_flat_pallet_cue_bridge_20261023_v92/ROOT_PHASE_RESULT_SEAL.json'
 assert sha(prior)=='939cdc4c1a372c9502c00967b23ce05ab945329bd64dad5511a39114932f0059'
 assert read(results)['fully_solved'] is False
 assert not V94.exists();V94.mkdir()
 scope=dict(schema='V94_MATCHED_ACTUAL_INITIAL_QUERY_SINGLE_SCALAR_SOURCE_CONTRACT_PREPARATION',created_UTC=now,goal_status='active',fully_solved=False,parent_report=bind(report),parent_results=bind(results),decision_draft=bind(P/'next_operator_candidate/DECISION_DRAFT_KO.json'),
  objective='Prepare same query-centred crop and single scalar source supervision without GT region/mosaic readout substitution; improve semantic visibility transfer using public real/synthetic sources.',
  first_stage='Saved source metadata/operator/expectedcue binding and availability contract only. No new fit/forward/current or actual source body read under this scope; freeze next data/operator/selector scope before them.',
  priority='Preserve actual public initializer query_id/q/L/native slot order and all invalid/MISS/U denominators. Existing public predicted queries are preferred over GT-centred source emulation.',
  cue_binding='Known target instance and expected cue support must be independently defined; absent binding remains UNKNOWN. Wholeboard or generic material presence is not native semantic gold.',
  source_emulation='Cuecentre+jitter is at most a separate AUX candidate; using r to generate q is GT-conditioned sampling even when r is absent from model tensors. It is not original initializer q and cannot silently replace it.',
  operator='Train/sourceVAL/deployment must read the same single scalar of the same original-q-centred inputs. No GT-region/mosaic readout relabelled as q-centre.',
  auxiliary_part_teacher='V93 part teacher remains optional geometric evidence, not mandatory native mapping or a reason to snap virtual refs to physical surfaces.',
  before_any_fit=['Define source query/expectedcue binding, source availability and GT-free deployment abstention separately.','Define controlled cover and cue-preserving sham, naturalforeign evidence tier, opacity/ownership uncertainty.','Group capture/instance/cue/beforeafter/donor families by partition; preserve used-source development exposure.','Freeze scale/operator/labels/selectors/checkpoint/threshold/source-gates before source body/model/fit; preserve failures.'],
  forbidden=['No current223 training/threshold/checkpoint/model selection.','No PnP, pose optimization or native q/ref coordinate correction.','No AI proxy/source weak annotation promoted to human native8 D/E/S gold.'],
  new_current_or_fit_or_forward_or_native_labels_or_PnP=0,no_user_approval_pending=True)
 write(V94/'SCOPE_BEFORE_MATCHED_QUERY_CONTRACT.json',scope)
 write(P/'ROOT_PHASE_COMPLETION_RECEIPT.json',dict(created_UTC=now,phase_state='COMPLETE_DIAGNOSTIC_NOT_CLASSIFIER_SOLUTION',goal_status='active',fully_solved=False,report=bind(report),results=bind(results),report_QA=bind(qa),next_scope=bind(V94/'SCOPE_BEFORE_MATCHED_QUERY_CONTRACT.json'),new_current_or_model_or_fit_or_PnP=0,V92_rootseal_unchanged=bind(prior)))
 index=P/'ARTIFACT_INDEX_PHASE_RESULTS.json';seal=P/'ROOT_PHASE_RESULT_SEAL.json'
 owned=[];excluded=[]
 for f in sorted(P.rglob('*')):
  if not f.is_file():continue
  rel=f.relative_to(P)
  if '__pycache__'in rel.parts or f.suffix=='.pyc':excluded.append(str(rel));continue
  if f in [index,seal]:continue
  assert not f.is_symlink(),str(f)
  owned.append(dict(relative_path=str(rel),**bind(f)))
 write(index,dict(schema='V93_ROOT_OWNED_ARTIFACT_INDEX',created_UTC=now,root=str(P),files=owned,excluded_runtime_bytecode=excluded,self_excluded=[index.name,seal.name],self_exclusion_reason='Avoid recursive checksum cycles; final rootseal binds index.',owned_files=len(owned)))
 for x in owned:assert sha(x['path'])==x['sha256']
 write(seal,dict(schema='V93_ROOT_PHASE_RESULT_SEAL',created_UTC=now,completed=True,PASS=True,phase_state='COMPLETE_SIXVIEW_PART_OWNER_AND_AUX_CROP_DIAGNOSTIC',goal_status='active',fully_solved=False,index=bind(index),owned_files=len(owned),all_owned_SHA_replay_PASS=True,report=bind(report),results=bind(results),geometry_audit=bind(P/'independent_geometry_review/GEOMETRY_AUDIT.json'),actual_raster_seal=bind(P/'raster_prototype/FIXED_PART152_SIX_CAMERAS/RASTER_OUTPUT_SEAL.json'),actual_buffer_audit=bind(P/'independent_actual_raster_audit/REVIEW.json'),actual_visual_review=bind(P/'independent_fixed6_visual_review/VISUAL_REVIEW_KO.json'),aux_crop_seal=bind(P/'saved_aux_query_crop_diagnosis/FINAL_SAVED_AUX_CROP_DIAGNOSIS_SEAL.json'),report_review=bind(P/'source_report_review/REVIEW.json'),report_visual_QA=bind(qa),next_scope=bind(V94/'SCOPE_BEFORE_MATCHED_QUERY_CONTRACT.json'),new_human_native_DES_gold=0,new_model_training_current_PnP=0,immutable_after_this_seal=True))
 for name in ['LATEST_STATUS_KO.json','ACTIVE_HANDOFF_KO.json']:
  path=BASE/name;d=read(path)
  d['V92_model_candidate']=d.pop('model_candidate')
  d.update(schema='PALLET_QUERY_VISIBILITY_V93_DIAGNOSTIC_COMPLETE_V94_MATCHED_QUERY_CONTRACT_PREPARATION',updated_UTC=now,goal_status='active',fully_solved=False,
   user_capture_provenance='Direct user: 직접 촬영했고 KU Leuven 자료와 무관합니다',latest_source_report=str(report),latest_source_report_QA=bind(qa),active_work=str(V94),latest_report=str(report),latest_phase_seal=bind(seal),latest_index=str(index),
   model_candidate=dict(state='NO_NEW_TRAINED_MODEL_IN_V93',latest_trained_current_evaluation='V91 unchanged',next='V94 matched actual query single-scalar source contract draft only; no fit'),
   previous_parent_pointers=str(P/'provenance/PARENT_POINTERS_AT_RASTER_START'),V93_state='COMPLETE_GEOMETRY_SIXVIEW_OWNER_RASTER_SAVED_BUFFER_AND_AUXCROP_DIAGNOSTIC_NOT_SEMANTIC_SOLUTION',V93_raster_entry=bind(P/'ROOT_ACTUAL_FIXED6_RASTER_ENTRY.json'),V93_root_seal=bind(seal),V93_actual_raster_seal=bind(P/'raster_prototype/FIXED_PART152_SIX_CAMERAS/RASTER_OUTPUT_SEAL.json'),V93_actual_buffer_audit=bind(P/'independent_actual_raster_audit/REVIEW.json'),V93_semantic_native8='UNKNOWN; part152 appears longdeckboard in pallet-like assembly. Wholepart pixelvisibility is not specificcorner/nativeD/E/S.',V93_total_diagnostic_pixel_areas=read(results)['total_diagnostic_pixel_areas'],V93_new_current223_access=0,V93_training_updates=0,V93_learned_model_forwards=0,V93_new_human_native_gold=0,V93_PnP=0,V94_state='SAVED_METADATA_AND_OPERATOR_CONTRACT_PREPARATION_ONLY_NO_FIT',V94_scope=bind(V94/'SCOPE_BEFORE_MATCHED_QUERY_CONTRACT.json'),
   next_required_actions=['Read V93report and V94scope; keep V91 current evaluation unchanged.','Audit saved public initializer q/L and expectedcue binding/availability for same query-centred single scalar train/VAL operator.','Retain UNKNOWN without cue/instance binding. Optional sourcecue jitter must be separate GT-conditioned emulation, never actual initializer substitute.','Freeze source-only label/operator/split/U/cover-sham/naturalforeign/gates before any actual body read/model/fit. No current223 training/threshold/checkpoint/model selection or PnP.'],
   known_limit='Goal unresolved: V91 nearD/external transfer errors remain. V92 testedQwen2B twoimage source outputs constantvisible. V93 verifies wholepart geometry and operator illustration only; no new semantic classifier.',no_user_approval_pending=True)
  tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
 print(json.dumps(dict(seal=bind(seal),index=bind(index),owned_files=len(owned),next_scope=bind(V94/'SCOPE_BEFORE_MATCHED_QUERY_CONTRACT.json'),goal_status='active',fully_solved=False),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
