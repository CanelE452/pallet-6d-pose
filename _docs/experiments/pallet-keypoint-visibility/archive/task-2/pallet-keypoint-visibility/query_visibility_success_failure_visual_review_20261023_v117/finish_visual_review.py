import hashlib
import json
import shutil
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

V=Path(__file__).resolve().parent
P=V.parent
S=P/'query_visibility_same_material_foreground_source_20261023_v116'
def bind(p):
    p=Path(p); b=p.read_bytes()
    return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def write(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
class Links(HTMLParser):
    def __init__(self): super().__init__(); self.links=[]
    def handle_starttag(self,tag,attrs):
        self.links += [v for k,v in attrs if k in ['href','src'] and v]
links=Links(); links.feed((V/'ROOT_INTERPRETATION_KO.html').read_text())
assert all((V/u.split('#')[0]).is_file() for u in links.links)
stamp=datetime.now(timezone.utc).isoformat()
qa={'completed':True,'created_UTC':stamp,'HTML':bind(V/'ROOT_INTERPRETATION_KO.html'),
 'root_actually_viewed_browser_screenshot':bind(V/'ROOT_INTERPRETATION_QA.png'),
 'root_actual_view_KO':'상단 한국어 설명·4개 실제 crop·좌표범례·관찰표 시작이 겹침 없이 읽힌다. 상세 원본9사례는 B/C가 전부 실제 확인했고 root는 contact3장과 중요한 퇴행2장, 요약을 확인했다.',
 'all_local_links_and_images_exist':True,'local_links_checked':len(links.links),
 'all9_raw_saved_crop_exactness':'Original sealed renderer certificate; no source pixels mutated by interpretation.',
 'no_new_inference_fit_private_tuning_PnP':True}
write(V/'ROOT_REPORT_VISUAL_QA.json',qa)
source_index=[S/f for f in ['ROOT_SOURCE_POLICY_BEFORE_DONORS_AND_PIXELS.json','ROOT_ACTUAL_METADATA_PREFLIGHT.json','ROOT_ACTUAL_IMAGE_CONSTRUCTION_ENTRY.json','ROOT_ACTUAL_FIXED_SOURCE_VISUAL_ENTRY.json','corpus_build_v1/CORPUS_OUTPUT_SEAL.json','corpus_build_v1/SUMMARY.json','independent_saved_corpus_review/REVIEW.json','fixed_source_visual_review_v1/RENDERING_SEAL.json','independent_source_visual_review/OBSERVATIONS_AND_RECOMMENDATION.json','ROOT_ACTUAL_EXECUTION_AND_FIXED_VISUAL_DECISION.json']]
write(S/'ARTIFACT_INDEX_PHASE_RESULTS.json',{'completed':True,'scope':'V116 public same-material foreground construction, replay, fixed visual review and bounded AUX admission; not a new model','artifacts':[bind(p) for p in source_index]})
write(S/'ROOT_PHASE_RESULT_SEAL.json',{'completed':True,'created_UTC':stamp,'phase_result':'COMPLETE_62_CONDITIONAL_AUX_PAIRS_386_FAILED_U_NO_FEATURES_OR_FIT','index':bind(S/'ARTIFACT_INDEX_PHASE_RESULTS.json'),'root_execution_and_decision':bind(S/'ROOT_ACTUAL_EXECUTION_AND_FIXED_VISUAL_DECISION.json'),'accepted_pairs':62,'failed_U_pairs':386,'new_model_fit_or_PnP':0,'fully_solved':False})
visual_index=[V/f for f in ['SCOPE_BEFORE_SELECTION_AND_PIXELS.json','SELECTION_LOCK_BEFORE_PIXELS.json','RENDER_RECEIPTS.json','RENDER_AND_QA_SEAL.json','ACTUAL_VISUAL_QA_AND_OBSERVATIONS.json','independent_visual_review/ACTUAL_LOCKED9_VISUAL_OBSERVATIONS_KO.json','ROOT_ACTUAL_VISUAL_INTERPRETATION.json','ROOT_INTERPRETATION_KO.html','ROOT_READOUT_KO.html','INLINE_FIXED4_SUMMARY.png','ROOT_REPORT_VISUAL_QA.json']]
write(V/'ARTIFACT_INDEX_PHASE_RESULTS.json',{'completed':True,'scope':'V117 user-requested visual diagnosis; saved results and crops only','artifacts':[bind(p) for p in visual_index]})
write(V/'ROOT_PHASE_RESULT_SEAL.json',{'completed':True,'created_UTC':stamp,'user_visual_request_handled':True,'index':bind(V/'ARTIFACT_INDEX_PHASE_RESULTS.json'),'root_interpretation':bind(V/'ROOT_ACTUAL_VISUAL_INTERPRETATION.json'),'report':bind(V/'ROOT_INTERPRETATION_KO.html'),'QA':bind(V/'ROOT_REPORT_VISUAL_QA.json'),'selected_cases':9,'new_model_forward_fit_private_selection_PnP':0,'overall_model_goal_solved':False,'source_progress':bind(S/'ROOT_PHASE_RESULT_SEAL.json')})
backup=V/'provenance/PARENT_POINTERS_BEFORE_V117_VISUAL_COMPLETE'
backup.mkdir(parents=True,exist_ok=True)
for name in ['LATEST_STATUS_KO.json','ACTIVE_HANDOFF_KO.json']:
    path=P/name
    assert not (backup/name).exists(), 'Do not overwrite provenance snapshot'
    shutil.copy2(path,backup/name)
    d=json.loads(path.read_text())
    latest_current=d['latest_current_evaluation']
    assert latest_current['version']=='V114_C12'
    d.update({
      'schema':'PALLET_QUERY_VISIBILITY_C12_UNRESOLVED_V116_SOURCE_COMPLETE_V117_VISUAL_DIAGNOSIS_COMPLETE',
      'updated_UTC':stamp,'goal_status':'active','fully_solved':False,
      'latest_report':str(V/'ROOT_INTERPRETATION_KO.html'),
      'latest_report_kind':'SAVED_CURRENT_SUCCESS_FAILURE_VISUAL_DIAGNOSIS_WITH_PUBLIC_SOURCE_PROGRESS_NOT_NEW_MODEL',
      'latest_results':bind(V/'ROOT_ACTUAL_VISUAL_INTERPRETATION.json'),
      'latest_phase_seal':bind(V/'ROOT_PHASE_RESULT_SEAL.json'),
      'latest_index':bind(V/'ARTIFACT_INDEX_PHASE_RESULTS.json'),
      'latest_source_report':bind(S/'ROOT_ACTUAL_EXECUTION_AND_FIXED_VISUAL_DECISION.json'),
      'latest_source_report_QA':bind(S/'independent_source_visual_review/OBSERVATIONS_AND_RECOMMENDATION.json'),
      'latest_source_phase_seal':bind(S/'ROOT_PHASE_RESULT_SEAL.json'),
      'V116_state':'COMPLETE_448_ATTEMPTS_62_CONDITIONAL_AUX_PAIRS_386_FAILURES_U_INDEPENDENT_REPLAY_AND_FIXED_VISUAL_REVIEW_NO_FIT',
      'V116_actual_execution_and_decision':bind(S/'ROOT_ACTUAL_EXECUTION_AND_FIXED_VISUAL_DECISION.json'),
      'V116_counts':{'attempts':448,'accepted_pairs':62,'failed_U_pairs':386,'occurrences':924,'D':90,'E':62,'U':772,'all_accepted_PALLOX':True,'LOCO_accepted':0},
      'V116_conditional_AUX_admission':True,'V116_new_features_or_fit':0,
      'V117_state':'COMPLETE_FIXED9_SAVED_CROP_VISUAL_DIAGNOSIS_AND_ROOT_INTERPRETATION',
      'V117_root_interpretation':bind(V/'ROOT_ACTUAL_VISUAL_INTERPRETATION.json'),
      'V117_report_QA':bind(V/'ROOT_REPORT_VISUAL_QA.json'),
      'V117_new_model_fit_private_model_selection_PnP':0,
      'active_work':str(S),
      'active_work_state':'V117_USER_VISUAL_REQUEST_COMPLETE; V116_SOURCE_READY_FOR_PROSPECTIVE_SOURCE_ONLY_FEATURE_HEAD_DESIGN_NO_FIT_STARTED',
      'next_required_actions':[
       'Continue public-source-only method work from V116 conditional AUX source; fix a prospective feature/head and source validation design before any new model fit.',
       'Evaluate absolute D/E precision/recall and U as well as cover-sham differences; retain failed U and disclose PALLOX-only accepted coverage and paste artifacts.',
       'Current223 and V112 remain excluded from training, normalizers, threshold, checkpoint or arm choice. Geometry self route stays separate; no q/L correction or PnP.'
      ],
      'latest_user_question_scope':'Success/failure actual visual comparison, common patterns, possible remedies and assessment of goal difficulty',
      'previous_goal_turn_classification':'PROGRESS_ACTUAL_PUBLIC_SOURCE_CONSTRUCTION_AND_USER_VISUAL_DIAGNOSIS_COMPLETED_GOAL_UNSOLVED',
      'previous_parent_pointers':str(backup),'no_user_approval_pending':True
    })
    assert d['latest_current_evaluation']==latest_current
    write(path,d)
print(json.dumps({'completed':True,'report':str(V/'ROOT_INTERPRETATION_KO.html'),'visual_phase':bind(V/'ROOT_PHASE_RESULT_SEAL.json'),'source_phase':bind(S/'ROOT_PHASE_RESULT_SEAL.json'),'all_links_checked':len(links.links),'goal':'active and unresolved'},ensure_ascii=False))
