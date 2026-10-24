from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import json,hashlib,datetime
V=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_separated_frozen_current_benchmark_20261024_v156')
def load(p):return json.loads(p.read_text())
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
s=load(V/'current_result_summary_agent/CURRENT_RESULTS_SUMMARY_KO.json');d=load(V/'ROOT_RESULTS_KO.json');phase=load(V/'ROOT_PHASE_RESULT.json');qa=load(V/'ROOT_REPORT_VISUAL_QA.json');visual=load(V/'ROOT_ACTUAL_FIXED_CROP_VISUAL_QA.json');html=(V/'ROOT_READOUT_KO.html').read_text()
assert d['completed'] and d['PASS'] and d['stage_complete'] and not d['draft_only']
assert d['primary']['cohort']==s['primary_cohort']
assert d['primary']['all7_global_and_material']==s['visibility']['paired_same_query_RGB']
assert d['same_slot_pipeline1248']==s['visibility']['same_slot_pipeline']
assert d['full_registered_labelled_DE']==s['full_registered_labelled_DE']
assert d['manual_near_D376']=={'scope':s['manual_near_D_scope'],'metrics':s['manual_near_D']}
assert d['full_inventory']=={'slots':1784,'original_available':1739,'original_input_U':45,'human_states':s['teacher_inventory'],'model_inventory':s['full_registered_model_inventory']}
assert d['auxiliary_response_diagnostic']==s['auxiliary_response_diagnostic']
q=d['quality_authority'];sq=s['secondary_fixed_reference_quality']
for k,sk in [('new_gold_known','new_primary_gold_known'),('new_gold_unknown','new_primary_gold_unknown'),('secondary_near','near'),('secondary_far','far'),('secondary_known','known'),('secondary_unknown','unknown'),('known_original_input_U','original_input_U_among_known')]:assert q[k]==sq[sk]
assert not q['physical_native_authority_certified']
for m,x in q['secondary_metrics'].items():assert all(sq['metrics']['ALL_ORIGINAL']['ALL'][m][k]==val for k,val in x.items())
expect=[r for r in s['paired_recovery_damage_ALL'] if r['cohort']=='paired_same_query_RGB' and r['baseline']=='C16' and r['truth_class'] in ('DIRECT_VISIBLE','EXTERNAL_OCCLUDED')]
assert d['paired_C16_to_candidates']==expect
assert d['source_qualification_unchanged']==s['source_qualification_unchanged']
assert d['goal_status']==phase['goal_status']=='active_unsolved'
assert d['selected_model'] is None and d['adopted_model'] is None and not d['goal_solved'] and not d['fully_solved']
assert phase['completed'] and phase['stage_complete'] and phase['execution_QA_PASS'] and not phase['fully_solved'] and not phase['source_scientific_qualification_PASS'] and phase['selected_model'] is None and phase['adopted_model'] is None
assert d['next_action_unchanged']==phase['next_action_unchanged'];n=d['next_action_unchanged'];assert not n['new_fit_authorized_by_this_report'] and n['chosen_before_current_outcomes']
c=n['commitment'];assert c['sha256']=='30c4b4f524e8d4968e0c14c70a30ab5d4d626dcf7c60b6cfac38f26320e9f3d0' and bind(Path(c['path']))==c
comm=load(Path(c['path']));assert n['original_scope'] in json.dumps(comm,ensure_ascii=False) and n['private_not_used_for_next_training'] in json.dumps(comm,ensure_ascii=False)
assert phase['all_original_slots']==1784 and phase['input_U']==45 and phase['primary_paired_N']==1237
assert phase['new_training_encoder_R0_PnP_or_q_L_correction']==0 and d['actual_executions']['new_training_encoder_initializer_PnP_or_q_L_correction_calls']==0
assert d['actual_executions']['new_candidate_head_calls_including_repeat']==8 and d['actual_executions']['original_C16_parity_head_calls']==2 and d['actual_executions']['all_jobs_terminal']
assert d['access']['phase_current_head_inference_from_saved_feature_packet'] and d['access']['phase_original_RGB_reads_only_for_fixed_case_visualization']
assert not d['access']['private_current223_training_calibration_threshold_checkpoint_arm_or_architecture_selection']
for x in d['inputs']:assert bind(Path(x['path']))==x
for x in d['independent_evidence'].values():assert bind(Path(x['path']))==x
for key in ('results','report','report_QA','prediction','evaluation','crop_render','root_crop_review'):assert bind(Path(phase[key]['path']))==phase[key]
assert qa['completed'] and qa['PASS'] and qa['root_actually_viewed_all3_scientific_PNGs_and_all8_crop_panels'] and not qa['HTML_browser_preview_performed']
assert visual['completed'] and visual['PASS'] and visual['all8_actually_viewed'] and visual['selected_case_ids_unchanged'] and not visual['root_display_edited_original_input_pixels']
assert len(visual['reviewed_panels'])==8 and all(x['actually_viewed_by_root'] for x in visual['reviewed_panels'])
class Parse(HTMLParser):
 def __init__(self):super().__init__();self.links=[];self.tables=[];self.table=None;self.row=None;self.cell=None;self.text=[]
 def handle_starttag(self,t,a):
  for k,v in a:
   if k in ('href','src'):self.links.append((t,k,v))
  if t=='table':self.table=[]
  elif t=='tr' and self.table is not None:self.row=[]
  elif t in ('td','th') and self.row is not None:self.cell=[]
 def handle_data(self,x):
  self.text.append(x)
  if self.cell is not None:self.cell.append(x)
 def handle_endtag(self,t):
  if t in ('td','th') and self.cell is not None:self.row.append(''.join(self.cell).strip());self.cell=None
  elif t=='tr' and self.row is not None:self.table.append(self.row);self.row=None
  elif t=='table' and self.table is not None:self.tables.append(self.table);self.table=None
p=Parse();p.feed(html);assert len(p.tables)==6
pct=lambda x:'—' if x is None else f'{100*x:.1f}%'
for ti,group in enumerate(('ALL','WOOD_PRIMARY105','SQUARE118')):
 expected=[]
 for m,x in s['visibility']['paired_same_query_RGB'][group].items():expected.append([m,' / '.join(map(str,x['CM_rows_D_E_columns_D_E_U'][0])),' / '.join(map(str,x['CM_rows_D_E_columns_D_E_U'][1])),*[pct(x[k]) for k in ('D_precision','D_recall','E_precision','E_recall','U_rate')]])
 assert p.tables[ti][1:]==expected
near=[]
for m,x in s['manual_near_D']['ALL'].items():near.append([m,str(x['pred_visible']),str(x['pred_binary_hidden']),str(x['unresolved']),str(s['manual_near_D']['WOOD_PRIMARY105'][m]['pred_binary_hidden']),str(s['manual_near_D']['SQUARE118'][m]['pred_binary_hidden'])])
assert p.tables[3][1:]==near
for row,m in zip(p.tables[4][1:],('C35','C36','C37','C38')):
 x=q['secondary_metrics'][m];cm=x['CM_rows_far_near_columns_far_near_U'];assert row==[m+('*' if m=='C35' else ''),f'{cm[0][0]}/39',f'{cm[1][1]}/376',pct(x['far_precision']),str(x['U_known'])]
assert [int(x[1]) for x in p.tables[5][1:]]==[1168,92,136,24,356,8]
refs=[]
for tag,attr,raw in p.links:
 u=urlsplit(raw)
 if u.scheme or raw.startswith('#'):continue
 target=(V/unquote(u.path)).resolve();assert '_v112' not in str(target).lower() and target.exists();refs.append({'tag':tag,'attribute':attr,'reference':raw,'path':str(target),'exists':True})
imgs=[x for x in refs if x['attribute']=='src'];assert len(imgs)==11
plotpaths={x['path'] for x in d['visual_status']['plots']};panelpaths={x['panel']['path'] for x in visual['reviewed_panels']};assert {x['path'] for x in imgs}==plotpaths|panelpaths
text=' '.join(p.text)
assert 'crop 시각화 복구 대기' not in text and '검토용 초안' not in text
assert 'blind test' in text and 'legacy' in text and 'CAL' in text and '중첩' in text
assert '0개' in text and '미확정1784개' in text and '물리 native·pixel gold' in text
assert '현재223 결과로 다음 구조·손실·문턱·checkpoint를 고르지 않습니다.' in text
out=V/'ROOT_FINAL_INDEPENDENT_REPORT_REVIEW.json'
x={'completed':True,'PASS':True,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'schema':'V156_FINAL_STORED_REPORT_INDEPENDENT_REVIEW_V1','results':bind(V/'ROOT_RESULTS_KO.json'),'HTML':bind(V/'ROOT_READOUT_KO.html'),'phase':bind(V/'ROOT_PHASE_RESULT.json'),'root_report_QA':bind(V/'ROOT_REPORT_VISUAL_QA.json'),'root_actual_crop_visual_QA':bind(V/'ROOT_ACTUAL_FIXED_CROP_VISUAL_QA.json'),'summary_numeric_and_figure_review':bind(V/'independent_saved_evaluation_review/SUMMARY_AND_FIGURE_REVIEW.json'),'independent_saved_evaluation_review':bind(V/'independent_saved_evaluation_review/REVIEW.json'),'legacy_metadata_scope':bind(V/'independent_saved_evaluation_review/LEGACY_COMPARISON_SCOPE_APPENDIX.json'),'review_code':bind(Path(__file__)),'checks':{'all7_primary_and_material_CM_precision_recall_U_exact_reuse':True,'same_slot1248_distinct_primary1237_full1784_inventory_exact':True,'manual_near376_reference_conditional_exact':True,'secondary415_reference_only_primary_gold0_all1784unknown':True,'paired_C16_candidate_transitions_exact':True,'source_failed_selectednull_no_adoption_goalactive_unsolved':True,'stage_execution_complete_does_not_imply_goal_success':True,'root_next_action_exact_precurrent_V155_commitment_unchanged':True,'no_new_fit_encoder_R0_PnP_coordinate_or_threshold_selection':True,'legacy_CAL_overlap_and_development_exposure_disclosed':True,'shared_vs_independent_different_budget_no_branch_only_causal_claim':True,'all3plot_and8panel_local_links_exist':True,'all6HTML_tables_match_verified_aggregates':True,'root_actual8panel_view_receipt_preserved':True,'earlier_crop_failure_and_fixed_case_reuse_documented':True,'no_stale_draft_or_crop_pending_banner':True},'links':refs,'actual_view_scope':{'summary3plots_previously_actually_viewed_by_this_reviewer':True,'original8RGB_crop_panels_viewed_by_root_per_bound_receipt':True,'new_original_RGB_or_crop_body_reads_by_this_final_review':0,'browser_screenshot_claimed':False},'scope':'Final reports, metadata receipts, existing verified aggregate summaries only. No raw label/score re-evaluation, original RGB/cache/feature values/weights/models or forbidden dataset access. Panel links checked by existence only; original crop pixel hashes and visual observations are attributed to renderer/root receipts.','material_issues':[],'selected_model':None,'goal_solved':False}
out.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');print(json.dumps(bind(out)))
