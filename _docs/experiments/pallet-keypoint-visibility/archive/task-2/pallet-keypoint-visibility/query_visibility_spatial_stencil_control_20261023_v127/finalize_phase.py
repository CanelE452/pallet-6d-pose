"""Record completed source experiment, keeping goal active and current C16 intact."""
from pathlib import Path
from datetime import datetime,timezone
from html.parser import HTMLParser
from urllib.parse import urlparse,unquote
import hashlib,json
V=Path(__file__).resolve().parent;P=V.parent
def read(p):return json.loads(p.read_text())
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
class Parser(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in('src','href'))
result=read(V/'ROOT_RESULTS_KO.json');report=V/'ROOT_READOUT_KO.html';parsed=Parser();parsed.feed(report.read_text());links=[]
for s in parsed.links:
 u=urlparse(s)
 if u.scheme:continue
 p=Path(unquote(u.path));p=p if p.is_absolute()else V/p
 assert p.exists(),p
 links.append(str(p))
for b in result['evidence'].values():assert bind(Path(b['path']))==b
assert bind(Path(result['figure']['path']))==result['figure']
for rel in ['saved_feature_review/REVIEW.json','saved_fit_review/REVIEW.json','report_consistency_review/REVIEW.json']:
 d=read(V/rel);assert d['PASS'] and not d.get('blockers',[])
now=datetime.now(timezone.utc).isoformat()
qa={'completed':True,'PASS':True,'created_UTC':now,'report':bind(report),'results':bind(V/'ROOT_RESULTS_KO.json'),'figure':result['figure'],
 'actual_root_visual_inspection':{'tool':'tools.view_image','summary_figure':'Viewed final C19_MATCHED_SPATIAL_COMPARISON_KO.png; repaired crowding by two-line count labels, all4panels readable; subsequent report rebuild figurebytes unchanged.','gallery':'Viewed all6raw contacts=24cases/72panels;2marked contacts and standalone CASE17_MARKED; q/r distinction readable.','public_intake':'Viewed6take contacts=24rawframes+nativebboxcrops and9-ruler contact plus individual6/16/22.','browser_render_performed':False},
 'root_actual_crop_observations':bind(V/'ROOT_C19_ACTUAL_CROP_OBSERVATIONS_KO.json'),'local_links_checked':links,'all_report_evidence_hashes_exact':True,'independent_report_consistency':bind(V/'report_consistency_review/REVIEW.json'),'new_model_fit_private_current_V112_PnP_calls_in_reporting':0,'limits':'PNG inspected; HTML staticchecks only, no browser render claim.'}
write(V/'ROOT_REPORT_VISUAL_QA.json',qa)
phase={'completed_phase':True,'goal_status':'active','fully_solved':False,'created_UTC':now,'version':'V127',
 'state':'COMPLETE_C19_MATCHED_SPATIAL_CONTROL_TRADEOFF_PUBLIC24_INTAKE_AND_NEW_REGISTERED_GEOMETRY_READY_FOR_RGB_RENDER',
 'report':bind(report),'results':bind(V/'ROOT_RESULTS_KO.json'),'report_QA':bind(V/'ROOT_REPORT_VISUAL_QA.json'),'actual_work':result['actual_work'],
 'source_control':bind(V/'runs/C19_MATCHED_STENCIL_FIXED1000/CONTROL_COMPLETE.json'),'feature_review':bind(V/'saved_feature_review/REVIEW.json'),'fit_review':bind(V/'saved_fit_review/REVIEW.json'),
 'source_outcome':{'VAL_E_qOUT_POOL_ORDERED':[229,279,483],'VAL_SHAM_D_qIN_POOL_ORDERED':[148,97,394],'both_arms_fixed_updates':1000,'joint_visibility_solution':False,'private_promotion':False,'old15_gate_PASS_claim':False},
 'new_registered_geometry':bind(V/'registered_native_teacher_feasibility/RESULT.json'),'root_geometry_review':bind(V/'ROOT_NEW_REGISTERED_GEOMETRY_REVIEW.json'),'registered_geometry_RGB_or_model_training_done':False,
 'public24':bind(V/'public_diversity_intake/FINAL_INTAKE_RECEIPT.json'),'public24_new_training_admission':False,
 'latest_current_unchanged':'V124_C16 D1137/1147 E35/90; no C19 current evaluation.',
 'next_safe_source_action':bind(V/'ROOT_NEXT_REGISTERED_RENDER_SCOPE_BEFORE_PIXELS.json'),'next_required_actions':result['next_actions_KO'],'goal_outcome':'UNSOLVED; external occlusion real transfer and target fidelity remain to be verified.','no_approval_or_genuine_blocker_pending':True}
write(V/'ROOT_PHASE_RESULT.json',phase)
for name in ['LATEST_STATUS_KO.json','ACTIVE_HANDOFF_KO.json']:
 p=P/name;d=read(p);before=json.dumps(d['latest_current_evaluation'],sort_keys=True,ensure_ascii=False)
 d.update({'schema':'PALLET_QUERY_VISIBILITY_V127_SOURCE_SPATIAL_CONTROL_COMPLETE_CURRENT_C16_UNCHANGED_UNSOLVED','updated_UTC':now,'goal_status':'active','fully_solved':False,
  'active_work':str(V),'active_work_state':phase['state'],'latest_report':str(report),'latest_report_kind':'PUBLIC_SOURCE_SPATIAL_CONTROL_AND_REGISTERED_GEOMETRY','latest_index':bind(V/'ROOT_RESULTS_KO.json'),'latest_phase_seal':bind(V/'ROOT_PHASE_RESULT.json'),'latest_source_phase_seal':bind(V/'ROOT_PHASE_RESULT.json'),'latest_source_report':bind(V/'ROOT_RESULTS_KO.json'),'latest_source_report_QA':bind(V/'ROOT_REPORT_VISUAL_QA.json'),
  'latest_source_training':{'version':'V127_C19_MATCHED_TWO_ARM_DIAGNOSTIC','control':phase['source_control'],'fixed_updates_each':1000,'selected_arm':None,'arms':['POOLED9_REPEAT8','ORDERED_OUTER8'],'trainable_parameters_each':418433,'current_inference_done':False,'private_model_promotion':False,'old15_source_regression_not_claimed':True},
  'known_limit':'Matchedstencil:VAL Eout229→279/483,butSHAMvisible148→97/394. Nojointsolution. CurrentC16 remainsD1137/1147,E35/90. Public24 screened9promising,notadmitted. TwoNEWnative8registeredproceduralassets passedgeometry/raychecks only; RGB/render/native-supportreviewandrealtransferremain.',
  'model_candidate':{'state':'C19_BOTH_SOURCE_ONLY_NOT_ADMITTED;NEW_REGISTERED_GEOMETRY_NOT_YET_RGB_TEACHER','latest_current_model':'V124_C16','latest_source_models':['V127_C19_POOLED9_REPEAT8','V127_C19_ORDERED_OUTER8'],'current_used_for_model_choice':False,'reliable_visibility_solved':False},
  'next_required_actions':result['next_actions_KO'],'next_safe_source_action':phase['next_safe_source_action'],'new_native_registered_geometry':phase['new_registered_geometry'],'new_native_registered_geometry_review':phase['root_geometry_review'],'latest_public_intake':phase['public24'],'V127_phase':bind(V/'ROOT_PHASE_RESULT.json'),'V127_new_current_or_V112_or_PnP':0,'previous_goal_turn_classification':'progress','no_user_approval_pending':True})
 assert json.dumps(d['latest_current_evaluation'],sort_keys=True,ensure_ascii=False)==before
 write(p,d)
print(json.dumps({'phase':bind(V/'ROOT_PHASE_RESULT.json'),'report_QA':bind(V/'ROOT_REPORT_VISUAL_QA.json'),'current_preserved':True,'goal_status':'active','fully_solved':False,'local_links_checked':len(links)},ensure_ascii=False))
