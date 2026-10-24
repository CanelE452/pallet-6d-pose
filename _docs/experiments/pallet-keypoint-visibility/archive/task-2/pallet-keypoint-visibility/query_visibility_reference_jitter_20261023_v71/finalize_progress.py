from pathlib import Path
import base64,datetime,hashlib,json
from html.parser import HTMLParser

V=Path(__file__).resolve().parent
P=V.parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(4*1024*1024),b''):h.update(part)
    return h.hexdigest()
def bind(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def read(p):return json.loads(p.read_text())
def write(p,d):
    assert not p.exists(),p
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

A=read(V/'source_audit/ACTUAL_PILOT_INDEPENDENT_AUDIT.json')
Y=read(V/'source_audit/INDEPENDENT_YCB_CUE_REVIEW.json')
assert A['status']=='PASS' and Y['mismatches']==0
assert A['GT_input_leakage_arrays']==0 and A['actual_rows']==260
assert A['contradictory_supervised_D_E_collisions']==[]
assert len(A['semantic_state_collisions'])==21
grid=read(V/'data/ycb_proposal_grid/GRID_RESULT.json')
assert grid['counts']['N']==1944 and grid['counts']['training']=={'U':1902,'D':42}
assert grid['counts']['availability']=={'MISS':1744,'PARTIAL':126,'FULL':74}
assert grid['by_split']['TRAIN']['training']=={'U':1274,'D':22}
assert grid['by_split']['VALIDATION']['training']=={'U':628,'D':20}
assert grid['E_count']==0 and grid['all_grid_proposals_preserved']
assert grid['input_GT_ref_state_mask_alpha_source_ID_offset_fields']==0
IG=read(V/'source_audit/INDEPENDENT_YCB_GRID_AUDIT.json')
assert IG['status']=='PASS' and IG['mismatches']==0
RR=read(V/'source_audit/PROGRESS_REPORT_REVIEW.json')
assert str(RR.get('status','')).startswith('PASS')
assert not RR.get('blockers',[]) and not RR.get('report_edits_required',False)
for b in RR['reviewed_report_bindings']:
    assert sha(b['path'])==b['sha256'], 'Report review must bind the actual current version'

class QA(HTMLParser):
    def __init__(self):super().__init__();self.images=[];self.links=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='img':self.images.append(attrs['src'])
        if tag=='a':self.links.append(attrs['href'])
q=QA();q.feed((V/'PROGRESS_KO.html').read_text())
pngs=[V/'data/synth_reference_jitter/BUILD_V2/ACTUAL_FIXED_REFERENCE_COUNTEREXAMPLES.png',V/'data/r2b_cue_review/EIGHT_RECORD_WOOD_CUE_RAW_MARKED_CONTACT_SHEET.png']
assert len(q.images)==2
for encoded,png in zip(q.images,pngs):assert hashlib.sha256(base64.b64decode(encoded.split(',',1)[1])).hexdigest()==sha(png)
for link in q.links:assert (V/link).resolve().is_file(),link
qa={'schema':'V71_PROGRESS_REPORT_QA','status':'PASS','first_page_root_viewed':True,'root_actual_source_images_viewed':['generated_Synthesis_D_E_PARTIAL_neutral_and_MISS','R2B8raw_marked','YCB12raw64_mask_controlled_3pages','YCBTRAIN8_VAL4overview'],'embedded_images_original_bytes_identical':2,'relative_links_exist':len(q.links),'report':bind(V/'PROGRESS_KO.html'),'first_page':bind(V/'FIRST_PAGE.png'),'new_forward_fit_current_PnP':0}
write(V/'source_audit/REPORT_QA.json',qa)
files=[V/x for x in ['PROGRESS_KO.html','PROGRESS_KO.json','FIRST_PAGE.png','build_progress_report.py','finalize_progress.py','source_audit/REPORT_QA.json','source_audit/ACTUAL_PILOT_INDEPENDENT_AUDIT.json','source_audit/REFERENCE_LABEL_EXECUTION_REVIEW.json','source_audit/INDEPENDENT_YCB_CUE_REVIEW.json','source_audit/ROOT_YCB_CUE_CONDITIONAL_TEACHER_REVIEW.json','data/synth_reference_jitter/PILOT_PROTOCOL.json','data/synth_reference_jitter/FAILED_V1_VALIDATION.json','data/synth_reference_jitter/build_actual_pilot.py','data/synth_reference_jitter/build_actual_pilot_v2.py','data/synth_reference_jitter/render_actual_pilot.py','data/synth_reference_jitter/BUILD_V2/PILOT_RESULT.json','data/synth_reference_jitter/BUILD_V2/EXECUTION_ENTRYLOCK.json','data/synth_reference_jitter/BUILD_V2/ROWS.jsonl','data/synth_reference_jitter/BUILD_V2/MODEL_INPUTS_ONLY.npz','data/synth_reference_jitter/BUILD_V2/GT_ONLY.npz','data/synth_reference_jitter/BUILD_V2/INTERVENTION_CERTIFICATES.json','data/synth_reference_jitter/BUILD_V2/INPUT_COLLISION_AUDIT.json','data/synth_reference_jitter/BUILD_V2/V1_RASTER_REPLAY_PARITY.json','data/ycb_cue_review/FROZEN_AI_CUE_PROPOSALS_GT_ONLY.json','data/ycb_cue_review/CUE_GEOMETRY_SEAL.json','data/ycb_cue_review/FROZEN_CUE_OWNERSHIP_SANITY/SEAL.json','data/ycb_cue_review/FROZEN_CUE_OWNERSHIP_SANITY/SANITY_REPORT.json','data/ycb_cue_review/YCB_CUE_COMPONENT_DATA_READINESS.json','data/r2b_cue_review/R2B_WOOD_CUE_PROPOSAL_LEDGER.json','data/r2b_cue_review/CUE_REVIEW_SEAL.json']]+pngs
files += sorted(p for p in (V/'data/ycb_proposal_grid').iterdir() if p.is_file())
files += [V/'source_audit/INDEPENDENT_YCB_GRID_AUDIT.json',V/'source_audit/PROGRESS_REPORT_REVIEW.json',V/'source_audit/REPORT_SCOPE_CLARIFICATION_V2.json',V/'source_audit/NEXT_PUBLIC_PALLET_INITIALIZER_FEASIBILITY.json']
bindings=[bind(p) for p in files]
cross={'schema':'V71_ACTUAL_DATA_PROGRESS_FINAL_CROSS_AUDIT','status':'PASS','goal_status':'active','fully_solved':False,'actual_data_preparation_completed':True,'training_completed':False,'actual_synth_rows':260,'actual_synth_rasters':20,'synth_label_D_E_U':[54,27,179],'synth_known_D_E_contradiction_groups':0,'synth_hidden_state_collision_groups_all_training_U':21,'original_YCB_eligible_TRAIN_D_E':0,'generic_YCB_reference_D_candidates_TRAIN_VAL':[4,3],'YCB_source_grid_rows':1944,'YCB_grid_D_E_U':[42,0,1902],'YCB_grid_independent_audit':'PASS','source_generic_native8_mapping':'UNKNOWN','R2B_AI_visible_cues':7,'R2B_U':1,'current223_RGB_GT_coordinates_scores_read':0,'current223_fit_threshold_or_checkpoint_selection':False,'new_model_forward':0,'optimizer_updates':0,'PnP':0,'independent_audits':['ACTUAL_PILOT_INDEPENDENT_AUDIT','INDEPENDENT_YCB_CUE_REVIEW','INDEPENDENT_YCB_GRID_AUDIT'],'report_review':'PASS','limitations':['New generic cue labels are conditional AI source teachers, not native8 human truth','Only3 learningSYNTHrefs/2RGB sourceframes, VAL0 for that pilot','Dense source querygrid is emulated from old approximate sourcebox/L, not actual current native8 output','New public real semantic E/independent native8/self supervision still need construction and verification'],'bindings':bindings,'created_at_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
write(V/'FINAL_CROSS_AUDIT.json',cross)
write(V/'ARTIFACT_INDEX.json',{'schema':'V71_ACTUAL_REFERENCE_QUERY_DATA_INDEX','goal_status':'active','fully_solved':False,'main_report':str(V/'PROGRESS_KO.html'),'bindings':bindings+[bind(V/'FINAL_CROSS_AUDIT.json')]})

latest=P/'LATEST_STATUS_KO.json';old=latest.read_bytes();oldsha=hashlib.sha256(old).hexdigest()
hist=V/'provenance/status_history';hist.mkdir(parents=True,exist_ok=True)
snapshot=hist/f'LATEST_STATUS_KO_{oldsha[:12]}.json';assert not snapshot.exists();snapshot.write_bytes(old)
status=json.loads(old)
status.update({'schema':'PALLET_QUERY_VISIBILITY_LATEST_STATUS_20261023_V71','latest_report':str(V/'PROGRESS_KO.html'),'latest_index':str(V/'ARTIFACT_INDEX.json'),'previous_pointer_preserved':str(snapshot),'latest_source_classification_report':str(P/'query_visibility_target_ownership_20261023_v70/RESULTS_KO.html'),'latest_source_classification_status':'V70 remainsHOLD; no new V71 model trained or current evaluated.','latest_data_preparation_result':'V71 actual260crop fixedref/jitter/opaque-cover pilot plus publicYCB12cues/sourcegrid1944inputrows andR2B7cue+1U. IndependentdataauditsPASS; no claimednewmodelimprovement/native8solution.','latest_data_preparation_report':str(V/'PROGRESS_KO.html'),'next_direction':'Keep fixed publicsource cues and input-only frozen querygrid; construct wholecue opaqueE and untouchedD controls with splitlineage preserved. Freeze semantic/availability model-loss-evaluation before newfit. Resolve deployment-native8-to-cue association and self evidence separately; no current fit/threshold/selection, no newPnP.','next_public_pallet_initializer_feasibility':str(V/'source_audit/NEXT_PUBLIC_PALLET_INITIALIZER_FEASIBILITY.json'),'V71_new_current_inference':0,'V71_new_training':0,'V71_new_PnP':0,'fully_solved':False,'goal_status':'active','updated_at_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()})
latest.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
write(V/'STATUS.json',{'schema':'V71_ACTUAL_DATA_PROGRESS_STATUS','data_preparation_completed':True,'model_training_completed':False,'goal_status':'active','fully_solved':False,'report':bind(V/'PROGRESS_KO.html'),'index':bind(V/'ARTIFACT_INDEX.json'),'final_audit':bind(V/'FINAL_CROSS_AUDIT.json'),'previous_pointer':bind(snapshot),'latest_pointer':bind(latest),'current223_new_inference':0,'new_PnP':0})
print(json.dumps({'status':bind(V/'STATUS.json'),'index':bind(V/'ARTIFACT_INDEX.json'),'audit':bind(V/'FINAL_CROSS_AUDIT.json')},ensure_ascii=False))
