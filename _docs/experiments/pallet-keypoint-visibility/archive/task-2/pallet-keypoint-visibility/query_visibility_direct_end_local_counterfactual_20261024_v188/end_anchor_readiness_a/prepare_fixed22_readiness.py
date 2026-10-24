"""Existing source code/JSON metadata only. Never opens any bound image/array asset."""
from pathlib import Path
from collections import Counter
import hashlib,json,os,stat,datetime

H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
S={
 'scope':V/'ROOT_SCOPE_BEFORE_DIRECT_END_SOURCE_CODE_AND_METADATA_PREPARATION.json',
 'regions':P/'query_visibility_train_native_part_witness_20261024_v163/reviewer_a/part_region_pilot/ALL22_COARSE_PART_REGION_OBSERVATIONS.jsonl',
 'plan':P/'query_visibility_native_part_counterfactual_20261024_v164/generator_preparation/FIXED22_FRAME_INPUTS_AND_66_ATTEMPTS_BEFORE_PIXELS.json',
 'proofs':P/'query_visibility_native_part_counterfactual_20261024_v164/source_part_pixels_v1/ALL66_PART_PIXEL_INPUT_PROOFS.jsonl',
 'grid':P/'query_visibility_public_part_supervision_readiness_20261024_v185/local96_region_intersection_b/actual_intersections_v1/ALL66_ORIGINAL_GRID_COARSE_REGION_INTERSECTIONS.jsonl',
 'mapping':P/'query_visibility_native_part_supervision_control_20261024_v166/controlled_fit_preparation/PART_NATIVE_MAPPING_COMPLETE.json',
 'inputs':P/'query_visibility_native_part_supervision_control_20261024_v166/controlled_fit_preparation/PART1144_INPUT_METADATA.jsonl',
 'weak':P/'query_visibility_native_part_supervision_control_20261024_v166/controlled_fit_preparation/PART1144_SEPARATE_NATIVE_WEAK_SUPERVISION.jsonl',
 'statuses':P/'query_visibility_native_part_supervision_control_20261024_v166/controlled_fit_preparation/ALL198_VARIANT_STATUSES_UNCHANGED.jsonl',
 'evidence':P/'query_visibility_train_native_part_witness_20261024_v163/evidence_recovered_v3/BLIND_CROSS40_ALL_REVIEWERS.json',
 'weak_instance':P/'query_visibility_source_selected_part_correspondence_20261024_v187/selected_instance_metadata_diagnostic_v2_nested_R/ALL185_FRAME_INSTANCE_ASSOCIATIONS.jsonl',
 'constructive_feasibility':P/'query_visibility_source_selected_part_correspondence_20261024_v187/witness_definition_b/STATIC_CONSTRUCTIVE_END_CF_FEASIBILITY.json',
 'constructive_code':P/'query_visibility_spatial_stencil_control_20261023_v127/registered_native_teacher_feasibility/build_registered_pallet.py',
 'crop_code':P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py',
 'generator_code':P/'query_visibility_native_part_counterfactual_20261024_v164/generator_preparation/build_native_part_counterfactual_pixels.py',
 'weak_trainer_code':P/'query_visibility_native_part_supervision_control_20261024_v166/controlled_fit_preparation/train_native_part_supervision_control.py'}

def require(v,s):
 if not bool(v):raise AssertionError(s)
def safe(path):
 p=Path(path);require(p.is_absolute()and '..'not in p.parts and P in p.parents,'Explicit project path')
 require(all('v112'not in x.casefold()for x in p.parts),'Forbidden before stat/open')
 require(p.suffix in {'.json','.jsonl','.py','.md','.txt'},'Text/code metadata only');return p
def opened(path):
 p=safe(path);fd=os.open(P,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  parts=p.relative_to(P).parts
  for part in parts[:-1]:
   new=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=new
  final=os.open(parts[-1],os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 require(stat.S_ISREG(os.fstat(final).st_mode),'Regular text file only');return os.fdopen(final,'rb')
def content(path):
 with opened(path)as f:return f.read()
def bind(path):return {'path':str(safe(path)),'sha256':hashlib.sha256(content(path)).hexdigest()}
def read(path):return json.loads(content(path))
def jl(path):return [json.loads(s)for s in content(path).splitlines()if s.strip()]
def write(name,obj):
 with (H/name).open('x')as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def writejl(name,rows):
 with (H/name).open('x')as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def count(rows,key):return dict(Counter(str(r[key])for r in rows))

def main():
 scope=read(S['scope']);regions=jl(S['regions']);source=read(S['plan']);proofs=jl(S['proofs']);grid=jl(S['grid'])
 mapping=read(S['mapping']);inputs=jl(S['inputs']);weak=jl(S['weak']);statuses=jl(S['statuses']);evidence=read(S['evidence'])
 associations={r['frame_index']:r for r in jl(S['weak_instance'])};images={r['case_id']:r for r in evidence['cases']}
 require(len(regions)==22 and len(source['attempts'])==len(proofs)==len(grid)==66,'Fixed22/66')
 require(len(inputs)==len(weak)==1144 and len(statuses)==198 and mapping['actual_occurrences']==1144,'Original1144/198')
 require(len({r['case_id']for r in regions})==22 and len({r['pair_id']for r in proofs})==66,'Identity uniqueness')
 require(Counter(r['region_status']for r in regions)=={'SUPPORTED':17,'UNKNOWN':5},'Original17/5')
 frames={r['frame_index']:r for r in source['frames']};require(len(frames)==21,'Original21 frames')
 cases={r['case_id']:r for r in regions};pf={r['pair_id']:r for r in proofs};gf={r['pair_id']:r for r in grid}
 require(all(r['split']=='SOURCE_DEVELOPMENT'for r in regions+inputs+statuses),'Source TRAIN only')
 require(sum(not r['query_available']for r in inputs)==178,'Original full-row U178')
 payloads=[r for r in statuses if r['constructed_payload']];missing=[r for r in statuses if not r['constructed_payload']]
 require(len(payloads)==143 and len(missing)==55 and Counter(r['native_slot']for r in inputs)=={k:143 for k in range(8)},'All payload/native rows includingU')
 for i,(r,w)in enumerate(zip(inputs,weak)):
  require(r['occurrence_index']==w['occurrence_index']==i and w['input_metadata']==r,'Exact old weak/input join')
  require(r['native_visibility_target']==r['point_quality_target']==r['owner_target']==-1,'Input packet targets untouched')
  require(w['native_visibility_target']==w['point_quality_target']==w['owner_target']==-1,'Separate oldweak only')
 known=[w for w in weak if w['part_native_weak_target']>=0]
 require(len(known)==92 and Counter(w['part_native_weak_target']for w in known)=={0:46,1:46},'Existing weak46+46')
 require(all(w['input_metadata']['native_slot']==w['selected_anchor_native_slot']and w['input_metadata']['pair_pixel_proof_PASS']and w['input_metadata']['variant']in {'COVER','ADJACENT_SHAM'}for w in known),'Only selected passed COVER/SHAM')
 require(all((w['part_native_weak_target']==1)==(w['input_metadata']['variant']=='COVER')for w in known),'Old E1/D0 encoding')
 original_slots=[]
 for fi,f in frames.items():
  require(f['split']=='SOURCE_DEVELOPMENT'and len(f['query_xy'])==len(f['L'])==8,'All8 original source inputs')
  for k in range(8):
   original_slots.append({'frame_index':fi,'native_slot':k,'original_global_index':f['global_index'][k],'query_id':f['query_id'][k],'query_xy':f['query_xy'][k],'L':f['L'][k],'original_query_available':f['native_continuous_valid'][k],'original_flags':{x:f[x][k]for x in ['feature_constructed','native_continuous_valid','nearest_valid','local_central4_available','context_available']},'raw_factual96_packet':f['existing_factual96_packet'],'original_fact':f['facts'][k]})
 slot_index={(r['frame_index'],r['native_slot']):r for r in original_slots};plan_cases=[]
 for r in regions:
  fi,k=r['frame_index'],r['native_slot'];f=frames[fi];e=images[r['case_id']];original=slot_index[(fi,k)]
  require(r['query_xy']==e['query_xy']==original['query_xy']and r['L']==e['L']==original['L']and r['query_id']==original['query_id'],'Original qL identity')
  require(r['selected_gt_instance_index']==e['selected_gt_instance_index'],'Inherited selectedGT unchanged')
  a=associations[fi]['diagnostic_association'];matches=None if a['selected_obj_id']!=1 else a['selected_gt_instance_index']==r['selected_gt_instance_index']
  inherited_images={x:e['images'][x]for x in ['card','nominal_ref_diagnostic','raw96','valid96']}
  plan_cases.append({'case_id':r['case_id'],'original_global_index':r['global_index'],'query_id':r['query_id'],'frame_index':fi,'source_id':r['source_id'],'split':r['split'],'native_slot':k,'native_name':r['native_name'],'original_query_xy':r['query_xy'],'original_L':r['L'],'original_query_available':original['original_query_available'],'original_flags':original['original_flags'],
   'original_selected_gt_instance_index':r['selected_gt_instance_index'],'selected_instance_authority':'Inherited V163 nominal selected-GT context plus assistant weak structural observations, not certified initializer-to-GT physical identity.',
   'V187_weak_bbox_diagnostic':{'status':a['status'],'selected_gt_instance_index':a['selected_gt_instance_index'],'selected_obj_id':a['selected_obj_id'],'same_as_original_selected_GT_when_obj1':matches,'authority':'Separate weak diagnostic; does not override original selectedGT or choose best native permutation.'},
   'original_expected_part':r['nominal_expected_part'],'existing_coarse_region':r,'existing_nominal_diagnostic_raw_xyxy':e['nominal_reference_diagnostic_raw_xyxy'],'existing_nominal_reference_xy':e['reference_xy'],
   'source_canonical_RGB_NPY':f['canonical_RGB_NPY'],'source_decoded_RGB_sha256_inherited':f['decoded_RGB_sha256'],'original_factual96_packet':f['existing_factual96_packet'],'original_selected_RGB96_sha256':e['RGB96_HWC_sha256'],'original_selected_valid96_sha256':e['valid96_sha256'],'existing_display_images_inherited_no_new_hash':inherited_images,
   'endpoint_locality_readiness':{'coarse_candidate_neighborhood_available':r['region_status']=='SUPPORTED','finite_end_enclosures':None,'all_plausible_endpoint_candidates_enclosed':'UNKNOWN_NOT_YET_OBSERVED','disjoint_visible_surrounding_same_part_witness':None,'material_mask_or_physical_corner_gold':False,'new_labels_or_training_admission':False},
   'fixed_original_attempt_ids':[a['pair_id']for a in source['attempts']if a['case_id']==r['case_id']]})
 attempts=[]
 for a in source['attempts']:
  q=pf[a['pair_id']];g=gf[a['pair_id']];r=cases[a['case_id']]
  require(q['case_id']==g['case_id']==a['case_id']and q['region']==r==a['region'],'Exact case/attempt region provenance')
  require(q['frame_index']==a['frame_index']==r['frame_index']and q['native_slot']==a['native_slot']==r['native_slot'],'Original attempt native/frame')
  attempts.append({'pair_id':a['pair_id'],'case_id':a['case_id'],'frame_index':a['frame_index'],'native_slot':a['native_slot'],'donor_id':a['donor_id'],'donor_metadata_and_asset_bindings_inherited':a['donor'],'selection_sha256':a['selection_sha256'],'original_PASS':q['PASS'],'original_constructed_triplet':q['constructed'],'original_reason':q['reason'],'variant_bindings_or_missing':q['variants'],'coarse_grid_geometry_metadata':g['geometry'],'coarse_grid_geometry_status':g['geometry_status'],'original_pixel_proof_metadata':q.get('proof'),'original_proof_detail_metadata':q.get('proof_detail'),'new_end_only_control_proof':'NOT_CREATED','new_training_admission':False})
 compact_frames=[{'frame_index':fi,'source_id':f['source_id'],'split':f['split'],'raw_hw':f['raw_hw'],'original_q8':f['query_xy'],'original_L8':f['L'],'original_global_indices':f['global_index'],'original_query_ids':f['query_id'],'original_available8':f['native_continuous_valid'],'canonical_RGB_NPY':f['canonical_RGB_NPY'],'original_factual96_packet':f['existing_factual96_packet'],'original_context_box_xyxy':f['context_box_xyxy']}for fi,f in frames.items()]
 population={'completed':True,'metadata_only':True,'scope':bind(S['scope']),'frames21':compact_frames,'original_slots168':original_slots,'cases22':plan_cases,'attempts66':attempts,'donor_catalog_binding_inherited':source['donor_records'],'all_native8_input_rows_preserved':True,'new_labels_or_training_admission':False}
 write('SOURCE_ONLY_POPULATION22_CASES66_ATTEMPTS.json',population)
 observation={'completed':True,'prepared_only':True,'scope':bind(S['scope']),'source_population':bind(H/'SOURCE_ONLY_POPULATION22_CASES66_ATTEMPTS.json'),'case_order':[r['case_id']for r in plan_cases],'cases22':plan_cases,
  'observation_contract':{'authority':'Post-historical assistant weak source RGB observation, not new blind vote/human gold/mesh certification. Old17/5 and weak92 remain immutable.','fields':['case_id','candidate_end_enclosures_all_alternatives_xyxy_fullRGB','selected_instance_and_material_part_binding_status_and_reason','surrounding_same_part_visible_witness_xyxy_fullRGB_or_UNKNOWN','uncertainty_and_other_plausible_end_candidates','actually_viewed_image_bindings','original_qL_native_flags_unchanged'],
   'locality':'Conservatively enclose a finite observed anatomical END location/neighborhood, recording all plausible candidates; distinguish it from broad deck/runner visibility. Do not set a corrected corner xy or center/size by q distance or nominal ref. Preserve UNKNOWN if topology is unresolved.','surrounding':'Identify separately visible material of the SAME selected physical component outside the end enclosure and within the original sampled crop when support is claimed. Do not silently use an adjacent block, a different board or another pallet.','display':'Use existing original CARD/raw96/valid and separately marked old nominal224 diagnostic; its raw offset permits coordinate conversion, not GT-centered model input. No new crop centering or q/L change.','sampling':'V90 F64 four-tap bilinear, zero padding, uint8 rounding is original RGB authority; nearest is display zoom/query validity only. Positive raw-contributor support must later be checked, not inferred from box overlap.'},
  'actual_pixel_reads_before_root_entry':0,'new_training_admission':False}
 write('FIXED22_SOURCE_OBSERVATION_PLAN_BEFORE_PIXELS.json',observation)
 allowed=[]
 for r in plan_cases:
  for key,b in r['existing_display_images_inherited_no_new_hash'].items():allowed.append({'case_id':r['case_id'],'kind':key,**b})
 write('ROOT_ACTUAL_FIXED22_SOURCE_OBSERVATION_ENTRY_TEMPLATE_APPROVED_FALSE.json',{'approved_actual_fixed22_source_RGB_weak_observation':False,'source_TRAIN_only':True,'scope':bind(S['scope']),'plan':bind(H/'FIXED22_SOURCE_OBSERVATION_PLAN_BEFORE_PIXELS.json'),'allowed_existing_image_bindings':allowed,'original_rawRGB_NPY_and_factual96_NPZ_bindings':'Inherited in plan for later explicitly scoped display if needed; not opened/hashed by readiness.','raw_RGB_NPY_NPZ_or_header_body_reads':False,'GT_mask_depth_mesh_model_score_eval_or_private_held_VAL_reads':False,'qL_native_or_original_annotation_changes':False,'new_training_admission':False})
 writejl('ALL198_ORIGINAL_VARIANT_STATUSES_UNCHANGED.jsonl',statuses)
 rows=[{'original_input_metadata':r,'existing_V166_part_native_weak_target':w['part_native_weak_target'],'existing_V166_weak_target_authority':w['target_semantics'],'existing_V166_root_design':w['root_fit_design'],'existing_V166_loss_admitted_under_that_design':w['part_native_loss_admitted_under_root_design'],'new_V188_direct_end_target':-1,'new_V188_support_target':[-1,-1],'new_V188_training_admission':False}for r,w in zip(inputs,weak)]
 writejl('ALL1144_ORIGINAL_INPUT_AND_EXISTING_WEAK_NATIVE_PROVENANCE.jsonl',rows)
 bynative=[]
 for k in range(8):
  cr=[r for r in regions if r['native_slot']==k];pr=[r for r in proofs if r['native_slot']==k];ir=[r for r in inputs if r['native_slot']==k];kw=[w for w in known if w['input_metadata']['native_slot']==k]
  bynative.append({'native_slot':k,'cases22':len(cr),'coarse_SUPPORTED':sum(r['region_status']=='SUPPORTED'for r in cr),'coarse_UNKNOWN':sum(r['region_status']=='UNKNOWN'for r in cr),'attempts66':len(pr),'pair_PASS':sum(r['PASS']for r in pr),'actual_input_rows1144':len(ir),'actual_input_U':sum(not r['query_available']for r in ir),'existing_weak_D0':sum(w['part_native_weak_target']==0 for w in kw),'existing_weak_E1':sum(w['part_native_weak_target']==1 for w in kw)})
 passed=[r for r in proofs if r['PASS']];failed=[r for r in proofs if r['region']['region_status']=='SUPPORTED'and not r['PASS']];unknown=[r for r in proofs if r['region']['region_status']=='UNKNOWN']
 require((len(passed),len(failed),len(unknown))==(46,5,15),'Original attempt partition')
 summary={'cases':22,'frames':21,'coarse_region_status':count(regions,'region_status'),'attempts':66,'passed_pairs':46,'placement_failed_pairs':5,'region_UNKNOWN_attempts':15,'actual_payloads':143,'potential_variant_statuses':198,'missing_payload_statuses':55,'potential_native_occurrences198x8':1584,'actual_native_input_rows':1144,'missing_native_occurrences55x8':440,'actual_inputU':178,'actual_input_available':966,'original_unique_frame_native_rows':168,'original_unique_inputU':sum(not r['original_query_available']for r in original_slots),'selected_case_original_inputU':sum(not r['original_query_available']for r in plan_cases),'old_weak_known':92,'old_weak_D0':46,'old_weak_E1':46,'old_weak_UNKNOWN':1052,'unique_donors_in66':len({r['donor_id']for r in proofs}),'payload_presence_by_variant':dict(Counter(r['variant']for r in payloads)),'missing_by_variant':dict(Counter(r['variant']for r in missing)),'failed_pair_ids':[r['pair_id']for r in failed],'UNKNOWN_case_ids':[r['case_id']for r in regions if r['region_status']=='UNKNOWN'],'per_native':bynative,
  'coarse_grid_only_PASS_pair_sample_overlap':{key:sum(gf[r['pair_id']]['geometry'][key]['any_valid_positive_overlap']for r in passed)for key in ['local96','central32','center4']},'V187_weak_bbox_same_originalGT':dict(Counter(str(r['V187_weak_bbox_diagnostic']['same_as_original_selected_GT_when_obj1'])for r in plan_cases))}
 write('ACTUAL_METADATA_CENSUS_AND_NATIVE8_TABLE.json',summary)
 authority={'completed':True,'source_code':bind(S['constructive_code']),'constructive_only_not_MR22':['V127 native_witness_points index0 is the declared selectedpart bound anchor. It is not original q or an inferred nearest vertex.','Indices1..9 are3axes x t=.2,.5,.8 edge points;10..36 are3faces x(.2,.5,.8)^2 points. All37 do not mean a single point or a full material component.','Three patch quads start at the anchor, axis-specific along -sign, step=min(.030,0.7*minimum selectedpart extent) metres in this base design. V180 broad support samples a registered patch, while a new finite near-anchor locality must be separately frozen.','region_teacher active witnesses always include0 and camera-facing eligible edge/face witnesses; all37 in-frame check, allactive D/E conjunction, partial/ties/self/unknown remain explicit.','V127 fixed_foreign covers broad all-three-face patch bbox plus3px; it is not end-only control evidence.'],
  'base_constructive_native_roles':[{'canonical_native_index':i,'WOOD_primary_part_id':([0,0,16,14,4,4,16,14][i]),'POLYMER_primary_part_id':([0,0,18,16,6,6,18,16][i]),'part_type':'top_deck_end'if i in [0,1,4,5]else'bottom_runner_end'}for i in range(8)],
  'MR22_authority':'17 historical assistant coarse visible expected MATERIAL-END neighborhoods;5UNKNOWN. No exact endpoint point, finite core, all-plausible endpoint set, exactmask or surrounding-samepart witness fields exist in these records. A new weak RGB witness can supply them without perfect mesh/pose/depth certification.',
  'next_actual_step':'Root approves exact22 existing image bindings; collective actual views record finite candidate enclosures/allalternatives and disjoint visible surrounding samecomponent witness orUNKNOWN. B then proves same-original-q bilinear support, opaque/front COVER and same-object disjoint SHAM, retaining every failed/UNKNOWN case.',
  'full_goal':'This real source pilot is Euro WOOD only, weak endpoint cases native0/1/4/5 and no native7case; raw payload rows still retain all8. SYNTH WOOD/SQUARE/allnative8 convention is separately fixed prospectively by B/C; this small real pilot is not full qualification.'}
 write('ENDPOINT_AND_SURROUNDING_WITNESS_AUTHORITY.json',authority)
 review={'completed':True,'PASS':True,'readiness_only':True,'code':bind(__file__),'scope':bind(S['scope']),'bindings':{k:bind(p)for k,p in S.items()},'census':summary,'outputs':{n:bind(H/n)for n in ['SOURCE_ONLY_POPULATION22_CASES66_ATTEMPTS.json','FIXED22_SOURCE_OBSERVATION_PLAN_BEFORE_PIXELS.json','ROOT_ACTUAL_FIXED22_SOURCE_OBSERVATION_ENTRY_TEMPLATE_APPROVED_FALSE.json','ALL198_ORIGINAL_VARIANT_STATUSES_UNCHANGED.jsonl','ALL1144_ORIGINAL_INPUT_AND_EXISTING_WEAK_NATIVE_PROVENANCE.jsonl','ACTUAL_METADATA_CENSUS_AND_NATIVE8_TABLE.json','ENDPOINT_AND_SURROUNDING_WITNESS_AUTHORITY.json']},'actual_new_RGB_NPY_NPZ_header_depth_mask_mesh_weights_body_or_asset_hash_reads':0,'new_source_pixel_views_or_render_forward_fit':0,'current_held_VAL_or_V182COMMON_fit_data_reads':0,'V112_access':0,'old_source_outputs_modified':False,'old_weak_labels_changed':False,'new_labels_or_admission':False,'selected_model':None,'full_goal_solved':False}
 write('REVIEW.json',review);print(json.dumps({'completed':True,'PASS':True,'review':bind(H/'REVIEW.json'),'observation_plan':review['outputs']['FIXED22_SOURCE_OBSERVATION_PLAN_BEFORE_PIXELS.json'],'false_entry':review['outputs']['ROOT_ACTUAL_FIXED22_SOURCE_OBSERVATION_ENTRY_TEMPLATE_APPROVED_FALSE.json'],'census':summary}),flush=True)

if __name__=='__main__':main()
