"""V92 existing-source metadata only; never decode images or load feature/model arrays."""
from pathlib import Path
import ast,collections,csv,datetime,hashlib,json,math
import numpy as np
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R=Path(__file__).resolve().parent
V73=P/'query_visibility_public_pallet_loco_20261023_v73/data/intake'
V74=P/'query_visibility_loco_native_context_20261023_v74'
I=V74/'public_initializer/DECODED_RGB_V2_RUN'
def load(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def bind(p):
 p=Path(p);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
def pure_function(path,name):
 tree=ast.parse(Path(path).read_text()); fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
 mod=ast.Module(body=[fn],type_ignores=[]); ns={'np':np};exec(compile(mod,str(path),'exec'),ns);return ns[name]

def main():
 assert not (R/'EXACT_SOURCE_PATHS_AND_NATIVE_QUERY_FEASIBILITY.json').exists()
 inventory=load(V73/'ARCHIVE_INVENTORY.json'); intake=load(V73/'INTAKE_RESULT.json')
 coco=load(V73/'annotations/loco-all-v1.json')
 im={x['id']:x for x in coco['images']}; assert len(im)==5097
 pallet=[a for a in coco['annotations'] if a['category_id']==7]; assert len(pallet)==120445
 assert len({a['image_id'] for a in pallet})==4727
 categories={x['id']:x['name'] for x in coco['categories']}; assert categories[7]=='pallet'
 archives=[z for z in inventory['files'] if not z['is_directory'] and z['name'].lower().endswith(('.jpg','.jpeg'))]
 assert len(archives)==5593
 group=rows(V74/'source_groups/CONSERVATIVE_SOURCE_GROUP_MANIFEST.jsonl'); assert len(group)==5593
 groups_by_member={x['archive_member']:x for x in group};assert len(groups_by_member)==5593
 assert set(groups_by_member)=={x['name'] for x in archives}
 hashrows=list(csv.DictReader((V73/'ALL_5593_RGB_FILE_SHA256.csv').open())); assert len(hashrows)==5593
 hby={x['archive_member']:x for x in hashrows};assert set(hby)==set(groups_by_member)
 for member,g in groups_by_member.items(): assert g['jpeg_file_SHA256']==hby[member]['sha256']
 sourcecounts=collections.Counter(g['official_split'] for g in group)
 subsetcounts=collections.Counter(str(g['subset']) for g in group)
 ip=rows(I/'INPUTS_RGB_ONLY.jsonl'); prop=rows(I/'FROZEN_PROPOSALS.jsonl');assert len(ip)==len(prop)==20
 ipby={x['id']:x for x in ip};assert len(ipby)==20
 ng=P.parent/'external_ray_proxy_head_20261022_v60/head_and_roi.py'
 qb=P/'query_visibility_cue_occlusion_20261023_v72/private_evaluation/current_input_adapter.py'
 native_grid=pure_function(ng,'native_grid'); boxfn=pure_function(qb,'initial_query_box')
 saved_rows=[]; sel_all=collections.Counter(); scores=[]; candidatekeys=set(); selectednativefinite=selectedcontinuous=selectednearest=selectednativevalid=0
 for frame,x in enumerate(prop):
  inp=ip[frame];assert x['input_id']==inp['id'] and x['image_sha256']==inp['image_sha256'];h,w=inp['raw_hw']
  member=inp['officialpath_lineage'];g=groups_by_member[member];assert hby[member]['sha256']==x['image_sha256'];assert g['raw_hw']==inp['raw_hw']
  candidates=x['candidates']; ss=[c['score'] for c in candidates]
  si=int(np.argmax(ss)) if ss else None;assert x['selected_index']==si and x['no_detection']==(si is None)
  for j,c in enumerate(candidates):
   candidatekeys.add(tuple(sorted(c)));assert c['candidate_index']==j
   assert np.asarray(c['keypoints_xy']).shape==(9,2);assert np.asarray(c['keypoints_conf']).shape==(9,)
   assert np.isfinite(np.asarray(c['keypoints_xy'])).all()
  out={'frame_index':frame,'input_id':x['input_id'],'source_subset':g['subset'],'official_split':g['official_split'],'archive_member':member,'image_path':inp['image'],'image_file_exists_stat_only':Path(inp['image']).is_file(),'image_sha256':inp['image_sha256'],'raw_hw':[h,w],'candidate_count':len(candidates),'selected_index':si,'state':'NO_DETECTION_U' if si is None else 'SAVED_PREDICTED_QUERYSET','GT_native_reference':'UNKNOWN','GT_semantic_visibility':'UNKNOWN'}
  if si is not None:
   c=candidates[si]; q=np.asarray(c['keypoints_xy'],float)[:8];finite=np.isfinite(q).all(1)&~(q==-1).all(1)
   cont=finite&(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h);p=np.floor(q+.5).astype(np.int64)
   nearest=finite&(p[:,0]>=0)&(p[:,0]<w)&(p[:,1]>=0)&(p[:,1]<h);valid=cont&nearest
   _,_,L=native_grid(c['keypoints_xy'],[h,w]);box=boxfn(q,[h,w])
   out.update(selected_score=c['score'],selected_native8_finite=int(finite.sum()),selected_native8_continuous_valid=int(cont.sum()),selected_native8_nearest_valid=int(nearest.sum()),selected_native8_pack_center_valid=int(valid.sum()),derived_roi_side=L,derived_initial_query_box=None if box is None else box.tolist(),selected_predictor_box=c['box_xyxy'])
   selectednativefinite+=int(finite.sum());selectedcontinuous+=int(cont.sum());selectednearest+=int(nearest.sum());selectednativevalid+=int(valid.sum());scores.append(float(c['score']))
  saved_rows.append(out);sel_all[g['official_split']]+=si is not None
 ref_summary=load(V74/'public_reference_review/COMBINED60_REFERENCE_REVIEW_ADDENDUM.json');references=[z for f in ref_summary['frames'] for z in f['reference_proposals']];conditional=[x for x in references if x.get('conditional_aux_candidate')];aux_rows=rows(V74/'public_reference_review/COMBINED60_AUX_CANDIDATES.jsonl');assert {x['cue_refID'] for x in aux_rows}=={x['cue_refID'] for x in conditional}
 assert len(references)==28 and len(conditional)==22
 prop_shas={x['image_sha256'] for x in prop}; ref_in20=[x for x in conditional if x['image_sha256'] in prop_shas]
 ref_noq=[x for x in ref_in20 if next(z for z in prop if z['image_sha256']==x['image_sha256'])['no_detection']]
 assert all(x['native8_mapping']=='UNKNOWN' and not x['human_gold'] and not x['admitted_training_teacher'] for x in references)
 protocol=load(I/'PROTOCOL.json'); inf=load(I/'INFERENCE_RESULT.json')
 # Check metadata bindings only; do not read checkpoint/image/feature bodies.
 for b in inf['outputs']:
  if str(b['path']).endswith('.jsonl'): assert bind(b['path'])['sha256']==b['sha256']
 path_files={
  'complete_source_archive':{'path':str(V73/'dataset.zip'),'exists_stat_only':(V73/'dataset.zip').is_file(),'bytes_stat_only':(V73/'dataset.zip').stat().st_size,'previously_bound_sha256':intake['archive']['sha256'],'fresh_archive_body_hash_read':False,'full_RGB_filesystem_extraction':False},
  'archive_directory':bind(V73/'ARCHIVE_INVENTORY.json'),'all5593_file_hashes':bind(V73/'ALL_5593_RGB_FILE_SHA256.csv'),'official_all_COCO':bind(V73/'annotations/loco-all-v1.json'),'official_schema_audit':bind(V73/'ACTUAL_COCO_SCHEMA_AUDIT.json'),'intake_result':bind(V73/'INTAKE_RESULT.json'),'intake_seal':bind(V73/'INTAKE_SEAL.json'),
  'source_group_manifest':bind(V74/'source_groups/CONSERVATIVE_SOURCE_GROUP_MANIFEST.jsonl'),'source_group_freeze':bind(V74/'source_groups/SOURCE_GROUP_FREEZE.json'),'official_split_evidence':bind(V74/'source_groups/OFFICIAL_ENVIRONMENT_SPLIT_EVIDENCE.json'),
  'saved_query_inputs':bind(I/'INPUTS_RGB_ONLY.jsonl'),'saved_query_proposals':bind(I/'FROZEN_PROPOSALS.jsonl'),'saved_query_result':bind(I/'INFERENCE_RESULT.json'),'saved_query_protocol':bind(I/'PROTOCOL.json'),'saved_query_producer':bind(I/'infer_frozen_initializer.py'),'saved_query_entrylock':bind(I/'RUN_ENTRYLOCK.json'),'coordinate_audit':bind(V74/'public_initializer/OUTPUT_AUDIT/COORDINATE_AND_IOU_INTERPRETATION.json'),'prior_bbox_diagnostic_only':bind(V74/'public_initializer/OUTPUT_AUDIT/OUTPUT_AUDIT_RESULT.json'),
  'prior_physical_cue_metadata':bind(V74/'public_reference_review/COMBINED60_AUX_CANDIDATES.jsonl'),'prior_cue_summary':bind(V74/'public_reference_review/COMBINED60_REFERENCE_REVIEW_ADDENDUM.json'),'native_grid_source_code':bind(ng),'initial_query_box_source_code':bind(qb),'license':bind(V73/'LICENSE_CC0_OFFICIAL.txt'),'local_official_README':bind(V73/'README_OFFICIAL.md')}
 derived={'schema':'V92_EXISTING_PUBLIC_LOCO_METADATA_AND_SAVED_QUERY_INVENTORY','completed':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope_policy':bind(R/'METADATA_READER_SCOPE_BEFORE_INVENTORY.json'),'reader_code':bind(Path(__file__)),'preliminary_read_disclosure':'Path discovery and selected pre-existing metadata were read before the dedicated inventory scope file; no RGB, feature, weight, current223 body or model was opened at any point in this task. The scope file is not claimed to precede these preliminary metadata reads.',
 'exact_source_paths':path_files,
 'count_correction':{'archive_RGB':5593,'official_COCO_images':5097,'unannotated_archive_RGB':496,'pallet_annotated_images':4727,'pallet_bbox_annotation_rows':120445,'note_ko':'4727은 팔레트 bbox 개수가 아니라 팔레트 annotation이 적어도 하나 있는 이미지 수입니다. 작은 적층 팔레트 개체가 많으므로 한 이미지가 여러 bbox를 가질 수 있습니다. bbox는 visible/amodal pixel mask나 D/E/native8 GT가 아닙니다.'},
 'actual_metadata_replay':{'archive_members_equal_group_manifest_equal_hash_inventory':True,'all5593_saved_file_SHA_join_exact':True,'all_saved20_input_order_SHA_member_HW_joins_exact':True,'all_saved20_selected_first_argmax_replay_exact':True,'all120_candidate_native9_shapes_finite':True,'archive_RGB_by_subset':dict(subsetcounts),'archive_RGB_by_official_split':dict(sourcecounts),'COCO_pallet_bbox_count':len(pallet),'COCO_pallet_images_count':len({a['image_id'] for a in pallet}),'COCO_image_metadata_key_set':sorted(coco['images'][0]),'COCO_annotation_key_set':sorted(coco['annotations'][0]),'official_split_annotation_files':[str(z) for z in sorted((V73/'annotations').glob('loco-sub*.json'))]},
 'available_saved_initializer':{'coverage_frames':20,'distinct_RGB':20,'all_candidates':sum(len(x['candidates']) for x in prop),'selected_nonempty_frames':sum(x['selected_index'] is not None for x in prop),'no_detection_frames':sum(x['selected_index'] is None for x in prop),'frames_per_official_split':dict(collections.Counter(g['official_split'] for g in (groups_by_member[x['officialpath_lineage']] for x in ip))),'selected_nonempty_per_official_split':dict(sel_all),'selected_slots_if_no_detection_retained_as_U':160,'selected_predicted_native_slots_available':selectednativefinite,'selected_native_continuous_valid':selectedcontinuous,'selected_native_nearest_valid':selectednearest,'selected_native_pack_center_valid':selectednativevalid,'all5593_saved_proposals_found':False,'extension40_saved_proposals_found':False,'search_scope':'Canonical FROZEN_PROPOSALS.jsonl files inventoried across P and V74 public_initializer producer/results; only the V74 successful decoded-RGB LOCO20 output matched LOCO. No global claim about arbitrary unindexed files outside P.','frame_key_set':sorted(prop[0]),'candidate_key_sets':[list(z) for z in sorted(candidatekeys)],'candidate_keypoint_shape':[9,2],'native8_use':'Preserve rows0..7 as predicted query slot tokens; row8 is predicted centroid, not a new native corner. No permutation/relabel/GT target choice.','coordinate_contract':'Already original native RGB coordinates; upstream100px reflective border already removed. Do not subtract pad again, clip or recenter. OOB remains factualU when needed.','selection_contract':protocol['predictor']['selection'],'predictor_box_vs_query_box':'Keep selected model box distinct from derived q8-envelope initial_query_box. Neither is the official GT bbox.','initializer_model_previously_bound_only':protocol['initializer_model'],'producer_wrapper_previously_bound_only':protocol['initializer_code'],'fresh_weight_or_feature_bytes_read':False,'derived_q_geometry_only':{'L':'qfinite excludes explicit[-1,-1]; L=clip(.25*max(ptp(qfinite8)),16,max(H,W)) if >=2points else16','query_box':'max(qmin,[0,0]) to min(qmax,[W-1,H-1]); require >=2points and span each>=2, elseNone','continuous_valid':'finite and not[-1,-1], 0<=qx<W,0<=qy<H','nearest_valid':'floor(q+.5) inside integernativeimage','pack_center_valid':'continuous_valid AND nearest_valid','crop':'No crops built or decoded in this inventory.'}},
 'saved20_frame_metadata':saved_rows,
 'existing_physical_cue_metadata_only':{'reviewed_frames_prior':60,'all_prior_proposals':28,'prior_conditional_candidates':22,'candidate_images':len({x['image_sha256'] for x in conditional}),'candidate_counts_per_source_split':dict(collections.Counter(x['source_split'] for x in conditional)),'candidate_rows_with_saved20_RGB':len(ref_in20),'candidate_images_with_saved20_RGB':len({x['image_sha256'] for x in ref_in20}),'candidate_rows_on_no_detection_saved20':len(ref_noq),'native8_mapping':'UNKNOWN for every proposal','human_gold':False,'official_native_visibility':False,'prior_training_admitted':False,'new_annotations_or_labels_created':0,'caution':'These were separately chosen physical cues tied to a fixed official representative bbox; a same-image predicted candidate may refer to a different instance. Same SHA alone is insufficient target correspondence. Existing geometry must remain unchanged; no nearest-q nativeID or maskpixel=D conversion.'},
 'split_and_capture_family':{'official_TRAIN_subsets':[2,3,5],'official_VAL_subsets':[1,4],'source_family_unit':'Whole author-defined subset/logistics environment, not image/timewindow random split','known_limit':'Saved initializer receipt still says UNASSIGNED_GROUP_HOLD because it predates later V74 conservative source-group freeze. Derive prospective metadata split via official archive_member+source group manifest; do not rewrite old output. Timestamp/date/take and COCOdataset_id do not establish independent physical site. Existing duplicate tests0 do not prove universal site independence.','unannotated496':'Remain UNKNOWN, not target-absence negatives','author_primary_evidence_scope':'Previously saved official README and indexed author paper excerpt; no fresh web verification in this task.'},
 'feasibility_verdict':{'metadata_and_existing_query_reuse':'AVAILABLE_FOR_FIXED20_ONLY','full_corpus_actual_frozen_queries':'NOT_AVAILABLE_IN_LOCATED_CANONICAL_OUTPUTS','target_physical_cue_training_labels':'No new admission; existing candidate metadata is weak observational and native8 correspondence unknown.','bridge_potential_ko':'기존 고정20장은 새 추론 없이 실제 frozen initial q8/L 운영조건을 따르는 소규모 source 시료정책을 세울 수 있습니다. 다만 no-detection2, OOB/target선택불확실성을 그대로 남기고, 팔레트 bbox 및 cue와 예측 candidate의 target 대응은 별도 증거로 확인해야 합니다. 나머지5573장과 extension40에는 저장 q가 없으므로 실제 initializer 조건을 맞추려면 별도 동결된 source-only 추론이 필요합니다. 아직 인가/실행하지 않았습니다.','required_before_any_future_RGB_review':['Freeze source sample/cohort using metadata before opening RGB and keep all assigned no-detection/hard cases.','Keep entire official environment groups for source split; preserve all bbox and candidate IDs separately.','Use original predicted q8 and derived L without reference-driven recentering or side enlargement.','Annotate observed physical cue/reference and whole uncertainty visible/covered/U only downstream; no native8 assignment without independent evidence.','If reference lies outside initial crop or target identity is unproved, preserve MISS/U rather than infer E.','Preserve no-detection queryset as U; do not synthesize q from GT bbox.','Keep masks, native semantic labels, self/foreign attribution unknown where dataset does not supply proof.']},
 'actions_this_task':{'RGB_body_reads_or_views':0,'feature_array_reads':0,'weight_body_or_tensor_reads':0,'new_extractions':0,'new_forward_fit_PnP':0,'current223_body_or_metrics_reads':0,'new_training_labels':0,'existing_artifacts_modified':0}}
 q=R/'EXACT_SOURCE_PATHS_AND_NATIVE_QUERY_FEASIBILITY.json';q.write_text(json.dumps(derived,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps({'report':bind(q),'counts':derived['available_saved_initializer']|{'saved20_frame_metadata':'inreport'}},ensure_ascii=False,default=str))
if __name__=='__main__': main()
