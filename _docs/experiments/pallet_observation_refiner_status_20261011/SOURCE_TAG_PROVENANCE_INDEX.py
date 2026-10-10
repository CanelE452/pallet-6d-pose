"""Read existing annotation provenance only. No GT fits, new labels, or RGB.

Saved phase maps the reference IDs; that phase does not certify physical IDs.
Original annotation JSON files remain external, read-only, and byte-preserved.
"""
import argparse,collections,datetime,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[3]
STATUS=ROOT/'_docs/experiments/pallet_observation_refiner_status_20261011'
def digest_bytes(raw):return hashlib.sha256(raw).hexdigest()
def read(p,root):
 if p.is_symlink() or any(q.is_symlink()for q in p.parents):raise RuntimeError('input symlink')
 raw=p.read_bytes();return json.loads(raw),{'path':str(p.relative_to(root)),'bytes':len(raw),'sha256':digest_bytes(raw)}
a=argparse.ArgumentParser();a.add_argument('--source-root',type=pathlib.Path,required=True);a.add_argument('--output',type=pathlib.Path,required=True);args=a.parse_args();source=args.source_root.absolute();output=args.output.absolute()
if output.exists()or output.is_symlink()or any(p.is_symlink()for p in output.parents):raise RuntimeError('output must initially be absent')
index,ib=read(STATUS/'REMAINING_THIRTEEN_QUERIES.json',ROOT)
mapping,mb=read(ROOT/'_docs/experiments/pallet_kp_corrected_supervision_20261010_v1/REFERENCE_ID_MAPPING.json',ROOT)
axis,ab=read(source/'data/pallet/results/paper_pose_metric_closure_v1/AXIS_REVIEW_MANIFEST.json',source)
if (ab['bytes'],ab['sha256'])!=(mapping['axis_manifest']['bytes'],mapping['axis_manifest']['sha256']):raise RuntimeError('source axis manifest differs from fixed public binding')
public={r['public_id']:r for r in mapping['rows']};frames={r['frame_id']:r for r in axis['frames_list']};cases=[];checked=[]
for case in index['cases']:
 link=public[case['frame_id']];f=frames[link['source_cache_id']];rel=pathlib.PurePosixPath(f['annotation'])
 if rel.is_absolute()or '..'in rel.parts:raise RuntimeError('annotation must be source-root relative')
 p=source/rel;j,b=read(p,source);checked.append((p,b));objects=j['objects']
 if len(objects)!=1:raise RuntimeError('single declared pallet object required')
 obj=objects[0];perm=case['mask_audit_saved']['native_reference_permutation'];k=case['corner_id']
 if len(perm)!=8 or sorted(perm)!=list(range(8)) or k not in range(8):raise RuntimeError('saved phase permutation contract')
 aid=perm[k];tags=obj['keypoint_annotations'];record=tags[aid]
 if not isinstance(tags,list)or len(tags)<8 or not isinstance(record,dict):raise RuntimeError('declared existing annotation point record missing')
 camera=obj.get('camera_facing_pnp',{})
 cases.append({'frame_id':case['frame_id'],'source_cache_id':link['source_cache_id'],'native_corner_id':k,'saved_native_reference_permutation':perm,'annotation_corner_index_by_saved_reference_phase':aid,'original_annotation_binding':{**b,'origin':'external_readonly_original_annotation_current_bytes'},'image_path_metadata_only':f['image'],'image_SHA_from_existing_mapping_not_fresh_RGB_hash':link['image_sha256'],'saved_human_state':case['human_state_saved'],'stored_annotation_point':record,'keypoint_frame':obj.get('keypoint_frame'),'gt_source':obj.get('gt_source'),'axis_assignment_saved':camera.get('axis_assignment'),'axis_assignment_confirmed_field_present':'axis_assignment_confirmed'in camera,'axis_assignment_confirmed_saved':camera.get('axis_assignment_confirmed'),'axis_review_saved':f['review'],'manual_review_reasons_saved':obj.get('manual_review_reasons'),'independent_physical_corner_and_incident_edge_certificate':'NOT_PRESENT_IN_THESE_DECLARED_RECORDS','physical_ID_mapping_certified':False})
if len(cases)!=13:raise RuntimeError('all thirteen queries required')
for p,b in checked:
 if digest_bytes(p.read_bytes())!=b['sha256']:raise RuntimeError('original annotation changed during read-only extraction')
counts=dict(collections.Counter(c['stored_annotation_point'].get('source','FIELD_MISSING')for c in cases))
result={'schema':'existing_source_tag_provenance_for_thirteen_queries_v18','complete':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'query_count':13,'inputs':{'saved_case_index':ib,'public_reference_mapping':mb,'external_axis_manifest':ab},'source_tag_literal_counts':counts,'cases':cases,'original_annotations_unchanged_during_extraction':True,'source_GT_builder_or_original_annotations_modified':False,'limits':{'new_metric_GT_fit_geometry_PnP_model_training_RGB_capture_annotation_timing_calls':0,'manual_click_is_input_provenance_not_physical_identity_or_edge_ownership_certificate':True,'saved_reference_phase_is_not_independent_physical_ID_approval':True,'independent_physical_pose_or_ownership_certification':False,'unknown_source_does_not_prove_the_coordinate_wrong':True,'no_claim_about_uninspected_other_records':True,'accuracy_goal_solved':False}}
output.parent.mkdir(parents=True,exist_ok=True)
with output.open('x')as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'complete':True,'queries':13,'source_tag_literal_counts':counts,'original_annotation_bytes_preserved':True,'new_numeric_model_RGB_calls':0},ensure_ascii=False))
