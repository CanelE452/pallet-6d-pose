"""Cached material-source metadata and byte hashes only; no pixel decoding/labels."""
from pathlib import Path
import collections,hashlib,json,sys
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
ROOT=V/'ROOT_MATERIAL_SOURCE_GEOMETRY_BEFORE_INTAKE.json'
INSTANCES=P/'query_visibility_public_real_20261022_v63/ycb/instances.jsonl'
PRIOR=P/'query_visibility_public_external_ray_20261023_v87/corpus_build_v1/SOURCE_DEPTH_MASK_BYTE_BINDINGS.json'
V76=P/'query_visibility_reference_local_support_20261023_v76/public_query_emulation/INPUTS_RGB_ONLY.jsonl'
V77=P/'query_visibility_controlled_pairs_20261023_v77/pair_generation'
V96=P/'query_visibility_query_anchored_cue_20261023_v96'
V99=P/'query_visibility_diverse_expected_cue_source_20261023_v99/background_catalog/BACKGROUND_ASSIGNMENTS96_INPUT_ONLY.jsonl'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x)for x in Path(p).read_text().splitlines()]
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def jsonl(p,xs):
 with Path(p).open('x')as f:
  for x in xs:f.write(json.dumps(x,allow_nan=False)+'\n')
def main():
 assert not(H/'SOURCE_FEASIBILITY.json').exists()
 root=read(ROOT);assert bind(INSTANCES)==root['instances']
 original=rows(INSTANCES);selected=sorted([(i,x)for i,x in enumerate(original)if x['object_id']in(16,21)],key=lambda z:(z[1]['scene_id'],z[1]['frame_id'],z[1]['instance_id']))
 assert len(selected)==400
 geometry=read(V/'SURFACE_CANDIDATE_RESULT.json');objects={o['object_id']:o for o in geometry['objects']}
 prior={x['path']:x for x in read(PRIOR)['files']}
 file_map={};scene_cache={};input_plan=[];lineage=[]
 def file(path,role,expected=None):
  path=str(Path(path).resolve())
  if path not in file_map:
   exists=Path(path).is_file();digest=sha(path)if exists else None;old=prior.get(path)
   file_map[path]=dict(path=path,exists=exists,sha256=digest,bytes=Path(path).stat().st_size if exists else None,
                       prior_V87_SHA=old['sha256']if old else None,prior_V87_SHA_match=(digest==old['sha256'])if old else None,roles=[])
  b=file_map[path]
  if role not in b['roles']:b['roles'].append(role)
  if expected is not None:
   b['source_RGB_expected_SHA']=expected;b['source_RGB_SHA_match']=b['sha256']==expected
  return dict(path=path,sha256=b['sha256'],exists=b['exists'])
 for index,(source_index,r)in enumerate(selected):
  scene=Path(r['image']).parent.parent
  if str(scene)not in scene_cache:
   scene_cache[str(scene)]=(read(scene/'scene_gt.json'),read(scene/'scene_camera.json'))
  gt,cam=scene_cache[str(scene)];all_gt=gt[str(r['frame_id'])];camera=cam[str(r['frame_id'])];own=r['instance_id']
  assert all_gt[own]['obj_id']==r['object_id']and all_gt[own]['cam_R_m2c']==r['cam_R_m2c']and all_gt[own]['cam_t_m2c']==r['cam_t_m2c']
  assert camera['cam_K']==r['K']and camera['depth_scale']==r['depth_scale']and r['depth_scale']>0
  rgb=file(r['image'],'RGB',r['image_sha256']);depth=file(r['depth'],'DEPTH');full=file(r['mask_full'],'OWN_FULL_MASK');vis=file(r['mask_vis'],'OWN_VISIBLE_MASK')
  owners=[dict(instance_index=j,object_id=g['obj_id'],visible_mask=file(scene/'mask_visib'/f'{r["frame_id"]:06d}_{j:06d}.png','ALL_INSTANCE_VISIBLE_MASK'))for j,g in enumerate(all_gt)]
  controls={name:file(scene/name,'SCENE_CONTROL_'+name)for name in('scene_gt.json','scene_camera.json','scene_gt_info.json')}
  identity=f'Y113_MATERIAL_{index:06d}'
  input_plan.append(dict(plan_row_index=index,input_id=identity,image={k:rgb[k]for k in('path','sha256')},raw_hw=[r['height'],r['width']],
                         original_split=r['split'],split=r['split'],source='PUBLIC_YCB_MATERIAL_SURFACE_CANDIDATE',
                         query_preparation='NOT_RUN_AWAIT_ROOT_POLICY',initial_points=None,roi_side=None,query_box_xyxy=None,
                         native_mapping='UNKNOWN',training_admission=False))
  complete=all(x['visible_mask']['exists']for x in owners)
  lineage.append(dict(plan_row_index=index,input_id=identity,original_source_row_index=source_index,
                      original_source_instance_id=f'YCB:{r["scene_id"]:06d}:{r["frame_id"]:06d}:{own:06d}',
                      scene_id=r['scene_id'],frame_id=r['frame_id'],instance_id=own,object_id=r['object_id'],original_split=r['split'],
                      image=rgb,depth=depth,own_full_mask=full,own_visible_mask=vis,complete_scene_GT_owner_count=len(all_gt),
                      all_visible_owners=owners,all_visible_owner_files_available=complete,controls=controls,
                      K=camera['cam_K'],cam_R_m2c=all_gt[own]['cam_R_m2c'],cam_t_m2c_mm=all_gt[own]['cam_t_m2c'],depth_scale_to_mm=camera['depth_scale'],
                      pose_camera_exact_instance_metadata_parity=True,model=objects[r['object_id']]['model'],
                      fixed_original_aux_surface_ids=[c['aux_surface_id']for c in objects[r['object_id']]['candidates']],
                      original_mesh_vertex_indices=[c['original_mesh_vertex_index']for c in objects[r['object_id']]['candidates']],
                      teacher_labels_generated=False,native_mapping='UNKNOWN',human_gold=False,training_admission=False))
 # Prior exposure is checked only through source/donor metadata, never scores.
 old=rows(V76);old_by_id={x['id']:x for x in old};pool=read(V77/'DONOR_POOL_LOCK.json')['all4200rows'];pairs=rows(V77/'PAIR_SELECTION_GT_ONLY.jsonl')
 cert_path=V96/'actual_query_anchored_image_intake/SELECTED_DONOR_RGBA_CERTIFICATES.jsonl';certs=rows(cert_path)
 assign_path=V96/'query_anchored_pair_preparation/DONOR_ASSIGNMENT960_K16_BEFORE_PIXELS.jsonl';assigns=rows(assign_path)
 backgrounds=rows(V99)
 scene9=[]
 for x in lineage:
  oldrow=old_by_id.get(x['original_source_instance_id']);x['V76_same_instance_input_reference']=oldrow is not None
  if oldrow is not None:assert oldrow['image']==x['image']['path']and oldrow['image_sha256']==x['image']['sha256']and oldrow['split']==x['original_split']
  if x['scene_id']!=9:continue
  digest=x['image']['sha256'];path=x['image']['path']
  a=[z for z in pool if z['image_sha256']==digest];b=[z for z in pairs if z['donor']['image_sha256']==digest]
  c=[z for z in certs if z['image']['sha256']==digest];d=[(z['base_row_index'],rank['donor_rank'])for z in assigns for rank in z['ranks']if rank['image_sha256']==digest]
  e=[z['base_id']for z in backgrounds if z['image']['sha256']==digest]
  scene9.append(dict(input_id=x['input_id'],scene_id=9,frame_id=x['frame_id'],instance_id=x['instance_id'],object_id=x['object_id'],image=x['image'],original_split=x['original_split'],
                     V76_same_instance_input=oldrow is not None,V76_same_RGB_input_rows=sum(z['image_sha256']==digest for z in old),
                     V77_donor_pool_same_RGB_rows=len(a),V77_selected_donor_assignments=len(b),
                     V96_actual_selected_donor_certificates=len(c),V96_fixed_donor_rank_assignments=len(d),
                     V99_background_base_ids=e,exposure_status='ALREADY_EXPOSED_PUBLIC_SOURCE_VALIDATION',fresh_holdout=False,
                     actual_use_scope='V76 source-input reference and V77/V96 donor lineage; no claim that every donor appearance entered a loss. No score values read.'))
 assert len(scene9)==100 and all(x['V76_same_instance_input']for x in scene9)
 jsonl(H/'INPUT_PLAN400_WITHOUT_QUERIES_INPUT_ONLY.jsonl',input_plan)
 jsonl(H/'INSTANCE400_LINEAGE_GT_ONLY.jsonl',lineage)
 jsonl(H/'SCENE9_PREVIOUS_EXPOSURE_METADATA.jsonl',scene9)
 write(H/'CACHE_FILE_BINDINGS.json',dict(completed=True,files=list(file_map.values()),pixel_decoding=False,previous_V87_bindings=bind(PRIOR)))
 rolecounts={}
 for role in sorted({s for x in file_map.values()for s in x['roles']}):
  rr=[x for x in file_map.values()if role in x['roles']]
  rolecounts[role]=dict(unique_paths=len(rr),available=sum(x['exists']for x in rr),SHA_computed=sum(x['sha256']is not None for x in rr),prior_V87_SHA_present=sum(x['prior_V87_SHA']is not None for x in rr),prior_V87_SHA_match=sum(x['prior_V87_SHA_match']is True for x in rr),source_RGB_SHA_match=sum(x.get('source_RGB_SHA_match')is True for x in rr))
 missing=[x['path']for x in file_map.values()if not x['exists']]
 mismatches=[x['path']for x in file_map.values()if x['prior_V87_SHA_match']is False or x.get('source_RGB_SHA_match')is False]
 counts={str(oid):dict(collections.Counter(x['original_split']for x in lineage if x['object_id']==oid))for oid in(16,21)}
 outputs=[bind(H/name)for name in('INPUT_PLAN400_WITHOUT_QUERIES_INPUT_ONLY.jsonl','INSTANCE400_LINEAGE_GT_ONLY.jsonl','SCENE9_PREVIOUS_EXPOSURE_METADATA.jsonl','CACHE_FILE_BINDINGS.json')]
 report=dict(completed=True,PASS=not missing and not mismatches,scope='Metadata and raw-file SHA only. No image/depth/mask decoding, teacher generation or learned model access.',
             code=bind(__file__),root=bind(ROOT),source_instances=bind(INSTANCES),fixed_geometry=bind(V/'SURFACE_CANDIDATE_RESULT.json'),outputs=outputs,
             exact_instances=400,unique_RGB_images=len({x['image']['path']for x in lineage}),all8_regions_per_instance_retained_for_future_policy=True,potential_surface_slots=3200,
             instances_by_object_and_original_split=counts,original_fold_unchanged=True,metadata_pose_camera_parity_instances=400,
             complete_visible_owner_sets=sum(x['all_visible_owner_files_available']for x in lineage),cache_by_role=rolecounts,unique_cache_files=len(file_map),missing_files=missing,SHA_mismatches=mismatches,
             prior_exposure_sources=[bind(z)for z in(V76,V77/'DONOR_POOL_LOCK.json',V77/'PAIR_SELECTION_GT_ONLY.jsonl',cert_path,assign_path,V99)],
             scene9_exposure=dict(instances=100,unique_RGB_images=100,V76_same_target_instance=100,
                  V77_pool_images=sum(x['V77_donor_pool_same_RGB_rows']>0 for x in scene9),
                  V77_selected_donor_images=sum(x['V77_selected_donor_assignments']>0 for x in scene9),V77_selected_donor_assignments=sum(x['V77_selected_donor_assignments']for x in scene9),
                  V96_actual_donor_images=sum(x['V96_actual_selected_donor_certificates']>0 for x in scene9),V96_actual_donor_certificates=sum(x['V96_actual_selected_donor_certificates']for x in scene9),
                  V96_rank_assignment_images=sum(x['V96_fixed_donor_rank_assignments']>0 for x in scene9),V96_rank_assignments=sum(x['V96_fixed_donor_rank_assignments']for x in scene9),
                  V99_background_images=sum(bool(x['V99_background_base_ids'])for x in scene9),status='ALREADY_EXPOSED_PUBLIC_SOURCE_VALIDATION',fresh_holdout=False),
             feasibility='All cached metadata and byte-bound evidence needed for unchanged V109 target-alone-ray/depth/ownership operations are inventoried. Await a prospective root cohort/query/teacher policy before generating q/r/L or labels.',
             next_source_constraints=['Preserve original train/val folds. Obj21 has no sourceVAL; do not move its TRAIN scenes to manufacture holdout.',
                  'Scene9 is old sourceVAL/donor/background development data, not a fresh scene-disjoint holdout.',
                  'Reuse fixed original mesh vertices and exact V109 geometry/depth/owner logic only under the later root policy. Surface/junction/native correspondence remains UNKNOWN.',
                  'Do not equate mesh surface, own mask pixel, material appearance or amodal support with physical-cue D/E.',
                  'V112 holdout is untouched and excluded from this plan; no fit/normalizer/checkpoint/threshold selection is performed.'],
             query_generation_run=False,teacher_labels_generated=False,source_RGB_depth_mask_pixels_decoded=False,new_network_bytes=0,
             learned_model_fit_private_current_PnP=False,native_mapping='UNKNOWN',human_gold=False,training_admission=False)
 write(H/'SOURCE_FEASIBILITY.json',report)
 write(H/'FEASIBILITY_SEAL.json',dict(completed=True,PASS=report['PASS'],report=bind(H/'SOURCE_FEASIBILITY.json'),outputs=outputs,code=bind(__file__),no_corpus_build=True))
 print(json.dumps(dict(report=bind(H/'SOURCE_FEASIBILITY.json'),seal=bind(H/'FEASIBILITY_SEAL.json'),counts=counts,roles=rolecounts,missing=len(missing),mismatches=len(mismatches),scene9=report['scene9_exposure'])),flush=True)
if __name__=='__main__':main()
