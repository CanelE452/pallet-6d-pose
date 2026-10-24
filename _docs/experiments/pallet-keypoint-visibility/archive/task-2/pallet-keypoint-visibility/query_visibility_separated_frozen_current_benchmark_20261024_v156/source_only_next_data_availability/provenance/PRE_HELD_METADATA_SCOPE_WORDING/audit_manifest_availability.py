"""Availability audit of official metadata only; no RGB/depth/mask/model reads."""
from pathlib import Path
from collections import Counter
import hashlib,json,re
H=Path(__file__).resolve().parent;P=H.parents[1]
OLD=P/'query_visibility_mr6d_public_native_intake_20261024_v151'
INFO=OLD/'metadata/HF_DATASET_INFO.json';META=OLD/'provided_metadata'
def bind(p):return dict(path=str(Path(p).resolve()),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def read(p):return json.loads(Path(p).read_text())
def main():
 d=read(INFO);commit=d['sha'];assert commit=='3c4ed8453312655499c3d3a3cf38b4929fae77ee'
 latest=read(H/'remote_metadata/LATEST_REPO_SHA_METADATA.json');assert latest['sha']==commit
 names=[r['rfilename']for r in d['siblings']];assert len(names)==92622 and len(set(names))==len(names)
 inventory={}
 for top in('val','dynamic','odyn','mr'):
  paths=[n for n in names if n.startswith(top+'/')]
  inventory[top]=dict(released_scene_directories=len({n.split('/')[1]for n in paths}),
   RGB_files=sum(bool(re.fullmatch(r'.+/rgb/[^/]+\.(jpg|jpeg|png)',n.lower()))for n in paths),
   depth_files=sum('/depth/'in n for n in paths),full_mask_files=sum('/mask/'in n for n in paths),
   visible_mask_files=sum('/mask_visib/'in n for n in paths),scene_camera_files=sum(n.endswith('/scene_camera.json')for n in paths),
   scene_gt_files=sum(n.endswith('/scene_gt.json')for n in paths),scene_gt_info_files=sum(n.endswith('/scene_gt_info.json')for n in paths),
   other_JSON=[n for n in paths if n.endswith('.json')and not n.endswith(('/scene_camera.json','/scene_gt.json','/scene_gt_info.json'))])
 gtfiles=sorted((META/'val').glob('*/scene_gt.json'));assert len(gtfiles)==16
 positives=[];observations=0;frames=0;counts=[]
 for path in gtfiles:
  gt=read(path);pp=0;oo=0;frames+=len(gt)
  for frame,rr in gt.items():
   nn=sum(r['obj_id']==1 for r in rr)
   if nn:positives.append(('val/'+path.parent.name,int(frame)));pp+=1;oo+=nn
  observations+=oo;counts.append(dict(scene=path.parent.name,all_frames=len(gt),obj1_GTpositive_frames=pp,obj1_observations=oo))
 assert frames==735 and len(positives)==230 and observations==303
 odd=read(H/'remote_metadata/ODD_WORLD_METADATA.json');oddids=[r['obj_id']for r in odd['w']];assert 1 not in oddids
 extra=sum(inventory[s]['RGB_files']for s in('dynamic','odyn','mr'));assert extra==13298
 trainpaths=[n for n in names if any(k in n.lower()for k in('train','pbr'))];assert not trainpaths
 gallery=[n for n in names if n.startswith('gallery_onboarding/obj_000001/')];assert len(gallery)==6
 md=list((H/'remote_metadata').glob('*.json'));total_new_metadata=sum(p.stat().st_size for p in md)
 report=dict(completed=True,PASS=True,scope='Read-only official public metadata/source-doc availability; not intake/training/model selection.',
  code=bind(__file__),pinned_manifest=bind(INFO),pinned_commit=commit,latest_official_API_commit_matches=True,metadata_only_new_files=[bind(p)for p in md],
  actual_new_saved_remote_metadata_bytes=total_new_metadata,new_RGB_depth_mask_or_model_downloads=0,current223_private_V112_or_learned_model_access=0,
  official_dataset='https://huggingface.co/datasets/anas-gouda/mr6d',
  official_pinned_tree='https://huggingface.co/datasets/anas-gouda/mr6d/tree/'+commit,
  official_latest_SHA_API='https://huggingface.co/api/datasets/anas-gouda/mr6d?expand%5B%5D=sha',
  primary_paper='https://arxiv.org/html/2508.13775v1',official_paper_publisher='https://openaccess.thecvf.com/content/ICCV2025W/R6D/html/Gouda_MR6D_Benchmarking_6D_Pose_Estimation_for_Mobile_Robots_ICCVW_2025_paper.html',
  publisher_fetch_status='403 via web tool; author arXiv HTML read successfully',
  license=dict(displayed='CC-BY-4.0',evidence=[bind(META/'README.md'),bind(INFO)],attribution='Anas Gouda et al., MR6D, ICCVW2025 R6D'),
  paper_claims=dict(no_dedicated_training_set=True,subsets_are_real_capture=True,
   mesh_rendering_is_a_possible_synthetic_training_source_not_additional_real_training=True,
   subset_capture='Validation/Dynamic: research halls; O3dyn: low-mounted robot, indoor/outdoor; MR: robot-like camera motion.',
   annotation_limit='Static pose accuracy described in low-centimeter range; pallets vary by centimeters. This does not certify pixel-exact native corners.',
   reference_identity_limit='Same modeled object IDs are used across subsets; physical pallet-unit disjointness is not established.'),
  release_manifest_counts=inventory,scene_directory_total=89,paper_scene_count=92,paper_vs_pinned_release_scene_difference='UNRESOLVED; do not substitute abstract count for available directories.',
  train_or_train_real_or_train_pbr_paths=0,synthetic_PBR_release_paths=0,
  existing_val=dict(all_RGB735=True,obj1_GTpositive_frames=230,obj1_observations=303,all_obj1_GTpositive_frames_already_in_fixed230=True,
   remaining_val_frames_without_obj1_in_provided_GT=505,remaining_val_not_certified_pixel_negative=True,scenes=counts),
  additional_real_scene_RGB_files=13298,additional_pallet_positive_frames='UNKNOWN: no per-frame scene_gt/object-ID mapping released in these folders.',
  additional_native_visibility_role_labels='NOT_PROVIDED; instance masks are not native8 D/E/S labels and mask filename instance suffix is not obj_id.',
  nonstandard_world_metadata=dict(relative_path='mr/000013/scene_gt_world (copy).json',provided_object_ids=oddids,pallet_present=False,
   evidence=bind(H/'remote_metadata/ODD_WORLD_METADATA.json'),not_a_hidden_extra_pallet_GT_source=True),
  pallet_onboarding_gallery=dict(image_paths=gallery,count=6,official_use='Object onboarding/reference imagery; not a dedicated training split.',
   per_image_real_vs_render='UNKNOWN without image/doc confirmation',pose_K_native_visibility_annotations_provided=False,physical_instance_nonoverlap='UNKNOWN'),
  physical_overlap=dict(object_id1_model_shared=True,train_test_physical_pallet_serial_IDs_provided=False,disjoint_physical_objects_or_backgrounds='NOT_CERTIFIED',
   sequential_scene_frames_are_not_independent_new_objects=True),
  HF_viewer_warning='8478 auto imagefolder validation rows equal735 RGB+735depth+7008mask images, not8478 RGB/pallet training samples.',
  official_research_repo_search='Author homepage and paper direct MR6D to official HF repository. No additional official MR6D training repository was identified in this bounded review; absence outside checked sources is not proved.',
  next_source_scope='Additional real RGB exists, but pallet identity/native-role supervision remains an annotation/GT-release gap. Availability alone does not authorize training or auto-labels.',
  original_source_validation45_reassigned_or_read=False,current_frozen_benchmark_changed=False,selected_model=None,
  limits=['Counts refer to exact current official commit matching the saved complete92622-path manifest.',
   'No new pixels were examined, so visibility/occluder diversity and real-vs-render status of six gallery images remain unverified.',
   'No maintainer message was sent; official test pose GT may exist outside the released files, but this review did not verify availability.'])
 path=H/'REVIEW.json'
 with path.open('x')as f:json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
 print(json.dumps(dict(completed=True,PASS=True,review=bind(path),additional_RGB_files=extra,additional_pallet_GTpositive_frames='UNKNOWN',new_metadata_bytes=total_new_metadata)))
if __name__=='__main__':main()
