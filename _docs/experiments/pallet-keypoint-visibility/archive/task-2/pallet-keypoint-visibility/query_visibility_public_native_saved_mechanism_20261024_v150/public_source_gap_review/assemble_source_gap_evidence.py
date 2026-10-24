"""Completed primary-source metadata research. No pixels/models/private data."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,collections
H=Path(__file__).resolve().parent;P=H.parents[1];M=H/'http_metadata'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(p),'sha256':sha(p)}
def write(p,x):
 with p.open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
previous=[
 P/'query_visibility_flat_pallet_cue_bridge_20261023_v92/public_source_research/PUBLIC_REAL_NATIVE_VISIBILITY_AND_POSE_SOURCE_RESEARCH.json',
 P/'query_visibility_training_duration_control_20261023_v124/public_intake/FINDINGS.json',
 P/'query_visibility_broad_source_geometry_control_20261024_v139/public_native_source_readiness/PUBLIC_NATIVE_READINESS_EXISTING_METADATA_ONLY.json',
 P/'query_visibility_public_native_reference_feasibility_20261024_v140/annotation_contract/PUBLIC_NATIVE_REFERENCE_ANNOTATION_CONTRACT.json']
assert all(p.is_file()for p in previous)
C=read(H/'MR6D_PINNED_REVISION.json')['revision'];assert C=='3c4ed8453312655499c3d3a3cf38b4929fae77ee'
base='https://huggingface.co/datasets/anas-gouda/mr6d/resolve/'+C+'/'
files={}
for stem in['MR6D_MODELS_TREE','MR6D_ROOT_TREE','MR6D_VAL_SCENE0_TREE','MR6D_VAL000000_RGB_TREE','MR6D_VAL000000_DEPTH_TREE','MR6D_VAL000000_MASK_TREE','MR6D_VAL000000_MASK_VISIB_TREE']:
 for x in read(M/(stem+'.body')):
  if x['type']=='file':files[x['path']]=x
paths=['README.md','models_names.json','models/models_info.json','models/obj_000001.ply','val/000000/scene_camera.json','val/000000/scene_gt.json','val/000000/scene_gt_info.json','val/000000/rgb/000000.jpg','val/000000/depth/000000.png','val/000000/mask/000000_000000.png','val/000000/mask/000000_000001.png','val/000000/mask_visib/000000_000000.png','val/000000/mask_visib/000000_000001.png']
tiny=[]
receipts=read(M/'MR6D_TINY_CONTRACT_RECEIPTS.json')
head={x['requested_url']:x for x in receipts if x['method']=='HEAD'}
for path in paths:
 x=files[path];url=base+path
 tiny.append({'path':path,'bytes':x['size'],'repository_git_blob_oid':x['oid'],'lfs':x.get('lfs'),'pinned_url':url,
 'HTTP_HEAD_verified_200':url in head and head[url].get('status')==200,'file_payload_read_in_this_review':path.endswith('.json')or path=='README.md'})
assert sum(x['bytes']for x in tiny)==1047772
obj1=read(M/'MR6D_MODELS_INFO.body')['1'];gt=read(M/'MR6D_VAL000000_SCENE_GT.body');cam=read(M/'MR6D_VAL000000_SCENE_CAMERA.body');info=read(M/'MR6D_VAL000000_SCENE_GT_INFO.body')
assert read(M/'MR6D_MODELS_NAMES.body')['1']=='wooden Euro pallet' and len(gt)==len(cam)==len(info)==73
assert [x['obj_id']for x in gt['0']]==[1,1,10,3]
counts={k:len([x for x in read(M/('MR6D_'+k.upper()+'_TREE.body'))if x['type']=='directory'])for k in['val','dynamic','odyn','mr']}
assert counts=={'val':16,'dynamic':40,'odyn':19,'mr':14}
langtree=read(M/'LANG_TREE.body');langfiles=[x for x in langtree['tree']if x['path'].startswith('static/datasets/')and x['type']=='blob'];langRGB=[x for x in langfiles if Path(x['path']).suffix.lower()in['.jpg','.png','.jpeg']]
assert len(langRGB)==129
MR=dict(
 id='MR6D_REAL_EURO_PALLET',priority=1,status='TINY_SOURCE_INTAKE_READY_NOT_NATIVE_VISIBILITY_TRAIN_READY',new_to_existing_read_reports=True,
 publisher='Anas Gouda et al.; TU Dortmund / Fraunhofer IML / LAMARR',
 primary_urls=['https://huggingface.co/datasets/anas-gouda/mr6d','https://arxiv.org/html/2508.13775v1','https://openaccess.thecvf.com/content/ICCV2025W/R6D/papers/Gouda_MR6D_Benchmarking_6D_Pose_Estimation_for_Mobile_Robots_ICCVW_2025_paper.pdf'],
 license=dict(dataset='CC-BY-4.0',verified_in_pinned_author_README=True,readme=bind(M/'MR6D_LICENSE_README_PINNED.body'),paper_license_not_substituted=True),
 pinned_revision=C,repo_public_access='No login/token used. Source metadata GET and first-payload HEAD return HTTP200.',
 primary_paper_claims='Real mobile-robot scenes include a Euro pallet, bins and people; low viewpoints, indoor/outdoor lighting and occlusion. Object1 is a manually designed Euro-pallet mesh. No dedicated TRAIN split. Static pose accuracy is described as low-centimeter, with real pallet dimension variation.',
 independently_observed_release=dict(scene_directories=counts,total_scene_directories=89,paper_scenes=92,count_mismatch_unresolved=True,
 obj1_name='wooden Euro pallet',obj1_model_file='models/obj_000001.ply',obj1_model_bytes=11994,obj1_models_info=obj1,
 obj1_dimensions_interpretation='800×1200×144 in BOP model millimeters; Z-up is the BOP convention, actual PLY/support geometry is not inspected yet.',
 val000000_frame_count=73,val000000_firstframe=0,val000000_firstframe_obj_ids=[1,1,10,3],two_distinct_same_class_pallet_instances=True,
 published_whole_instance_visibility_fractions_first_two=[info['0'][0]['visib_fract'],info['0'][1]['visib_fract']],
 sample_gt_R_shape='nested3x3, not flat9; reshape explicitly without changing values',
 sample_gt_t_m2c_scale='First obj1 t≈[179.60,474.81,1540.72], compatible with BOP mm; actual RGB alignment remains unchecked.',
 perframe_cam_K_present=True,perframe_depth_scale=cam['0']['depth_scale'],
 optional_camera_world_extrinsics_caution='cam_t_w2c magnitudes differ from object t scale in sample. Do not use/assume these optional world extrinsics for native projection; native projection uses cam_R_m2c/cam_t_m2c plus K.',
 test_subset_first_scene_GT='dynamic/000000,odyn/000000,mr/000000 directory listings contain camera.json and modalities, but no scene_gt or scene_gt_info; do not infer all test GT public. Only val000000 GT bodies were verified.',
 hf_viewer_not_frame_count='Viewer8478 imagefolder rows mix modalities; not8478 RGB frames.'),
 provided_annotation_contract=dict(provided=['scene_camera.cam_K/depth_scale','scene_gt.obj_id/cam_R_m2c/cam_t_m2c','scene_gt_info.bbox_obj/bbox_visib/px_count_*/visib_fract','instance-indexed mask and mask_visib filenames','registered object mesh plus models_info'],
 absent=['project native0..7 ordering','per-native D/E/S/U','explicit external blocker identity per corner','native structural support uncertainty'],
 mask_meaning='BOP mask is projected silhouette and mask_visib is visible silhouette; IMID_GTID indexes a scene_gt entry. This supports instance association, not automatic per-corner semantic D/E/S. Mask0/depth discrepancy is never an E label.',
 source_format_primary='https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md'),
 query_to_native_bridge=dict(feasibility='CONDITIONAL_BUT_CONCRETE',
 reason='Flat Euro-pallet model dimensions and per-instance supplied object-to-camera pose can define the original normal-facing virtual8, after explicit source model axes/support audit. This directly avoids the Pallox fullcrate-vs-base upper-role ambiguity.',
 required_before_labels=['Check source PLY axes/extent and RGB projection using provided pose only, no fitting/PnP/snapping.','Keep both obj1 instances; associate original firstargmax target using entity/extent evidence without proposal reselection or GT-nearest permutation.','Review registered-reference meaning against raw RGB, distinguish foreign blocking, self state and U; preserve all8 and all noquery/OOB.'],
 unsupported_claims=['Supplied native8 visibility gold','mask presence equals corner visibility','all92paper scenes are released withGT','new pallet design diversity beyond one standardized Euro type','independent current223 hashes checked']),
 capture_split=dict(official='val/dynamic/odyn/mr; no TRAIN per paper',
 planned_use='If converted into training/development source, prospectively freeze disjoint whole scenes/capture groups and renounce MR6D benchmark performance claims on those reused groups. Do not randomly split neighboring frames.',
 independence='Distinct named publisher/capture pipeline from KU Leuven Pallox. Current223 overlap not tested because no current data were accessed; provenance and future source hash audit must remain separate.'),
 next_action_KO='V151의 별도 tiny-intake 정책 아래 val의 고정 scene/frame과 source obj1 mesh를 제한 수집해 source axes·투영·가림 의미를 실제 확인한다. 이 연구에서 학습행이나 D/E 라벨은 만들지 않았다.',
 tiny_intake_example=dict(fixed_selection_basis='Literal lexicographic first scene/frame metadata example, not a best visibility selection; root V151 will prospectively choose its own bounded cohort.',
 total_bytes=1047772,total_files=13,all13_file_paths_and_sizes_verified_from_tree=True,files=tiny,
 all_pixel_depth_mask_mesh_payload_GETs_in_this_review=0),
 local_evidence=[bind(M/x)for x in['MR6D_ROOT_TREE.body','MR6D_MODELS_TREE.body','MR6D_MODELS_NAMES.body','MR6D_MODELS_INFO.body','MR6D_VAL000000_SCENE_CAMERA.body','MR6D_VAL000000_SCENE_GT.body','MR6D_VAL000000_SCENE_GT_INFO.body','MR6D_TINY_CONTRACT_RECEIPTS.json','MR6D_FILE_CONTRACT_RECEIPTS.json','MR6D_ARXIV_V1.body']])
LANG=dict(id='LANG2LIFT_OUTDOOR_REAL',priority=2,status='RGB_DOWNLOAD_CONCRETE_MASK_LICENSE_RELEASE_GAP',
 primary_urls=['https://eric-nguyen1402.github.io/lang2lift.github.io/','https://github.com/Eric-nguyen1402/lang2lift.github.io/tree/'+langtree['sha']+'/static/datasets','https://arxiv.org/html/2508.15427v2'],
 observed=dict(pinned_commit=langtree['sha'],RGB_original_files=129,prompt_text_files=4,counts_by_folder=dict(collections.Counter(x['path'].split('/')[2]for x in langRGB)),tree_total_RGB_and_prompt_bytes=sum(x['size']for x in langfiles),public_mask_K_pose_files_in_complete_tree=0),
 primary_paper_claim='129 images and387 human prompts under sunny/snow/lowlight/occlusion; images combine author testsite and nonAI internet sources; masks manually annotated in Roboflow. These claims do not establish mask files in the released tree.',
 license='README explicitly says Website License CC-BY-SA4.0, repository API license=null. Dataset-specific license/provenance of internet-origin images is not confirmed.',
 native_join='No supplied native8/pose mapping in public tree. Needs source instance masks/target identity then independent native semantic review; RGB/prompt alone cannot directly supervise native E.',
 next_action='Use exact author dataset pointer to request or locate the missing Roboflow mask export and per-image capture/provenance/license manifest. Do not contact authors without explicit authorization; no message sent.',
 admission='Not training-ready from currently verified release. It is an actual outdoor appearance/foreign-occlusion candidate rather than a Pallox or generic YCB repack.',
 local_evidence=[bind(M/x)for x in['LANG_REPO.body','LANG_TREE.body','LANG_README.body','LANG_PROJECT.body','LANG_PROMPTS_OCC.body','LANG_ARXIV_V2.body']])
RGBD=dict(id='AUTHOR_RGBD_PALLET_HOANG',priority=3,status='PREVIOUSLY_KNOWN_ACCESS_AND_LICENSE_STILL_UNRESOLVED',
 primary_urls=['https://github.com/hoangcuongbk80/pallet_pose_est','https://cloud.oru.se/s/4JPf5s4YE4Gbx3A'],
 duplicate_of='V92/V139 candidate; same source commit1a18539ee288abfeb782a1bf2c41827ba6d05f70',
 verified_loader=['color/depth/labelPNG','cls_indexes/poses/factor_depth MAT','models/<class>/points.xyz','pallet_free/half_pallet_free/pallet_occluded/half_pallet_occluded classes'],
 not_labels='free/occluded are whole-class names, not native visibility states; label==class does not certify same-class instance ownership.',
 current_access='Bounded15s metadata landing GET timed out again; README says outside-Sweden VPN may be needed. No VPN or other workaround attempted, no dataset payload.',
 license='Dataset license not in complete author tree/README; no rootLICENSE. Paper license cannot replace dataset terms.',
 next_action='Resolve publisher access and dataset reuse terms plus version/file manifest before intake; do not keep spending GPU on guessed native labels.',
 native_join='Potential supplied pose/model reference mapping, conditional on actual release schema/axes/K/unique-instance proof. Not immediately usable now.',
 local_evidence=[bind(M/x)for x in['RGBD_REPO.body','RGBD_TREE.body','RGBD_LOADER.body','RGBD_CLASSES.body','RETRIEVAL_RECEIPTS.json']])
OTHER=[
 dict(id='LCS_NET_PALLET_PARTS500',url='https://pmc.ncbi.nlm.nih.gov/articles/PMC11991395/',status='REQUEST_ONLY',
  concrete='500 real pallet images; VOC/JSON part pixels0background/1stringerboard/2slot. Includes flat/steel forms. Data availability states upon request; no public dataset file pointer found in article.',
  gap='Useful part-ownership auxiliary if obtained and licensed; no native8 external/self labels. Current source action would be a human-authorized request, not automatic training.',evidence=bind(M/'LCS_FULLTEXT.body')),
 dict(id='RF_PALLET_CORNER_DETECTION',url='https://universe.roboflow.com/matthews-workspace-y8ofl/pallet-corner-detection',status='SEARCH_DISCOVERY_NOT_SCHEMA_OR_DOWNLOAD_VERIFIED',
  concrete='Creator page indexed as keypoint Front class,200images/version1 generated360,CC-BY4.0. Current direct page and version requests return403; web current main also returned404, while indexed snapshot exists.',
  gap='No exact keypoint names/order/visibility codes, original-camera provenance, real-vs-rendered status or raw export verified. Cannot assert a ready native dataset or independence from image mirrors.'),
 dict(id='RF_PALLET_SEG240',url='https://universe.roboflow.com/philipp-charisis/pallet-seg-4ossp',status='SEARCH_DISCOVERY_NOT_EXPORT_VERIFIED',
  concrete='Creator indexed instance-segmentation page,woodenpallet class,240images,CC-BY4.0. Bounded direct metadata requests403.',
  gap='No raw annotation export or capture provenance verified. Pixel instance masks would still require native role/semantic annotation.'),
 dict(id='UNLOGICAL_PALLET_DATABASE',url='https://github.com/unlogical0327/Pallet_database',status='REAL_APPEARANCE_FILES_BUT_NO_CLEAR_LICENSE_OR_ANNOTATION',
  concrete='Complete tree has213RGB/depth-associated image filenames under Harbin/Suzhou/Ningbo captures, plastic andwood named folders; README only statesRGBdepth dataset. Repository license=null and no license file found.',
  gap='No verified mask/pose/native corner fields. Folder names are metadata evidence, not inspected physical material gold. Not a label-ready source.',local_evidence=[bind(M/'UNLOGICAL_TREE.body'),bind(M/'UNLOGICAL_README.body')]),
 dict(id='HYPERGRAPH_PALLET16233',url='https://doi.org/10.1111/nyas.70219',status='RELEASE_PROMISE_NOT_VERIFIED_PUBLIC_FILES',
  concrete='Paper data-availability promises>=3000anonymized keypoint images spanningocclusion/stacking and refers full set byrequest; page supplies no concrete GitHub dataset URL.',
  gap='No release manifest/license/keypointorder/visibility definition verified. Do not count promised subset as downloaded or annotation-ready.'),
 dict(id='MENDELEY_DEEP_KEYPOINT_DOCKING',url='https://data.mendeley.com/datasets/m8d6236xpn/1',status='EMBARGO_UNTIL_2028_07_30',
  concrete='DOI10.17632/m8d6236xpn.1 official record gives embargo30July2028. Data description/files notpublic now.',gap='Not an immediate source.'),
 dict(id='PALLET_BLOCK32965',url='https://zenodo.org/records/6358607',status='DOWNLOADABLE_PART_IDENTITY_BUT_NO_FULL_NATIVE_CONTEXT',
  concrete='CC-BY4.0 publisherrecord,131860images of32965chipwoodblocks with repeatedcameras;13,518,893,025B archive listed.',
  gap='Reidentification of croppedblock identity, not full pallet instance/pose/native8/foreignE labels. Large texture expansion would not close current task gap.',evidence=bind(M/'PALLETBLOCK_ZENODO.body')),
 dict(id='PALLOC6D_NVIDIA_STAER_SYNTHETIC',status='NOT_ADDITIONAL_REAL_SOURCE',
  urls=['https://tore.tuhh.de/entities/product/4b51fbbb-6b30-4468-80c3-fd6d25419310','https://github.com/NVIDIA-AI-IOT/sdg_pallet_model','https://staer.ai/news/an-open-warehouse-dataset-for-physical-ai-research/'],
  gap='Synthetic release or inference model is not new real pallet native supervision.'),
 dict(id='LOCO_TOMIE_INDUSTRIAL_LOADS',status='PREVIOUSLY_AUDITED_NOT_NEW_DISCOVERY',
  gap='LOCO cachedTRAIN originals remain an executable manual native-label route; official bbox is not visibility. TOMIE source association/pose gaps and IndustrialLateralLoads load-vs-pallet mask limitation are already documented. Repeating them as new native gold would add no evidence.')]
result=dict(completed=True,PASS=True,phase='V150',UTC=datetime.now(timezone.utc).isoformat(),scope='Primary official/research/author dataset metadata+small sourceJSON only; no image/depth/mask/mesh/weight/features/predictions/private bodies, no large downloads, no model/fit/PnP.',
 conclusion_KO='새로 확인한 가장 실행 가능한 원천은 MR6D의 실제 Euro pallet이다. obj1 mesh·K·제공 pose·instance masks와 소규모 파일 경로가 실제 공개되어 tiny source 정합 검토를 바로 진행할 수 있다. 다만 native8 D/E는 제공되지 않아 pose 기반 참조와 실제 구조/외부가림 판정을 새로 연결해야 한다. 이번 검색에서 즉시 투입 가능한 완성 native8 가림 gold는 확인하지 못했다.',
 previous_local_reports_read_before_search=[bind(p)for p in previous],ranked_candidates=[MR,LANG,RGBD],other_candidates_and_nonadmissions=OTHER,
 actions=dict(public_METADATA_HTTP_and_web_reads=True,public_small_pose_camera_JSON_reads=True,source_pixel_depth_mask_mesh_payload_GET=0,new_model_fit_encoder=0,new_current223_or_prohibited_phase_access=0,large_downloads=0,author_messages=0,new_labels=0,admission_or_threshold_or_fold_changes=0),
 guards=['Source instance masks and object visibility fraction remain separate from native semantic D/E/S/U.','No public source convention is matched to current q by closest-point permutation or proposal reselection.','Source actual q/L and allslots/noquery/OOB must be preserved in later independent producer.','Source/development reuse of official MR6D VAL/test cannot be reported as untouched MR6D benchmark performance.','No claims that current private223 overlap was checked in this task.'],
 receipts=[bind(M/x)for x in['RETRIEVAL_RECEIPTS.json','ADDITIONAL_RETRIEVAL_RECEIPTS.json','MR6D_RETRIEVAL_RECEIPTS.json','MR6D_FILE_CONTRACT_RECEIPTS.json','MR6D_TINY_CONTRACT_RECEIPTS.json']])
write(H/'PUBLIC_SOURCE_GAP_AND_ACTION_EVIDENCE.json',result)
write(H/'MR6D_MINIMUM_PUBLIC_FILE_MANIFEST_NOT_DOWNLOADED.json',dict(completed=True,revision=C,example_only=True,files=tiny,total_bytes=1047772,actual_pixel_depth_mask_mesh_downloads=0))
script='''#!/usr/bin/env bash
# NOT EXECUTED. Literal tiny example, not V151 prospective selection.
# Parent V151 policy/entry determines any actual acquisition.
set -euo pipefail
source_out="${1:?Pass a fresh dedicated public-source intake directory}"
mkdir -p "$source_out"
source_base='''+"'"+base+"'\n"+'''while IFS= read -r rel; do
  mkdir -p "$source_out/$(dirname "$rel")"
  test ! -e "$source_out/$rel"
  curl --fail --location --max-time 45 --max-filesize 1000000 --output "$source_out/$rel" "$source_base$rel"
done <<'SOURCE_FILES'
'''+''.join(x+'\n'for x in paths)+'''SOURCE_FILES
'''
with(H/'MINIMAL_MR6D_SOURCE_INTAKE_NOT_EXECUTED.sh').open('x')as f:f.write(script)
qa=dict(completed=True,PASS=True,result=bind(H/'PUBLIC_SOURCE_GAP_AND_ACTION_EVIDENCE.json'),minimum_manifest=bind(H/'MR6D_MINIMUM_PUBLIC_FILE_MANIFEST_NOT_DOWNLOADED.json'),unexecuted_command=bind(H/'MINIMAL_MR6D_SOURCE_INTAKE_NOT_EXECUTED.sh'),code=bind(Path(__file__).resolve()),
 checks=dict(primary_sources_only_for_claims=True,prior_reports_bound=True,pinned_MR6D_commit_and_license=True,all13_exact_file_sizes_and_paths=True,minimum_bundle_bytes=1047772,first_two_instances_obj1_confirmed=True,release89_vs_paper92_explicit=True,test_GT_availability_not_assumed=True,fullnative_gold_not_claimed=True,current_overlap_not_certified=True,pixel_or_model_or_current_access=0))
write(H/'REVIEW_COMPLETE.json',qa)
print(json.dumps({'result':bind(H/'PUBLIC_SOURCE_GAP_AND_ACTION_EVIDENCE.json'),'QA':bind(H/'REVIEW_COMPLETE.json'),'manifest':bind(H/'MR6D_MINIMUM_PUBLIC_FILE_MANIFEST_NOT_DOWNLOADED.json')},ensure_ascii=False))
