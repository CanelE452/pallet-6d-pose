"""V185 direct-lineage TRAIN inventory. Reads JSON/JSONL/code only; payload stat only.

Does not import producer/model code or open/hash/header-probe image, depth, mask,
mesh, pose, cached pixel/feature/target NPZ, checkpoint, or any held/VAL payload.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V=P/'query_visibility_public_part_supervision_readiness_20261024_v185'
H=V/'local_train_asset_inventory_b'
V181=P/'query_visibility_local_support_raw_head_20261024_v181'
V153=P/'query_visibility_mr6d_full_public_source_20261024_v153'
V161=P/'query_visibility_public_train_owner_counterfactual_20261024_v161'
SCOPE=V/'ROOT_SCOPE_BEFORE_PUBLIC_SOURCE_TARGET_OR_ASSET_BODIES.json'


def require(value,message):
    if not value:raise AssertionError(message)


def safe(path):
    p=Path(path)
    require(not any(x in str(p).lower() for x in ('_v112','current223','/private_evaluation/')),'Forbidden path')
    require(not any(x in str(p.resolve()).lower() for x in ('_v112','current223','/private_evaluation/')),'Forbidden target')
    return p


def metadata_sha(path):
    p=safe(path);require(p.suffix in ('.json','.jsonl','.py'),'Only metadata/source body SHA')
    return hashlib.sha256(p.read_bytes()).hexdigest()


def bind(path):return dict(path=str(Path(path).resolve()),sha256=metadata_sha(path))
def read(path):return json.loads(safe(path).read_text())
def lines(path):return [json.loads(x) for x in safe(path).read_text().splitlines() if x]
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


assets={}
def stat_asset(path,category,domain,binding=None,context=None,schema=None):
    p=safe(path);require('/val/000001/' not in str(p),'Exclude held scene even from payload stat')
    key=str(p)
    if key not in assets:
        exists=p.exists();info=p.stat() if exists else None
        assets[key]=dict(path=key,exists=exists,is_file=p.is_file() if exists else False,bytes_stat_only=info.st_size if info else None,
            inherited_sha256=(binding or {}).get('sha256'),body_header_or_payload_SHA_read=False,categories=[],domains=[],contexts=[],declared_schema=schema)
    record=assets[key]
    if category not in record['categories']:record['categories'].append(category)
    if domain not in record['domains']:record['domains'].append(domain)
    if context is not None and context not in record['contexts']:record['contexts'].append(context)
    return record


def main():
    scope=read(SCOPE);require(scope['prospective_scope_only'] and scope['no_model_data_architecture_or_threshold_selection_from_real_eval'],'Frozen source-only scope')
    handoff=P/'ACTIVE_HANDOFF_KO.json';active=read(handoff)
    # Follow only the active handoff V153 phase pointer, never other phase/results.
    phase_binding=active['V153_phase_result'];phase=read(phase_binding['path']);require(bind(phase_binding['path'])==phase_binding,'Active V153 pointer')
    intake_binding=phase['bindings']['source_intake'];download_binding=phase['bindings']['download'];policy_binding=phase['bindings']['policy']
    intake=read(intake_binding['path']);download=read(download_binding['path']);policy=read(policy_binding['path'])
    require(all(bind(b['path'])==b for b in (intake_binding,download_binding,policy_binding)),'Source seal metadata immutable')
    selection=read(intake['selection']['path']);rgbmanifest=read(intake['RGB_manifest']['path'])
    cachepath=V181/'source_raw96_cache_v1/COMPLETE.json';cache=read(cachepath)
    pub_binding=cache['domains']['PUBLIC_TRAIN1480']['metadata'];cf_binding=cache['domains']['CF3816']['metadata']
    pub=lines(pub_binding['path']);cf=lines(cf_binding['path'])
    require(len(pub)==1480 and len(cf)==3816 and all(r['split']=='SOURCE_DEVELOPMENT' for r in pub+cf),'Only original sourceTRAIN cache rows')
    require(sum(not r['original_query_available'] for r in pub)==194 and sum(not r['original_query_available'] for r in cf)==360,'All original U retained')
    train_frames={r['original']['frame_index'] for r in pub};require(len(train_frames)==185,'Public185 frames')
    selected=[r for r in selection['frames'] if r['split']=='SOURCE_DEVELOPMENT']
    require({r['frame_index'] for r in selected}==train_frames and {r['scene'] for r in selected}=={'000000','000002','000005','000006'},'Exact original TRAIN split')
    selected_rgb={r['frame_index']:r for r in rgbmanifest['rows'] if r['split']=='SOURCE_DEVELOPMENT'}
    require(set(selected_rgb)==train_frames,'Original RGB manifest same185')
    by_relative={r['relative_path']:r for r in download['files']}
    frame_inventory=[]
    for f in selected:
        sf=f['source_files'];idx=f['frame_index'];rgb=selected_rgb[idx]
        require(rgb['source_id']==f['source_id'] and rgb['source']=='MR6D','Original MR dataset image join')
        items={}
        for category,relative in [('RGB',sf['RGB']),('depth',sf['depth'])]:
            b=by_relative[relative];require(b['completed'] and b['source_content_ID_verified'],'Prior original download provenance')
            items[category]=stat_asset(b['local_path'],'MR_'+category,'PUBLIC_TRAIN1480',b,dict(frame_index=idx,source_id=f['source_id']),
                'RGB uint8 HWC[720,1280,3], pinnedcv2' if category=='RGB' else 'depth PNG uint16[720,1280], scale in scene_camera; declared by intake code')
        mask_records={}
        for source_key,category in [('full_masks','MR_instance_fullmask'),('visible_masks','MR_instance_visiblemask')]:
            mask_records[source_key]=[]
            for m in sf[source_key]:
                b=by_relative[m['path']];stat_asset(b['local_path'],category,'PUBLIC_TRAIN1480',b,dict(frame_index=idx,gt_instance_index=m['gt_instance_index'],obj_id=1),
                    'PNG same image H/W; nonzero foreground, nonexclusive instances; declared intake')
                mask_records[source_key].append(dict(gt_instance_index=m['gt_instance_index'],binding=dict(path=b['local_path'],sha256=b['sha256'])))
        stat_asset(rgb['canonical_RGB_NPY']['path'],'MR_canonical_RGB','PUBLIC_TRAIN1480',rgb['canonical_RGB_NPY'],dict(frame_index=idx),'uint8 HWC[720,1280,3], no header opened')
        frame_inventory.append(dict(frame_index=idx,id=f['id'],source_id=f['source_id'],internal_split='SOURCE_DEVELOPMENT',upstream_split=f['official_split'],
            scene=f['scene'],frame_key=f['frame_key'],obj_id=1,all_obj1_GT_instance_indices=f['obj1_gt_instance_indices'],
            RGB=dict(path=rgb['image'],sha256=rgb['image_sha256']),depth=dict(path=items['depth']['path'],sha256=items['depth']['inherited_sha256']),
            canonical_RGB_NPY=rgb['canonical_RGB_NPY'],**mask_records))
    mesh=by_relative['models/obj_000001.ply'];stat_asset(mesh['local_path'],'MR_nominal_mesh','PUBLIC_TRAIN1480',mesh,schema='PLY obj1 nominal800x1200x144mm wood Euro pallet; part annotation not declared')
    projection_binding=phase['bindings']['source_references'];projection=read(projection_binding['path'])
    for scene in sorted({r['scene'] for r in selected}):
        for category,b in projection['source_metadata'][scene].items():
            stat_asset(b['path'],'MR_'+category+'_metadata','PUBLIC_TRAIN1480',b,dict(scene=scene),
                'scene_gt: frame->[obj_id,cam_R_m2c,cam_t_m2c]; scene_camera: frame->cam_K/depth_scale; source code declaration')
        gt=Path(projection['source_metadata'][scene]['GT']['path'])
        stat_asset(gt.parent/'scene_gt_info.json','MR_GT_info_metadata','PUBLIC_TRAIN1480',context=dict(scene=scene),schema='BOP per-frame GT instance info; path directly constructed by pinned intake code')
    stat_asset(projection['models_info']['path'],'MR_nominal_models_info','PUBLIC_TRAIN1480',projection['models_info'],schema='models_info[1]: min_x/y/z,size_x/y/z; declared producer dimensions800,1200,144mm')
    for r in pub:stat_asset(r['raw_packet']['path'],'MR_factual96_packet','PUBLIC_TRAIN1480',r['raw_packet'],dict(frame_index=r['original']['frame_index']),
        'rgb96 uint8[8,96,96,3];valid96 bool[8,96,96];global_index int32[8];crop_constructed bool[8], source producer schema')
    for r in cf:
        m=r['original'];require(m['split']=='SOURCE_DEVELOPMENT' and m['original_frame_index'] in train_frames,'Every CF source is TRAIN')
        stat_asset(r['raw_packet']['path'],'CF_variant_payload','CF3816',r['raw_packet'],dict(pair_id=m['pair_id'],variant=m['variant']),
            'V161 coherent full_RGB/rgb96/valid96/central32/target224/q8/L8/native/flags/pasted_alpha; no depth/partmap/triangle/rendered3Dpose key')
    for domain in ('PUBLIC_TRAIN1480','CF3816'):
        for kind in ('inputs','targets'):
            b=cache['domains'][domain][kind];stat_asset(b['path'],'V181_'+kind,domain,b,schema='INPUTS raw10 separate from TARGETS5; no NPZ header/body read')
    cfcomplete=read(V161/'source_pixels_v1/COMPLETE.json');assign=lines(cfcomplete['assignments']['path']);donorcomplete=read(cfcomplete['donor_proof_complete']['path']);donors=lines(donorcomplete['records']['path'])
    require(len(assign)==159 and len(donors)==59 and all(x['split']=='SOURCE_DEVELOPMENT' for x in assign) and all(x['split']=='TRAIN' for x in donors),'All fixed source anchors/donors')
    donor_ids={x['donor_id'] for x in donors};require({a['donor_id'] for a in assign}==donor_ids,'Exact59 fixed donor IDs reused')
    for r in donors:
        for category,key in [('YCB_original_RGB','original_RGB'),('YCB_compact_RGBA','compact_RGBA_cutout'),('YCB_exclusive_alpha_proof','native_alpha_proof')]:
            stat_asset(r[key]['path'],category,'CF3816',r[key],dict(donor_id=r['donor_id']),r['runtime_cutout_contract'] if key=='compact_RGBA_cutout' else None)
        inv=r['complete_mask_inventory'];stat_asset(inv['scene_gt']['path'],'YCB_scene_GT_metadata','CF3816',inv['scene_gt'],dict(scene_id=inv['scene_id']),
            'BOP frame GT instance/object pose list; no per-frame body read')
        for ins in inv['instances']:stat_asset(ins['mask']['path'],'YCB_all_instance_visiblemask','CF3816',ins['mask'],dict(donor_id=r['donor_id'],instance_id=ins['instance_id'],object_id=ins['object_id']),
            'uint8 0/255 visible instance mask per donor proof; not pallet native part map')
    counts=Counter(c for r in assets.values() for c in r['categories'])
    missing=[r['path'] for r in assets.values() if not r['exists'] or not r['is_file']]
    unique_anchors={a['anchor_query_id']:a for a in assign};anchor_slots=Counter(a['native_slot'] for a in unique_anchors.values())
    require(len(unique_anchors)==53 and len({a['frame_index'] for a in assign})==46 and len({r['raw_packet']['path'] for r in cf})==477,'Original fixed CF scale retained')
    metadata_paths=[SCOPE,Path(__file__),handoff,Path(phase_binding['path']),Path(intake_binding['path']),Path(download_binding['path']),Path(policy_binding['path']),Path(intake['selection']['path']),
        Path(intake['RGB_manifest']['path']),cachepath,Path(pub_binding['path']),Path(cf_binding['path']),Path(projection_binding['path']),V181/'raw_cache_preparation/build_source_raw96_cache.py',
        V181/'raw_cache_preparation/PROTOCOL_BEFORE_RAW_SAM_AND_TARGET_BODIES.json',V153/'full_intake_preparation/intake_full_source.py',V153/'native_reference_preparation/project_all_source_native_references.py',
        V161/'source_pixels_v1/COMPLETE.json',Path(cfcomplete['assignments']['path']),Path(cfcomplete['donor_proof_complete']['path']),Path(donorcomplete['records']['path']),
        V161/'pixel_builder_preparation/PROTOCOL_BEFORE_SOURCE_PIXEL_BODIES.json']
    assetpath=H/'LOCAL_SOURCE_TRAIN_ASSET_STAT_INVENTORY.jsonl'
    with assetpath.open('x') as f:
        for record in sorted(assets.values(),key=lambda x:x['path']):f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
    framepath=H/'MR185_SOURCE_TRAIN_FRAME_PATHS.jsonl'
    with framepath.open('x') as f:
        for record in frame_inventory:f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
    report=dict(completed=True,PASS=True,audit_type='Direct-lineage sourceTRAIN metadata/code+file stat only, asset bodies/headers/hashes not opened',scope=bind(SCOPE),
        source_pointer_lineage=dict(active_handoff=bind(handoff),used_field='V153_phase_result -> bindings.source_intake/download/policy/source_references only',V153_phase=phase_binding),
        metadata_and_code_bindings=[bind(x) for x in metadata_paths],
        source_cache=bind(cachepath),all_original_rows_retained=dict(PUBLIC_TRAIN1480=1480,CF3816=3816,PUBLIC_inputU194=194,CF_inputU360=360,native8_preserved=True),
        domains=dict(PUBLIC_TRAIN1480=dict(dataset='MR6D, not YCB-V',frames185=185,upstream_split='val',internal_split='SOURCE_DEVELOPMENT',scenes=sorted({r['scene'] for r in selected}),
            held_scene000001_excluded=True,source_commit=policy['source_commit'],official_source=policy['official_source'],selected_object_id=1,
            selection_rule=policy['selection_rule'],GT_instance_observations=sum(len(r['obj1_gt_instance_indices']) for r in selected),
            RGB_depth_fullmask_visiblemask_GTpose_camera_mesh_paths_present=not missing,
            owner_known253=cache['domains']['PUBLIC_TRAIN1480']['owner_hard_counts'],support_targets_all_UNKNOWN=True,
            schema=dict(RGB='uint8HWC720x1280',depth='uint16PNG with camera depth_scale',GT_pose='cam_R_m2c/cam_t_m2c; p_cam=R p_obj+t',camera='cam_K',
                instance='BOP GT-list index and per-instance full/visible masks; instances may overlap',partmap='NOT_DECLARED_IN_REVIEWED_REAL_INTAKE',
                triangle_map='NOT_DECLARED_IN_REVIEWED_REAL_INTAKE',mesh='obj_000001.ply nominal wood Euro-pallet',selected_query_to_GT_instance='NOT_ESTABLISHED_BY_INPUT_ROW_CACHE',
                native_correspondence='Provided pose+models_info+V74 dynamic normal-face permutation creates nominal8 CAD references; not certified physical native/part correspondences and no exact query->GTinstance match.'),
            exact_reference_paths=dict(projection_complete=projection_binding,GT_metadata_SOURCE_TRAIN={s:projection['source_metadata'][s] for s in sorted({r['scene'] for r in selected})},models_info=projection['models_info'],nominal_mesh=dict(path=mesh['local_path'],sha256=mesh['sha256'])),
            limits=['Actual source is nominal Euro WOOD; no real SQUARE supervision established by this intake.',
                'Available full/visible instance masks and sensor depth do not themselves annotate the intended native corner or selected-part visibility/external/self state.',
                'q8 proposal is an input, not GT instance authority; multiple obj1 GT instances are retained. Pixel/depth/pose registration and intended-part association need separate source-only proof before new supervision.']),
            CF3816=dict(dataset='Controlled MR6D-background + YCB-Video train_real foreground composite, not standalone real pallet GT',pairs159=159,payloads477=477,rows3816=3816,
                anchors53=53,anchor_frames46=46,anchor_native_slot_counts=dict(anchor_slots),missing_anchor_native_slots=[s for s in range(8) if s not in anchor_slots],donors59=59,
                variants=dict(Counter(r['original']['variant'] for r in cf)),pair_proof_pass=cfcomplete['passed_pixel_input_proofs'],pair_proof_fail_retained=cfcomplete['failed_attempts_retained'],
                known_owner314=cache['domains']['CF3816']['owner_hard_counts'],support_targets_all_UNKNOWN=True,new_native_and_quality_minus1=True,
                donor_authority='YCB original TRAIN RGB + selected mask_visib minus union of all other instance masks, erode1 alpha; controlled paste opacity proof only.',
                donor_selected_object_ids=sorted({int(a['donor']['object_id']) for a in assign}),YCB_is_occluder_not_pallet_target=True,
                stored_pasted_alpha=True,fullRGB_and_all8_original_qL_flags_retained=True,
                no_composited_depth_partmap_triangle_or_3Dpose_declared=True,query_anchor_authority='Original assistantweak T center observation, not native-part visible/physical3px gold.',
                donor_GTpose_paths='Present scene_gt in donor lineage, not a pose for the 2D resized/translated composite. Donor depth/mesh/camera paths are not declared in the reviewed proof and were not guessed/searched.',
                nativepart_correspondence='No selected pallet part mask or native end region is certified by the owner253/CF314 metadata.',
                original_training_admission_flags_preserved=True)),
        synthetic_distinction='SYNTH_TRAIN4096 is a separate procedural scene/source raster causal supervision domain. Its part/depth/triangle authority is not evidence that MR6D or YCB composites contain equivalent real native-part labels.',
        payload_stat_inventory=bind(assetpath),MR185_frame_paths=bind(framepath),unique_assets_stat=len(assets),counts_by_declared_asset_category=dict(counts),missing_paths=missing,
        unconfirmed=['Public q8/native slot -> selected GT instance -> intended physical pallet end/part correspondence.',
            'Validity of measured/rendered depth for exact intended-part external vs self separation; no depth/mask/mesh/pose body was read in this inventory.',
            'Donor depth/mesh/camera outside the explicitly reviewed lineage; no filesystem-wide absence claim.',
            'Real SQUARE/native8 direct part supervision remains unestablished by the EuroWOOD-only public source.'],
        authority='Inventory confirms paths/provenance and producer-declared schemas only, not current pixel correctness, new gold, new supervision or training admission.',
        access=dict(broad_recursive_searches=0,actual_pixel_mask_depth_pose_mesh_or_NPZ_body_header_or_payload_hash_reads=0,held_VAL_current_payload_stat_or_body_reads=0,
            source_code_or_manifest_only=True,V112_read_hash_search=0,model_forward_fit_download_crop_or_label_generation=0),
        prior_labels_queries_or_admission_changed=False,new_training_admission=False,selected_model=None,fully_solved=False)
    write(H/'REVIEW.json',report)
    print(json.dumps(dict(completed=True,PASS=True,review=bind(H/'REVIEW.json'),counts=dict(counts),missing=len(missing))))


if __name__=='__main__':main()
