"""Existing official metadata/README only. No network, payload or model access."""
from pathlib import Path,PurePosixPath
from collections import Counter,defaultdict
import hashlib,json,re

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OFFICIAL=ROOT/'OFFICIAL_DATAVERSE_DATASET_METADATA.json'
README=ROOT/'OFFICIAL_README_SOURCE.txt'
PRIOR=ROOT/'source_annotation_contract_review_b/REVIEW.json'

def b(p):return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
    d=json.loads(OFFICIAL.read_text())['data'];v=d['latestVersion'];prior=json.loads(PRIOR.read_text())
    assert b(OFFICIAL)=={k:prior['inputs'][OFFICIAL.name][k] for k in ('path','sha256')}
    assert b(README)=={k:prior['inputs'][README.name][k] for k in ('path','sha256')}
    assert hashlib.md5(README.read_bytes()).hexdigest()=='477a6514dc7ce51e504a6c984de34f0b'
    files=[];by=defaultdict(dict);groupaux=defaultdict(dict);counts=Counter();sizes=Counter();ids=set();paths=set();hashgroups=defaultdict(list)
    for f in v['files']:
        z=f['dataFile'];path=str(PurePosixPath(f.get('directoryLabel',''))/z['filename']).lstrip('/')
        assert z['id'] not in ids and path not in paths
        ids.add(z['id']);paths.add(path)
        record=dict(official_file_id=z['id'],relative_path=path,declared_bytes=z['filesize'],content_type=z['contentType'],declared_checksum=z['checksum'],restricted=f['restricted'],
                    download_API_URL='https://rdr.kuleuven.be/api/access/datafile/'+str(z['id']),download_URL_method='Constructed standard official Dataverse datafile endpoint from registered file ID; no HTTP request made',payload_SHA256_not_read=None)
        files.append(record);parts=path.split('/');modality=None
        if len(parts)==3 and parts[1] in ('rgb','depth','masks','annotated_poses'):
            modality=parts[1];pattern=r'(\d{4})_sam1_mask\.png' if modality=='masks' else r'(\d{4})\.'+('txt' if modality=='annotated_poses' else 'png')
            m=re.fullmatch(pattern,parts[2]);assert m,(path,'Unexpected official filename')
            key=(parts[0],m.group(1));assert modality not in by[key];by[key][modality]=record
        elif len(parts)==2 and parts[1] in ('cam_K.txt','visibility_scores.csv'):
            modality='calibration' if parts[1]=='cam_K.txt' else 'visibility_csv';groupaux[parts[0]][modality]=record
        elif path.startswith('assets/'):modality='mesh_asset'
        elif path=='README.md':modality='readme'
        else:raise AssertionError(path)
        counts[modality]+=1;sizes[modality]+=z['filesize'];hashgroups[z['checksum']['value']].append(path)
    assert counts==dict(rgb=322,depth=322,masks=290,annotated_poses=284,calibration=6,visibility_csv=6,mesh_asset=3,readme=1)
    frames=[];groups={}
    for split in sorted({k[0] for k in by}):
        records=[(k,fs) for k,fs in sorted(by.items()) if k[0]==split]
        group=dict(acquisition_group=split,**groupaux[split],registered_counts={m:sum(m in fs for k,fs in records) for m in ('rgb','depth','masks','annotated_poses')},
                   registered_RGB_mask_intersection=sum('rgb' in fs and 'masks' in fs for k,fs in records),registered_all4_intersection=sum(all(m in fs for m in ('rgb','depth','masks','annotated_poses')) for k,fs in records))
        groups[split]=group
        for (s,frame),fs in records:
            assert s==split and 'rgb' in fs and 'depth' in fs
            frames.append(dict(source_id=split+':'+frame,acquisition_group=split,frame_id=frame,files={m:fs.get(m) for m in ('rgb','depth','masks','annotated_poses')},
                RGB_mask_registered='masks' in fs,all4_registered=all(m in fs for m in ('rgb','depth','masks','annotated_poses')),
                original_initializer_query8=None,original_query_L=None,original_input_availability='NOT_RUN_OR_READ_IN_THIS_AUDIT',training_admission=False))
    assert len(frames)==322 and sum(r['RGB_mask_registered'] for r in frames)==290 and sum(r['all4_registered'] for r in frames)==284
    assert {s:g['registered_counts'] for s,g in groups.items()}=={s:{'rgb':p['RGB'],'depth':p['depth'],'masks':p['mask'],'annotated_poses':p['pose']} for s,p in prior['six_split_filename_joins'].items()}
    manifest=HERE/'ALL322_REGISTERED_FRAME_BINDINGS_METADATA_ONLY.jsonl'
    manifest.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in frames))
    review=dict(completed=True,PASS=True,PASS_scope='Independent official-metadata filename/count/URL/byte readiness only; no payload validity, strong physical label certification or training admission',
        code=b(Path(__file__)),inputs={'official_metadata':b(OFFICIAL),'official_README':b(README),'V179_metadata_scope':b(ROOT/'ROOT_METADATA_SCOPE_BEFORE_OFFICIAL_SOURCE_RESPONSE.json'),'prior_independent_contract':b(PRIOR)},
        official=dict(DOI='10.48804/DJLDFS',dataset_URL='https://doi.org/10.48804/DJLDFS',dataset_id=d['id'],version=[v['versionNumber'],v['versionMinorNumber']],releaseTime=v['releaseTime'],license=v['license'],all_registered_files_unrestricted=all(not f['restricted'] for f in files)),
        counts=dict(registered_RGB=322,registered_depth=322,registered_masks=290,registered_poses=284,RGB_depth_mask_registered_intersection=290,all4_registered_intersection=284,RGB_without_mask=32,RGB_without_pose=38,mask_with_no_pose=6,pose_with_no_mask=0),
        declared_bytes_by_modality=dict(sizes),declared_total_file_bytes=sum(sizes.values()),registered_modalities=dict(counts),acquisition_groups=groups,all322_metadata_manifest=b(manifest),
        declared_repeated_checksum_groups=sum(len(x)>1 for x in hashgroups.values()),declared_checksum_repeat_not_payload_duplicate_proof=True,
        resolution_and_units=dict(README_RGB_HW=[480,640],decoded_dimensions_not_verified=True,depth='Declared uint16 PNG millimeters; not read',pose='Declared 4x4 SE3 row-major, translations in meters; not read'),
        split_contract=dict(official_groups='Six cameraheight x hall/lab acquisition groups, not an established ML TRAIN/VAL/TEST allocation',prospective_split_required_before_any_payload_or_result_selection=True,frame_correlation_and_duplicate_pose_checksums_retained=True,split_frame_is_key_not_basename=True),
        camera_convention=dict(conflicting_README_phrase='camera-to-object',explicit_README_name='ob_in_cam',explicit_equation='p_cam = T @ p_obj',official_metadata_phrase='object-in-camera',operational_pose_direction='UNRESOLVED; not applied',all6_calibration_declared_same_MD5=len({g['calibration']['declared_checksum']['value'] for g in groups.values()})==1,pose_not_required_for_binary_literal_owner_raster_auxiliary=True),
        mask_authority=dict(README_definition='Binary PNG: white255 Pallox pallet bin, black0 background',scope='Whole bin instance, not selected endpoint/native8/flat pallet part',sam1_filename_not_human_QA_proof=True,human_review_generation_visible_vs_amodal_and_occluder_boundary_authority='UNKNOWN_NOT_VERIFIED',binary_non_target_cannot_separate_FOREIGN_from_BACKGROUND=True,whole_mesh_visibility_CSV_not_native_or_query_owner_gold=True,mask_dtype_values_shape_RGB_alignment_and_CSV_keys_not_read=True),
        source_selection_origin='Inherited direct human confirmation: privately captured223 are unrelated to this previously named KU Leuven public source. No private/held outcomes were read or used to choose this source.',
        owner_auxiliary_readiness=dict(status='CONDITIONAL_NOT_ADMITTED',potential_registered_RGB_mask_pairs=290,pose_complete_subset_not_mandatory_for_owner_only=284,
            certifiable_now='Official registration, license text, filenames/declared sizes/checksums and explicit limitations only',
            conditional_supervision='After prospective intake and payload alignment/visible-mask/instance authority checks, sample provided target mask at the ORIGINAL frozen initializer q/L grid for T-v-nonT raster auxiliary. Keep boundary/padding/unknown and every original failure row. This is mask-defined membership, not physical native visibility.',
            still_unknown='FOREIGN-v-BG in natural images; end-part identity/visibility; selected-instance association; physical position error and native8 correspondence. No strong human/pixel ownership certification is available from metadata alone.'),
        controlled_intervention_readiness=dict(status='FEASIBLE_IN_PRINCIPLE_AFTER_SOURCE_AUTHORITY_CHECKS_NOT_EXECUTED',
            prerequisites=['Prospectively freeze source-only split and all322-frame census; preserve missingmask32 and original initializer-U/OOF/noDetection rows.',
                'Run unchanged original fullRGB initializer to obtain q8/L/native ordering; do not substitute GT crop centres, GTbox prompts, nearestGT or GT-bestnative/permutation.',
                'For actual controlled opaque donor occlusion, BEFORE must have verified target material at the SAME sampled q/crop region. Foreign pixels alone do not prove that the target was occluded.',
                'Use source-authorized foreground cutout/opaque alpha and preserve transform/provenance. Retain failed/unknown attempts, all original q/L and BEFORE pixel parity.',
                'ADJACENT_SHAM uses the same donor transform but its center/crop ownership must be measured, not assumed; it may change q-owner. Record changed pixels reaching raw96 versus context separately.',
                'Any generated owner or crop-presence supervision remains separate from original nativeD/E/S, coordinate quality and fixed-end support. No native/quality labels are created from whole-bin masks.'],
            no_new_generation_download_forward_or_fit=True),
        exact_future_source_paths=dict(official_manifest=str(OFFICIAL),frame_binding_manifest=str(manifest),RGB_pattern='{group}/rgb/{frame}.png',mask_pattern='{group}/masks/{frame}_sam1_mask.png',pose_pattern='{group}/annotated_poses/{frame}.txt',intrinsics_pattern='{group}/cam_K.txt',visibility_pattern='{group}/visibility_scores.csv',retrieval='Use declared official_file_id API URLs in metadata manifest only after separate root entry; signed README URL is not a durable asset binding'),
        local_payload_presence_or_existing_subset_query_alignment='NOT_SCANNED_OR_VERIFIED; this audit uses existing local metadata only, not a project-wide payload inventory',
        new_payload_download_hash_or_read=0,new_feature_weight_score_private_held_or_V112_access=0,new_forward_fit_label_or_threshold_change=0,training_admission=False,selected_model=None,adopted=False,fully_solved=False)
    write(HERE/'REVIEW.json',review)
    print(json.dumps({'review':b(HERE/'REVIEW.json'),'manifest':b(manifest),'counts':review['counts'],'declared_bytes_by_modality':dict(sizes)},ensure_ascii=False))

if __name__=='__main__':main()
